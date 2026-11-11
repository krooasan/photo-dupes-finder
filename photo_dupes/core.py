"""
Всё, кроме окна: ищем картинки и видео, превращаем каждую в «отпечаток» SSCD (512 чисел)
через ONNX Runtime, сравниваем отпечатки, собираем группы копий, выбираем оригинал,
переносим / удаляем. Torch для работы не нужен.
"""
from __future__ import annotations

import csv
import os
import shutil
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC с айфонов; без пакета такие файлы просто не откроются и попадут в ошибки
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = None  # огромные панорамы и длинные скриншоты тоже открываем

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".wmv"}
EXTS = IMAGE_EXTS | VIDEO_EXTS
KIND_EXTS = {"photo": IMAGE_EXTS, "video": VIDEO_EXTS, None: EXTS}
# сколько кадров берём из видео: по одному на 5 секунд, но от 8 до 32
VIDEO_FRAMES_MIN, VIDEO_FRAMES_MAX, VIDEO_SEC_PER_FRAME = 8, 32, 5
DUP_DIR_NAME = "Дубли"
CACHE_NAME = ".photo_dupes_cache.npz"
CACHE_VERSION = 2
MODEL_NAME = "sscd_disc_mixup.onnx"
PAIR_FLOOR = 0.6   # ниже ползунок не опускается, поэтому более слабые пары не храним
TOP_K = 64         # сколько самых похожих соседей помним для каждого файла
BARS_RATIO = 0.9   # если реальная картинка занимает меньше 90% кадра — значит, есть рамки/полосы
MAX_SIDE = 1728    # длинная сторона входа в модель; иначе скриншот 1080×30000 не влезет в видеопамять
CHUNK = 64         # сколько файлов декодируем за раз (иначе готовые картинки копятся в памяти)
# Декодер видео (особенно 4K) сам держит сотни мегабайт буферов. 8 таких одновременно — это 3,5 ГБ,
# поэтому видео читаем не больше чем по два за раз; фото по-прежнему идут во все потоки.
_video_slots = threading.Semaphore(2)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

Progress = Callable[[str, int, int], None]  # (этап, сделано, всего)


def resource_dir() -> Path:
    """Папка с файлами программы: внутри exe (PyInstaller) или корень проекта."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def find_model() -> Path:
    for p in (resource_dir() / "models" / MODEL_NAME,
              Path(sys.executable).parent / "models" / MODEL_NAME):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Не найдена модель {MODEL_NAME}. Положите её в папку models/ "
        f"(или выполните tools/export_onnx.py).")


# ----------------------------------------------------------------------------- файлы
def _hidden(st: os.stat_result, name: str) -> bool:
    attrs = getattr(st, "st_file_attributes", 0)  # есть только на Windows
    return name.startswith(".") or bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))


def collect_files(folder: Path, recursive: bool, kind: str | None = None) -> list[Path]:
    """Фото (kind="photo"), видео (kind="video") или всё сразу (None). Скрытые и системные файлы,
    пустые файлы, Корзину ($RECYCLE.BIN), «System Volume Information» и нашу папку «Дубли» пропускаем."""
    exts = KIND_EXTS[kind]
    out = []
    # onerror=None: папки без доступа просто молча пропускаются, а не роняют поиск
    for root, dirs, names in os.walk(folder, onerror=None):
        if recursive:
            dirs[:] = [d for d in dirs if not (d.startswith((".", "$")) or d == DUP_DIR_NAME
                                               or d == "System Volume Information")]
        else:
            dirs[:] = []
        for name in names:
            if Path(name).suffix.lower() not in exts:
                continue
            p = Path(root) / name
            try:
                st = p.stat()
            except OSError:  # файл удалили прямо во время обхода или нет доступа
                continue
            if st.st_size == 0 or _hidden(st, name):
                continue
            out.append(p)
    return sorted(out)


def file_sig(p: Path) -> tuple[int, int] | None:
    """(время изменения, размер) — по ним понимаем, что файл не менялся и можно взять из кэша.
    None, если файла уже нет."""
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def video_frames(p: Path, n: int | None = None) -> tuple[list[Image.Image], int, int]:
    """n кадров, равномерно по всему видео (самое начало и конец пропускаем — там часто
    чёрный экран), плюс размер видео."""
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # не сыпать в консоль ругань на битые видео
    import cv2  # грузим только если в папке есть видео, библиотека тяжёлая

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise ValueError("не удалось открыть видео")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n is None:  # длиннее видео — больше кадров, чтобы обрезанная копия всё равно совпала
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            n = int(min(VIDEO_FRAMES_MAX, max(VIDEO_FRAMES_MIN, count / fps / VIDEO_SEC_PER_FRAME)))
        positions = [int(count * (i + 1) / (n + 1)) for i in range(n)] if count > n else [None]
        frames = []
        for pos in positions:
            if pos is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, f = cap.read()
            if ok:
                # Сразу уменьшаем: 32 кадра 4K — это ~800 МБ, а таких видео декодируется 8 одновременно
                # (так однажды улетело 6 ГБ памяти). Модели всё равно нужно только 288 px.
                short = min(f.shape[:2])
                if short > 576:
                    k = 576 / short
                    f = cv2.resize(f, (round(f.shape[1] * k), round(f.shape[0] * k)), interpolation=cv2.INTER_AREA)
                frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError("в видео нет читаемых кадров")
        return frames, w, h
    finally:
        cap.release()


def open_picture(p: Path, max_side: int | None = None) -> Image.Image:
    """Картинка файла в RGB: сама фотка или средний кадр видео."""
    if is_video(p):
        frames, _, _ = video_frames(p, 1)
        im = frames[0]
    else:
        with Image.open(p) as src:
            if max_side:
                src.draft("RGB", (max_side, max_side))  # JPEG сразу декодируется уменьшенным — сильно быстрее
            im = ImageOps.exif_transpose(src).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def _to_input(im: Image.Image) -> np.ndarray:
    """Картинка -> вход модели.

    Должно в точности совпадать с тем, на чём учили SSCD (torchvision Resize(288) + Normalize),
    иначе отпечатки «поплывут». Проверял против torchvision — массивы совпадают байт в байт.
    Короткая сторона становится 288, поэтому у картинок с разными пропорциями разный размер.
    """
    iw, ih = im.size
    s = 288
    if iw <= ih:
        nw, nh = s, int(s * ih / iw)
    else:
        nw, nh = int(s * iw / ih), s
    if max(nw, nh) > MAX_SIDE:  # очень длинный скриншот: ужимаем, чтобы не съесть всю видеопамять
        k = MAX_SIDE / max(nw, nh)
        nw, nh = max(1, int(nw * k)), max(1, int(nh * k))
    a = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((a.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)


def preprocess(p: Path):
    """Всё, что нужно от файла для поиска: вход модели ((3, H, W) для фото,
    (кадры, 3, H, W) для видео), размер и доля «реальной» картинки без рамок."""
    if is_video(p):
        with _video_slots:
            frames, w, h = video_frames(p)
        return np.stack([_to_input(f) for f in frames]), w, h, content_ratio(frames[len(frames) // 2])
    with Image.open(p) as im:
        w, h = im.size
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return _to_input(rgb), w, h, content_ratio(rgb)


# ----------------------------------------------------------------------------- кэш
def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d["version"]) != CACHE_VERSION:
                return {}
            # ratio появился позже: в старых кэшах его нет, тогда досчитаем при поиске
            ratios = d["ratio"] if "ratio" in d.files else np.full(len(d["paths"]), np.nan)
            return {str(p): {"sig": (int(s[0]), int(s[1])), "w": int(wh[0]), "h": int(wh[1]),
                             "emb": e.astype(np.float32), "ratio": float(r)}
                    for p, s, wh, e, r in zip(d["paths"], d["sigs"], d["wh"], d["emb"], ratios)}
    except Exception:  # битый кэш — не страшно, просто посчитаем заново
        return {}


def save_cache(path: Path, items: dict):
    if not items:
        return
    keys = list(items)
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(tmp, version=np.array(CACHE_VERSION),
                 paths=np.array(keys, dtype=str),
                 sigs=np.array([items[k]["sig"] for k in keys], dtype=np.int64),
                 wh=np.array([(items[k]["w"], items[k]["h"]) for k in keys], dtype=np.int64),
                 emb=np.stack([items[k]["emb"] for k in keys]).astype(np.float32),
                 ratio=np.array([items[k].get("ratio", np.nan) for k in keys], dtype=np.float32))
        os.replace(tmp, path)  # пишем во временный файл и подменяем — так кэш не побьётся при вылете
    except OSError:  # папка только для чтения и т.п. — работаем без кэша
        return
    if os.name == "nt":
        try:  # прячем кэш, чтобы не мозолил глаза в Проводнике
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)
        except Exception:
            pass


# ----------------------------------------------------------------------------- модель
class Embedder:
    def __init__(self, use_gpu: bool = True):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # DirectML работает на любой видеокарте с DX12 (NVIDIA / AMD / Intel). Он не умеет
        # memory pattern и параллельное выполнение — в доке onnxruntime просят их выключить.
        providers = ["CPUExecutionProvider"]
        if use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(find_model()), so, providers=providers)
        except Exception:  # кривой драйвер видеокарты и т.п. — откатываемся на процессор
            if providers[0] == "CPUExecutionProvider":
                raise
            self.sess = ort.InferenceSession(str(find_model()), so, providers=["CPUExecutionProvider"])
        self.device = "gpu" if self.sess.get_providers()[0] == "DmlExecutionProvider" else "cpu"

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        e = self.sess.run(None, {"image": batch})[0].astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------- поиск
@dataclass
class ScanResult:
    files: list[Path]
    wh: np.ndarray       # (n, 2) ширина и высота
    emb: np.ndarray      # (n, 512) отпечатки, длина каждого = 1
    ratio: np.ndarray    # (n,) доля кадра без рамок и полос
    size: np.ndarray     # (n,) размер файла в байтах
    mtime: np.ndarray    # (n,) время изменения, секунды
    # похожие пары тремя массивами: i < j и их сходство. Массивы, а не словарь {(i, j): s} —
    # словарь Python на сотнях тысяч пар весит в десятки раз больше.
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_s: np.ndarray
    errors: list[tuple[Path, str]] = field(default_factory=list)
    device: str = ""
    seconds: float = 0.0

    def sim(self, a: int, b: int) -> float:
        # отпечатки нормированы, поэтому скалярное произведение = косинусное сходство
        return 1.0 if a == b else float(self.emb[a] @ self.emb[b])


def scan(folder: Path, recursive: bool, progress: Progress, cancel: Callable[[], bool],
         use_gpu: bool = True, batch_size: int = 16, kind: str | None = None) -> ScanResult:
    t0 = time.time()
    progress("files", 0, 0)
    files = collect_files(folder, recursive, kind)
    cache_path = folder / CACHE_NAME
    cache = load_cache(cache_path)
    sigs = {p: file_sig(p) for p in files}
    todo = [p for p in files if sigs[p] is not None
            and ((c := cache.get(str(p))) is None or c["sig"] != sigs[p])]

    errors: list[tuple[Path, str]] = []
    device = "cache"
    if todo:
        progress("model", 0, len(todo))
        emb = Embedder(use_gpu)
        device = emb.device
        progress("analyze", 0, len(todo))
        # В одну пачку для модели можно сложить только картинки одного размера, а после
        # preprocess у них разные пропорции. Поэтому раскладываем по «корзинам» по размеру
        # и запускаем корзину, когда в ней набралось 16 штук (скриншоты с одного монитора
        # все одного размера, так что пачки почти всегда полные).
        buckets: dict[tuple, list] = {}
        done = 0
        last_save = time.time()

        def put(p, w, h, ratio, e):
            cache[str(p)] = {"sig": sigs[p], "w": w, "h": h, "emb": e, "ratio": ratio}

        def flush(key):
            nonlocal done
            items = buckets.pop(key)
            try:
                out = emb(np.stack([it[1] for it in items]))
                for (p, _, w, h, r), e in zip(items, out):
                    put(p, w, h, r, e)
            except Exception:
                # пачка не прошла (например, не хватило видеопамяти) — пробуем по одному,
                # чтобы один кривой файл не утянул за собой всю пачку
                for p, arr, w, h, r in items:
                    try:
                        put(p, w, h, r, emb(arr[None])[0])
                    except Exception as ex:
                        errors.append((p, f"{type(ex).__name__}: {ex}"))
            done += len(items)

        def load(p):
            try:
                return p, *preprocess(p), ""
            except Exception as e:  # битый файл, обрезанная загрузка и т.п. — пропускаем, не падаем
                return p, None, 0, 0, 1.0, f"{type(e).__name__}: {e}"

        # Декодирование PNG/JPEG — самое медленное, поэтому идёт в потоках, пока видеокарта считает.
        # Файлы отдаём порциями по CHUNK: если отдать все сразу, потоки декодируют быстрее,
        # чем считает модель, и готовые картинки (~1,7 МБ каждая) копятся в памяти гигабайтами.
        with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as pool:
            for c0 in range(0, len(todo), CHUNK):
                if cancel():
                    break
                for p, arr, w, h, ratio, err in pool.map(load, todo[c0:c0 + CHUNK]):
                    if err:
                        errors.append((p, err))
                        done += 1
                    elif arr.ndim == 4:
                        # видео: все кадры — одна пачка, отпечаток видео — их среднее.
                        # Так пересжатая или обрезанная по времени копия всё равно оказывается рядом.
                        try:
                            e = emb(arr).mean(axis=0)
                            put(p, w, h, ratio, e / max(np.linalg.norm(e), 1e-12))
                        except Exception as ex:
                            errors.append((p, f"{type(ex).__name__}: {ex}"))
                        done += 1
                    else:
                        key = arr.shape
                        buckets.setdefault(key, []).append((p, arr, w, h, ratio))
                        if len(buckets[key]) >= batch_size:
                            flush(key)
                        elif sum(len(v) for v in buckets.values()) > 4 * batch_size:
                            # много картинок разных размеров: не держим их все, запускаем самую полную корзину
                            flush(max(buckets, key=lambda k: len(buckets[k])))
                    # счётчик двигаем после каждого файла: порция из 64 файлов с видео
                    # может считаться 5-10 секунд, и без этого кажется, что всё зависло
                    progress("analyze", done + sum(len(v) for v in buckets.values()), len(todo))
                if time.time() - last_save > 30:  # чтобы «Стоп» или вылет не обнулили работу
                    save_cache(cache_path, cache)
                    last_save = time.time()
            for key in list(buckets):
                if cancel():
                    break
                flush(key)
                progress("analyze", done, len(todo))
        save_cache(cache_path, cache)
        if cancel():
            raise InterruptedError

    files = [p for p in files if sigs[p] is not None and str(p) in cache and cache[str(p)]["sig"] == sigs[p]]
    n = len(files)
    items = [cache[str(p)] for p in files]
    E = np.stack([c["emb"] for c in items]).astype(np.float32) if n else np.zeros((0, 512), np.float32)
    pi, pj, ps = _similar_pairs(E, progress)

    # В старых кэшах нет ratio. Досчитываем только для файлов, у которых вообще есть пары, —
    # остальным оригинал выбирать не придётся.
    ratio = np.array([c.get("ratio", np.nan) for c in items], dtype=np.float32)
    need = np.unique(np.concatenate([pi, pj]))
    need = need[np.isnan(ratio[need])] if len(need) else need
    if len(need):
        for k in need:
            try:
                ratio[k] = content_ratio(open_picture(files[k], 512))
            except Exception:
                ratio[k] = 1.0
            cache[str(files[k])]["ratio"] = float(ratio[k])
        save_cache(cache_path, cache)
    ratio = np.nan_to_num(ratio, nan=1.0)

    return ScanResult(
        files=files,
        wh=np.array([(c["w"], c["h"]) for c in items], dtype=np.int64).reshape(-1, 2),
        emb=E, ratio=ratio,
        size=np.array([c["sig"][1] for c in items], dtype=np.int64),
        mtime=np.array([c["sig"][0] / 1e9 for c in items], dtype=np.float64),
        pair_i=pi, pair_j=pj, pair_s=ps,
        errors=errors, device=device, seconds=time.time() - t0)


def _similar_pairs(E: np.ndarray, progress: Progress):
    """Похожие пары: для каждого файла — до TOP_K самых похожих соседей со сходством >= PAIR_FLOOR.

    Полная матрица n×n на 50 000 фото — это 10 ГБ, поэтому считаем полосами строк и сразу
    выкидываем лишнее. Хранить все пары выше порога нельзя: в папке из 10 000 однотипных
    скриншотов их 50 миллионов (проверял — 2,2 ГБ памяти). Для групп хватает ближайших соседей:
    копии одного снимка всегда друг у друга в топе.
    """
    n = len(E)
    if n < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    k = min(TOP_K, n - 1)
    step = max(256, min(2048, 50_000_000 // n))  # полоса матрицы не больше ~200 МБ
    out_i, out_j, out_s = [], [], []
    for s in range(0, n, step):
        progress("compare", s, n)
        S = E[s:s + step] @ E.T
        rows = np.arange(len(S))
        S[rows, rows + s] = -1  # сам с собой не пара
        nb = np.argpartition(-S, k - 1, axis=1)[:, :k]  # k ближайших без полной сортировки
        sim = np.take_along_axis(S, nb, axis=1)
        keep = sim >= PAIR_FLOOR
        i = np.broadcast_to(rows[:, None] + s, nb.shape)[keep]
        j = nb[keep]
        out_i.append(np.minimum(i, j).astype(np.int32))
        out_j.append(np.maximum(i, j).astype(np.int32))
        out_s.append(sim[keep].astype(np.float32))
    i, j, sim = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)
    # пара A-B могла найтись и у A, и у B — оставляем одну
    _, first = np.unique(i.astype(np.int64) * n + j, return_index=True)
    return i[first], j[first], sim[first]


# ----------------------------------------------------------------------------- выбор оригинала
def _largest_run(mask: np.ndarray):
    """(начало, конец) самой длинной подряд идущей серии True."""
    best, start, se = 0, None, (0, len(mask))
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, se = i - start, (start, i)
            start = None
    return se


def _bar_lines(a: np.ndarray, axis: int) -> np.ndarray:
    """Строки/столбцы, похожие на полосу: почти однотонные и очень тёмные или очень светлые
    (чёрные поля, тёмный интерфейс приложения, белая страница вокруг фото). Пороги подобраны на глаз."""
    mean, std = a.mean(axis=axis), a.std(axis=axis)
    return (std < 6) & ((mean < 45) | (mean > 240))


def content_ratio(im: Image.Image) -> float:
    """Какую долю кадра занимает реальная картинка (без чёрных/белых полос и интерфейса вокруг)."""
    try:
        small = im.convert("L")
        small.thumbnail((512, 512))
        a = np.asarray(small, dtype=np.float32)
        r0, r1 = _largest_run(~_bar_lines(a, axis=1))
        if r1 <= r0:
            return 1.0
        c0, c1 = _largest_run(~_bar_lines(a[r0:r1], axis=0))
        return ((r1 - r0) * (c1 - c0)) / (a.shape[0] * a.shape[1])
    except Exception:
        return 1.0


def original_key(res: ScanResult, k: int):
    """Ключ сортировки: у «лучшей» копии группы он самый большой. Сравниваем по очереди:
    нет рамок/полос -> больше пикселей реальной картинки -> не PNG (обычно скриншот) -> файл больше."""
    r = float(res.ratio[k])
    w, h = res.wh[k]
    return (r >= BARS_RATIO, round(r * int(w) * int(h)), res.files[k].suffix.lower() != ".png", int(res.size[k]))


def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Номер компоненты связности для каждого файла (A~B и B~C -> одна компонента).

    Обычный union-find в цикле Python на десятках миллионов пар идёт минутами, поэтому
    делаем то же самое векторно: каждому узлу раз за разом присваиваем минимальную метку
    среди соседей, пока метки не перестанут меняться.
    """
    labels = np.arange(n)
    if len(i) == 0:
        return labels
    while True:
        old = labels.copy()
        m = np.minimum(labels[i], labels[j])
        np.minimum.at(labels, i, m)
        np.minimum.at(labels, j, m)
        labels = labels[labels]  # «перепрыгиваем» по цепочке меток — сходится за несколько шагов
        if np.array_equal(labels, old):
            return labels


def group(res: ScanResult, threshold: float) -> list[list[int]]:
    """Группы индексов: сначала оригинал, потом остальные по убыванию сходства с ним."""
    n = len(res.files)
    keep = res.pair_s >= threshold
    labels = _components(n, res.pair_i[keep], res.pair_j[keep])
    order = np.argsort(labels, kind="stable")
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    out = []
    for comp in np.split(order, splits):
        if len(comp) < 2:
            continue
        rest = comp
        # Компоненты бывают цепочками: A~B, B~C, но A и C — разные фото (кадры одной серии).
        # Поэтому внутри компоненты оставляем только тех, кто похож на выбранный оригинал,
        # а остальных так же перегруппировываем между собой.
        while len(rest) > 1:
            orig = max(rest, key=lambda k: original_key(res, int(k)))
            sims = res.emb[rest] @ res.emb[orig]
            close = (sims >= threshold) | (rest == orig)
            members = rest[close]
            if len(members) > 1:
                others = sorted((int(k) for k in members if k != orig), key=lambda k: -res.sim(int(orig), k))
                out.append([int(orig)] + others)
            rest = rest[~close]
    out.sort(key=len, reverse=True)
    return out


# ----------------------------------------------------------------------------- действия
def unique_dest(dest: Path) -> Path:
    """Свободное имя: если файл уже есть, добавляем _1, _2…"""
    if not dest.exists():
        return dest
    k = 1
    while (c := dest.with_name(f"{dest.stem}_{k}{dest.suffix}")).exists():
        k += 1
    return c


def apply(folder: Path, res: ScanResult, groups: list[list[int]], selected: set[int],
          mode: str) -> tuple[int, Path | None]:
    """mode: 'move' -> <папка>/Дубли/group_NNNN/, 'trash' -> Корзина.
    Пишет отчёт CSV. Возвращает (сколько файлов обработано, путь к отчёту)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dup_root = folder / DUP_DIR_NAME
    rows, count = [], 0
    gnum = 1
    if mode == "move" and dup_root.exists():  # продолжаем нумерацию групп с прошлых переносов
        nums = [int(d.name[6:]) for d in dup_root.iterdir()
                if d.is_dir() and d.name.startswith("group_") and d.name[6:].isdigit()]
        gnum = max(nums, default=0) + 1
    if mode == "trash":
        from send2trash import send2trash

    for g in groups:
        chosen = [k for k in g if k in selected]
        if not chosen:
            continue
        orig = g[0]
        gname = f"group_{gnum:04d}"
        gdir = dup_root / gname
        if mode == "move":
            gdir.mkdir(parents=True, exist_ok=True)
            # кладём рядом копию оригинала, чтобы в папке группы было с чем сравнить
            if orig not in selected and res.files[orig].exists():
                shutil.copy2(res.files[orig], unique_dest(gdir / f"ORIGINAL_{res.files[orig].name}"))
        for k in g:
            p = res.files[k]
            done_to = ""
            if k in selected and p.exists():
                if mode == "move":
                    dest = unique_dest(gdir / p.name)
                    shutil.move(str(p), str(dest))
                    done_to = str(dest)
                else:
                    send2trash(str(p))
                    done_to = "Корзина"
                count += 1
            rows.append({"group": gname, "role": "ORIGINAL" if k == orig else "DUPLICATE",
                         "file": str(p), "width": int(res.wh[k][0]), "height": int(res.wh[k][1]),
                         "sim_to_original": f"{res.sim(orig, k):.4f}",
                         "moved_to": done_to if mode == "move" else "",
                         "action": ("moved" if mode == "move" else "trash") if done_to else "kept"})
        gnum += 1
    report = None
    if rows:
        report = folder / f"dupes_report_{stamp}.csv"
        # utf-8-sig — чтобы Excel правильно открыл кириллицу
        with open(report, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
            w.writeheader()
            w.writerows(rows)
    return count, report


def undo(folder: Path, report: Path) -> tuple[int, int]:
    """Возвращает файлы на место по отчёту переноса и убирает копии ORIGINAL_."""
    with open(report, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    back = skipped = 0
    for r in rows:
        if r.get("moved_to"):
            src, dst = Path(r["moved_to"]), Path(r["file"])
            # не затираем файл, если на старом месте уже лежит что-то другое
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                back += 1
            else:
                skipped += 1
    dup_root = folder / DUP_DIR_NAME
    for r in rows:
        if r["role"] != "ORIGINAL":
            continue
        orig, gdir = Path(r["file"]), dup_root / r["group"]
        c = gdir / f"ORIGINAL_{orig.name}"
        # удаляем копию оригинала, только если сам оригинал на месте и это точно та же фотка
        if c.exists() and orig.exists() and c.stat().st_size == orig.stat().st_size:
            c.unlink()
        if gdir.is_dir() and not any(gdir.iterdir()):
            gdir.rmdir()
    if dup_root.is_dir() and not any(dup_root.iterdir()):
        dup_root.rmdir()
    return back, skipped
