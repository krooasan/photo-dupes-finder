"""
Everything except the window: find images, turn each one into an SSCD "fingerprint"
(512 numbers) with ONNX Runtime, compare fingerprints, group copies, pick the original,
move / delete. No torch needed at runtime.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC from iPhones; without the package those files are just skipped
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".wmv"}
EXTS = IMAGE_EXTS | VIDEO_EXTS
# frames taken from a video, evenly spread over its length: one per 5 s, but 8..32
VIDEO_FRAMES_MIN, VIDEO_FRAMES_MAX, VIDEO_SEC_PER_FRAME = 8, 32, 5
DUP_DIR_NAME = "Дубли"
CACHE_NAME = ".photo_dupes_cache.npz"
CACHE_VERSION = 2
MODEL_NAME = "sscd_disc_mixup.onnx"
PAIR_FLOOR = 0.55  # store pairs above this; the threshold slider works without recompute
BARS_RATIO = 0.9
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

Progress = Callable[[str, int, int], None]  # (stage, done, total)


def resource_dir() -> Path:
    """Folder with bundled files (PyInstaller) or the project root."""
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


# ----------------------------------------------------------------------------- files
def collect_files(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    out = []
    for p in it:
        try:
            if not p.is_file() or p.name.startswith("._") or p.suffix.lower() not in EXTS:
                continue
            if DUP_DIR_NAME in p.relative_to(folder).parts:
                continue
        except OSError:
            continue
        out.append(p)
    return sorted(out)


def file_sig(p: Path) -> tuple[int, int]:
    st = p.stat()
    return st.st_mtime_ns, st.st_size


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def video_frames(p: Path, n: int | None = None) -> tuple[list[Image.Image], int, int]:
    """n frames spread over the video (skipping the very start/end, often black), and its size."""
    import cv2  # imported lazily: only needed when the folder has videos

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open video")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n is None:  # more frames for longer videos, so a short trimmed copy still overlaps enough
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            n = int(min(VIDEO_FRAMES_MAX, max(VIDEO_FRAMES_MIN, count / fps / VIDEO_SEC_PER_FRAME)))
        positions = [int(count * (i + 1) / (n + 1)) for i in range(n)] if count > n else [None]
        frames = []
        for pos in positions:
            if pos is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, f = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError("no readable frames")
        return frames, w, h
    finally:
        cap.release()


def open_picture(p: Path, max_side: int | None = None) -> Image.Image:
    """RGB picture of a file: the image itself, or the middle frame of a video."""
    if is_video(p):
        frames, _, _ = video_frames(p, 1)
        im = frames[0]
    else:
        with Image.open(p) as src:
            if max_side:
                src.draft("RGB", (max_side, max_side))  # JPEG: decode at reduced size, much faster
            im = ImageOps.exif_transpose(src).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def _to_input(im: Image.Image) -> np.ndarray:
    """Picture -> model input.

    Must match what SSCD saw in training (torchvision Resize(288) + Normalize), otherwise
    fingerprints drift. Checked against torchvision: the arrays are identical.
    Note: only the short side becomes 288, so different aspect ratios give different shapes.
    """
    iw, ih = im.size
    s = 288
    if iw <= ih:
        nw, nh = s, int(s * ih / iw)
    else:
        nw, nh = int(s * iw / ih), s
    a = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((a.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)


def preprocess(p: Path):
    """Model input for a file: (3, H, W) for an image, (frames, 3, H, W) for a video."""
    if is_video(p):
        frames, w, h = video_frames(p)
        return np.stack([_to_input(f) for f in frames]), w, h
    with Image.open(p) as im:
        w, h = im.size
        return _to_input(ImageOps.exif_transpose(im).convert("RGB")), w, h


# ----------------------------------------------------------------------------- cache
def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d["version"]) != CACHE_VERSION:
                return {}
            return {str(p): {"sig": (int(s[0]), int(s[1])), "w": int(wh[0]), "h": int(wh[1]),
                             "emb": e.astype(np.float32)}
                    for p, s, wh, e in zip(d["paths"], d["sigs"], d["wh"], d["emb"])}
    except Exception:
        return {}


def save_cache(path: Path, items: dict):
    if not items:
        return
    keys = list(items)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, version=np.array(CACHE_VERSION),
             paths=np.array(keys, dtype=str),
             sigs=np.array([items[k]["sig"] for k in keys], dtype=np.int64),
             wh=np.array([(items[k]["w"], items[k]["h"]) for k in keys], dtype=np.int64),
             emb=np.stack([items[k]["emb"] for k in keys]).astype(np.float32))
    os.replace(tmp, path)
    if os.name == "nt":
        try:  # hide the cache file in Explorer
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)
        except Exception:
            pass


# ----------------------------------------------------------------------------- model
class Embedder:
    def __init__(self, use_gpu: bool = True):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # DirectML runs on any DX12 GPU (NVIDIA / AMD / Intel). It doesn't support memory
        # patterns or parallel execution, onnxruntime docs ask to turn both off.
        providers = ["CPUExecutionProvider"]
        if use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(find_model()), so, providers=providers)
        except Exception:  # broken GPU driver etc. -> just fall back to CPU
            if providers[0] == "CPUExecutionProvider":
                raise
            self.sess = ort.InferenceSession(str(find_model()), so, providers=["CPUExecutionProvider"])
        self.device = "gpu" if self.sess.get_providers()[0] == "DmlExecutionProvider" else "cpu"

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        e = self.sess.run(None, {"image": batch})[0].astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------- scan
@dataclass
class ScanResult:
    files: list[Path]
    wh: np.ndarray                      # (n, 2)
    pairs: dict[tuple[int, int], float]  # i<j -> cosine, only >= PAIR_FLOOR
    errors: list[tuple[Path, str]] = field(default_factory=list)
    device: str = ""
    seconds: float = 0.0

    def sim(self, a: int, b: int) -> float:
        if a == b:
            return 1.0
        return self.pairs.get((a, b) if a < b else (b, a), 0.0)


def scan(folder: Path, recursive: bool, progress: Progress, cancel: Callable[[], bool],
         use_gpu: bool = True, batch_size: int = 16) -> ScanResult:
    t0 = time.time()
    progress("files", 0, 0)
    files = collect_files(folder, recursive)
    cache_path = folder / CACHE_NAME
    cache = load_cache(cache_path)
    todo = [p for p in files if (c := cache.get(str(p))) is None or c["sig"] != file_sig(p)]

    errors: list[tuple[Path, str]] = []
    device = "cache"
    if todo:
        progress("model", 0, len(todo))
        emb = Embedder(use_gpu)
        device = emb.device
        # A batch has to be one shape, but images keep their aspect ratio after preprocess.
        # So we collect images into buckets by shape and run a bucket once it has 16 items
        # (screenshots from one monitor all share a shape, so most batches are full).
        buckets: dict[tuple, list] = {}
        done = 0
        last_save = time.time()

        def flush(key):
            nonlocal done
            items = buckets.pop(key)
            out = emb(np.stack([it[1] for it in items]))
            for (p, _, w, h), e in zip(items, out):
                cache[str(p)] = {"sig": file_sig(p), "w": w, "h": h, "emb": e}
            done += len(items)

        def load(p):
            try:
                return p, *preprocess(p), ""
            except Exception as e:
                return p, None, 0, 0, f"{type(e).__name__}: {e}"

        # decoding PNG/JPEG is the slow part, so it runs in threads while the GPU works
        with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as pool:
            for p, arr, w, h, err in pool.map(load, todo):
                if cancel():
                    break
                if err:
                    errors.append((p, err))
                    done += 1
                elif arr.ndim == 4:
                    # video: its frames form one batch; the video's fingerprint is their average,
                    # so a re-encoded / trimmed copy still lands close to the original
                    e = emb(arr).mean(axis=0)
                    cache[str(p)] = {"sig": file_sig(p), "w": w, "h": h, "emb": e / max(np.linalg.norm(e), 1e-12)}
                    done += 1
                else:
                    key = arr.shape
                    buckets.setdefault(key, []).append((p, arr, w, h))
                    if len(buckets[key]) >= batch_size:
                        flush(key)
                    elif sum(len(v) for v in buckets.values()) > 4 * batch_size:
                        # many odd shapes: don't hold them all in memory, run the biggest bucket
                        flush(max(buckets, key=lambda k: len(buckets[k])))
                progress("analyze", done, len(todo))
                if time.time() - last_save > 30:  # so Stop / a crash doesn't lose the work
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

    files = [p for p in files if str(p) in cache and cache[str(p)]["sig"] == file_sig(p)]
    n = len(files)
    wh = np.array([(cache[str(p)]["w"], cache[str(p)]["h"]) for p in files], dtype=np.int64).reshape(-1, 2)
    pairs: dict[tuple[int, int], float] = {}
    if n > 1:
        E = np.stack([cache[str(p)]["emb"] for p in files])
        # Fingerprints are L2-normalised, so a dot product is the cosine similarity.
        # A full n×n matrix would be 10 GB for 50k photos, so we go in blocks of rows
        # and keep only pairs above PAIR_FLOOR.
        step = 2048
        for s in range(0, n, step):
            progress("compare", s, n)
            S = E[s:s + step] @ E.T
            ii, jj = np.nonzero(S >= PAIR_FLOOR)
            for a, b in zip(ii, jj):
                i, j = s + int(a), int(b)
                if i < j:
                    pairs[(i, j)] = float(S[a, b])
    return ScanResult(files, wh, pairs, errors, device, time.time() - t0)


# ----------------------------------------------------------------------------- original choice
def _largest_run(mask: np.ndarray):
    """(start, end) of the longest run of True values."""
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
    """Rows/columns that look like a bar: almost one colour and very dark or very light
    (black letterbox, dark app UI, white page around a picture). Thresholds are hand-tuned."""
    mean, std = a.mean(axis=axis), a.std(axis=axis)
    return (std < 6) & ((mean < 45) | (mean > 240))


_content_memo: dict[str, float] = {}


def content_ratio(p: Path) -> float:
    """Share of the frame that is real picture (without black/white bars and UI)."""
    key = str(p)
    if key in _content_memo:
        return _content_memo[key]
    ratio = 1.0
    try:
        a = np.asarray(open_picture(p, 512).convert("L"), dtype=np.float32)
        r0, r1 = _largest_run(~_bar_lines(a, axis=1))
        if r1 > r0:
            c0, c1 = _largest_run(~_bar_lines(a[r0:r1], axis=0))
            ratio = ((r1 - r0) * (c1 - c0)) / (a.shape[0] * a.shape[1])
    except Exception:
        pass
    _content_memo[key] = ratio
    return ratio


def original_key(res: ScanResult, k: int):
    """Sort key: the "best" copy of a group gets the biggest tuple. Compared in order:
    no bars/frames -> more real picture pixels -> not a PNG (usually a screenshot) -> bigger file."""
    p = res.files[k]
    r = content_ratio(p)
    w, h = res.wh[k]
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return (r >= BARS_RATIO, round(r * int(w) * int(h)), p.suffix.lower() != ".png", size)


class _UF:
    """Union-find: merges files into groups through pairs (A~B and B~C -> one group)."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group(res: ScanResult, threshold: float) -> list[list[int]]:
    """Groups of indices, original first, then by similarity to the original."""
    n = len(res.files)
    uf = _UF(n)
    for (a, b), s in res.pairs.items():
        if s >= threshold:
            uf.union(a, b)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)
    out = []
    for g in comps.values():
        rest = g
        # Union-find can chain: A~B, B~C, but A and C are different photos (two frames of a burst).
        # So inside each component we keep only files close to the chosen original
        # and regroup the leftovers the same way.
        while len(rest) > 1:
            orig = max(rest, key=lambda k: original_key(res, k))
            members = [k for k in rest if k == orig or res.sim(orig, k) >= threshold]
            if len(members) > 1:
                out.append([orig] + sorted((k for k in members if k != orig),
                                           key=lambda k: -res.sim(orig, k)))
            taken = set(members)
            rest = [k for k in rest if k not in taken]
    out.sort(key=len, reverse=True)
    return out


# ----------------------------------------------------------------------------- actions
def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    k = 1
    while (c := dest.with_name(f"{dest.stem}_{k}{dest.suffix}")).exists():
        k += 1
    return c


def apply(folder: Path, res: ScanResult, groups: list[list[int]], selected: set[int],
          mode: str) -> tuple[int, Path | None]:
    """mode: 'move' -> <folder>/Дубли/group_NNNN/, 'trash' -> Recycle Bin.
    Writes a CSV report. Returns (files processed, report path)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dup_root = folder / DUP_DIR_NAME
    rows, count = [], 0
    gnum = 1
    if mode == "move" and dup_root.exists():
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
        with open(report, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
            w.writeheader()
            w.writerows(rows)
    return count, report


def undo(folder: Path, report: Path) -> tuple[int, int]:
    """Bring files back by a 'move' report; remove ORIGINAL_ copies."""
    with open(report, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    back = skipped = 0
    for r in rows:
        if r.get("moved_to"):
            src, dst = Path(r["moved_to"]), Path(r["file"])
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
        if c.exists() and orig.exists() and c.stat().st_size == orig.stat().st_size:
            c.unlink()
        if gdir.is_dir() and not any(gdir.iterdir()):
            gdir.rmdir()
    if dup_root.is_dir() and not any(dup_root.iterdir()):
        dup_root.rmdir()
    return back, skipped
