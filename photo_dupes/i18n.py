"""UI strings (ru / en) and user settings."""
from __future__ import annotations

import json
import locale
import os
from pathlib import Path

LANGS = ("ru", "en")

S = {
    "ru": {
        "app_sub": "поиск дублей фото",
        "folder": "Папка с фото",
        "pick": "Выбрать папку",
        "recursive": "Искать в подпапках",
        "gpu": "Считать на видеокарте",
        "gpu_hint": "В 3–4 раза быстрее процессора. Подойдёт любая видеокарта.",
        "gpu_more": "Что это?",
        "gpu_title": "Что значит «считать на видеокарте»",
        "gpu_text": (
            "Чтобы найти дубли, нейросеть «смотрит» на каждую картинку и превращает её в набор из "
            "512 чисел — отпечаток. У копий одного снимка отпечатки почти совпадают.\n\n"
            "Это тысячи однотипных вычислений. Видеокарта делает их параллельно, поэтому она "
            "в 3–4 раза быстрее процессора: на тесте 58 картинок в секунду против 16.\n\n"
            "Подходит любая видеокарта — NVIDIA, AMD или встроенная Intel. Работает через DirectML, "
            "который уже есть в Windows, ничего доустанавливать не нужно.\n\n"
            "Когда выключать: если на этапе анализа появляется ошибка, если ноутбук работает от батареи "
            "или видеокарта занята игрой. Найденные дубли от этого не меняются — меняется только скорость."
        ),
        "strict": "Строгость",
        "options": "Параметры",
        "folder_none": "Папка не выбрана",
        "folder_click": "Нажмите, чтобы выбрать",
        "title_sub": "Найдите копии одних и тех же снимков и освободите место",
        "more_found": "больше находок",
        "less_err": "меньше ошибок",
        "recommend": "0.85 — лучший вариант для большинства папок",
        "reset": "Вернуть 0.85",
        "find": "Найти дубли",
        "stop": "Остановить",
        "undo_prev": "Откатить прошлый перенос…",
        "lang": "Язык",
        "title_empty": "Дубли фото",
        "groups_head": "Группы дублей",
        "groups_title": ("{n} группа дублей", "{n} группы дублей", "{n} групп дублей"),
        "sub": ("{n} лишняя копия", "{n} лишние копии", "{n} лишних копий"),
        "sub_of": "среди {m} файлов",
        "no_dupes_title": "Дублей нет",
        "checked": "Проверено файлов: {n}",
        "select": "Отметить:",
        "sel_dupes": "Все копии",
        "sel_older": "Старые",
        "sel_newer": "Новые",
        "sel_lowres": "Меньше разрешение",
        "sel_hires": "Больше разрешение",
        "sel_clear": "Снять всё",
        "note_dupes": "Отмечены все копии — в каждой группе останется оригинал.",
        "note_older": "Отмечены старые файлы — в каждой группе останется самый новый.",
        "note_newer": "Отмечены новые файлы — в каждой группе останется самый старый.",
        "note_lowres": "Отмечены файлы поменьше — в каждой группе останется самый большой по разрешению.",
        "note_hires": "Отмечены файлы побольше — в каждой группе останется самый маленький по разрешению.",
        "note_clear": "Отметки сняты.",
        "marked": "Отмечено {n}  ·  освободится {size}",
        "none_marked": "Ничего не отмечено",
        "move": "Перенести в «Дубли»",
        "trash": "Удалить в Корзину",
        "group": "Группа {i}",
        "files": ("{n} файл", "{n} файла", "{n} файлов"),
        "original": "Оригинал",
        "sim": "сходство {p}",
        "remove": "Удалить",
        "no_preview": "нет превью",
        "start_title": "Выберите папку с фото",
        "start_text": ("Программа найдёт копии одних и тех же снимков и видео: скриншоты с рамками и полосами, обрезки, "
                       "пересжатые копии из мессенджеров. В каждой группе оставит лучший оригинал, остальное "
                       "предложит убрать. Без вашего подтверждения ничего не удаляется."),
        "start_btn": "Выбрать папку и начать",
        "scan_title": "Ищу дубли…",
        "scan_text": ("Первый раз каждая картинка анализируется нейросетью. Результат запоминается — "
                      "повторный поиск в этой папке займёт секунды."),
        "prep": "Подготовка…",
        "st_files": "Поиск файлов",
        "st_model": "Загрузка нейросети",
        "st_analyze": "Анализ изображений",
        "st_compare": "Сравнение",
        "of": "{d} из {t}",
        "eta_min": "осталось ~{m} мин {s} с",
        "eta_sec": "осталось ~{s} с",
        "stopping": "Останавливаю… прогресс сохранится",
        "stopped": "Остановлено. В следующий раз продолжит с того же места.",
        "nodup_title": "Дублей не найдено",
        "nodup_text": ("Если уверены, что копии есть, сдвиньте «Строгость» левее — группы пересчитаются "
                       "сразу, без повторного анализа."),
        "done_in": "Готово за {s} с · {dev}",
        "dev_gpu": "видеокарта",
        "dev_cpu": "процессор",
        "dev_cache": "из кэша",
        "unread": "Не удалось прочитать: {n}",
        "err_title": "Не получилось",
        "nofolder_title": "Папка не найдена",
        "nofolder_text": "Папки «{f}» нет. Выберите другую кнопкой «Выбрать папку».",
        "ok": "Понятно",
        "cancel": "Отмена",
        "move_title": "Перенести в «Дубли»?",
        "move_text": ("{n} ({size}) переедут в подпапку «Дубли» рядом с фото. Вернуть их можно кнопкой "
                      "«Откатить прошлый перенос»."),
        "move_ok": "Перенести",
        "trash_title": "Удалить в Корзину?",
        "trash_text": "{n} ({size}) отправятся в Корзину Windows. Пока Корзина не очищена, их можно восстановить.",
        "trash_ok": "Удалить",
        "warn_all": ("\n\nВ {g} отмечены все файлы вместе с оригиналом — от этих снимков не останется "
                     "ни одной копии."),
        "groups_in": ("{n} группе", "{n} группах", "{n} группах"),
        "done_move": "{n} перенесено в «Дубли» · {size}",
        "done_trash": "{n} удалено в Корзину · {size}",
        "undo_title": "Отчёт прошлого переноса",
        "report": "Отчёт",
        "undone": "Возвращено: {n}",
        "skipped": ", пропущено: {n}",
        "pick_title": "Папка с фото",
        "open_photos": "Открыть в «Фото»",
        "open_video": "Открыть в плеере",
        "drop_hint": "или просто перетащите папку в это окно",
        "drop_short": "Папку можно перетащить в окно",
        "ready_title": "Папка выбрана",
        "ready_text": "«{name}». Нажмите «Найти дубли», чтобы начать поиск. Перед этим можно поменять строгость и параметры слева.",
        "show_folder": "Показать в папке",
        "counter": "{i} из {n}   ←  →  ·  пробел — отметить",
        "loading": "Загрузка…",
        "cant_open": "Не удалось открыть файл: {e}",
        "units": ("Б", "КБ", "МБ", "ГБ"),
    },
    "en": {
        "app_sub": "duplicate photo finder",
        "folder": "Photo folder",
        "pick": "Choose folder",
        "recursive": "Include subfolders",
        "gpu": "Use graphics card",
        "gpu_hint": "3–4× faster than the processor. Any graphics card works.",
        "gpu_more": "What is this?",
        "gpu_title": "What “use graphics card” means",
        "gpu_text": (
            "To find duplicates, a neural network “looks” at every picture and turns it into 512 numbers — "
            "a fingerprint. Copies of the same shot get almost identical fingerprints.\n\n"
            "That is thousands of identical calculations. A graphics card runs them in parallel, so it is "
            "3–4× faster than the processor: 58 pictures per second versus 16 in our test.\n\n"
            "Any graphics card works — NVIDIA, AMD or built-in Intel. It runs through DirectML, which ships "
            "with Windows, so there is nothing to install.\n\n"
            "When to turn it off: if an error appears during analysis, on a laptop running on battery, or "
            "while a game is using the card. The duplicates found stay the same — only the speed changes."
        ),
        "strict": "Strictness",
        "options": "Options",
        "folder_none": "No folder selected",
        "folder_click": "Click to choose",
        "title_sub": "Find copies of the same shots and free up space",
        "more_found": "finds more",
        "less_err": "fewer mistakes",
        "recommend": "0.85 is the best choice for most folders",
        "reset": "Back to 0.85",
        "find": "Find duplicates",
        "stop": "Stop",
        "undo_prev": "Undo last move…",
        "lang": "Language",
        "title_empty": "Duplicate photos",
        "groups_head": "Duplicate groups",
        "groups_title": ("{n} duplicate group", "{n} duplicate groups"),
        "sub": ("{n} extra copy", "{n} extra copies"),
        "sub_of": "among {m} files",
        "no_dupes_title": "No duplicates",
        "checked": "Files checked: {n}",
        "select": "Select:",
        "sel_dupes": "All copies",
        "sel_older": "Older",
        "sel_newer": "Newer",
        "sel_lowres": "Lower resolution",
        "sel_hires": "Higher resolution",
        "sel_clear": "Clear",
        "note_dupes": "All copies selected — each group keeps its original.",
        "note_older": "Older files selected — each group keeps the newest one.",
        "note_newer": "Newer files selected — each group keeps the oldest one.",
        "note_lowres": "Smaller files selected — each group keeps the highest resolution.",
        "note_hires": "Larger files selected — each group keeps the lowest resolution.",
        "note_clear": "Selection cleared.",
        "marked": "{n} selected  ·  frees {size}",
        "none_marked": "Nothing selected",
        "move": "Move to “Дубли”",
        "trash": "Move to Recycle Bin",
        "group": "Group {i}",
        "files": ("{n} file", "{n} files"),
        "original": "Original",
        "sim": "similarity {p}",
        "remove": "Remove",
        "no_preview": "no preview",
        "start_title": "Choose a photo folder",
        "start_text": ("The app finds copies of the same shots and videos: screenshots with frames and bars, crops, re-compressed "
                       "copies from messengers. Each group keeps the best original and offers to remove the rest. "
                       "Nothing is deleted without your confirmation."),
        "start_btn": "Choose folder and start",
        "scan_title": "Looking for duplicates…",
        "scan_text": ("The first time, every picture is analysed by a neural network. The result is remembered — "
                      "searching this folder again takes seconds."),
        "prep": "Preparing…",
        "st_files": "Looking for files",
        "st_model": "Loading the neural network",
        "st_analyze": "Analysing images",
        "st_compare": "Comparing",
        "of": "{d} of {t}",
        "eta_min": "~{m} min {s} s left",
        "eta_sec": "~{s} s left",
        "stopping": "Stopping… progress will be kept",
        "stopped": "Stopped. Next time it continues where it left off.",
        "nodup_title": "No duplicates found",
        "nodup_text": ("If you are sure there are copies, move “Strictness” to the left — groups are recalculated "
                       "instantly, without a new analysis."),
        "done_in": "Done in {s} s · {dev}",
        "dev_gpu": "graphics card",
        "dev_cpu": "processor",
        "dev_cache": "from cache",
        "unread": "Could not read: {n}",
        "err_title": "Something went wrong",
        "nofolder_title": "Folder not found",
        "nofolder_text": "“{f}” does not exist. Pick another one with “Choose folder”.",
        "ok": "Got it",
        "cancel": "Cancel",
        "move_title": "Move to “Дубли”?",
        "move_text": ("{n} ({size}) will move to the “Дубли” subfolder next to the photos. You can bring them back "
                      "with “Undo last move”."),
        "move_ok": "Move",
        "trash_title": "Move to Recycle Bin?",
        "trash_text": "{n} ({size}) will go to the Windows Recycle Bin. Until it is emptied you can restore them.",
        "trash_ok": "Remove",
        "warn_all": "\n\nIn {g} every file including the original is selected — no copy of those shots will remain.",
        "groups_in": ("{n} group", "{n} groups"),
        "done_move": "{n} moved to “Дубли” · {size}",
        "done_trash": "{n} moved to Recycle Bin · {size}",
        "undo_title": "Report of the last move",
        "report": "Report",
        "undone": "Restored: {n}",
        "skipped": ", skipped: {n}",
        "pick_title": "Photo folder",
        "open_photos": "Open in Photos",
        "open_video": "Open in player",
        "drop_hint": "or just drag a folder into this window",
        "drop_short": "You can drag a folder into the window",
        "ready_title": "Folder selected",
        "ready_text": "“{name}”. Press “Find duplicates” to start. You can change strictness and options on the left first.",
        "show_folder": "Show in folder",
        "counter": "{i} of {n}   ←  →  ·  space — select",
        "loading": "Loading…",
        "cant_open": "Could not open the file: {e}",
        "units": ("B", "KB", "MB", "GB"),
    },
}


class Tr:
    def __init__(self, lang: str):
        self.lang = lang if lang in S else "en"

    def __call__(self, key: str, **kw) -> str:
        v = S[self.lang].get(key, S["en"].get(key, key))
        return v.format(**kw) if kw and isinstance(v, str) else v

    def n(self, key: str, n: int, **kw) -> str:
        """Plural form: ru -> (one, few, many), en -> (one, other)."""
        forms = S[self.lang][key]
        if self.lang == "ru":
            n10, n100 = n % 10, n % 100
            i = 0 if n10 == 1 and n100 != 11 else 1 if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else 2
        else:
            i = 0 if n == 1 else 1
        return forms[i].format(n=n, **kw)

    def size(self, n: float) -> str:
        units = S[self.lang]["units"]
        for i, u in enumerate(units):
            if n < 1024 or i == len(units) - 1:
                s = f"{n:.0f} {u}" if i == 0 else f"{n:.1f} {u}"
                return s.replace(".", ",") if self.lang == "ru" else s
            n /= 1024
        return ""


# ------------------------------------------------------------------ settings
def _settings_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "PhotoDupesFinder"
    return base / "settings.json"


def default_lang() -> str:
    try:
        loc = (locale.getlocale()[0] or "") + (locale.getdefaultlocale()[0] or "")
    except Exception:
        loc = ""
    return "ru" if "ru" in loc.lower() or "russian" in loc.lower() else "en"


def load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        p = _settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
