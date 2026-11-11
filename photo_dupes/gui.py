"""Окно программы на CustomTkinter. Внешний вид подсмотрен у лаунчера Hydra."""
from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from . import __version__, core
from .i18n import LANGS, Tr, default_lang, load_settings, save_settings

# ------------------------------------------------------------------ цвета (как в globals.scss у Hydra)
# Всё чёрно-белое: важные кнопки залиты белым, а состояние плитки видно по рамке и яркости
# (белая рамка — оригинал, затемнённая картинка — пойдёт в дубли), без цветных акцентов.
BG = "#121212"          # фон основной области
SIDE = "#0d0d0d"        # левая панель, заголовок окна, полосы
CARD = "#171717"        # карточка группы
TILE = "#1c1c1c"        # подложка превью, поля ввода, обычные кнопки
BORDER = "#272727"      # рамки: белый с прозрачностью 8% поверх фона
BORDER_HI = "#3f3f3f"   # наведение и контурные кнопки
HOVER = "#222222"
BRIGHT = "#f0f1f7"      # заголовки и белые кнопки
TEXT = "#d0d1d7"        # обычный текст
MUTED = "#8e919b"       # подписи и подсказки (контраст не ниже 4.5:1 на любом фоне)
ACCENT, ACCENT_H, ON_ACCENT = BRIGHT, "#ffffff", "#121212"
DIM = 0.32              # яркость превью, отмеченного в дубли
DISABLED_TX = "#55575e"
BACKDROP = "#070707"

# Два шрифта, как в Hydra: Noto Sans для текста, Space Grotesk для цифр.
TEXT_FONT, NUM_FONT, FALLBACK_FONT = "Noto Sans", "Space Grotesk", "Segoe UI"
FONT_FILES = ("NotoSans-400.ttf", "NotoSans-500.ttf", "NotoSans-700.ttf", "SpaceGrotesk-700.ttf")

# размеры как у Hydra
T_BODY, T_HINT, T_SMALL = 14, 13, 12

SIDEBAR_W = 360
TW, TH = 224, 160
GROUPS_PER_PAGE = 20
DEFAULT_THR = 0.85

# коды значков из шрифта Segoe Fluent Icons / MDL2
I_FOLDER, I_SEARCH, I_STOP, I_CLOSE = "\uE838", "\uE721", "\uE71A", "\uE711"
I_PREV, I_NEXT, I_TRASH, I_MOVE = "\uE76B", "\uE76C", "\uE74D", "\uE8DE"
I_UNDO, I_PHOTO, I_CHECK, I_INFO = "\uE7A7", "\uE91B", "\uE73E", "\uE946"
I_STAR, I_LANG, I_EDIT, I_PLAY, I_VIDEO = "\uE734", "\uE774", "\uE70F", "\uE768", "\uE714"
I_DROP = "\uE896"  # стрелка вниз: «бросай сюда»

_fonts_ok = False


def load_fonts():
    global _fonts_ok
    d = core.resource_dir() / "assets" / "fonts"
    ok = True
    for f in FONT_FILES:
        try:
            ok &= bool(ctk.FontManager.load_font(str(d / f)))
        except Exception:
            ok = False
    _fonts_ok = ok


def F(size=T_BODY, weight="normal"):
    if not _fonts_ok:
        return ctk.CTkFont(family=FALLBACK_FONT, size=size, weight="bold" if weight == "bold" else "normal")
    if weight == "medium":
        return ctk.CTkFont(family=f"{TEXT_FONT} Medium", size=size)
    return ctk.CTkFont(family=TEXT_FONT, size=size, weight=weight)


def N(size=T_BODY):
    return ctk.CTkFont(family=NUM_FONT if _fonts_ok else FALLBACK_FONT, size=size, weight="bold")


def tkfont(size, weight="normal", num=False, s=1.0):
    # CTk сам масштабирует шрифты под DPI, а обычному tk нужен размер в пикселях
    if not _fonts_ok:
        fam = FALLBACK_FONT
    elif num:
        fam = NUM_FONT
        weight = "bold"
    elif weight == "medium":
        fam, weight = f"{TEXT_FONT} Medium", "normal"
    else:
        fam = TEXT_FONT
    return (fam, -round(size * s), weight)


# ------------------------------------------------------------------ значки
_icon_font_path = next((p for p in (r"C:\Windows\Fonts\SegoeIcons.ttf", r"C:\Windows\Fonts\segmdl2.ttf")
                        if os.path.exists(p)), None)
ICON_FAMILY = "Segoe Fluent Icons" if _icon_font_path and "SegoeIcons" in _icon_font_path else "Segoe MDL2 Assets"
_icon_cache: dict = {}


def icon(code: str, color: str = TEXT, size: int = 18):
    # Рисуем значок из системного шрифта Windows в картинку. Если шрифта нет — None.
    if not _icon_font_path:
        return None
    key = (code, color, size)
    if key not in _icon_cache:
        px = size * 3
        im = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(_icon_font_path, int(px * 0.86))
        x0, y0, x1, y1 = d.textbbox((0, 0), code, font=font)
        d.text(((px - (x1 - x0)) / 2 - x0, (px - (y1 - y0)) / 2 - y0), code, font=font, fill=color)
        _icon_cache[key] = ctk.CTkImage(light_image=im, dark_image=im, size=(size, size))
    return _icon_cache[key]


STYLES = {
    "normal": dict(fg_color=TILE, hover_color=HOVER, border_color=BORDER, border_width=1, text_color=TEXT),
    "accent": dict(fg_color=ACCENT, hover_color=ACCENT_H, text_color=ON_ACCENT, border_width=0),
    "danger": dict(fg_color=TILE, hover_color=HOVER, text_color=BRIGHT, border_width=1, border_color=BORDER_HI),
    "ghost": dict(fg_color="transparent", hover_color=HOVER, text_color=TEXT, border_width=0),
}


def button(parent, text, command, kind="normal", ico=None, size=T_BODY, height=40, ico_size=16, **kw):
    opts = dict(corner_radius=8, height=height, text_color_disabled=DISABLED_TX,
                font=F(size, "bold" if kind == "accent" else "medium"))
    opts.update(STYLES[kind])
    if "text_color" in kw:
        opts["text_color"] = kw.pop("text_color")
    fg = opts["text_color"]
    if ico:
        opts.update(image=icon(ico, fg, ico_size), compound="left")
    opts.update(kw)
    b = ctk.CTkButton(parent, text=text, command=command, **opts)
    b._ico = (ico, fg, ico_size)
    b._fg = opts["fg_color"]
    return b


def set_enabled(b: ctk.CTkButton, on: bool):
    ico, fg, sz = getattr(b, "_ico", (None, None, 18))
    kw = {"state": "normal" if on else "disabled"}
    if ico:
        kw["image"] = icon(ico, fg if on else DISABLED_TX, sz)
    if getattr(b, "_fg", None) not in (None, "transparent"):  # белая кнопка в выключенном виде становится серой
        kw["fg_color"] = b._fg if on else TILE
    b.configure(**kw)


def label(parent, text="", size=T_BODY, color=TEXT, weight="normal", **kw):
    return ctk.CTkLabel(parent, text=text, font=F(size, weight), text_color=color, **kw)


def link(parent, text, command, size=T_HINT):
    # Ссылка текстом, а не кнопкой: у кнопки внутренние отступы, и она не встаёт ровно под текст
    w = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(family=F(size).cget("family"), size=size,
                                                          weight="bold", underline=True),
                     text_color=BRIGHT, cursor="hand2", anchor="w", height=24)
    w.bind("<Button-1>", lambda _e: command())
    w.bind("<Enter>", lambda _e: w.configure(text_color="#ffffff"))
    w.bind("<Leave>", lambda _e: w.configure(text_color=BRIGHT))
    return w


def checkbox(parent, text, var, size=20, fsize=T_BODY):
    return ctk.CTkCheckBox(parent, text=text, variable=var, font=F(fsize), text_color=TEXT,
                           border_color=BORDER_HI, corner_radius=5, checkbox_width=size, checkbox_height=size,
                           fg_color=ACCENT, hover_color=ACCENT_H, checkmark_color=ON_ACCENT, border_width=2)


def load_image(p: Path, max_side: int) -> Image.Image:
    return core.open_picture(p, max_side)  # у видео — кадр из середины


def open_external(p: Path):
    try:
        os.startfile(str(p))
    except Exception:
        pass


def show_in_folder(p: Path):
    try:
        subprocess.Popen(["explorer", "/select,", str(p)])
    except Exception:
        pass


def pct(s: float) -> str:
    return f"{min(s, 1.0) * 100:.0f}%"


def elide_left(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-(n - 1):]


def dark_title_bar(win, color=SIDE, text=BRIGHT):
    # Красим системный заголовок окна в наши цвета. Работает только на Windows 11.
    try:
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())

        def rgb(c):
            c = c.lstrip("#")
            return ctypes.c_int(int(c[4:6], 16) << 16 | int(c[2:4], 16) << 8 | int(c[0:2], 16))

        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        # 35 — фон заголовка, 36 — текст, 34 — рамка (DWMWA_* из dwmapi)
        for attr, val in ((35, rgb(color)), (36, rgb(text)), (34, rgb(color))):
            dwm(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


# ------------------------------------------------------------------ всплывающие окна
class Modal(ctk.CTkFrame):
    # Слой поверх всего окна, закрывается по Esc. blur=True — сзади размытый снимок окна.

    def __init__(self, app: "App", blur: bool = False):
        super().__init__(app, fg_color=BACKDROP, corner_radius=0)
        self.app = app
        self._bg_img = None
        if blur:
            try:
                from PIL import ImageFilter, ImageGrab

                app.update()
                x, y, w, h = app.winfo_rootx(), app.winfo_rooty(), app.winfo_width(), app.winfo_height()
                shot = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
                shot = shot.resize((max(1, w // 4), max(1, h // 4))).filter(ImageFilter.GaussianBlur(4))
                shot = ImageEnhance.Brightness(shot).enhance(0.35)
                s = self._get_widget_scaling()
                self._bg_img = ctk.CTkImage(light_image=shot, dark_image=shot, size=(int(w / s), int(h / s)))
                ctk.CTkLabel(self, text="", image=self._bg_img).place(relx=0, rely=0, relwidth=1, relheight=1)
            except Exception:
                pass
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        app.modals.append(self)

    def close(self):
        if self in self.app.modals:
            self.app.modals.remove(self)
        self.destroy()

    def on_key(self, key: str):
        if key == "Escape":
            self.close()


class Dialog(Modal):
    def __init__(self, app, title, text, ok_text, kind="accent", on_ok=None, cancel=True, ok_icon=None,
                 width=500):
        super().__init__(app, blur=not app.modals)
        self._on_ok = on_ok
        panel = ctk.CTkFrame(self, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=12)
        panel.place(relx=0.5, rely=0.5, anchor="center")
        head = ctk.CTkFrame(panel, fg_color="transparent")
        head.pack(fill="x", padx=(28, 14), pady=(18, 0))
        label(head, title, 16, BRIGHT, "bold").pack(side="left")
        button(head, "", self.close, "ghost", ico=I_CLOSE, width=36, height=36).pack(side="right")
        label(panel, text, T_BODY, TEXT, justify="left", wraplength=width, anchor="w").pack(
            fill="x", padx=28, pady=(10, 24))
        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.pack(fill="x", padx=28, pady=(0, 24))
        button(row, ok_text, self.ok, kind, ico=ok_icon, width=160).pack(side="right")
        if cancel:
            button(row, app.t("cancel"), self.close, width=110).pack(side="right", padx=(0, 10))

    def ok(self):
        self.close()
        if self._on_ok:
            self._on_ok()

    def on_key(self, key):
        if key == "Escape":
            self.close()
        elif key == "Return":
            self.ok()


class Toast(ctk.CTkFrame):

    def __init__(self, app, text, ico=I_CHECK):
        super().__init__(app, fg_color=CARD, border_color=BORDER_HI, border_width=1, corner_radius=10)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(padx=18, pady=14)
        ctk.CTkLabel(row, text="", image=icon(ico, BRIGHT, 20)).pack(side="left", padx=(0, 12))
        label(row, text, T_BODY, BRIGHT, justify="left").pack(side="left")
        self.place(relx=1.0, rely=1.0, x=-32, y=-120, anchor="se")
        self.lift()
        self.after(4000, self.destroy)


class Preview(Modal):
    def __init__(self, app: "App", group: list[int], pos: int):
        super().__init__(app)
        t = app.t
        self.group, self.pos = group, pos
        self._pil: Image.Image | None = None
        self._token = 0
        self._fit_job = None
        self._keep: list = []

        top = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, height=64)
        top.pack(fill="x")
        top.pack_propagate(False)
        self.lbl_title = label(top, "", 16, BRIGHT, "bold", anchor="w")
        self.lbl_title.pack(side="left", padx=(24, 12))
        self.badge = ctk.CTkLabel(top, text="", font=F(T_SMALL, "bold"), corner_radius=6, height=26)
        self.badge.pack(side="left")
        button(top, "", self.close, "ghost", ico=I_CLOSE, width=44, height=44).pack(side="right", padx=10)
        self.lbl_count = label(top, "", T_HINT, MUTED)
        self.lbl_count.pack(side="right", padx=10)

        body = ctk.CTkFrame(self, fg_color=BACKDROP, corner_radius=0)
        body.pack(fill="both", expand=True)
        self.btn_prev = button(body, "", lambda: self.step(-1), "ghost", ico=I_PREV, width=56, height=110,
                               ico_size=24)
        self.btn_prev.pack(side="left", padx=10)
        self.btn_next = button(body, "", lambda: self.step(1), "ghost", ico=I_NEXT, width=56, height=110,
                               ico_size=24)
        self.btn_next.pack(side="right", padx=10)
        self.view = ctk.CTkLabel(body, text="", fg_color=BACKDROP)
        self.view.pack(fill="both", expand=True, pady=18)
        self.view.bind("<Configure>", lambda _e: self._schedule_fit())

        bot = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, height=84)
        bot.pack(fill="x")
        bot.pack_propagate(False)
        self.lbl_info = label(bot, "", T_HINT, MUTED, anchor="w", justify="left")
        self.lbl_info.pack(side="left", padx=28, fill="x", expand=True)
        self.btn_open = button(bot, t("open_photos"), self.open_photos, "accent", ico=I_PHOTO, width=190)
        self.btn_open.pack(side="right", padx=(10, 28))
        button(bot, t("show_folder"), self.reveal, ico=I_FOLDER, width=180).pack(side="right", padx=4)
        self.chk = checkbox(bot, t("remove"), None)
        self.chk.pack(side="right", padx=18)
        self.show()

    @property
    def k(self) -> int:
        return self.group[self.pos]

    def show(self):
        app, t = self.app, self.app.t
        res, k = app.res, self.k
        p = res.files[k]
        orig = self.group[0]
        self.lbl_title.configure(text=p.name)
        self.btn_open.configure(text=t("open_video" if core.is_video(p) else "open_photos"),
                                image=icon(I_PLAY if core.is_video(p) else I_PHOTO, ON_ACCENT, 16))
        if k == orig:
            self.badge.configure(text=f"  {t('original')}  ", fg_color=ACCENT, text_color=ON_ACCENT)
        else:
            self.badge.configure(text=f"  {t('sim', p=pct(res.sim(orig, k)))}  ", fg_color=TILE, text_color=BRIGHT)
        self.lbl_count.configure(text=t("counter", i=self.pos + 1, n=len(self.group)))
        w, h = res.wh[k]
        self.lbl_info.configure(text=f"{w} × {h}  ·  {t.size(app.size_of(k))}  ·  "
                                     f"{p.suffix.upper().lstrip('.')}  ·  {app.date_of(k)}\n{p.parent}")
        self.chk.configure(variable=app.checks[k])
        set_enabled(self.btn_prev, self.pos > 0)
        set_enabled(self.btn_next, self.pos < len(self.group) - 1)
        self._pil = None
        self.view.configure(image=None, text=t("loading"), text_color=MUTED, font=F(T_BODY))
        self._token += 1
        token = self._token

        def work():
            try:
                im = load_image(p, 2560)
            except Exception as e:
                im = e
            app.q.put(("preview", self, token, im))

        threading.Thread(target=work, daemon=True).start()

    def set_image(self, token, im):
        if token != self._token or not self.winfo_exists():
            return
        if isinstance(im, Exception):
            self.view.configure(image=None, text=self.app.t("cant_open", e=im))
            return
        self._pil = im
        self._fit()

    def _schedule_fit(self):
        if self._fit_job:
            self.after_cancel(self._fit_job)
        self._fit_job = self.after(60, self._fit)

    def _fit(self):
        self._fit_job = None
        if self._pil is None:
            return
        s = self.view._get_widget_scaling()
        bw, bh = self.view.winfo_width() / s, self.view.winfo_height() / s
        iw, ih = self._pil.size
        r = min(bw / iw, bh / ih, 2.0 / s)  # маленькие картинки растягиваем не больше чем вдвое
        img = ctk.CTkImage(light_image=self._pil, dark_image=self._pil,
                           size=(max(1, int(iw * r)), max(1, int(ih * r))))
        self._keep.append(img)  # держим ссылку, иначе сборщик мусора удалит картинку и Tk покажет пустоту
        del self._keep[:-4]
        self.view.configure(image=img, text="")

    def step(self, d):
        if 0 <= self.pos + d < len(self.group):
            self.pos += d
            self.show()

    def open_photos(self):
        open_external(self.app.res.files[self.k])

    def reveal(self):
        show_in_folder(self.app.res.files[self.k])

    def on_key(self, key):
        if key == "Escape":
            self.close()
        elif key == "Left":
            self.step(-1)
        elif key == "Right":
            self.step(1)
        elif key in ("Delete", "space"):
            v = self.app.checks[self.k]
            v.set(not v.get())

    def close(self):
        super().close()
        self.app.refresh_marks()


# ------------------------------------------------------------------ главное окно
SELECT_MODES = ("dupes", "older", "newer", "lowres", "hires", "clear")


class App(ctk.CTk):
    def __init__(self, start_folder: str | None = None):
        ctk.set_appearance_mode("dark")
        load_fonts()
        super().__init__(fg_color=BG)
        self.title("Photo Dupes")
        w, h = 1400, 900
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        s = self._get_window_scaling()
        w, h = min(w, int(sw / s) - 40), min(h, int(sh / s) - 80)
        self.geometry(f"{w}x{h}+{max(0, int((sw / s - w) / 2))}+{max(0, int((sh / s - h) / 2) - 20)}")
        self.minsize(1100, 780)
        try:
            self.iconbitmap(default=str(core.resource_dir() / "assets" / "icon.ico"))
        except Exception:
            pass
        self.after(30, lambda: dark_title_bar(self))
        self.after(60, self._enable_drop)

        st = load_settings()
        self.lang = st.get("lang") if st.get("lang") in LANGS else default_lang()
        self.t = Tr(self.lang)
        self.folder = ctk.StringVar(value=start_folder or st.get("folder", ""))
        self.recursive = ctk.BooleanVar(value=bool(st.get("recursive", False)))
        self.use_gpu = ctk.BooleanVar(value=bool(st.get("gpu", True)))
        self.threshold = ctk.DoubleVar(value=float(st.get("threshold", DEFAULT_THR)))

        self.res: core.ScanResult | None = None
        self.scan_folder: Path | None = None
        self.groups: list[list[int]] = []
        self.checks: dict[int, ctk.BooleanVar] = {}
        self.orig_set: set[int] = set()
        self._bulk = False
        self.sel_mode = "dupes"
        self._note_mode = None
        self.page = 0
        self.modals: list[Modal] = []
        self.q: queue.Queue = queue.Queue()
        self._cancel = False
        self._worker: threading.Thread | None = None
        self._regroup_job = None
        self._render_id = 0
        self._tiles: dict[int, ctk.CTkLabel] = {}
        self._tile_imgs: dict[int, tuple] = {}
        self._tile_rm: dict[int, tk.Label] = {}
        self._thumb_imgs: list = []
        self._thumb_cache: dict[str, Image.Image] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._stage_t0 = (None, 0.0)
        self._progress_lbl = None
        self._last_progress = None
        self._frames: list = []

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.threshold.trace_add("write", lambda *_a: self._on_threshold())
        self.folder.trace_add("write", lambda *_a: self._paint_folder())
        self._build_ui()
        self.bind_all("<Key>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    # ---------------------------------------------------------- настройки и язык
    def _save(self):
        save_settings({"lang": self.lang, "folder": self.folder.get(), "recursive": self.recursive.get(),
                       "gpu": self.use_gpu.get(), "threshold": round(self.threshold.get(), 2)})

    def _on_close(self):
        self._save()
        self.destroy()

    def set_lang(self, lang):
        if lang == self.lang:
            return
        self.lang, self.t = lang, Tr(lang)
        self._save()
        for m in list(self.modals):
            m.close()
        self._build_ui()

    def _build_ui(self):
        for f in self._frames:
            f.destroy()
        self._build_sidebar()
        self._build_main()
        self._build_status()
        self._frames = [self.sidebar, self.main, self.status]
        self._on_threshold(regroup=False)
        self._paint_folder()
        self._render()
        if self._note_mode and self.groups:
            self.lbl_note.configure(text=self.t(f"note_{self._note_mode}"))
        if self._busy() and self._last_progress:
            self._progress(*self._last_progress)

    # ---------------------------------------------------------- строка состояния
    def _build_status(self):
        st = self.status = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, height=30)
        st.grid(row=1, column=0, columnspan=2, sticky="ew")
        st.pack_propagate(False)
        tk.Frame(st, bg=BORDER, height=1).pack(fill="x", side="top")
        self.lbl_summary = label(st, "", T_SMALL, MUTED, anchor="w", height=26)
        self.lbl_summary.pack(side="left", padx=18)
        label(st, f"Photo Dupes · v{__version__}", T_SMALL, MUTED, height=26).pack(side="right", padx=18)

    # ---------------------------------------------------------- левая панель
    def _section(self, parent, text, top=20):
        label(parent, text, 16, BRIGHT, "bold", anchor="w").pack(fill="x", padx=28, pady=(top, 10))

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(22, 0))

    def _build_sidebar(self):
        """Left column reads top to bottom as the workflow: folder → strictness → options → search."""
        t = self.t
        sb = self.sidebar = ctk.CTkFrame(self, fg_color=SIDE, corner_radius=0, width=SIDEBAR_W)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.pack_propagate(False)
        tk.Frame(sb, bg=BORDER, width=1).place(relx=1.0, rely=0, relheight=1, anchor="ne")
        pad = dict(padx=28, fill="x")

        # 1. папка — одна большая карточка, по ней можно кликнуть
        self._section(sb, t("folder"), top=24)
        card = self.folder_card = ctk.CTkFrame(sb, fg_color=TILE, corner_radius=10, border_width=1,
                                               border_color=BORDER, height=68, cursor="hand2")
        card.pack(**pad)
        drop = ctk.CTkFrame(sb, fg_color="transparent")
        drop.pack(pady=(8, 0), **pad)
        ctk.CTkLabel(drop, text="", image=icon(I_DROP, MUTED, 14)).pack(side="left")
        label(drop, t("drop_short"), T_SMALL, MUTED, anchor="w").pack(side="left", padx=(8, 0))
        card.pack_propagate(False)
        self.folder_icon = ctk.CTkLabel(card, text="", image=icon(I_FOLDER, BRIGHT, 22), width=34)
        self.folder_icon.pack(side="left", padx=(18, 10))
        txt = ctk.CTkFrame(card, fg_color="transparent")
        txt.pack(side="left", fill="both", expand=True, pady=12)
        self.folder_name = label(txt, "", T_BODY, BRIGHT, "bold", anchor="w")
        self.folder_name.pack(fill="x")
        self.folder_path = label(txt, "", T_SMALL, MUTED, anchor="w")
        self.folder_path.pack(fill="x")
        self.folder_edit = ctk.CTkLabel(card, text="", image=icon(I_EDIT, MUTED, 16), width=30)
        self.folder_edit.pack(side="right", padx=(0, 16))
        for w in (card, self.folder_icon, txt, self.folder_name, self.folder_path, self.folder_edit):
            w.bind("<Button-1>", lambda _e: self._pick())
            w.bind("<Enter>", lambda _e: card.configure(border_color=BORDER_HI, fg_color=HOVER))
            w.bind("<Leave>", lambda _e: card.configure(border_color=BORDER, fg_color=TILE))

        # 2. строгость
        self._divider(sb)
        thr = ctk.CTkFrame(sb, fg_color="transparent")
        thr.pack(pady=(20, 6), **pad)
        label(thr, t("strict"), 16, BRIGHT, "bold").pack(side="left")
        self.lbl_thr = ctk.CTkLabel(thr, text="", font=N(16), text_color=BRIGHT)
        self.lbl_thr.pack(side="right")
        ctk.CTkSlider(sb, from_=0.6, to=0.97, number_of_steps=37, variable=self.threshold,
                      button_color=BRIGHT, button_hover_color="#ffffff", progress_color=BRIGHT,
                      fg_color=BORDER_HI, height=16, button_length=0).pack(pady=(4, 4), **pad)
        ends = ctk.CTkFrame(sb, fg_color="transparent")
        ends.pack(**pad)
        label(ends, t("more_found"), T_SMALL, MUTED).pack(side="left")
        label(ends, t("less_err"), T_SMALL, MUTED).pack(side="right")
        rec = ctk.CTkFrame(sb, fg_color=TILE, corner_radius=10)
        rec.pack(pady=(12, 0), **pad)
        rrow = ctk.CTkFrame(rec, fg_color="transparent")
        rrow.pack(fill="x", padx=14, pady=12)
        ctk.CTkLabel(rrow, text="", image=icon(I_STAR, BRIGHT, 16)).pack(side="left", anchor="n", pady=3)
        label(rrow, t("recommend"), T_HINT, TEXT, anchor="w", justify="left", wraplength=250).pack(
            side="left", padx=(10, 0), fill="x")

        # 3. настройки
        self._divider(sb)
        self._section(sb, t("options"))
        checkbox(sb, t("recursive"), self.recursive).pack(pady=(0, 12), **pad)
        gpu_row = ctk.CTkFrame(sb, fg_color="transparent")
        gpu_row.pack(**pad)
        checkbox(gpu_row, t("gpu"), self.use_gpu).pack(side="left")
        button(gpu_row, "", self._gpu_info, "ghost", ico=I_INFO, width=32, height=32,
               text_color=MUTED).pack(side="right")
        hint = ctk.CTkFrame(sb, fg_color="transparent")
        hint.pack(padx=(58, 28), fill="x")
        label(hint, t("gpu_hint"), T_SMALL, MUTED, anchor="w", justify="left", wraplength=260).pack(fill="x")
        link(hint, t("gpu_more"), self._gpu_info, T_SMALL).pack(anchor="w", pady=(2, 0))

        # 4. запуск
        self._divider(sb)
        self.btn_scan = button(sb, t("find"), self._scan, "accent", ico=I_SEARCH, height=44)
        self.btn_scan.pack(pady=(22, 0), **pad)

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=18, pady=(0, 18))
        lang = ctk.CTkFrame(bottom, fg_color="transparent")
        lang.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(lang, text="", image=icon(I_LANG, MUTED, 18)).pack(side="left")
        label(lang, t("lang"), T_HINT, MUTED).pack(side="left", padx=(10, 0))
        for code in reversed(LANGS):
            on = code == self.lang
            button(lang, code.upper(), lambda c=code: self.set_lang(c), "accent" if on else "normal",
                   width=52, height=32, size=T_SMALL).pack(side="right", padx=(6, 0))
        button(bottom, t("undo_prev"), self._undo, "ghost", ico=I_UNDO, anchor="w", size=T_HINT,
               text_color=MUTED).pack(fill="x")

    def _paint_folder(self):
        if not hasattr(self, "folder_name") or not self.folder_name.winfo_exists():
            return
        f = self.folder.get().strip().strip('"')
        if f:
            p = Path(f)
            self.folder_name.configure(text=elide_left(p.name or f, 22))
            self.folder_path.configure(text=elide_left(str(p.parent), 28))
        else:
            self.folder_name.configure(text=self.t("folder_none"))
            self.folder_path.configure(text=self.t("folder_click"))

    # ---------------------------------------------------------- правая часть
    def _build_main(self):
        t = self.t
        main = self.main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=36, pady=(26, 0))
        titles = ctk.CTkFrame(head, fg_color="transparent")
        titles.pack(side="left")
        hrow = ctk.CTkFrame(titles, fg_color="transparent")
        hrow.pack(anchor="w")
        self.lbl_head = label(hrow, "", 20, BRIGHT, "bold", anchor="w")
        self.lbl_head.pack(side="left")
        self.pill = ctk.CTkLabel(hrow, text="", font=N(T_SMALL), fg_color=TILE, text_color=BRIGHT,
                                 corner_radius=6, height=24, width=32)
        self.lbl_sub = label(titles, "", T_BODY, MUTED, anchor="w")
        self.lbl_sub.pack(anchor="w", pady=(2, 0))
        self.nav = nav = ctk.CTkFrame(head, fg_color="transparent")
        self.btn_prev = button(nav, "", lambda: self._go(-1), ico=I_PREV, width=42)
        self.btn_prev.pack(side="left")
        self.lbl_page = label(nav, "", T_HINT, MUTED, width=100)
        self.lbl_page.pack(side="left")
        self.btn_next = button(nav, "", lambda: self._go(1), ico=I_NEXT, width=42)
        self.btn_next.pack(side="left")

        # быстрый выбор: в каждой группе оставляем один файл по правилу, остальные отмечаем
        self.selbar = ctk.CTkFrame(main, fg_color="transparent")
        chips = ctk.CTkFrame(self.selbar, fg_color="transparent")
        chips.pack(fill="x")
        label(chips, t("select"), T_BODY, MUTED).pack(side="left", padx=(0, 12))
        self.chips = {}
        for m in SELECT_MODES:
            b = ctk.CTkButton(chips, text=t(f"sel_{m}"), command=lambda m=m: self._select(m), height=36,
                              width=0, corner_radius=8, font=F(T_BODY, "medium"), fg_color=TILE, hover_color=HOVER,
                              border_width=1, border_color=BORDER, text_color=TEXT)
            b.pack(side="left", padx=(0, 8))
            self.chips[m] = b
        self.lbl_note = label(self.selbar, "", T_HINT, MUTED, anchor="w")
        self.lbl_note.pack(fill="x", pady=(8, 0))

        self.results = ctk.CTkScrollableFrame(main, fg_color=BG, corner_radius=0,
                                              scrollbar_button_color=BORDER_HI,
                                              scrollbar_button_hover_color="#5a5a5a")
        self.results.grid(row=2, column=0, sticky="nsew", padx=(28, 8), pady=(14, 0))

        bar = ctk.CTkFrame(main, fg_color=SIDE, corner_radius=0, height=68)
        bar.grid(row=3, column=0, sticky="ew")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x", side="top")
        self.lbl_sel = label(bar, "", T_BODY, BRIGHT, "medium")
        self.lbl_sel.pack(side="left", padx=36)
        self.btn_trash = button(bar, t("trash"), lambda: self._apply("trash"), "danger", ico=I_TRASH)
        self.btn_trash.pack(side="right", padx=(10, 36))
        self.btn_move = button(bar, t("move"), lambda: self._apply("move"), "accent", ico=I_MOVE)
        self.btn_move.pack(side="right")

    # ---------------------------------------------------------- мелочи
    def _on_key(self, e):
        if self.modals:
            self.modals[-1].on_key(e.keysym)

    def _on_threshold(self, regroup=True):
        v = round(self.threshold.get(), 2)
        if not hasattr(self, "lbl_thr") or not self.lbl_thr.winfo_exists():
            return
        self.lbl_thr.configure(text=f"{v:.2f}")
        if regroup and self.res:
            if self._regroup_job:
                self.after_cancel(self._regroup_job)
            self._regroup_job = self.after(250, self._regroup)

    def _gpu_info(self):
        Dialog(self, self.t("gpu_title"), self.t("gpu_text"), self.t("ok"), cancel=False, width=540)

    def _pick(self):
        d = filedialog.askdirectory(title=self.t("pick_title"), initialdir=self.folder.get() or None)
        if d:
            self.folder.set(os.path.normpath(d))
            if not self._busy():  # новая папка: старые результаты не подходят, показываем «Папка выбрана»
                self.res, self.groups, self.checks = None, [], {}
                self._render()
            return True
        return False

    def _enable_drop(self):
        try:
            import windnd

            # Этот обработчик вызывается изнутри оконной процедуры Windows. Если тут тронуть Tk
            # (даже after()), Python падает. Поэтому просто кладём пути в очередь, её читает _poll().
            windnd.hook_dropfiles(self, func=lambda paths: self.q.put(("drop", list(paths))), force_unicode=True)
        except Exception:
            pass  # нет windnd или не Windows — остаётся кнопка «Выбрать папку»

    def _on_drop(self, paths):
        if not paths or self._busy():
            return
        p = Path(paths[0] if isinstance(paths[0], str) else paths[0].decode("utf-8", "replace"))
        folder = p if p.is_dir() else p.parent  # бросили файл, а не папку — берём папку, где он лежит
        for m in list(self.modals):
            m.close()
        # Только выбираем папку, поиск не запускаем: вдруг человек хочет сначала поменять строгость.
        # Старые результаты относятся к прошлой папке — выкидываем.
        self.folder.set(os.path.normpath(str(folder)))
        self.res, self.groups, self.checks = None, [], {}
        self._render()
        self.folder_card.configure(border_color=BRIGHT)
        self.after(1200, lambda: self.folder_card.winfo_exists() and self.folder_card.configure(border_color=BORDER))

    def _pick_and_scan(self):
        if self._pick():
            self._scan()

    def info(self, title, text):
        Dialog(self, title, text, self.t("ok"), cancel=False)

    def _busy(self):
        return self._worker is not None and self._worker.is_alive()

    def size_of(self, k: int) -> int:
        return int(self.res.size[k])  # размер и дата уже есть после поиска, диск не трогаем

    def date_of(self, k: int) -> str:
        return time.strftime("%d.%m.%Y %H:%M", time.localtime(float(self.res.mtime[k])))

    def _update_controls(self):
        t = self.t
        busy = self._busy()
        if busy:
            self.btn_scan.configure(text=t("stop"), fg_color=TILE, hover_color=HOVER, text_color=BRIGHT,
                                    border_width=1, border_color=BORDER_HI, image=icon(I_STOP, BRIGHT, 16))
        else:
            self.btn_scan.configure(text=t("find"), fg_color=ACCENT, hover_color=ACCENT_H, text_color=ON_ACCENT,
                                    border_width=0, image=icon(I_SEARCH, ON_ACCENT, 16))
        has = bool(self.groups) and not busy
        if has:
            self.selbar.grid(row=1, column=0, sticky="ew", padx=36, pady=(18, 0))
        else:
            self.selbar.grid_forget()
        for m, b in self.chips.items():
            on = m == self.sel_mode
            b.configure(fg_color=ACCENT if on else TILE, hover_color=ACCENT_H if on else HOVER,
                        text_color=ON_ACCENT if on else TEXT, border_color=ACCENT if on else BORDER)
        marked = has and any(v.get() for v in self.checks.values())
        for b in (self.btn_trash, self.btn_move):
            set_enabled(b, marked)
        pages = max(1, -(-len(self.groups) // GROUPS_PER_PAGE))
        if len(self.groups) > GROUPS_PER_PAGE and not busy:
            self.nav.pack(side="right", anchor="n", pady=(6, 0))
            set_enabled(self.btn_prev, self.page > 0)
            set_enabled(self.btn_next, self.page < pages - 1)
            self.lbl_page.configure(text=t("of", d=self.page + 1, t=pages))
        else:
            self.nav.pack_forget()

    def _update_stats(self):
        t, r = self.t, self.res
        if not r:
            self.lbl_head.configure(text=t("title_empty"))
            self.pill.pack_forget()
            self.lbl_sub.configure(text=t("title_sub"))
            self.lbl_sel.configure(text="")
            return
        n_dup = sum(len(g) - 1 for g in self.groups)
        if self.groups:
            self.lbl_head.configure(text=t("groups_head"))
            self.pill.configure(text=f" {len(self.groups)} ")
            self.pill.pack(side="left", padx=(10, 0), pady=(2, 0))
            self.lbl_sub.configure(text=f"{t.n('sub', n_dup)} {t('sub_of', m=len(r.files))}")
        else:
            self.lbl_head.configure(text=t("no_dupes_title"))
            self.pill.pack_forget()
            self.lbl_sub.configure(text=t("checked", n=len(r.files)))
        marked = [k for k, v in self.checks.items() if v.get()]
        self.lbl_sel.configure(text=t("marked", n=len(marked), size=t.size(sum(self.size_of(k) for k in marked)))
                               if marked else (t("none_marked") if self.groups else ""))
        dev = t({"gpu": "dev_gpu", "cpu": "dev_cpu"}.get(r.device, "dev_cache"))
        err = "  ·  " + t("unread", n=len(r.errors)) if r.errors else ""
        self.lbl_summary.configure(text=t("done_in", s=f"{r.seconds:.1f}", dev=dev) + err)

    # ---------------------------------------------------------- поиск
    def _scan(self):
        t = self.t
        if self._busy():
            self._cancel = True
            if self._progress_lbl and self._progress_lbl.winfo_exists():
                self._progress_lbl.configure(text=t("stopping"))
            return
        if not self.folder.get().strip():
            self._pick_and_scan()
            return
        folder = Path(self.folder.get().strip().strip('"'))
        if not folder.is_dir():
            self.info(t("nofolder_title"), t("nofolder_text", f=folder))
            return
        self._save()
        self._cancel = False
        self.scan_folder = folder
        self.groups, self.res, self.checks = [], None, {}
        self._last_progress = None
        self.lbl_summary.configure(text="")
        rec, gpu = self.recursive.get(), self.use_gpu.get()

        def work():
            try:
                res = core.scan(folder, rec, lambda s, d, n: self.q.put(("progress", s, d, n)),
                                lambda: self._cancel, use_gpu=gpu)
                self.q.put(("done", res))
            except InterruptedError:
                self.q.put(("cancelled",))
            except Exception as e:
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self._render()

    def _progress(self, stage, done, total):
        self._last_progress = (stage, done, total)
        if self._progress_lbl is None or not self._progress_lbl.winfo_exists():
            return
        t = self.t
        if self._stage_t0[0] != stage:
            self._stage_t0 = (stage, time.time())
        name = t(f"st_{stage}")
        if total:
            self._pbar.set(done / total)
            eta = ""
            el = time.time() - self._stage_t0[1]
            if done > 20 and el > 2:
                left = (total - done) * el / done
                eta = "  ·  " + (t("eta_min", m=int(left // 60), s=int(left % 60)) if left >= 60
                                 else t("eta_sec", s=int(left)))
            self._progress_lbl.configure(text=f"{name}: {t('of', d=done, t=total)}{eta}")
            self.lbl_summary.configure(text=f"{name}: {t('of', d=done, t=total)}")
        else:
            self._pbar.set(0)
            self._progress_lbl.configure(text=name + "…")

    def _poll(self):
        try:
            for _ in range(200):
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self._progress(*msg[1:])
                elif kind == "done":
                    self.res = msg[1]
                    self._regroup()
                elif kind == "cancelled":
                    self._render()
                    self.lbl_summary.configure(text=self.t("stopped"))
                elif kind == "error":
                    self._render()
                    self.info(self.t("err_title"), msg[1])
                elif kind == "thumb":
                    _, rid, lbl, im, k = msg
                    if rid == self._render_id and lbl.winfo_exists():
                        self._set_thumb(lbl, im, k)
                elif kind == "drop":
                    self._on_drop(msg[1])
                elif kind == "preview":
                    _, pv, token, im = msg
                    if pv.winfo_exists():
                        pv.set_image(token, im)
                if kind in ("done", "cancelled", "error"):
                    self._update_controls()
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _regroup(self):
        self._regroup_job = None
        if not self.res:
            return
        self.groups = core.group(self.res, round(self.threshold.get(), 3))
        self.orig_set = {g[0] for g in self.groups}
        self.checks = {}
        for g in self.groups:
            for k in g:
                v = ctk.BooleanVar(value=k != g[0])
                v.trace_add("write", lambda *_a, k=k: self._on_check(k))
                self.checks[k] = v
        self.sel_mode, self._note_mode = "dupes", None
        self.page = 0
        self._render()
        self.lbl_note.configure(text="")

    # ---------------------------------------------------------- выбор файлов
    def _select(self, mode: str):
        self.sel_mode = mode
        mt = self.res.mtime
        # Без этого флага каждая из тысяч галочек пересчитывала бы статистику целиком.
        # Пока он стоит, _on_check только перекрашивает плитку, а итоги считаем один раз в конце.
        self._bulk = True
        for g in self.groups:
            if mode == "clear":
                keep = None
            elif mode == "dupes":
                keep = g[0]
            elif mode == "older":    # оставляем самый новый
                keep = max(g, key=lambda k: (mt[k], k == g[0]))
            elif mode == "newer":    # оставляем самый старый
                keep = min(g, key=lambda k: (mt[k], k != g[0]))
            elif mode == "lowres":   # оставляем самое большое разрешение
                keep = max(g, key=lambda k: (int(self.res.wh[k][0]) * int(self.res.wh[k][1]), k == g[0]))
            else:                    # hires: оставляем самое маленькое
                keep = min(g, key=lambda k: (int(self.res.wh[k][0]) * int(self.res.wh[k][1]), k != g[0]))
            for k in g:
                v = self.checks[k]
                want = keep is not None and k != keep
                if v.get() != want:
                    v.set(want)
        self._bulk = False
        self._update_stats()
        self._note_mode = mode
        self.lbl_note.configure(text=self.t(f"note_{mode}"))
        self._update_controls()

    # ---------------------------------------------------------- результаты
    def _go(self, d):
        self.page += d
        self._render()

    def _marked(self, k) -> bool:
        v = self.checks.get(k)
        return v is not None and v.get()

    def _paint_tile(self, k):
        # белая рамка — оригинал; затемнение и значок корзины — файл пойдёт в дубли
        tl = self._tiles.get(k)
        if tl is None or not tl.winfo_exists():
            return
        tl.configure(border_color=self._tile_color(k))
        pair = self._tile_imgs.get(k)
        if pair:
            tl.configure(image=pair[1] if self._marked(k) else pair[0])
        chip = self._tile_rm.get(k)
        if chip is not None:
            if self._marked(k):
                chip.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")
            else:
                chip.place_forget()

    def _on_check(self, k):
        self._paint_tile(k)
        if self._bulk:
            return
        self._update_stats()
        self._update_controls()

    def refresh_marks(self):
        for k in list(self._tiles):
            self._paint_tile(k)
        self._update_stats()
        self._update_controls()

    def _tile_color(self, k, hover=False):
        if k in self.orig_set and not self._marked(k):
            return ACCENT
        return BORDER_HI if hover else TILE

    def _clear(self):
        self._render_id += 1
        for w in self.results.winfo_children():
            w.destroy()
        self._tiles.clear()
        self._tile_imgs.clear()
        self._tile_rm.clear()
        self._thumb_imgs.clear()
        self._progress_lbl = None
        try:
            self.results._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    def _empty_state(self, title, text, action=None):
        box = ctk.CTkFrame(self.results, fg_color="transparent")
        box.pack(pady=(120, 0))
        label(box, title, 18, BRIGHT, "bold").pack()
        label(box, text, T_BODY, MUTED, justify="center", wraplength=540).pack(pady=(10, 24))
        if action:
            action(box).pack()
        return box

    def _render(self):
        t = self.t
        self._clear()
        self._update_controls()
        self._update_stats()
        if self._busy():
            box = self._empty_state(t("scan_title"), t("scan_text"))
            self._pbar = ctk.CTkProgressBar(box, width=460, height=8, corner_radius=4, fg_color=BORDER_HI,
                                            progress_color=BRIGHT)
            self._pbar.set(0)
            self._pbar.pack()
            self._progress_lbl = label(box, t("prep"), T_HINT, MUTED)
            self._progress_lbl.pack(pady=(12, 0))
            self._stage_t0 = (None, 0.0)
            return
        folder = self.folder.get().strip().strip('"')
        if not self.res and folder and Path(folder).is_dir():
            # папка уже выбрана (кликом, перетаскиванием или с прошлого раза), но поиска ещё не было
            self._empty_state(t("ready_title"), t("ready_text", name=Path(folder).name or folder),
                              lambda p: button(p, t("find"), self._scan, "accent", ico=I_SEARCH,
                                               height=44, width=220))
            return
        if not self.res:
            box = self._empty_state(t("start_title"), t("start_text"),
                                    lambda p: button(p, t("start_btn"), self._pick_and_scan, "accent",
                                                     ico=I_FOLDER, height=44, width=250))
            label(box, t("drop_hint"), T_HINT, MUTED).pack(pady=(12, 0))
            return
        if not self.groups:
            self._empty_state(t("nodup_title"), t("nodup_text"))
            return
        s = self.results._get_widget_scaling()
        # сразу после перестройки фрейм ещё не размечен и его ширина 0 — берём ширину окна минус панель
        avail = max(self.results.winfo_width() / s, self.winfo_width() / s - SIDEBAR_W - 40, 400)
        cols = max(1, int((avail - 60) // (TW + 16)))
        start = self.page * GROUPS_PER_PAGE
        todo = list(enumerate(self.groups[start:start + GROUPS_PER_PAGE], start + 1))
        self._render_chunk(self._render_id, todo, cols, s)

    def _render_chunk(self, rid, todo, cols, s):
        # Группы добавляются по 3 штуки через after(), чтобы окно не зависало, пока страница заполняется.
        # Каждый виджет CTk — это отдельный canvas, создаётся медленно (страница из 20 групп рисовалась
        # ~3,7 с), поэтому подписи на плитках — обычные tk.Label, так раза в два быстрее.
        # rid нужен, чтобы выкинуть недорисованные куски, если пользователь уже переключил страницу.
        if rid != self._render_id or not todo:
            return
        t = self.t
        for gi, g in todo[:3]:
            box = ctk.CTkFrame(self.results, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)
            box.pack(fill="x", pady=(0, 14), padx=(0, 8))
            hd = tk.Frame(box, bg=CARD)
            hd.pack(fill="x", padx=22, pady=(16, 0))
            tk.Label(hd, text=t("group", i=gi), font=tkfont(T_BODY, "bold", s=s), fg=BRIGHT, bg=CARD).pack(side="left")
            tk.Label(hd, text="   " + t.n("files", len(g)), font=tkfont(T_HINT, s=s), fg=MUTED,
                     bg=CARD).pack(side="left")
            grid = tk.Frame(box, bg=CARD)
            grid.pack(fill="x", padx=16, pady=(12, 16))
            for pos, k in enumerate(g):
                self._tile(grid, g, pos, k, s).grid(row=pos // cols, column=pos % cols, padx=6, pady=4, sticky="n")
        self.after(1, self._render_chunk, rid, todo[3:], cols, s)

    def _tile(self, parent, g, pos, k, s):
        t, res = self.t, self.res
        p = res.files[k]
        cell = tk.Frame(parent, bg=CARD)
        img = ctk.CTkLabel(cell, text="", width=TW, height=TH, fg_color=TILE, corner_radius=8,
                           border_width=2, border_color=self._tile_color(k), cursor="hand2", bg_color=CARD)
        img.pack()
        self._tiles[k] = img
        img.bind("<Button-1>", lambda _e: Preview(self, g, pos))
        img.bind("<Enter>", lambda _e: img.configure(border_color=self._tile_color(k, hover=True)))
        img.bind("<Leave>", lambda _e: img.configure(border_color=self._tile_color(k)))

        # плашки поверх картинки, как значки на обложках в библиотеке Hydra
        if pos == 0:
            c = tk.Label(img, text=t("original"), font=tkfont(T_SMALL, "bold", s=s), fg=ON_ACCENT, bg=ACCENT,
                         padx=round(7 * s), pady=round(1 * s))
        else:
            c = tk.Label(img, text=pct(res.sim(g[0], k)), font=tkfont(T_SMALL, num=True, s=s), fg=BRIGHT,
                         bg="#0b0b0b", padx=round(7 * s), pady=round(1 * s))
        c.place(x=round(10 * s), y=round(10 * s))
        if core.is_video(p):  # плашка «видео» в левом нижнем углу, чтобы видео не терялись среди фото
            v = tk.Label(img, text=I_VIDEO, font=(ICON_FAMILY, -round(13 * s)), fg=BRIGHT, bg="#0b0b0b",
                         padx=round(6 * s), pady=round(4 * s))
            v.place(x=round(10 * s), rely=1.0, y=-round(10 * s), anchor="sw")
            v.bind("<Button-1>", lambda _e: Preview(self, g, pos))
        c.bind("<Button-1>", lambda _e: Preview(self, g, pos))
        self._tile_rm[k] = tk.Label(img, text=I_TRASH, font=(ICON_FAMILY, -round(15 * s)), fg=BRIGHT,
                                    bg="#0b0b0b", padx=round(6 * s), pady=round(4 * s))
        self._request_thumb(img, p, k)

        name = p.name if len(p.name) <= 28 else p.name[:25] + "…"
        tk.Label(cell, text=name, font=tkfont(T_HINT, "medium", s=s), fg=BRIGHT, bg=CARD,
                 anchor="w").pack(fill="x", pady=(8, 0))
        w, h = res.wh[k]
        tk.Label(cell, text=f"{w}×{h}  ·  {t.size(self.size_of(k))}", font=tkfont(T_SMALL, s=s), fg=MUTED,
                 bg=CARD, anchor="w").pack(fill="x", pady=(1, 0))
        checkbox(cell, t("remove"), self.checks[k], size=18, fsize=T_HINT).pack(anchor="w", pady=(8, 0))
        self._paint_tile(k)
        return cell

    def _request_thumb(self, lbl, p: Path, k: int):
        key = str(p)
        if key in self._thumb_cache:
            self._set_thumb(lbl, self._thumb_cache[key], k)
            return
        rid = self._render_id

        def job():
            try:
                im = load_image(p, TW * 2)
            except Exception:
                im = None
            if im is not None:
                if len(self._thumb_cache) > 600:
                    self._thumb_cache.clear()
                self._thumb_cache[key] = im
            self.q.put(("thumb", rid, lbl, im, k))

        self._pool.submit(job)

    def _set_thumb(self, lbl, im, k):
        if im is None:
            lbl.configure(text=self.t("no_preview"), text_color=MUTED, font=F(T_SMALL))
            return
        r = min((TW - 8) / im.width, (TH - 8) / im.height)
        size = (max(1, int(im.width * r)), max(1, int(im.height * r)))
        dim = ImageEnhance.Brightness(ImageOps.grayscale(im).convert("RGB")).enhance(DIM)
        pair = (ctk.CTkImage(light_image=im, dark_image=im, size=size),
                ctk.CTkImage(light_image=dim, dark_image=dim, size=size))
        self._thumb_imgs.extend(pair)
        self._tile_imgs[k] = pair
        self._paint_tile(k)

    # ---------------------------------------------------------- перенос и удаление
    def _apply(self, mode: str):
        t = self.t
        sel = {k for k, v in self.checks.items() if v.get()}
        if not sel:
            return
        size = t.size(sum(self.size_of(k) for k in sel))
        lost = sum(1 for g in self.groups if all(k in sel for k in g))
        warn = t("warn_all", g=t.n("groups_in", lost)) if lost else ""
        n = t.n("files", len(sel))

        def do():
            try:
                done, _report = core.apply(self.scan_folder, self.res, self.groups, sel, mode)
            except Exception as e:
                self.info(t("err_title"), f"{type(e).__name__}: {e}")
                return
            Toast(self, t(f"done_{mode}", n=t.n("files", done), size=size))
            self._scan()

        if mode == "move":
            Dialog(self, t("move_title"), t("move_text", n=n, size=size) + warn, t("move_ok"), "accent",
                   on_ok=do, ok_icon=I_MOVE)
        else:
            Dialog(self, t("trash_title"), t("trash_text", n=n, size=size) + warn, t("trash_ok"), "accent",
                   on_ok=do, ok_icon=I_TRASH)

    def _undo(self):
        t = self.t
        folder = self.folder.get().strip().strip('"') or None
        rep = filedialog.askopenfilename(title=t("undo_title"), initialdir=folder,
                                         filetypes=[(t("report"), "dupes_report_*.csv")])
        if not rep:
            return
        try:
            back, skipped = core.undo(Path(rep).parent, Path(rep))
        except Exception as e:
            self.info(t("err_title"), f"{type(e).__name__}: {e}")
            return
        Toast(self, t("undone", n=back) + (t("skipped", n=skipped) if skipped else ""), I_UNDO)
        if self.res:
            self._scan()


def main():
    import sys

    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
