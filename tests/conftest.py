"""Общие заготовки для тестов: генерируем картинки и видео сами, свои фото не нужны."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photo_dupes import core  # noqa: E402


def scene(seed: int, size=(960, 640)) -> Image.Image:
    """Случайная «картинка»: градиент, цветные фигуры и немного шума. У разных seed — разные сцены."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    c0, c1 = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
    base = (c0 * (1 - xx[..., None]) + c1 * yy[..., None]).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(base)
    d = ImageDraw.Draw(im)
    for _ in range(14):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(30, 220)
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            d.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=col)
        else:
            d.rectangle((x0 - r, y0 - r // 2, x0 + r, y0 + r // 2), fill=col)
    im = im.filter(ImageFilter.GaussianBlur(1.5))
    noise = rng.normal(0, 6, (h, w, 3))
    return Image.fromarray((np.asarray(im, dtype=np.float32) + noise).clip(0, 255).astype(np.uint8))


def with_bars(im: Image.Image, pad=0.2) -> Image.Image:
    """Та же картинка внутри чёрной рамки — как скриншот видео на весь экран."""
    w, h = im.size
    out = Image.new("RGB", (int(w * (1 + 2 * pad)), int(h * (1 + 2 * pad))), (0, 0, 0))
    out.paste(im, (int(w * pad), int(h * pad)))
    return out


def write_video(path: Path, seed: int, frames=60, size=(320, 240), fps=10, start=0):
    """Короткое mp4: сцена, по которой ползёт квадрат. start — пропустить первые кадры (обрезка по времени)."""
    import cv2

    # сцена всегда одного размера и потом уменьшается: иначе у копии другого разрешения
    # получилась бы совсем другая картинка (фигуры раскладываются в зависимости от размера)
    base = scene(seed, (640, 480))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(start, frames):
        f = np.asarray(base).copy()
        x = int(600 * i / frames)
        f[200:280, x:x + 40] = (255, 255, 255)
        f = np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR))
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def hide(p: Path):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x2)  # FILE_ATTRIBUTE_HIDDEN


has_model = pytest.mark.skipif(
    not (Path(core.resource_dir()) / "models" / core.MODEL_NAME).exists(),
    reason="нет models/sscd_disc_mixup.onnx — сначала tools/export_onnx.py")


@pytest.fixture
def quiet():
    """progress и cancel для scan(), которые ничего не делают."""
    return (lambda *a: None), (lambda: False)
