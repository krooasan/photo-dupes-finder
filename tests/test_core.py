"""Тесты поиска: сбор файлов, группировка, выбор оригинала, видео, кэш, перенос и откат.

Запуск: py -3.11 -m pytest tests
Тесты с нейросетью пропускаются, если нет models/sscd_disc_mixup.onnx.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import has_model, hide, scene, with_bars, write_video
from photo_dupes import core

THR = 0.8


def names(res, groups):
    return [sorted(res.files[k].name for k in g) for g in groups]


def group_of(res, groups, name):
    for g in groups:
        if any(res.files[k].name == name for k in g):
            return sorted(res.files[k].name for k in g)
    return None


# ----------------------------------------------------------------------------- сбор файлов
def test_collect_skips_hidden_system_empty_and_service_dirs(tmp_path):
    im = scene(1, (200, 150))
    im.save(tmp_path / "ok.jpg")
    im.save(tmp_path / ".dot.jpg")
    im.save(tmp_path / "._mac.jpg")
    (tmp_path / "empty.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("не картинка")
    im.save(tmp_path / "hidden.jpg")
    hide(tmp_path / "hidden.jpg")
    for d in ("$RECYCLE.BIN", ".git", "System Volume Information", core.DUP_DIR_NAME, "sub"):
        (tmp_path / d).mkdir()
        im.save(tmp_path / d / "inside.jpg")

    flat = {p.name for p in core.collect_files(tmp_path, recursive=False)}
    deep = {str(p.relative_to(tmp_path)) for p in core.collect_files(tmp_path, recursive=True)}

    expected_flat = {"ok.jpg"} if os.name == "nt" else {"ok.jpg", "hidden.jpg"}
    assert flat == expected_flat
    assert deep == expected_flat | {os.path.join("sub", "inside.jpg")}


def test_collect_by_kind(tmp_path):
    scene(1, (200, 150)).save(tmp_path / "a.jpg")
    scene(2, (200, 150)).save(tmp_path / "b.PNG")
    write_video(tmp_path / "c.mp4", 3, frames=10)
    assert {p.name for p in core.collect_files(tmp_path, False, "photo")} == {"a.jpg", "b.PNG"}
    assert {p.name for p in core.collect_files(tmp_path, False, "video")} == {"c.mp4"}
    assert len(core.collect_files(tmp_path, False)) == 3


def test_file_sig_of_missing_file_is_none(tmp_path):
    assert core.file_sig(tmp_path / "nope.jpg") is None


def test_unique_dest(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "a_1.jpg").write_bytes(b"1")
    assert core.unique_dest(tmp_path / "b.jpg").name == "b.jpg"
    assert core.unique_dest(tmp_path / "a.jpg").name == "a_2.jpg"


# ----------------------------------------------------------------------------- без нейросети
def test_content_ratio_sees_bars():
    im = scene(5, (400, 300))
    assert core.content_ratio(im) > core.BARS_RATIO
    assert core.content_ratio(with_bars(im, 0.25)) < core.BARS_RATIO


def test_components_join_chains():
    labels = core._components(6, np.array([0, 1, 4]), np.array([1, 2, 5]))
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_similar_pairs_and_group_on_synthetic_vectors():
    """10 000 векторов, из них 100 пар почти одинаковых: находим ровно их, быстро и без лишних пар."""
    rng = np.random.default_rng(0)
    n = 10_000
    E = rng.normal(size=(n, 512)).astype(np.float32)
    E[1::100] = E[0::100] + rng.normal(scale=0.05, size=E[0::100].shape)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    t = time.time()
    i, j, s = core._similar_pairs(E, lambda *a: None)
    assert time.time() - t < 60
    assert len(i) <= n * core.TOP_K
    res = core.ScanResult(files=[Path(f"{k}.jpg") for k in range(n)], wh=np.full((n, 2), 100),
                          emb=E, ratio=np.ones(n, np.float32), size=np.ones(n), mtime=np.zeros(n),
                          pair_i=i, pair_j=j, pair_s=s)
    groups = core.group(res, THR)
    assert sorted(sorted(g) for g in groups) == [[a, a + 1] for a in range(0, n, 100)]


def test_original_prefers_no_bars_then_resolution_then_not_png():
    E = np.ones((4, 512), np.float32) / np.sqrt(512)
    res = core.ScanResult(
        files=[Path("big.png"), Path("bars.jpg"), Path("small.jpg"), Path("big.jpg")],
        wh=np.array([(1000, 800), (1400, 1000), (500, 400), (1000, 800)]),
        emb=E, ratio=np.array([1.0, 0.5, 1.0, 1.0], np.float32), size=np.array([5, 9, 1, 3]),
        mtime=np.zeros(4), pair_i=np.array([0, 0, 0, 1, 1, 2]), pair_j=np.array([1, 2, 3, 2, 3, 3]),
        pair_s=np.ones(6, np.float32))
    (g,) = core.group(res, THR)
    assert res.files[g[0]].name == "big.jpg"


# ----------------------------------------------------------------------------- с нейросетью
@pytest.fixture(scope="module")
def photo_folder(tmp_path_factory):
    """Папка с тремя «снимками» и разными их копиями, плюс несколько непохожих файлов."""
    d = tmp_path_factory.mktemp("photos")
    a, b = scene(10), scene(20)
    a.save(d / "a.jpg", quality=95)
    a.resize((480, 320)).save(d / "a_small.jpg", quality=90)
    a.save(d / "a_telegram.jpg", quality=35)
    a.crop((40, 30, 920, 610)).save(d / "a_crop.jpg", quality=92)
    with_bars(a, 0.1).save(d / "a_bars.png")
    b.save(d / "b.png")
    b.save(d / "b_copy.webp", quality=80)
    for s in range(30, 36):
        scene(s).save(d / f"single_{s}.jpg", quality=92)
    (d / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (d / "truncated.png").write_bytes((d / "b.png").read_bytes()[:500])
    return d


@has_model
def test_scan_groups_copies_and_keeps_singles_apart(photo_folder, quiet):
    res = core.scan(photo_folder, False, *quiet, kind="photo")
    groups = core.group(res, THR)
    assert group_of(res, groups, "a.jpg") == sorted(
        ["a.jpg", "a_small.jpg", "a_telegram.jpg", "a_crop.jpg", "a_bars.png"])
    assert group_of(res, groups, "b.png") == ["b.png", "b_copy.webp"]
    assert all(group_of(res, groups, f"single_{s}.jpg") is None for s in range(30, 36))
    assert {p.name for p, _ in res.errors} == {"broken.jpg", "truncated.png"}


@has_model
def test_original_is_full_size_jpeg_without_bars(photo_folder, quiet):
    res = core.scan(photo_folder, False, *quiet, kind="photo")
    groups = core.group(res, THR)
    g = next(g for g in groups if len(g) == 5)
    assert res.files[g[0]].name == "a.jpg"
    # при одинаковых пикселях PNG проигрывает: обычно это скриншот, а не исходная фотка
    g = next(g for g in groups if len(g) == 2)
    assert res.files[g[0]].name == "b_copy.webp"


@has_model
def test_cache_is_used_and_updated(tmp_path, quiet):
    scene(1).save(tmp_path / "x.jpg")
    scene(1).save(tmp_path / "x2.jpg", quality=60)
    r1 = core.scan(tmp_path, False, *quiet)
    assert r1.device in ("gpu", "cpu")
    assert (tmp_path / core.CACHE_NAME).exists()
    r2 = core.scan(tmp_path, False, *quiet)
    assert r2.device == "cache"
    assert np.allclose(r1.emb, r2.emb)
    # файл поменялся — его надо посчитать заново, а остальное взять из кэша
    time.sleep(0.05)
    scene(2).save(tmp_path / "x2.jpg")
    r3 = core.scan(tmp_path, False, *quiet)
    assert r3.device in ("gpu", "cpu")
    assert core.group(r3, THR) == []


@has_model
def test_file_deleted_between_listing_and_reading(tmp_path, quiet, monkeypatch):
    scene(1).save(tmp_path / "a.jpg")
    scene(1).save(tmp_path / "b.jpg", quality=70)
    real = core.collect_files

    def listing_then_delete(*a, **kw):
        files = real(*a, **kw)
        (tmp_path / "b.jpg").unlink()
        return files

    monkeypatch.setattr(core, "collect_files", listing_then_delete)
    res = core.scan(tmp_path, False, *quiet)
    assert [p.name for p in res.files] == ["a.jpg"]


@has_model
def test_cancel_raises_and_keeps_partial_cache(tmp_path):
    for s in range(70):
        scene(s, (320, 240)).save(tmp_path / f"{s}.jpg")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(InterruptedError):
        core.scan(tmp_path, False, lambda *a: None, cancel)


@has_model
def test_videos(tmp_path, quiet):
    write_video(tmp_path / "v.mp4", 50, frames=80, size=(640, 480))
    write_video(tmp_path / "v_trimmed.mp4", 50, frames=80, size=(640, 480), start=15)
    # 75% размера. Копия в 50% (320×240) после mp4v даёт сходство ~0.78 — найдётся только на строгости 0.75
    write_video(tmp_path / "v_small.mp4", 50, frames=80, size=(480, 360))
    write_video(tmp_path / "other.mp4", 60, frames=80, size=(640, 480))
    (tmp_path / "broken.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 nope")
    scene(50).save(tmp_path / "photo.jpg")  # в режиме «Видео» фото не участвует
    res = core.scan(tmp_path, False, *quiet, kind="video")
    groups = core.group(res, THR)
    assert all(core.is_video(p) for p in res.files)
    assert group_of(res, groups, "v.mp4") == ["v.mp4", "v_small.mp4", "v_trimmed.mp4"]
    assert group_of(res, groups, "other.mp4") is None
    assert [p.name for p, _ in res.errors] == ["broken.mp4"]


@has_model
def test_move_and_undo(tmp_path, quiet):
    a = scene(7)
    a.save(tmp_path / "keep.jpg", quality=95)
    a.save(tmp_path / "copy1.jpg", quality=50)
    a.resize((480, 320)).save(tmp_path / "copy2.jpg")
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.jpg")}
    res = core.scan(tmp_path, False, *quiet)
    groups = core.group(res, THR)
    (g,) = groups
    selected = set(g[1:])

    count, report = core.apply(tmp_path, res, groups, selected, "move")
    assert count == 2
    assert {p.name for p in tmp_path.glob("*.jpg")} == {"keep.jpg"}
    gdir = tmp_path / core.DUP_DIR_NAME / "group_0001"
    assert {p.name for p in gdir.iterdir()} == {"copy1.jpg", "copy2.jpg", "ORIGINAL_keep.jpg"}
    assert report.exists()

    back, skipped = core.undo(tmp_path, report)
    assert (back, skipped) == (2, 0)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.jpg")} == before
    assert not (tmp_path / core.DUP_DIR_NAME).exists()


@has_model
def test_second_move_continues_group_numbers(tmp_path, quiet):
    for s in (1, 2):
        scene(s).save(tmp_path / f"{s}.jpg")
        scene(s).save(tmp_path / f"{s}_copy.jpg", quality=50)
    res = core.scan(tmp_path, False, *quiet)
    groups = core.group(res, THR)
    assert len(groups) == 2
    core.apply(tmp_path, res, groups[:1], set(groups[0][1:]), "move")
    core.apply(tmp_path, res, groups[1:], set(groups[1][1:]), "move")
    assert sorted(p.name for p in (tmp_path / core.DUP_DIR_NAME).iterdir()) == ["group_0001", "group_0002"]


@has_model
def test_trash_uses_recycle_bin(tmp_path, quiet, monkeypatch):
    import send2trash

    trashed = []
    monkeypatch.setattr(send2trash, "send2trash", lambda p: (trashed.append(Path(p).name), os.remove(p)))
    scene(3).save(tmp_path / "a.jpg")
    scene(3).save(tmp_path / "b.jpg", quality=50)
    res = core.scan(tmp_path, False, *quiet)
    groups = core.group(res, THR)
    count, _ = core.apply(tmp_path, res, groups, set(groups[0][1:]), "trash")
    assert count == 1 and len(trashed) == 1
    assert not (tmp_path / core.DUP_DIR_NAME).exists()


@has_model
def test_long_screenshot_and_cyrillic_path(tmp_path, quiet):
    d = tmp_path / "Скриншоты отпуск"
    d.mkdir()
    long = scene(9, (1080, 1600)).resize((1080, 16000))
    long.save(d / "длинный.png")
    long.resize((540, 8000)).save(d / "длинный_копия.jpg")
    res = core.scan(d, False, *quiet)
    assert not res.errors
    assert len(core.group(res, THR)) == 1


@has_model
def test_cpu_and_gpu_give_same_fingerprints(tmp_path):
    scene(4).save(tmp_path / "a.jpg")
    scene(4).save(tmp_path / "b.jpg", quality=50)
    gpu = core.scan(tmp_path, False, lambda *a: None, lambda: False, use_gpu=True)
    (tmp_path / core.CACHE_NAME).unlink()
    cpu = core.scan(tmp_path, False, lambda *a: None, lambda: False, use_gpu=False)
    assert cpu.device == "cpu"
    assert np.allclose(gpu.emb, cpu.emb, atol=1e-2)


@has_model
def test_rotated_phone_photo_reports_real_size(tmp_path, quiet):
    """Телефон пишет пиксели «лёжа» и ставит в EXIF флаг поворота. Размер должен быть как на экране."""
    im = scene(8, (800, 600))
    exif = Image.Exif()
    exif[0x0112] = 6  # повернуть на 90°
    im.save(tmp_path / "portrait.jpg", exif=exif)
    im.rotate(-90, expand=True).save(tmp_path / "portrait_copy.jpg", quality=70)
    res = core.scan(tmp_path, False, *quiet)
    assert [tuple(map(int, wh)) for wh in res.wh] == [(600, 800), (600, 800)]
    assert len(core.group(res, THR)) == 1
