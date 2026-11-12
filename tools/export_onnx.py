"""
Переводит модель Meta SSCD (sscd_disc_mixup, TorchScript) в ONNX и проверяет,
что ONNX выдаёт те же векторы, что и оригинал.

torch нужен только здесь, один раз. Самой программе он не нужен:
    py -3.11 -m pip install torch onnx onnxruntime
    py -3.11 tools/export_onnx.py
"""
import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
MODEL_SHA256 = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"
SRC = Path.home() / ".cache" / "sscd" / "sscd_disc_mixup.torchscript.pt"
OUT = Path(__file__).resolve().parent.parent / "models" / "sscd_disc_mixup.onnx"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    SRC.parent.mkdir(parents=True, exist_ok=True)
    tmp = SRC.with_suffix(".part")
    try:
        torch.hub.download_url_to_file(MODEL_URL, str(tmp))
    except Exception as e:
        # В ноябре 2026 у dl.fbaipublicfiles.com истёк сертификат. Тогда качаем без проверки
        # сертификата — подмену всё равно поймает сверка SHA-256 ниже.
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        print("certificate check failed, downloading without it and verifying SHA-256")
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if sha256(tmp) != MODEL_SHA256:
        tmp.unlink()
        sys.exit("downloaded model does not match SHA-256")
    tmp.replace(SRC)


def main():
    if not SRC.exists() or sha256(SRC) != MODEL_SHA256:
        download()
    model = torch.jit.load(str(SRC), map_location="cpu").eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 288, 384)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # сверяем на разных размерах и пачках: косинус между выходом torch и onnx должен быть ~1
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    worst = 1.0
    with torch.inference_mode():
        for shape in [(1, 3, 288, 288), (4, 3, 288, 512), (2, 3, 640, 288)]:
            x = torch.randn(*shape)
            a = torch.nn.functional.normalize(model(x), dim=1).numpy()
            b = sess.run(None, {"image": x.numpy()})[0]
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            cos = float((a * b).sum(axis=1).min())
            worst = min(worst, cos)
            print(f"  {shape}: min cosine torch vs onnx = {cos:.6f}")
    if worst < 0.9999:
        sys.exit("ONNX output differs from TorchScript!")
    print("OK")


if __name__ == "__main__":
    main()
