"""
Convert Meta SSCD (sscd_disc_mixup, TorchScript) to ONNX and check that the
ONNX model gives the same embeddings as the original.

Needs torch (only for this one-time conversion, the app itself does not):
    py -3.11 -m pip install torch onnx onnxruntime
    py -3.11 tools/export_onnx.py
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SRC = Path.home() / ".cache" / "sscd" / "sscd_disc_mixup.torchscript.pt"
OUT = Path(__file__).resolve().parent.parent / "models" / "sscd_disc_mixup.onnx"


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(MODEL_URL, str(SRC))
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

    # check: several shapes and batch sizes, cosine similarity torch vs onnx
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
