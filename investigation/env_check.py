"""Environment check: register QNN EP via Windows ML and run resnet-50.onnx.

This is a pre-flight check for the LoRA/QNN investigation. It does NOT install
anything. It uses the WinML ``EpCatalog`` registration path (the same one the
repo's ``ep_registry.py`` uses) because this ORT build ships QNN as a WinML
plugin EP rather than the classic ``backend_path=QnnCpu.dll`` provider option.

If this runs an inference successfully, the environment is good to proceed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from windowsml import EpCatalog

QNN = "QNNExecutionProvider"


def register_qnn_devices() -> list[ort.OrtEpDevice]:
    """Register the QNN EP DLL with ORT and return its EpDevices."""
    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            if provider.name != QNN:
                continue
            provider.ensure_ready()  # download/install MSIX if missing; no-op when ready
            ort.register_execution_provider_library(provider.name, provider.library_path)
            print(f"[env] registered {provider.name} from {provider.library_path}")
            break
        else:
            raise RuntimeError(f"{QNN} not found in WinML EpCatalog")

    devices = [d for d in ort.get_ep_devices() if d.ep_name == QNN]
    for d in devices:
        print(f"[env] QNN EpDevice: hw_type={d.device.type} vendor={getattr(d.device, 'vendor', '?')}")
    if not devices:
        raise RuntimeError(f"No EpDevice exposed for {QNN}")
    return devices


def main() -> int:
    model_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("resnet-50.onnx").resolve()
    if not model_path.exists():
        print(f"[env] model not found: {model_path}")
        return 2

    print(f"[env] ort={ort.__version__} python={sys.version.split()[0]}")
    devices = register_qnn_devices()

    so = ort.SessionOptions()
    # Pick the first QNN device WinML exposes (NPU on Snapdragon). Functional check.
    so.add_provider_for_devices([devices[0]], {})

    print(f"[env] creating session for {model_path.name} on {QNN} ...")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), so)
    t_build = time.perf_counter() - t0
    print(f"[env] session built in {t_build * 1000:.1f} ms")

    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.rand(*shape).astype(np.float32)
        print(f"[env] input {inp.name} shape={shape} dtype={inp.type}")

    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    t_run = time.perf_counter() - t0
    print(f"[env] run OK in {t_run * 1000:.1f} ms")
    for o, arr in zip(sess.get_outputs(), outputs):
        print(f"[env] output {o.name} shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
    print("[env] === ENVIRONMENT OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
