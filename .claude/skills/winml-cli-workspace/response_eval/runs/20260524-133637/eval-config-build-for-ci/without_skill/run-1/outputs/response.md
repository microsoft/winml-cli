# CI Job: Rebuild Optimized ResNet50 for QNN on Config Change

Here's a clean, reproducible setup that uses a single JSON config in the repo, plus a deterministic build step that CI runs whenever the config changes.

## 1. Repo layout

```
repo/
  models/
    resnet50/
      config.json            # single source of truth
      build.py               # reproducible build script
      requirements.txt       # pinned tool versions
  .github/workflows/
    build-resnet50-qnn.yml   # CI workflow
```

Keep the config, build script, and pinned tool versions co-located so a change to any one is reviewable in a single PR.

## 2. The single JSON config (`models/resnet50/config.json`)

Put everything the build needs in one declarative file. Nothing about the build should be implicit.

```json
{
  "model": {
    "name": "resnet50",
    "source": "https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz",
    "sha256": "REPLACE_WITH_KNOWN_HASH",
    "opset": 17,
    "input_shape": [1, 3, 224, 224],
    "input_name": "data",
    "input_dtype": "float32"
  },
  "target": {
    "execution_provider": "QNNExecutionProvider",
    "backend": "HTP",
    "soc_model": "SM8650",
    "htp_arch": "73",
    "precision": "int8"
  },
  "optimization": {
    "graph_optimization_level": "all",
    "shape_inference": true,
    "constant_folding": true,
    "transpose_optimization": true
  },
  "quantization": {
    "mode": "static",
    "activation_type": "uint8",
    "weight_type": "int8",
    "per_channel": true,
    "symmetric_weights": true,
    "calibration": {
      "method": "minmax",
      "dataset_uri": "azureml://datasets/imagenet-calib-256",
      "num_samples": 256,
      "preprocessing": {
        "resize": 256,
        "center_crop": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225]
      }
    }
  },
  "output": {
    "artifact_name": "resnet50_qnn_int8.onnx",
    "context_binary": "resnet50_qnn_int8.bin",
    "save_intermediate": false
  },
  "reproducibility": {
    "seed": 0,
    "onnx_version": "1.16.1",
    "onnxruntime_version": "1.18.0",
    "qnn_sdk_version": "2.22.0"
  }
}
```

Why this shape:
- One file -> one watched path in CI.
- `sha256` on the source model makes the input bit-exact.
- `reproducibility` block pins every tool so reruns produce identical artifacts.
- `target.*` captures QNN HTP backend specifics (SoC, HTP arch) needed for context-binary caching.

Validate the config in CI against a JSON Schema (`config.schema.json`) so malformed PRs fail fast.

## 3. Reproducible build step (`models/resnet50/build.py`)

The script must be a pure function of `config.json` + pinned tools.

```python
# models/resnet50/build.py
import hashlib, json, os, random, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

CFG = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(os.environ.get("BUILD_OUT", "build"))
OUT.mkdir(parents=True, exist_ok=True)

# 1. Determinism
random.seed(CFG["reproducibility"]["seed"])
np.random.seed(CFG["reproducibility"]["seed"])

# 2. Fetch + verify source model
src = OUT / "source.onnx"
urllib.request.urlretrieve(CFG["model"]["source"], src)
got = hashlib.sha256(src.read_bytes()).hexdigest()
assert got == CFG["model"]["sha256"], f"sha mismatch: {got}"

# 3. Optimize (graph opts, shape inference, constant folding)
opt = OUT / "optimized.onnx"
# ... ORT SessionOptions with graph_optimization_level=ORT_ENABLE_ALL,
# write optimized model via session_options.optimized_model_filepath

# 4. Calibration data reader (driven entirely by config)
class Reader(CalibrationDataReader):
    ...  # implements preprocessing from CFG["quantization"]["calibration"]

# 5. Static QDQ quantization for QNN HTP
q_out = OUT / CFG["output"]["artifact_name"]
quantize_static(
    model_input=str(opt),
    model_output=str(q_out),
    calibration_data_reader=Reader(CFG),
    quant_format="QDQ",
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=CFG["quantization"]["per_channel"],
)

# 6. Generate QNN HTP context binary (offline EP context cache)
# Use ORT with provider="QNNExecutionProvider" and provider_options:
#   backend_path=QnnHtp.dll, qnn_context_cache_enable=1,
#   qnn_context_cache_path=<context_binary>, htp_arch=<from config>, soc_model=<from config>
# Loading the quantized model once with these options emits the .bin.

# 7. Emit a build manifest with input + tool hashes for traceability
manifest = {
    "config_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
    "artifact_sha256": hashlib.sha256(q_out.read_bytes()).hexdigest(),
    "tooling": CFG["reproducibility"],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

Key reproducibility properties:
- Inputs hashed (source model + config).
- All randomness seeded.
- Tool versions pinned in `requirements.txt` (`onnx==1.16.1`, `onnxruntime-qnn==1.18.0`, etc.).
- Output is hashed and recorded in a `manifest.json` next to the artifact.

## 4. CI workflow (`.github/workflows/build-resnet50-qnn.yml`)

Triggers only when the config (or build script) changes. Runs on a Windows ARM64 runner that has the QNN SDK installed (HTP context binaries must be generated on the target architecture).

```yaml
name: build-resnet50-qnn

on:
  pull_request:
    paths:
      - "models/resnet50/config.json"
      - "models/resnet50/build.py"
      - "models/resnet50/requirements.txt"
  push:
    branches: [main]
    paths:
      - "models/resnet50/config.json"
      - "models/resnet50/build.py"
      - "models/resnet50/requirements.txt"
  workflow_dispatch:

jobs:
  build:
    runs-on: [self-hosted, windows, arm64, qnn]   # runner with QNN SDK + HTP
    steps:
      - uses: actions/checkout@v4

      - name: Validate config against schema
        run: |
          python -m pip install jsonschema
          python -c "import json,jsonschema; jsonschema.validate(
            json.load(open('models/resnet50/config.json')),
            json.load(open('models/resnet50/config.schema.json')))"

      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }

      - name: Install pinned tools
        run: pip install -r models/resnet50/requirements.txt

      - name: Cache calibration data
        uses: actions/cache@v4
        with:
          path: .cache/calib
          key: calib-${{ hashFiles('models/resnet50/config.json') }}

      - name: Build optimized + quantized model
        env:
          BUILD_OUT: build/resnet50
          QNN_SDK_ROOT: ${{ env.QNN_SDK_ROOT }}
        run: python models/resnet50/build.py models/resnet50/config.json

      - name: Smoke-test with QNN EP
        run: python models/resnet50/smoke_test.py build/resnet50/resnet50_qnn_int8.onnx

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: resnet50-qnn-${{ github.sha }}
          path: |
            build/resnet50/resnet50_qnn_int8.onnx
            build/resnet50/resnet50_qnn_int8.bin
            build/resnet50/manifest.json
```

Notes:
- `paths:` filter is what gives you "rebuild when the config changes" — nothing else triggers it.
- Runner label `qnn` selects a machine that has the Qualcomm QNN SDK installed and (ideally) the HTP backend available for context-binary generation. If you can't run on real hardware in CI, you can still emit the QDQ-quantized ONNX and defer context-binary generation to a device-side job.
- Artifacts are uploaded with the commit SHA so each rebuild is traceable.

## 5. Make rebuilds verifiable (optional but recommended)

- Have CI re-run the build twice in parallel and assert `manifest.json` hashes match. This catches non-determinism quickly (common culprits: thread count, calibration shuffling, ORT graph-opt randomness).
- Gate merge on an accuracy check (top-1 on a small ImageNet val slice) and a latency budget on the QNN HTP runner. Store thresholds in the same `config.json` under an `acceptance` block so reviewers see proposed accuracy/latency changes alongside config changes.
- Publish artifacts to a model registry (Azure Blob, GitHub Releases, or an internal feed) keyed by `config_sha256` so consumers can pull "the model built from this exact config".

## 6. Things to watch out for with QNN specifically

- HTP context binaries are tied to the SoC + HTP arch declared in the config; if you target multiple devices, parameterize the workflow with a matrix on `target.soc_model`.
- QNN expects QDQ-format quantized models; do not use the operator-quantized format.
- Per-channel weights + symmetric int8 weights with uint8 activations is the standard combo for QNN HTP — keep that in the config so reviewers can see it.
- Pin `onnxruntime-qnn` (not vanilla `onnxruntime`) in `requirements.txt`, and pin the QNN SDK version on the runner image.

## Summary

- One `models/resnet50/config.json` declares model source (+ hash), target (QNN/HTP/SoC), optimization, quantization, calibration, and pinned tool versions.
- `build.py` is a pure function of that config; it hashes inputs and outputs into `manifest.json`.
- CI triggers on `paths:` changes to the config or build script, runs on a QNN-capable runner, validates the config, builds, smoke-tests, and uploads the artifact + manifest.
- Determinism comes from: pinned tools, seeded RNGs, hashed inputs/outputs, and (optionally) a double-build hash check.
