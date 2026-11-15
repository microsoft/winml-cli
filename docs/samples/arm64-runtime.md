# Native Windows Arm64 Runtime preview

The native Arm64 installation supports GGUF text generation through
`winml perf --runtime winml-runtime`: Windows ML Tasks drives the Runtime
llama.cpp backend. Use native Arm64 Python 3.11, not x64 Python under emulation.
The experimental wheel drop must contain both `windowsml` and
`windowsml-llama-core` **2.9.31a0** for `win_arm64`.

## Implementation

1. Select dependencies by the Python process architecture. On Windows Arm64,
   install the CLI dependencies plus the matching Runtime and llama.cpp core.
   Exclude ONNX, ORT, PyTorch, NumPy and the export/evaluation stack. The x64
   dependency set is retained.
2. Reuse the existing GGUF Runtime/Task session and benchmark command. No
   model-specific inference implementation or ONNX conversion is required.
3. Record Python architecture and version in Task performance reports, and
   validate installation, import isolation and real generation independently
   from the main test suite, whose fixtures require ONNX and PyTorch.

With the optional `runtime-classic` extra, prebuilt source ONNX models can also
run through `winml perf --runtime winml-runtime --backend cgc` on the GPU.
The CLI uses the Runtime session directly, without ORT, Torch or model-export
wrappers. ONNX schema utilities and performance context types load independently.

This profile does not support ONNX export, ONNX evaluation, ORT inference, or
the complete developer dependency group. CPU GGUF is verified; GPU requires a
compatible backend and hardware. The supplied CUDA wheel is not needed for CPU
and does not provide acceleration on an Adreno GPU.

## Install

Extract the supplied ZIP and locate its `wheels` directory. From a source
checkout, with a standard native Arm64 Python 3.11 installation:

```powershell
python -c "import platform; print(platform.machine())" # must print ARM64
python -m venv .venv-arm64
.venv-arm64/Scripts/python.exe -m pip install -e . --find-links C:/path/to/drop/wheels
.venv-arm64/Scripts/python.exe -m pip check
.venv-arm64/Scripts/python.exe -m winml.modelkit perf --help
```

Use `pip` against this dedicated environment. `uv sync` also resolves the
full developer groups and unrelated platforms; it is not the minimal preview
installation workflow. Do not install `windowsml[with-ort]` or the
`windowsml-llama` CUDA bundle for this CPU profile. The CLI metadata selects
the exact matching core wheel automatically from `--find-links`.

## Classic source ONNX on GPU

Install the extra using the same wheel directory:

```powershell
.venv-arm64/Scripts/python.exe -m pip install -e ".[runtime-classic]" --find-links C:/path/to/drop/wheels
.venv-arm64/Scripts/python.exe -m winml.modelkit perf `
  --model C:/models/clip-image/model.onnx --runtime winml-runtime --backend cgc `
  --device gpu --iterations 3 --warmup 1 --input-data C:/models/clip-image/inputs.npz `
  --format json --output temp/clip-online.json
```

The native Arm64 extra uses NumPy >=2.4 and ONNX >=1.23 wheels instead of the
x64 pins. It does not install ORT or Torch. Source ONNX is compiled through
the Runtime ModelCompiler during this invocation; no prebuilt MLIR is required.

NPZ keys, dtypes and shapes must match the model. Without an NPZ, floating-point
inputs are generated with NumPy; integer inputs require value-range metadata.
Use `--shape-config` for symbolic dimensions, or provide concrete NPZ inputs.
`--device-luid` selects a GPU explicitly; the default uses DXCore hardware GPU
selection. Reports include placement diagnostics and exclude warmup timings.

This minimal classic profile supports fixed iteration counts, source ONNX,
GPU selection, input data, shape overrides, text/JSON output and RSS memory
checkpoints. Device-memory measurements are explicitly unavailable. Advanced
monitoring, ETW, duration mode and build/export options are rejected when
explicitly supplied; they are not silently ignored.

The CLI checks finite outputs, not model accuracy. The CLIP image encoder passed
the separate ORT CPU comparison; the text encoder has a known numerical mismatch
in this preview. Successful performance runs do not mark that mismatch resolved.

`tests/runtime_minimal/test_classic_cli.py` exercises both encoders via the real
CLI with an import hook that rejects ORT, Torch, Transformers and Optimum.
Set `WINML_CLIP_TEST_ROOT` to a fixture root containing `models/image/model.onnx`,
`models/text/model.onnx` and the input binary fixtures used by
`test_classic_online.py`. `WINML_CLIP_TEST_OUTPUT` optionally saves the CLI reports.

## Verify a small LLM

Download `Qwen3-0.6B-Q8_0.gguf` from the official
[Qwen model repository](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF).
Pass the local file to the existing command:

```powershell
.venv-arm64/Scripts/python.exe -m winml.modelkit perf `
  --model C:/models/Qwen3-0.6B-Q8_0.gguf --runtime winml-runtime --device cpu `
  --prompt "Name three primary colors. /no_think" --max-new-tokens 64 `
  --context-length 512 --warmup 1 --iterations 3 --no-memory `
  --format json --output temp/qwen3-arm64.json
```

Run the isolated tests, including the native hardware test:

```powershell
.venv-arm64/Scripts/python.exe -m pip install pytest packaging
$env:WINML_ARM64_TEST_MODEL = "C:/models/Qwen3-0.6B-Q8_0.gguf"
.venv-arm64/Scripts/python.exe -m pytest -c tests/runtime_minimal/pytest.ini `
  --confcutdir tests/runtime_minimal tests/runtime_minimal
```

`WINML_ARM64_TEST_REPORT` optionally names a new JSON output file for the test.
The test checks native process architecture, matching packages, CPU placement,
nonempty generation, token limits, request counts and positive timings. Without
a model path, only the hardware test is skipped. These are execution checks,
not an accuracy benchmark. Task TTFT includes native Task processing; stream
token throughput is not a model-only kernel measurement.
