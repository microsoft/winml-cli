# GGUF — llama.cpp on Windows ML Runtime (Experimental)

`winml perf` can benchmark a GGUF language model on the llama.cpp backend of the
Windows ML Runtime API (`windowsml.runtime`). The Runtime selects llama.cpp from
the `.gguf` extension, and winml-cli drives the model as one decoder stage with
greedy decoding. The report uses the same prefill, time-to-first-token (TTFT), and
decode metrics as `--runtime ort-genai`, so GGUF and onnxruntime-genai runs of the
same model can be compared field by field.

!!! warning "Experimental"
    llama.cpp support ships only in the preview **Windows ML 2.7 experimental**
    packages. winml-cli does not install or pin them: its own dependencies keep the
    stable `windowsml` release. Install the preview packages yourself, in a separate
    virtual environment, only when you need to benchmark GGUF models. The preview
    Runtime API can change between releases.

## Prerequisites

- Windows x64 and Python 3.11 (the winml-cli requirements).
- winml-cli, from PyPI or from a source checkout.
- The preview packages below, all at the **same version**. Each llama.cpp
  distribution pins the exact `windowsml` version it was built with.

| Package | Version | Provides |
|---|---|---|
| `windowsml[with-ort]` | `2.7.2021a0` | The Runtime API (`windowsml.runtime`) and the matching `onnxruntime-windowsml` build |
| `windowsml-llama-core` | `2.7.2021a0` | The Runtime llama.cpp adapter, the llama.cpp and ggml modules, and the CPU backend |
| `windowsml-llama-cuda` (optional) | `2.7.2021a0` | The NVIDIA CUDA backend for `--device gpu`; depends on NVIDIA's `nvidia-cublas` wheel |

- A local GGUF file. For example, the Windows ML samples use
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` from the `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
  Hugging Face repository. Review the model license before you download it:

    ```powershell
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='models'))"
    ```

    For a sharded model, keep every shard in one folder and pass the first shard.

## Step 1: Create an experimental environment

The preview packages replace the `windowsml` version that winml-cli pins (and can
replace its `onnxruntime-windowsml` version), so keep them out of your regular
environment.

=== "uv (source checkout)"

    ```powershell
    uv venv .venv-gguf --python 3.11
    .venv-gguf\Scripts\activate
    uv pip install -e .
    uv pip install --prerelease=allow "windowsml[with-ort]==2.7.2021a0" "windowsml-llama-core==2.7.2021a0"
    ```

=== "pip"

    ```powershell
    py -3.11 -m venv .venv-gguf
    .venv-gguf\Scripts\activate
    python -m pip install winml-cli
    python -m pip install --pre "windowsml[with-ort]==2.7.2021a0" "windowsml-llama-core==2.7.2021a0"
    ```

`pip` (or `uv pip check`) reports that winml-cli requires a different `windowsml`
version. That conflict is expected in this environment.

!!! note "Run `winml` from the activated environment"
    Run `winml` directly in the activated environment. `uv run` and `uv sync`
    synchronize the project environment back to the pinned packages, which removes
    the preview Runtime.

Check the installation:

```powershell
python -c "import windowsml.runtime; print('Windows ML Runtime API is available')"
uv pip show windowsml-llama-core   # or: python -m pip show windowsml-llama-core
```

## Step 2: Benchmark the model

```powershell
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --device cpu `
  --max-new-tokens 64 --prompt "Name three primary colors."
```

`--runtime auto` (the default) selects `winml-runtime` for `.gguf` files, and
`--device auto` runs GGUF models on the CPU. Without explicit values, GGUF runs use
2 warmup and 10 timed generations, like `--runtime ort-genai`.

The console report lists the runtime and backend, the device, the installed preview
package versions, the prompt and generated token counts with the stop reason, the
model response, startup spans, a TTFT table, and prefill and decode throughput. The
JSON report is written to `~/.cache/winml/perf/<model-file-stem>/<timestamp>.json`
unless you pass `--output`.

These options apply to GGUF models:

| Option | Behavior with a `.gguf` model |
|---|---|
| `--model` / `-m` | Path to a local `.gguf` file (the first shard of a sharded model). |
| `--runtime` | `auto` or `winml-runtime`. Other runtimes are rejected. |
| `--device` | `auto` (CPU), `cpu`, or `gpu`. `npu` is not supported. |
| `--prompt`, `--prompt-file` | The prompt for every generation. |
| `--apply-template/--no-apply-template` | Render the prompt with the chat template stored in the GGUF file. Without a template, the raw prompt is used and the report sets `chat_template_applied` to `false`. |
| `--max-new-tokens` | Upper bound on generated tokens per generation. |
| `--iterations`, `--warmup` | Timed and warmup generations (defaults `10` and `2`). |
| `--memory/--no-memory` | Process RAM at the load and inference checkpoints. |
| `--output`, `--overwrite`, `--format json` | Same as for the other runtimes. |

Options that select another benchmark target are rejected: `--ep`, `--ep-options`,
`--backend`, `--device-luid`, `--input-data`, `--module`, and `--submodel`.
Build-only and ONNX-only options, such as `--precision`, `--task`, `--compile`,
`--op-tracing`, `--duration`, `--batch-size`, and `--monitor`, are ignored with a
warning.

## Context capacity

winml-cli encodes the prompt first and sets the decoder's sequence capacity to the
prompt tokens plus `--max-new-tokens` before it builds the pipeline. Without this
hint, llama.cpp can reserve a KV cache for the model's full training context. The
capacity the Runtime reports is saved as `benchmark_info.context_length`.

A generation stops at the end-of-sequence token, at `--max-new-tokens`, or at the
capacity. The stop reason of the first timed generation is saved as
`benchmark_info.stop_reason`, and `aggregate.stop_reason_counts` counts all timed
generations.

## Run on a GPU

The `windowsml-llama-core` wheel ships only the CPU backend. `--device gpu` needs a
llama.cpp GPU backend in the `LlamaBackends` folder beside the Runtime
(`windowsml\lib\LlamaBackends\<family>` in the environment):

- **NVIDIA:** install the CUDA backend. It depends on NVIDIA's `nvidia-cublas`
  wheel, published under the NVIDIA software license, which installing it accepts.

    ```powershell
    uv pip install --prerelease=allow "windowsml-llama-cuda==2.7.2021a0"
    # or: python -m pip install --pre "windowsml-llama-cuda==2.7.2021a0"
    winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --device gpu
    ```

- **Other backend families**, such as Vulkan or OpenCL: build the backend with
  [`Build-LlamaBackend.ps1`](https://github.com/microsoft/WindowsML/blob/main/Tools/llama/README.md)
  from the Windows ML repository, and pass the environment's `windowsml\lib` folder
  as `-OutputRoot`. Rebuild the backend after you upgrade `windowsml`. `pip uninstall`
  does not remove a backend you built; delete its folder yourself.

When only the CPU backend is installed, `winml perf --device gpu` stops before it
loads the model. If the Runtime still finds no usable GPU backend, the load fails
with HRESULT `0x80070032`.

## Report fields

GGUF reports use the GenAI phase-based schema (`schema_version: 2`, `load`,
`requests`, `aggregate`, optional `memory`), so the
[GenAI metric definitions](../commands/perf.md#genai-metric-definitions) apply. The
differences are:

| Field | GGUF value |
|---|---|
| `benchmark_info.runtime` / `backend` / `model_format` | `winml-runtime` / `llama.cpp` / `gguf` |
| `benchmark_info.experimental` | `true` |
| `benchmark_info.ep` | `null`. llama.cpp does not use an ONNX Runtime execution provider. |
| `benchmark_info.packages` | Installed versions of `windowsml`, `windowsml-llama-core`, and `windowsml-llama-cuda` (`null` when absent). |
| `benchmark_info.llama_backends` | Backend families found under `windowsml\lib\LlamaBackends`. |
| `load.model_create_duration_ms` | `Runtime.load_model` for the GGUF file. |
| `load.pipeline_build_duration_ms` | Building the one-stage decoder pipeline. `native_load_duration_ms` is model load plus pipeline build. |
| `load.weight_upload_*` | `null` / `unavailable`. The Runtime does not expose weight-upload timing. |
| `requests[].tokenization_duration_ms` | Chat-template rendering and tokenization, which the Runtime performs in one call. `template_duration_ms` is `0`. |
| `requests[].generator_create_duration_ms` | Resetting the decoder's sequence state for a new generation. |
| `requests[].decode_token_durations_ms` | One entry per generated token after the first. A step that selects the end-of-sequence token is not recorded. |
| `memory` | RAM fields only. VRAM fields are omitted because llama.cpp does not report which adapter a GPU backend uses. |

## Limitations

- Local `.gguf` files only; Hugging Face Hub references are not resolved.
- Greedy decoding with batch size 1.
- No NPU target, `--monitor`, or operator tracing.
- Targets the Windows ML 2.7 experimental packages (`2.7.2021a0`). Later previews
  can change the Runtime API.

## Troubleshooting

| Message | What to do |
|---|---|
| `GGUF models need the preview 'windowsml' package with the Runtime API` | Install `windowsml[with-ort]` at the preview version in the active environment. |
| `GGUF models need the llama.cpp backend for the Windows ML Runtime` | Install `windowsml-llama-core` at the same version as `windowsml`. |
| `--device gpu needs a llama.cpp GPU backend`, or HRESULT `0x80070032` | Install `windowsml-llama-cuda` or build a backend (see [Run on a GPU](#run-on-a-gpu)), or use `--device cpu`. |
| `the prompt (...) does not fit the model sequence capacity` | The model's maximum context is smaller than the prompt plus `--max-new-tokens`. Shorten the prompt or lower `--max-new-tokens`. |
| `The installed 'windowsml' package does not ship the Runtime native library` | The installed `windowsml` is not a preview build with the Runtime API. Reinstall the preview version. |

To leave the experimental setup, delete the `.venv-gguf` environment. Your regular
environment is unchanged.

## See also

- [winml perf](../commands/perf.md): full flag reference and the GenAI metric definitions
- [Qwen3 — Genai Bundle](qwen3-genai-bundle.md): the onnxruntime-genai path for decoder models
- [GGUF language models](https://github.com/microsoft/WindowsML/blob/main/docs/Runtime/gguf-models.md) in the Windows ML documentation
