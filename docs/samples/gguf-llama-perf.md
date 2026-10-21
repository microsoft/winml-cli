# GGUF — llama.cpp on Windows ML Runtime (Experimental)

`winml perf` can benchmark a GGUF language model through the Windows ML Task API
(`windowsml.tasks`). The Task uses one decoder stage on the Runtime's llama.cpp
backend and manages greedy generation, token streaming, and sequence state. The
Runtime still selects llama.cpp from the `.gguf` extension. The report measures
time to the first Task token update, intervals between token updates, and total
generation time. These include Task processing and are distinct from the model
phase timings reported by `--runtime ort-genai`.

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
| `windowsml[with-ort]` | `2.7.2021a0` | The Runtime and Task APIs (`windowsml.runtime`, `windowsml.tasks`) and the matching `onnxruntime-windowsml` build |
| `windowsml-llama-core` | `2.7.2021a0` | The Runtime llama.cpp adapter, the llama.cpp and ggml modules, and the CPU backend |

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

`pip` (or `uv pip check`) reports that winml-cli requires different `windowsml`
and `onnxruntime-windowsml` versions. These conflicts are expected in this
environment: the preview's `with-ort` extra installs ONNX Runtime 1.30, above
winml-cli's `<1.28` constraint.

!!! note "Run `winml` from the activated environment"
    Run `winml` directly in the activated environment. `uv run` and `uv sync`
    synchronize the project environment back to the pinned packages, which removes
    the preview Runtime.

Check the installation:

```powershell
python -c "import windowsml.runtime, windowsml.tasks; print('Windows ML Runtime and Task APIs are available')"
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
model response, startup spans, a Task TTFT table, and token-update throughput. The
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
| `--gguf-prefill-trace` | Fresh JSONL path for an externally instrumented native prefill writer. This option does not add timing support to stock Runtime/Task binaries. See [Native prefill trace](#native-prefill-trace). |

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
generations. The Task session is reset before every warmup and timed request, so
each generation evaluates its prompt independently.

## Run on a GPU

The `windowsml-llama-core` wheel ships only the CPU backend. Windows ML does not
publish a `windowsml-llama-cuda` wheel for this release. Build a matching GPU
backend with the Windows ML
[`Build-LlamaBackend.ps1` tool](https://github.com/microsoft/WindowsML/blob/v2.7.2021-experimental/Tools/llama/README.md),
using the `v2.7.2021-experimental` checkout for the preview packages above. The
backend is installed under `windowsml\lib\LlamaBackends\<family>` in the environment.

For NVIDIA CUDA, the build requires:

- PowerShell 7, Git, CMake, and Visual Studio 2022 or 2026 with the **v143** C++
  toolset (MSVC 14.30 through 14.49). A newer toolset alone does not satisfy the
  release's build script.
- Windows long paths enabled (`LongPathsEnabled`), as required by the official
  backend build guide.
- An installed NVIDIA CUDA Toolkit and a driver that supports that toolkit. A
  GPU driver alone does not provide the compiler and libraries needed to build
  the backend.
- The signed
  [`Microsoft.Windows.AI.MachineLearning.LibLlama.Core` NuGet package, version `0.5.0.2022-experimental`](https://www.nuget.org/packages/Microsoft.Windows.AI.MachineLearning.LibLlama.Core/0.5.0.2022-experimental),
  its verified SHA-256, and a .NET SDK for the script's `dotnet nuget verify --all`
  check. This NuGet version belongs to the same Windows ML release as the
  `windowsml-llama-core==2.7.2021a0` wheel; the version strings differ.

Open PowerShell 7 with the Visual Studio developer environment and v143 toolset
active (Developer PowerShell). The script requires `dumpbin.exe` on `PATH` before
it discovers Visual Studio; an ordinary PowerShell session is insufficient.
Activate the experimental Python environment in that session. From the matching
Windows ML checkout root, set the paths for the downloaded Core package and
installed CUDA Toolkit, then run:

```powershell
$corePackage = '<path to Microsoft.Windows.AI.MachineLearning.LibLlama.Core.0.5.0.2022-experimental.nupkg>'
$expectedSha256 = '<verified 64-character SHA-256 of that package>'
$cudaToolkit = '<installed CUDA Toolkit directory>'
$library = python -c "import pathlib, windowsml; print(pathlib.Path(windowsml.__file__).parent / 'lib')"

.\Tools\llama\Build-LlamaBackend.ps1 `
  -Backend cuda `
  -Architecture x64 `
  -CMakeOption "-DCUDAToolkit_ROOT=$cudaToolkit" `
  -CorePackage $corePackage `
  -ExpectedPackageSha256 $expectedSha256 `
  -OutputRoot $library `
  -WorkRoot (Join-Path $PWD 'temp\llama-cuda')
```

The script checks that the environment's Core payload matches the NuGet package,
builds the backend from the package's exact llama.cpp commit, and writes
`ggml-cuda.dll` with its matching `WinMLLlamaBackendManifest.json`. An arbitrary
llama.cpp CUDA DLL cannot replace this build. Copy the required NVIDIA runtime
DLLs, including their dependencies, from the installed toolkit into
`$library\LlamaBackends\cuda`. The script's warnings name direct dependencies;
check their imports too. In the verified CUDA 13.4.2 build, `ggml-cuda.dll` imports
`cublas64_13.dll`, which in turn imports `cublasLt64_13.dll`, so both are needed. Names can differ
with the toolkit version. NVIDIA's toolkit and runtime license terms apply to
those files.

Then run the benchmark from the directory containing the `models` folder, or
pass an absolute model path:

```powershell
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --device gpu
```

Other families, such as Vulkan or OpenCL, use the same script with their backend
name and SDK. Follow the release's GPU backend guide for those prerequisites.
Rebuild after upgrading the Core package: the backend manifest must match its
llama.cpp commit and runtime contract. `pip uninstall` does not remove a backend
you built; delete its folder yourself.

When only the CPU backend is installed, `winml perf --device gpu` stops before it
loads the model. If the Runtime still finds no usable GPU backend, the load fails
with HRESULT `0x80070032`.

!!! note "Preview CUDA dependency warning"
    With `windowsml==2.7.2021a0`, a backend you built can trigger a warning that
    `WinMLLlamaCudaDependencies.json` is missing and suggest installing
    `windowsml-llama-cuda`. The Python loader expects a wheel-specific dependency
    record that the official backend build script does not create. This warning
    alone does not establish whether native CUDA loading failed. Confirm a
    successful generation, `benchmark_info.effective_device: gpu`, and native
    `ggml_cuda_init` output identifying the GPU. The suggested CUDA wheel is not
    published for this release; follow the build and DLL setup above.

## Report fields

GGUF Task reports use `schema_version: 3`, retaining the `load`, `requests`,
`aggregate`, and optional `memory` sections. Only GGUF reports use this version;
`ort-genai` reports remain at version 2. Earlier GGUF version 2 reports measured
individual Runtime calls and Python token selection. Version 3 measures the Task
stream, which includes native sampling and detokenization. Do not interpret its
TTFT or token-update intervals as the earlier model-compute phases.

| Field | GGUF value |
|---|---|
| `benchmark_info.runtime` / `backend` / `model_format` | `winml-runtime` / `llama.cpp` / `gguf` |
| `benchmark_info.api` / `timing_source` | `windowsml.tasks` / `task-stream` |
| `benchmark_info.experimental` | `true` |
| `benchmark_info.ep` | `null`. llama.cpp does not use an ONNX Runtime execution provider. |
| `benchmark_info.packages` | Installed versions of `windowsml` and `windowsml-llama-core`. The legacy `windowsml-llama-cuda` entry is retained in the report schema and is `null` for this release; a locally built CUDA backend does not install a Python distribution. |
| `benchmark_info.llama_backends` | Backend families found under `windowsml\lib\LlamaBackends`. |
| `load.model_create_duration_ms` | `Runtime.load_model` for the GGUF file. |
| `load.pipeline_build_duration_ms` | Building the one-stage decoder pipeline. `native_load_duration_ms` is model load plus pipeline build. |
| `load.task_create_duration_ms` | Creating the Task configuration, Task, and generation session. |
| `load.weight_upload_*` | `null` / `unavailable`. The Runtime does not expose weight-upload timing. |
| `requests[].tokenization_duration_ms` | Chat-template rendering and tokenization, which the Runtime performs in one call. `template_duration_ms` is `0`. |
| `requests[].generator_create_duration_ms` | Resetting the Task session before a new generation. |
| `requests[].task_stream_create_duration_ms` | The `generate_tokens` call that creates the pull stream. |
| `requests[].task_ttft_duration_ms` | From immediately before `generate_tokens` through the first token update; includes stream creation. |
| `requests[].request_ttft_duration_ms` | Tokenization plus session reset plus Task TTFT. |
| `requests[].decode_token_durations_ms` | Arrival intervals between consecutive token updates, including Task sampling and detokenization. There is one interval per token after the first. |
| `requests[].steady_state_decode_tokens_per_second` | Tokens after the first divided by the sum of token-update intervals. TPOT uses those same intervals. |
| `requests[].task_stream_completion_duration_ms` | From the last token update to the terminal read returning `None`; when no token is produced, from stream creation's end. Includes any final stop-token processing. |
| `requests[].result_fetch_duration_ms` | Fetching the terminal Task result after the stream completes. |
| `requests[].task_generation_duration_ms` | From immediately before `generate_tokens` through the terminal read and result fetch. |
| `requests[].request_duration_ms` | Tokenization plus session reset plus Task generation duration. |
| `memory` | RAM fields only. VRAM fields are omitted because llama.cpp does not report which adapter a GPU backend uses. |

`aggregate` summarizes timed requests only; warmups remain in `requests` but do not
contribute to its distributions. Task, request, and cold-start TTFT are `null`
when the corresponding generation produces no token update.
Decode throughput and TPOT are `null` when there is no interval after the first
token. An empty text fragment still represents a token update.

The stock Task API does not expose separate prefill, model-compute, or
detokenization timers. Without an instrumented native writer, the following
request fields and their aggregate counterparts remain `null`, rather than
estimated from stream timing:

- `prefill_duration_ms` and `prefill_tokens_per_second`
- `first_token_duration_ms` and `detokenization_duration_ms`
- `model_ttft_duration_ms` and `model_compute_duration_ms`
- `response_eval_duration_ms` and `response_eval_tokens_per_second`
- `sequence_fetch_duration_ms`

### Native prefill trace

`--gguf-prefill-trace` adds **CLI ingestion and reporting**, not native backend
instrumentation. The stock Windows ML packages described above do not emit this
protocol. Enabling the option with those binaries still produces null prefill
metrics and the reason `native_prefill_event_unavailable`. A Runtime/backend
change is required to supply the events.

An instrumented writer must measure prompt computation through synchronized
backend completion **before the first sampling operation**. Tokenization,
session reset, sampling and Task output delivery are outside that native span.
Timing `generate_tokens` or the first stream update instead is not prefill.
Existing llama.cpp `prompt_ms` must not be relabeled when it includes sampling.

With a compatible instrumented build, select a fresh trace path:

```powershell
winml perf -m models\model.gguf --device gpu --warmup 10 --iterations 25 `
  --gguf-prefill-trace .\capture\prefill.jsonl --output .\capture\perf.json
```

The trace and its sibling `prefill.jsonl.context.json` must not already exist;
`--overwrite` applies to the report and does not authorize reusing trace files.
The option is available only for GGUF on `winml-runtime`. Model, prompt, report,
trace and context paths must not overlap.

Before the session is created, the CLI publishes the context path through
`WINMLCLI_GGUF_PREFILL_CONTEXT`. Before each generation, outside the recorded
Task timer, the context identifies an active request with a unique run/request
ID, process ID, entry-point model-file SHA-256, prompt-ID SHA-256, token count, warmup/timed
kind and index. The native writer snapshots that identity before prompt
execution and appends one completed JSON object per line to `trace_path`. It
must finish writing before the corresponding Task request returns.

The writer protocol is `winml-gguf-prefill-event-1`. Each event must copy the
active context's `run_id`, `pid`, `model_sha256` and request fields
(`request_id`, `kind`, `index`, `prompt_sha256`, `prompt_tokens`), and supply:

| Field | Required value |
|---|---|
| `phase` | `prefill` |
| `boundary` | `backend_prompt_start_to_synchronized_completion_before_sampling` |
| `synchronized` / `before_first_sampling` | Both `true`, established by the native implementation |
| `duration_ns` | Positive integer duration from one native clock domain |
| `clock_monotonic` | `true`; the measured native duration uses a monotonic clock |
| `source` / `backend_build` / `clock_domain` | Nonempty native instrumentation, build and clock identifiers; consistent within the run |

These declarations are an interface contract, not proof that a stock backend
implements that boundary. Native instrumentation must be reviewed and validated
separately. Host and native timestamps are never subtracted from one another.
For sharded GGUF, the model hash identifies only the entry-point file; it does
not establish the other shards' identity. Hashing that file before model load
can warm the file cache, so trace-enabled load timing is not a cold-cache
measurement. The metadata records both limitations.

The CLI rejects malformed or duplicate events, identity/count mismatches,
unqualified boundaries and durations longer than the containing Task generation.
It preserves the trace bytes and their SHA-256 for inspection.

For valid events, `requests[].prefill_duration_ms` is the native duration, and
`requests[].prefill_tokens_per_second` is the actual input-token count divided
by that duration in seconds. `requests[].native_prefill` retains the unrounded
nanoseconds and timing provenance. Missing events stay null with
`prefill_missing_reason`; no value is filled from TTFT.

The matching aggregate fields summarize **per-request** values, excluding
warmups and unavailable observations. Prefill P50/P95 use nearest-rank
`ceil(percentile * valid_count / 100)`, with explicit
`prefill_valid_request_count` and `prefill_missing_request_count`. This native
prefill definition does not change existing Task/GenAI percentile behavior.
`benchmark_info.prefill_measurement` records the instrumented protocol, boundary,
run identity, trace/context paths and trace hash. Task timings collected in this
mode include any native trace-writing overhead and must retain that label.

## Limitations

- Local `.gguf` files only; Hugging Face Hub references are not resolved.
- Greedy decoding with batch size 1; Task sampling and speculation options are
  not exposed by this command.
- Task stream timings include native sampling and detokenization. Native prefill
  is available only with the opt-in trace and an independently instrumented
  backend; other model phases remain unavailable.
- No NPU target, `--monitor`, or operator tracing.
- Targets the Windows ML 2.7 experimental packages (`2.7.2021a0`). Later previews
  can change the Runtime API.

## Troubleshooting

| Message | What to do |
|---|---|
| `GGUF models need the preview 'windowsml' package with the Runtime API` | Install `windowsml[with-ort]` at the preview version in the active environment. |
| `GGUF models need the llama.cpp backend for the Windows ML Runtime` | Install `windowsml-llama-core` at the same version as `windowsml`. |
| `--device gpu needs a llama.cpp GPU backend`, or HRESULT `0x80070032` | Build a matching backend and provide its vendor runtime DLLs (see [Run on a GPU](#run-on-a-gpu)), or use `--device cpu`. For CUDA, check that the toolkit dependencies are present under `LlamaBackends\cuda` and the driver supports that toolkit. |
| `the prompt (...) does not fit the model sequence capacity` | The model's maximum context is smaller than the prompt plus `--max-new-tokens`. Shorten the prompt or lower `--max-new-tokens`. |
| `The installed 'windowsml' package does not ship the Runtime native library` | The installed `windowsml` is not a preview build with the Runtime API. Reinstall the preview version. |

To leave the experimental setup, delete the `.venv-gguf` environment. Your regular
environment is unchanged.

## See also

- [winml perf](../commands/perf.md): full flag reference and runtime-specific metric definitions
- [Qwen3 — Genai Bundle](qwen3-genai-bundle.md): the onnxruntime-genai path for decoder models
- [GGUF language models](https://github.com/microsoft/WindowsML/blob/main/docs/Runtime/gguf-models.md) in the Windows ML documentation
