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
| `--prompt-ids-file` | A JSON array of token IDs (UTF-8 with or without a byte order mark, or UTF-16 as Windows PowerShell 5.1 writes it) submitted unchanged on every generation, with no chat template and no tokenization. Cannot be combined with `--prompt`, `--prompt-file`, or `--apply-template`. |
| `--context-length` | Sequence capacity to load the model with, instead of the prompt tokens plus `--max-new-tokens`. The run fails before the first generation unless the Runtime reports exactly this capacity. |
| `--ignore-eos/--no-ignore-eos` | Create the Task with an empty end-of-sequence set, so each generation runs to `--max-new-tokens` unless it reaches the sequence capacity. The run fails if the Task still reports end-of-sequence tokens. |
| `--evidence` | Write raw `gguf-task-events-1.1` request evidence to this JSON file and to a `.rows.jsonl` file next to it. Both must be new unless you pass `--overwrite`. Neither can be the `--output` file, the model, or the `--prompt-ids-file` file; `--output` cannot be either input either. Needs `--warmup` of at least `1` and turns off `--memory`; an explicit `--memory` is an error. |
| `--iterations`, `--warmup` | Timed and warmup generations (defaults `10` and `2`). |
| `--memory/--no-memory` | Process RAM at the load and inference checkpoints. |
| `--output`, `--overwrite`, `--format json` | Same as for the other runtimes. |

`--prompt-ids-file`, `--context-length`, `--ignore-eos`, and `--evidence` apply only
to GGUF models; `winml perf` rejects them for other inputs.

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

## Protocol runs with exact token IDs

To compare the Task API with another llama.cpp host on the same workload, fix the
prompt tokens, the sequence capacity, and the output length, and record raw
per-request evidence:

```powershell
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --device gpu `
  --prompt-ids-file prompt-ids.json --max-new-tokens 128 --context-length 4096 `
  --ignore-eos --warmup 10 --iterations 25 `
  --evidence runs\task-events.json --format json --output runs\winml-perf.json
```

`prompt-ids.json` holds a JSON array of token IDs, for example `[151644, 872, 198]`.
The IDs are checked against the model vocabulary after load and submitted unchanged
on every request. `--context-length` must hold the prompt plus `--max-new-tokens`,
and `--ignore-eos` makes every request generate exactly `--max-new-tokens` tokens
unless it reaches the capacity. Each request uses greedy Task options with every
field set explicitly.

`--evidence PATH` writes the `gguf-task-events-1.1` format:

| File | Contents |
|---|---|
| `PATH` | One JSON document: run status, workload, effective configuration, initialization spans, every request row, and the captured native log. |
| `PATH` with `.rows.jsonl` in place of `.json` | One request row per line, appended and flushed after each request. |
| `PATH.tmp` | Present only while the evidence document is being replaced. Every write goes to this file first, so a reader never sees a partial document. |

The effective configuration includes every environment variable whose name starts
with `WINML_`, `GGML_`, `LLAMA_`, or `CUDA_`. Evidence is meant to be shared, so the
value of a variable whose name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`,
`PASSWD`, `CREDENTIAL`, or `AUTH` (for example `LLAMA_API_KEY`) is written as
`<redacted>`. The same applies to credentials in the URL winml-cli was installed
from: its user information and every query value, such as a signed link's `sig=`.

The document is first written with status `running` before the model is hashed and
before the Runtime is imported, and it is rewritten with a final status when the run
ends:

| Status | Meaning | Exit code |
|---|---|---|
| `running` | The process did not finish; treat the run as failed and read completed rows from the `.rows.jsonl` file. | — |
| `complete` | Every planned request succeeded. | 0 |
| `partial` | Every planned request ran, but at least one did not succeed. The schema 3 report is still saved. | 1 |
| `failed` | Configuration, initialization, or finalization failed, a session reset failed, or a request raised an unexpected error. Requests that never ran are recorded as `incomplete`, and no schema 3 report is saved. | 1 |

A request that fails after it was submitted is kept in the evidence and the run
continues, because a reset before the next request starts it from a clean session.
A failed reset ends the run. On a `partial` run, the schema 3 report's `requests`
and `aggregate` include only the requests that succeeded, so
`aggregate.timed_request_count` can be lower than `--iterations`; the evidence rows
hold every request. Its cold-start fields come from the first request that succeeded,
which is not a cold start when warmup 0 failed. `benchmark_info.stop_reason`,
`prompt_tokens`, and `generated_tokens` come from the first measured request that
succeeded, and are `null` or `0` when none did.

Timestamps are `time.perf_counter_ns()` readings in the machine-wide QPC domain, so
another process on the same machine can align its own samples with them. During an
evidence run:

- Native llama.cpp log lines are captured from model load through warmup 0, then
  the Runtime's own log callback is restored. Backend registration lines that the
  Runtime writes before its logger is installed are not captured.
- Python garbage collection is off during measured requests and runs between them.
- No in-process resource sampling runs; an explicit `--memory` is an error.
- `load` and `initialization.ready` are `null`, because Windows ML 2.7.2021a0
  exposes no initialization-to-ready signal and creates the native context lazily
  on the first request. Each row's `prefill` is `null`, because the Task API does
  not expose synchronized prefill timing. `missing_reasons` records these reasons.

With any of `--prompt-ids-file`, `--context-length`, `--ignore-eos`, or
`--evidence`, the schema 3 report adds `benchmark_info.input_mode`,
`prompt_ids_sha256`, `context_length_requested`, `eos_policy`, `evidence_path`, and
`evidence_status`. With `--prompt-ids-file`, `benchmark_info.prompt` is `null`.

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

The Task API does not expose separate prefill, model-compute, or detokenization
timers. The following request fields, and their aggregate counterparts where
present, are `null` rather than estimated from stream timing:

- `prefill_duration_ms` and `prefill_tokens_per_second`
- `first_token_duration_ms` and `detokenization_duration_ms`
- `model_ttft_duration_ms` and `model_compute_duration_ms`
- `response_eval_duration_ms` and `response_eval_tokens_per_second`
- `sequence_fetch_duration_ms`

## Limitations

- Local `.gguf` files only; Hugging Face Hub references are not resolved.
- Greedy decoding with batch size 1; Task sampling and speculation options are
  not exposed by this command.
- Speculative decoding keeps the Runtime default (off). An evidence run records the
  draft-prediction settings the stage reports and the per-request speculative
  statistics, but does not change them.
- Task stream timings include native sampling and detokenization; separate model
  phase timings are unavailable.
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
| `--context-length N resolved to an effective sequence capacity of M` | The Runtime did not apply the requested capacity, for example because the model supports less. Both values are in the evidence under `configuration.context`. Use a `--context-length` the model supports. |
| `--ignore-eos requested an empty EOS set, but the Task reports EOS token IDs` | The installed Runtime does not honor an explicit empty EOS set. Use a Windows ML version that does, or run without `--ignore-eos`. |
| `prompt token ... is outside the model vocabulary` | A token ID in `--prompt-ids-file` is not valid for this model. Regenerate the IDs with this model's tokenizer. |

To leave the experimental setup, delete the `.venv-gguf` environment. Your regular
environment is unchanged.

## See also

- [winml perf](../commands/perf.md): full flag reference and runtime-specific metric definitions
- [Qwen3 — Genai Bundle](qwen3-genai-bundle.md): the onnxruntime-genai path for decoder models
- [GGUF language models](https://github.com/microsoft/WindowsML/blob/main/docs/Runtime/gguf-models.md) in the Windows ML documentation
