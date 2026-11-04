# CGC — Text Generation on Windows ML (Experimental)

`winml perf --backend cgc` benchmarks text generation of a GGUF file or a local
Hugging Face safetensors checkpoint folder on the Windows ML CGC backend, through
model-backed Windows ML Tasks: `Runtime.load_model` (`windowsml.runtime`) loads the
model, and `TextGenerationConfiguration.set_model` (`windowsml.tasks`) binds it to one
GPU and creates its tokenizer from the model files. Windows ML compiles the model's
GPU programs when the first Task session is created. Greedy generation then streams
token updates through the same Task API as the
[GGUF llama.cpp benchmark](gguf-llama-perf.md), and the report uses the same
schema 3 layout with `benchmark_info.backend: cgc`.

CGC is opt-in: pass `--backend cgc`. A `.gguf` file without `--backend` still runs on
llama.cpp, and a checkpoint folder without it is still exported to ONNX.

!!! warning "Experimental"
    Model-backed text generation needs a preview Windows ML build; this page uses
    **Windows ML 2.8.183.dev0**. winml-cli does not install or pin it: its own
    dependencies keep the stable `windowsml` release. Install the preview yourself, in
    a separate virtual environment, only when you need this benchmark. The
    model-backed Task API can change between builds, and Windows ML, not winml-cli,
    decides which models and GPUs it supports.

## Prerequisites

- Windows x64 and Python 3.11 (the winml-cli requirements).
- A hardware GPU. CGC text generation runs only on a GPU.
- winml-cli, from PyPI or from a source checkout.
- The preview `windowsml` wheel, `windowsml-2.8.183.dev0-py3-none-win_amd64.whl`, in
  a local folder. This build is not published on PyPI.

The tested environment holds winml-cli with its own dependencies plus the preview
`windowsml`. These two packages matter for CGC text generation:

| Package | Version | Provides |
|---|---|---|
| `windowsml` | `2.8.183.dev0` | The Runtime and Task APIs with model-backed text generation, and the CGC backend. It also installs `comtypes`. |
| `onnxruntime-windowsml` | The version winml-cli installs (1.27 in the tested environment) | ONNX Runtime. Windows ML loads its native library when it loads a model. |

CGC text generation needs neither the `with-ort` extra of `windowsml` nor a llama.cpp
package such as `windowsml-llama-core`. The tested environment had neither, and the
`with-ort` extra was not tested with this path.

- A model: a local `.gguf` file, or a local Hugging Face checkpoint folder. A
  checkpoint folder needs `config.json`, its safetensors weights (a single
  `*.safetensors` file, or `model.safetensors.index.json` and every shard it lists),
  `tokenizer.json`, and `tokenizer_config.json`. For example, download the GGUF file
  and the checkpoint of Qwen2.5-0.5B-Instruct. Review the model license before you
  download it:

    ```powershell
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='models'))"
    python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', allow_patterns=['*.json', '*.safetensors'], local_dir='models/Qwen2.5-0.5B-Instruct'))"
    ```

## Step 1: Create an experimental environment

The preview replaces the `windowsml` version that winml-cli pins, so keep it out of
your regular environment. In the commands below, set `$wheels` to the folder that
holds the `windowsml` wheel.

=== "uv (source checkout)"

    ```powershell
    $wheels = '<folder that holds windowsml-2.8.183.dev0-py3-none-win_amd64.whl>'
    uv venv .venv-cgc --python 3.11
    .venv-cgc\Scripts\activate
    uv pip install -e .
    uv pip install --find-links $wheels --prerelease=allow "windowsml==2.8.183.dev0"
    ```

=== "pip"

    ```powershell
    $wheels = '<folder that holds windowsml-2.8.183.dev0-py3-none-win_amd64.whl>'
    py -3.11 -m venv .venv-cgc
    .venv-cgc\Scripts\activate
    python -m pip install winml-cli
    python -m pip install --pre --find-links $wheels "windowsml==2.8.183.dev0"
    ```

`pip` (or `uv pip check`) reports that winml-cli requires a different `windowsml`
version. This conflict is expected in this environment. Keep the
`onnxruntime-windowsml` version that winml-cli installed.

!!! note "Run `winml` from the activated environment"
    Run `winml` directly in the activated environment. `uv run` and `uv sync`
    synchronize the project environment back to the pinned packages, which removes
    the preview.

Check the installation:

```powershell
python -c "import windowsml.runtime, windowsml.tasks as tasks; print('Model-backed Task API:', hasattr(tasks.TextGenerationConfiguration, 'set_model'))"
uv pip show windowsml   # or: python -m pip show windowsml
```

The first command prints `Model-backed Task API: True`, and the second shows
`windowsml` version `2.8.183.dev0`.

## Step 2: Benchmark the model

Pass `--backend cgc` with a `.gguf` file or a checkpoint folder:

```powershell
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --backend cgc
winml perf -m models\Qwen2.5-0.5B-Instruct --backend cgc
```

`--backend cgc` selects CGC text generation with `--runtime auto` (the default). A
`.gguf` file also runs on CGC with `--runtime winml-runtime --backend cgc`, but a
checkpoint folder with `--runtime winml-runtime` is still exported to ONNX.
`--device auto` (the default) means the GPU here, while the llama.cpp benchmark runs
a `.gguf` file on the CPU with `--device auto`. Without explicit values, runs use 2
warmup and 10 timed generations, as on llama.cpp.

The first Task session compiles the GPU programs, so without a program cache every
run spends a while loading before the first generation. In the sample runs on an
NVIDIA GeForce RTX 5080, the first session took 11 to 16 seconds for
Qwen2.5-0.5B-Instruct and 27 to 34 seconds for the two-shard Qwen2.5-3B-Instruct
checkpoint. To reuse the compiled programs in later runs, see
[Program cache](#program-cache).

The console report extends the llama.cpp report with the pinned GPU (`Adapter`), the
model format, the sequence capacity, the program cache, and the `set_model` and
first-session (GPU compile) spans. Abbreviated output of the first command from one
sample run on an RTX 5080 (numbers depend on the GPU, the driver, and the model, and
are not guarantees):

```text
Runtime:        winml-runtime (cgc, experimental)
API:            Windows ML Tasks (task-stream timing)
Device:         auto -> gpu
Model:          models\qwen2.5-0.5b-instruct-q4_k_m.gguf
Adapter:        NVIDIA GeForce RTX 5080 (0x00000000_0x00190171, high-performance)
Format:         gguf
Capacity:       168 tokens
Program cache:  off (cold compile every run)
Packages:       windowsml 2.8.183.dev0, winml-cli 0.4.0
Prompt:         40 tokens   Generated: 128 tokens (max_new_tokens=128, stop=MAX_TOKENS)
...
Startup
  Session load: 15275.56 ms  |  Native load: 14722.48 ms
  set_model: 1783.05 ms  |  First session (GPU compile): 12926.91 ms
  Cold TTFT: 15298.75 ms  |  Cold total: 16650.29 ms
...
Decode (Task stream): 93.79 tokens/sec  |  10.66 ms/token (TPOT)
...
Memory:
  RAM:  1699.6 MB -> model load: +1646.9 MB  |  inference: +5.0 MB  |  total: +1651.9 MB
  GPU local/shared: 1156.5/111.2 MiB after load  |  1156.5/112.2 MiB after inference
```

`Cold TTFT` and `Cold total` include the load, so they include the GPU compile. The
input format can change CGC throughput a lot: in the sample runs on the RTX 5080, the
safetensors checkpoint of Qwen2.5-0.5B-Instruct decoded at about 542 tokens/sec, and
its q4_k_m GGUF file at about 93 tokens/sec.

The JSON report is written to
`~/.cache/winml/perf/<file stem or folder name>/<timestamp>.json` unless you pass
`--output`.

These options apply to CGC text generation:

| Option | Behavior with `--backend cgc` |
|---|---|
| `--model` / `-m` | A local `.gguf` file, or a local Hugging Face checkpoint folder. Hugging Face model IDs and single `.safetensors` files are rejected. |
| `--runtime` | `auto` (the default). A `.gguf` file also accepts `winml-runtime`; a checkpoint folder with `--runtime winml-runtime` is exported to ONNX instead. |
| `--device` | `auto` or `gpu`; both mean the GPU. `cpu` and `npu` are rejected. |
| `--device-luid` | The GPU to pin, by its LUID from `winml sys`; see [Choose the GPU](#choose-the-gpu). |
| `--prompt`, `--prompt-file` | The prompt for every generation. |
| `--apply-template/--no-apply-template` | Render the prompt with the model's chat template, which Windows ML reads from the model files. Without a usable template, the raw prompt is used, the console `Prompt` row adds `(no chat template; raw prompt)`, and the report sets `chat_template_applied` to `false`. |
| `--max-new-tokens` | Upper bound on generated tokens per generation. |
| `--prompt-ids-file` | A JSON array of token IDs submitted unchanged on every generation, with no chat template and no tokenization. Cannot be combined with `--prompt`, `--prompt-file`, or `--apply-template`. The IDs are checked against the vocabulary only when Windows ML reports its size; see [Report fields](#report-fields). |
| `--context-length` | Sequence capacity to load the model with, instead of the prompt tokens plus `--max-new-tokens`; see [Context capacity](#context-capacity). |
| `--ignore-eos/--no-ignore-eos` | Create the Task with an empty end-of-sequence set, so each generation runs to `--max-new-tokens` unless it reaches the capacity. The run fails if the Task still reports end-of-sequence tokens. |
| `--program-cache` | Directory for compiled GPU programs that later runs reuse; see [Program cache](#program-cache). |
| `--iterations`, `--warmup` | Timed and warmup generations (defaults `10` and `2`). |
| `--memory/--no-memory` | Process RAM and the pinned GPU's memory at the load and inference checkpoints; see [Report fields](#report-fields). |
| `--output`, `--overwrite`, `--format json` | Same as for the other runtimes, except that `--output` cannot name the `.gguf` file, any of its shards, or the `--prompt-ids-file` input, and cannot lie inside a checkpoint folder, even with `--overwrite`. |

Options that CGC text generation cannot honor are rejected: `--ep`, `--ep-options`,
`--input-data`, `--module`, and `--submodel`, which select another benchmark target;
`--evidence`, which records the llama.cpp native log; and `--memory-backend dxgi`,
`--etw-tracing`, and `--warmup-seconds`, which belong to the single-inference ONNX
measurements. Build-only and ONNX-only options, such as `--precision`, `--task`,
`--compile`, `--op-tracing`, `--duration`, `--batch-size`, `--monitor`, and
`--config`, are ignored with a warning.

## Choose the GPU

CGC text generation runs on one GPU, which winml-cli pins before Windows ML loads the
model:

- Without `--device-luid`, winml-cli lists the hardware GPUs through DXCore, orders
  them by DXCore's high-performance preference, which puts a discrete or external GPU
  ahead of an integrated one, and pins the first. If Windows cannot rank the GPUs,
  winml-cli uses the only hardware GPU, or stops and asks for `--device-luid` when
  there are several.
- With `--device-luid`, winml-cli pins the hardware GPU with that LUID.

Software adapters, such as the Microsoft Basic Render Driver, are never selected. The
console `Adapter` row and `benchmark_info.adapter` report the GPU's name, its LUID,
and how it was selected: `high-performance`, `only-gpu`, or `device-luid`. To use
another GPU, copy its LUID from the device list of `winml sys`:

```powershell
winml sys
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --backend cgc `
  --device-luid 0x00000000_0x00190171
```

Replace the example LUID with the value that `winml sys` lists for your GPU. LUIDs are
local to the current Windows boot and can change after a reboot or a driver update.

Windows ML decides whether a GPU can run the model. On the test machine, which has an
NVIDIA GeForce RTX 5080 and an Intel integrated GPU, the default selection pinned the
RTX 5080, and pinning the integrated GPU with `--device-luid` failed when the Task
session was created.

## Context capacity

winml-cli encodes the prompt first and sets the model's logical capacity to the
prompt tokens plus `--max-new-tokens` (`set_logical_capacity`) before it creates the
Task session. Without this hint, Windows ML can size the session for the model's full
trained context. The capacity Windows ML reports back (`logical_capacity`) is saved as
`benchmark_info.context_length` and shown in the console `Capacity` row.

`--context-length N` sets the capacity to `N` instead. It must hold the prompt plus
`--max-new-tokens`, and the run fails before the first request unless Windows ML
reports exactly `N`. In the sample runs, CGC applied the requested capacity exactly:
256 and 300 were read back unchanged, so no multiple of 256 is needed here. CGC does
not clamp a capacity above the model's trained context length either: a capacity of
40000 for Qwen2.5-0.5B-Instruct, which was trained for 32768 tokens, was rejected when
the Task session was created, with a message that the selected GPU or the requested
options cannot run this model.

If Windows ML reports no capacity, winml-cli prints a warning, saves `context_length`
as `null`, and skips the check that the prompt fits. With `--context-length`, the run
fails instead, because the requested capacity cannot be verified.

For a fixed workload, combine exact token IDs, a pinned capacity, and an empty
end-of-sequence set:

```powershell
winml perf -m models\qwen2.5-0.5b-instruct-q4_k_m.gguf --backend cgc `
  --prompt-ids-file prompt-ids.json --max-new-tokens 64 --context-length 256 --ignore-eos
```

A generation stops at the end-of-sequence token, at `--max-new-tokens`, or at the
capacity. The stop reason of the first timed generation is saved as
`benchmark_info.stop_reason`, and `aggregate.stop_reason_counts` counts all timed
generations. The Task session is reset before every warmup and timed request, so each
generation evaluates its prompt independently.

## Program cache

By default winml-cli gives Windows ML no program cache directory, so every run
compiles the GPU programs again and measures that cold compile.
`--program-cache DIR` passes `DIR` to Windows ML as its compiled program directory
(`compiled_program_directory`): winml-cli creates the directory when it is missing,
Windows ML stores the compiled programs there, and later runs with the same directory
reuse them. The report records the directory as an absolute path in
`benchmark_info.program_cache_dir` (`null` without the option), and the console shows
it in the `Program cache` row. With a checkpoint folder, the directory cannot lie
inside that folder.

winml-cli never deletes the directory. Delete it yourself to measure a cold compile
again or to free the space: in the sample below, it held 9 files and 962,415,331
bytes, about twice the size of the 491,400,032-byte GGUF file it was compiled from.

One sample on an RTX 5080, Qwen2.5-0.5B-Instruct q4_k_m GGUF file, two runs with the
same `--program-cache` directory (not a guarantee):

| Run | `set_model` | First session (`session_create_duration_ms`) | Native load | Cold TTFT |
|---|---|---|---|---|
| First run, empty directory | 1883.94 ms | 15019.29 ms | 16909.48 ms | 17465.85 ms |
| Second run, same directory | 1957.10 ms | 1376.85 ms | 3340.56 ms | 3944.60 ms |

The cache shortened only the first session. Task TTFT (20.75 and 19.05 ms) and decode
throughput (93.74 and 93.86 tokens/sec) were about the same in both runs.

## Report fields

CGC reports use `schema_version: 3` with the same `load`, `requests`, `aggregate`, and
optional `memory` sections as GGUF reports on llama.cpp. Request and aggregate fields
have the same definitions; see the
[GGUF report fields](gguf-llama-perf.md#report-fields). These fields differ:

| Field | CGC value |
|---|---|
| `benchmark_info.runtime` / `backend` / `model_format` | `winml-runtime` / `cgc` / `gguf` or `safetensors` |
| `benchmark_info.api` / `timing_source` | `windowsml.tasks` / `task-stream` |
| `benchmark_info.experimental` | `true` |
| `benchmark_info.ep` | `null`. CGC text generation does not use an ONNX Runtime execution provider. |
| `benchmark_info.device` / `effective_device` | The `--device` value as passed (`auto` or `gpu`) / `gpu` |
| `benchmark_info.adapter` | The pinned GPU: `name`, `luid`, and `selection` (`high-performance`, `only-gpu`, or `device-luid`). |
| `benchmark_info.context_length` | The logical capacity Windows ML reported (`logical_capacity`) after winml-cli set it; `null`, with a warning, when Windows ML reported none. |
| `benchmark_info.vocabulary_size` | The tokenizer vocabulary size Windows ML reported; every prompt token ID must be below it. `null`, with a warning, when Windows ML reports none, as for both safetensors checkpoints in the sample runs; prompt token IDs, including those from `--prompt-ids-file`, are then not range-checked. |
| `benchmark_info.program_cache_dir` | The `--program-cache` directory as an absolute path, or `null` without it. |
| `benchmark_info.packages` | Installed versions of `windowsml` and `winml-cli`. CGC text generation uses no llama.cpp package. |
| `load.runtime_create_duration_ms` | Creating the Windows ML `Runtime`. |
| `load.target_create_duration_ms` | Creating the execution target on the pinned GPU. |
| `load.model_create_duration_ms` | `Runtime.load_model` for the `.gguf` file or the checkpoint folder. |
| `load.set_model_duration_ms` | `TextGenerationConfiguration.set_model`, which binds the model to the GPU and creates its tokenizer. |
| `load.session_create_duration_ms` | The first `create_session`, where the GPU programs compile, or load from the `--program-cache` directory. |
| `load.native_load_duration_ms` | `model_create_duration_ms` plus `set_model_duration_ms` plus `session_create_duration_ms`. |
| `load.session_load_duration_ms` | The whole load, from creating the `Runtime` through the first `create_session`, without the prompt encoding in between. |
| `load.weight_upload_*` | `null` / `unavailable`. Windows ML does not expose weight-upload timing. |
| `memory` | Process RAM and the pinned GPU's memory (below). |

With `--memory` (the default), `memory` holds the process working set (`rss_*`) and
the process's local and shared memory on the pinned GPU (`vram_local_*` and
`vram_shared_*`, from the Windows GPU performance counters, PDH), recorded before the
load, after the load, and after inference. Every `_mb` field uses MiB (bytes /
1,048,576). The console `RAM:` line says `MB` and its `GPU local/shared:` line says
`MiB`, but both lines show MiB.

The process has no GPU memory counter for the pinned GPU until Windows ML allocates
memory on it, and this path creates no Direct3D device to make the counter appear
earlier. So the GPU baseline is `null`, and so is every value computed from it:
`vram_*_baseline_mb`, `vram_*_checkpoint_peak_mb`, `vram_*_model_load_delta_mb`, and
`vram_*_total_delta_mb`. The readings after load (`vram_*_after_compile_mb`) and after
inference (`vram_*_after_inference_mb`), and the inference deltas, are measured. In
the sample runs on an RTX 5080, the process held about 1156 MiB of local GPU memory
after loading the Qwen2.5-0.5B-Instruct GGUF file and about 6071 MiB after loading the
Qwen2.5-3B-Instruct checkpoint.

## Troubleshooting

A failed run exits with a non-zero code and writes no report. winml-cli never retries
it on llama.cpp or the ONNX path.

| Message | What to do |
|---|---|
| `CGC text generation needs windowsml 2.8.183.dev0 or newer` | The active environment has no `windowsml`, the stable release, or an older preview. Create the environment in [Step 1](#step-1-create-an-experimental-environment) and run `winml` from it. |
| `Windows ML could not load its Runtime API` or `could not load a native dependency` | Importing or starting Windows ML failed for the reason the message gives, for example a missing `comtypes` package. Reinstall `windowsml` 2.8.183.dev0, which also installs `comtypes`, in the experimental environment. |
| `The installed windowsml package is missing a native library` | The installation is incomplete. Reinstall `windowsml` 2.8.183.dev0. |
| `CGC text generation needs a hardware GPU, but DXCore reports none` | Windows reports no hardware GPU; software adapters, such as the Microsoft Basic Render Driver, do not count. Install or update the GPU driver and check that `winml sys` lists the GPU. To run a `.gguf` file on the CPU instead, omit `--backend cgc`. |
| `Windows cannot rank the GPUs by performance here` | Pass `--device-luid` with the LUID of one of the GPUs the message lists, as `winml sys` shows it. |
| `No hardware GPU has LUID ...` | The LUID is not a hardware GPU, or it changed after a reboot or a driver update. Copy a current GPU LUID from `winml sys` and pass it with `--device-luid`. |
| `Windows ML CGC cannot run this model on ...` | `load_model` or `set_model` rejected the model: the installed CGC backend does not support it (in the sample runs, an encoder-only checkpoint and a split GGUF model). winml-cli does not try another engine. To run a `.gguf` file on llama.cpp, omit `--backend cgc`. |
| `The selected GPU ... or the requested options cannot run this model` | Creating the Task session failed on the pinned GPU, which the message names: that GPU cannot run the model, or the capacity is larger than the model supports. On the test machine, the Intel integrated GPU failed this way. Select another GPU with `--device-luid` (see `winml sys`), or lower `--context-length` or `--max-new-tokens`. |
| `Windows ML could not find a model file` | A model file is missing. A checkpoint folder needs `config.json`, its safetensors weights (or `model.safetensors.index.json` and every shard it lists), `tokenizer.json`, and `tokenizer_config.json`. |
| `is not a Hugging Face checkpoint folder` | The folder has no `config.json` or no safetensors weights. Pass a checkpoint folder, or pass a `.gguf` file itself with `-m`. |
| `Windows ML created no tokenizer for this model after set_model` | Add `tokenizer.json` and `tokenizer_config.json` to the checkpoint folder. |
| `--backend cgc takes a checkpoint folder, not a single .safetensors file` | Pass the folder that holds the weights with `-m`. |
| `--backend cgc accepts only a local checkpoint folder path` | Hugging Face model IDs are not supported. Download the checkpoint (see [Prerequisites](#prerequisites)) and pass its folder with `-m`. |
| `--device cpu is not supported with --backend cgc` (or `npu`) | CGC text generation runs only on a GPU. Use `--device gpu` or `auto`. To run a `.gguf` file on the CPU, omit `--backend cgc`. |
| `cannot be used with --backend cgc text generation` | Remove the options the message lists. `--evidence` works only on llama.cpp: omit `--backend cgc` to use it with a `.gguf` file. |
| `Options that only apply to GGUF models (.gguf) or to --backend cgc text generation` | `--prompt-ids-file`, `--context-length`, and `--ignore-eos` need a `.gguf` file or `--backend cgc`. For a checkpoint folder, add `--backend cgc`. |
| `Windows ML did not report the tokenizer vocabulary size` (warning) | Expected for safetensors checkpoints with 2.8.183.dev0. The run continues with `vocabulary_size` set to `null`, and `--prompt-ids-file` IDs are not range-checked, so take them from this model's tokenizer. |
| `--context-length N resolved to an effective sequence capacity of M` | Windows ML did not apply the requested capacity. Choose a capacity that the model supports. |
| `--context-length N could not be verified` | Windows ML reported no logical capacity, so the requested capacity cannot be checked. Run without `--context-length`. |
| `--context-length N is smaller than the P prompt tokens plus --max-new-tokens M` | Raise `--context-length`, lower `--max-new-tokens`, or shorten the prompt. |
| `--ignore-eos requested an empty EOS set, but the Task reports EOS token IDs` | The installed Windows ML does not honor an explicit empty EOS set. Run without `--ignore-eos`. |
| `prompt token ... is outside the model vocabulary` | A token ID in `--prompt-ids-file` is not valid for this model. Regenerate the IDs with this model's tokenizer. |

To leave the experimental setup, delete the `.venv-cgc` environment and any
`--program-cache` directory you created. Your regular environment is unchanged.

## Limitations

- Local inputs only: a `.gguf` file or a checkpoint folder. Hugging Face model IDs are
  not resolved, and a single `.safetensors` file is rejected; pass its folder.
- GPU only: no CPU or NPU target. To run a `.gguf` file on the CPU, omit
  `--backend cgc` to use llama.cpp.
- Windows ML decides which models run. winml-cli keeps no list of supported
  architectures and never falls back to llama.cpp or the ONNX path. In the sample runs
  with 2.8.183.dev0:
    - Dense decoder models ran: Qwen2.5-0.5B-Instruct as a q4_k_m GGUF file and as a
      single-file safetensors checkpoint, and Qwen2.5-3B-Instruct as a sharded
      checkpoint (`model.safetensors.index.json` plus two shards).
    - A split GGUF model (`qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`) is not
      supported: `set_model` returns `0x80070032`. winml-cli checks that every shard
      exists, passes the file given with `-m` to Windows ML, and reports that Windows
      ML CGC cannot run this model. Omit `--backend cgc` to run it on llama.cpp.
    - An encoder-only checkpoint (`sentence-transformers/all-MiniLM-L6-v2`) fails in
      `set_model` with `0x80004005` (E_FAIL), which winml-cli also reports as a model
      that Windows ML CGC cannot run.
- `--evidence`, `--memory-backend dxgi`, and `--etw-tracing` are not supported.
- The GPU memory baseline, peak, and load and total deltas are `null`; see
  [Report fields](#report-fields).
- Safetensors checkpoints report no vocabulary size, so `vocabulary_size` is `null`
  and `--prompt-ids-file` cannot range-check their token IDs.
- Greedy decoding with batch size 1; Task sampling options and speculative decoding
  are not exposed by this command.
- Task stream timings include native sampling and detokenization; separate model
  phase timings are unavailable.
- Targets the preview `windowsml` 2.8.183.dev0 build, tested on one machine:
  Windows 11 build 26719 with an NVIDIA GeForce RTX 5080 and an Intel integrated GPU.
  Later builds can change the API and the models it supports.

## See also

- [winml perf](../commands/perf.md): full flag reference and runtime-specific metric definitions
- [GGUF — llama.cpp on Windows ML Runtime](gguf-llama-perf.md): the default path for `.gguf` files, and the Task stream report fields both paths share
- [winml sys](../commands/sys.md): the GPUs on this machine and their LUIDs
