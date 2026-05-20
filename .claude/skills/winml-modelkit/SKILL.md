---
name: winml-modelkit
description: Build, optimize, quantize, compile, and benchmark ONNX models for Windows ML using the `winml` CLI. Covers the Build-Your-Own-Model (BYOM) pipeline across NPU (Qualcomm QNN, Intel OpenVINO, AMD VitisAI), GPU, and CPU execution providers. Use this skill whenever the user wants to run a Hugging Face or ONNX model on a Windows AI PC, target an NPU, prepare a model for on-device inference, benchmark latency on Snapdragon X Elite / Intel Core Ultra / AMD Ryzen AI, or troubleshoot operator/EP compatibility — even when they don't say "ModelKit" or "winml" by name. If a user mentions running models on Windows hardware, NPU acceleration, or low-latency on-device inference, this skill applies. **Skip for generative models** — LLMs (GPT, LLaMA, Phi, Mistral), Stable Diffusion, Whisper, or any decoder-only / seq2seq architecture are out of scope (planned for late 2026).
---

# WinML ModelKit

ModelKit ships a CLI called `winml` that turns a source model — a Hugging Face ID or a local ONNX file — into a portable, performant artifact that runs on any Windows execution provider. This skill teaches you the *shape* of that workflow. The CLI is the source of truth for current commands and flags.

## Installing the CLI

**Default behavior: lead any walkthrough with a brief install section.** Unless the user signals they already have `winml` working, include the install steps below (or a clear "prereq: install winml first" pointer to them) as the first thing in your response. First-timers shouldn't have to guess what they need.

**Skip the install section only if the user clearly signals they're past install:**
- They quote a `winml <command>` they ran, with output or an error from it.
- They say they "already" / "previously" exported, built, optimized, etc. with winml.
- They share an artifact path that came out of an earlier winml run.
- They're asking a debugging or comparison question that presumes a working install.

When in doubt, include it — a five-line prereq block is cheaper than a stuck user.

ModelKit pins **Python 3.10 exactly** (`>=3.10,<3.11`) — use `uv` to create an isolated venv so you don't pollute system Python or land on a 3.11+ environment that won't resolve.

**1. Create a Python 3.10 environment**

```bash
uv venv --python 3.10
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (Git Bash / WSL)
source .venv/Scripts/activate
```

**2. Install the `winml-cli` wheel**

Today the wheel ships locally with AI Toolkit (AITK), not from PyPI. Install it from the AITK cache:

```powershell
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
```

When `winml-cli` is published to PyPI (planned), replace step 2 with:

```bash
uv pip install winml-cli
```

**3. Verify**

```bash
winml --help
winml sys --list-ep
```

`--help` should print the command list, and `sys --list-ep` should show the execution providers registered on this machine.

## Discover the CLI before doing anything else

The command set and flags evolve. Don't memorize them and don't guess them — read them from the tool itself:

- **`winml --help`** — current top-level command list with one-line descriptions.
- **`winml <command> --help`** — current flags, arguments, and defaults for that command.
- **`winml sys --list-device --list-ep`** — what hardware and execution providers are actually present on this machine.

Run these *before quoting any command to the user*, not after. "I'll check `--help` if anything looks off" is too late — the user has already copy-pasted a broken command and come back annoyed. If you're about to write `winml <something>` in your reply, run `winml <something> --help` first.

**The CLI is flag-based, not positional.** Model IDs and paths go through `-m` / `--model`, not as bare positional arguments. `winml inspect microsoft/resnet-50` will error — you need `winml inspect -m microsoft/resnet-50`. This shape is stable across the toolkit; the specific flag spelling per command isn't, which is why you still read `--help`.

Inventing plausible-sounding flags (a `--preset`, `--profile`, `--mode=fast`) is the most common way to waste the user's time — the command will reject them and the user has to come back. When in doubt, `--help`.

## The mental model

ModelKit organizes work as a pipeline. Each stage is its own primitive command, and the output of one stage feeds the next:

```
inspect → export → analyze → optimize → quantize → compile → perf
```

You don't have to run every stage. Enter wherever the user's input lives (already have an ONNX file? skip `export`) and exit when you have what you need (just want a latency number? stop at `perf`). Several stages are EP- or hardware-sensitive — `compile` is documented as requiring an NPU device (per README's Scope & Limitations: "winml compile requires an NPU device"). `winml compile --help` does expose `--device` and `--ep` values for CPU/GPU, but treat NPU as the assumed target unless the user says otherwise.

Sitting on top of the primitives are two **shortcut commands** that wrap the whole pipeline:

- A **config** command auto-detects every setting the pipeline needs and writes a JSON file.
- A **build** command reads that config and runs the stages in order.

Together they replace the seven primitives with two.

The names above (`inspect`, `export`, `analyze`, `optimize`, `quantize`, `compile`, `perf`, plus the config/build pair) are stable concepts — they map to subcommands of `winml`. Confirm exact spelling and current flags via `winml --help` before you write any command.

## The golden rule: inspect first

Before any other command, run the inspect subcommand on the user's model. Per `winml inspect --help`, it reads the model configuration *without downloading weights* and shows the loader, exporter, WinML inference class, I/O specs, and the build resolution the pipeline will use. Pass `-f json` for machine-readable output.

Inspect tells you whether the toolkit knows how to handle the architecture. But **always cross-check against the scope section below** — a model that inspect accepts can still be out of scope. The scope rule overrides anything inspect prints; for example, an LLM may have a usable loader/exporter via TasksManager defaults but is still not a fit.

Skipping inspect and jumping to export or build is the most common cause of confusing failures three stages in, because the cost of finding out a model is unsupported climbs at every later stage.

## Choosing a path

Once inspect passes, pick one of two paths based on what the user is trying to do. Default to **config + build** unless the user explicitly wants to fiddle with a single stage.

**Primitive commands — one stage at a time.** Right when the user is exploring, debugging a specific stage, or tweaking settings between runs. They get fine-grained control at the cost of running more commands.

**Config + build — two commands for the whole thing.** Right when the user wants a clean, reproducible, end-to-end build for production, CI, or sharing with a teammate. The generated config is the single source of truth — they edit it to override defaults, version-control it, and replay deterministically.

If the user is unsure, default to config + build unless they say "I want to try different settings" or "something failed and I need to debug a specific stage."

## Hardware and execution providers

The right execution provider depends on the user's machine. Status as of 2026-05-20:

| Hardware | Execution provider | Status |
|---|---|---|
| Qualcomm NPU (Snapdragon X Elite) | QNN | 🟢 Ready |
| Intel NPU (Meteor Lake / Lunar Lake / Core Ultra) | OpenVINO | 🟢 Ready |
| AMD NPU (Ryzen AI: Phoenix / Hawk Point / Strix) | VitisAI | 🟢 Ready |
| NVIDIA discrete GPU | NvTensorRTRTX | 🟢 Ready |
| Hardware-agnostic GPU | DirectML (Dml) | 🟢 Ready |
| AMD discrete GPU | MIGraphX | 🔶 Planned |
| CPU | CPU EP | ⚪ Always available |

If the user has hardware whose EP is **Planned** (currently only MIGraphX for AMD discrete GPUs), say so — recommend CPU or DML as the working fallback rather than pretending the planned EP is ready. The README's Supported Hardware table may lag behind this status; trust `winml sys --list-ep` on the user's machine for what's actually registered.

For the **current flag spelling, supported status, and device-selection options** (including any auto-pick mode), consult `winml <command> --help` and `winml sys`. Don't hardcode flag values from this skill into your suggestions — read them live.

If you don't know what hardware the user has, ask, or run `winml sys` and read the output.

## Common patterns

**"Just benchmark this model on my hardware."** A single perf invocation against the source model is enough — `winml perf` builds artifacts on the fly (see `--rebuild`, `--ignore-cache`, `--no-quantize` in `winml perf --help`). You don't have to chain primitives manually. For live NPU utilization during the run, look for the `--monitor` flag in `winml perf --help`.

**"What's the latency on NPU vs CPU?"** Build once, then run perf twice — once against the EP-compiled artifact on the NPU, once against the optimized (pre-compile) artifact on CPU. Compiled artifacts are tied to the EP they were compiled for, so run the CPU comparison against the pre-compile optimized ONNX, not the compiled NPU artifact.

**"Will this model work with my hardware?"** Inspect, then analyze. The analyzer's linter classifies every operator as supported / partial / unsupported per EP — that's the cheapest way to find out a build will succeed before paying the full export cost.

**"My optimize/quantize step just blew up."** Most operator-pattern failures land at these stages even when export succeeded. Re-run analyze against the exported ONNX; the linter will usually name the offending op pattern. Don't hand-edit the ONNX graph — try a different optim or quantization configuration to dodge the unsupported pattern, or escalate to "this model isn't a fit for this EP."

## Scope — what's in and what's out

**In scope.** Classic deep learning models — CNNs, encoders, vision transformers, NLP classifiers, NER, object detection, segmentation. Concretely: ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, Table Transformer, SegFormer families. If the user passes one of these, the pipeline is designed to handle it.

**Out of scope.** Generative and decoder-only architectures: GPT, LLaMA, Phi, Mistral, Stable Diffusion, any seq2seq generator. If a user asks ModelKit to handle one of these, **stop and say so** — the pipeline will fail mid-way and the error won't always make the cause obvious. LLM support (with LoRA) is on the public roadmap for late 2026; don't pretend it works today.

If you're genuinely unsure whether a model is in scope, the inspect command is the source of truth. Trust its verdict over your guess.

## Things that catch people out

- **Confirm the target EP is registered before compiling.** Run `winml sys --list-ep` first; if your `--ep <foo>` isn't in the list, compile won't produce a usable artifact for that EP. Compile also runs validation by default (see `--validate / --no-validate` in `winml compile --help`).
- **Compile defaults to external EP-context storage.** Per `winml compile --help`, the default writes EP context to a `.bin` file co-located with the output `.onnx`; pass `--embed` to inline it instead. If you move a non-embedded artifact, move the `.bin` alongside.
- **CLI flags override the config file, not the other way around.** Every primitive that accepts `-c, --config` says so in its `--help`: "Provides defaults; explicit CLI options take precedence." For repeatable builds, edit the JSON; for one-off overrides, pass the flag at build time.
- **Output paths are explicit on the pipeline-building commands.** `export`, `optimize`, `quantize`, `compile`, `perf`, `config`, and `build` each take an `-o` / `--output` (or `--output-dir`). There's no implicit "current directory" convention — tell the user where files will land. `inspect`, `sys`, and `hub` print to stdout and don't require an output path.
- **EP-compiled models are tied to their target EP.** Don't try to perf a QNN-compiled artifact against the CPU EP — the result is at best meaningless. For cross-EP comparison, use the pre-compile optimized ONNX.
- **Don't fabricate flags.** If a flag isn't in `winml <command> --help`, it doesn't exist. Find a real one or change approach.

## When things go sideways

Read the error before suggesting a next step. ModelKit error messages are usually specific (op name, EP, stage). When you don't know what to do:

1. `winml <failing-command> --help` to confirm you used real flags.
2. `winml sys --list-ep` to confirm the EP is actually registered on this machine.
3. `winml inspect` and `winml analyze` to confirm the model is supported and the operator pattern is buildable.

The CLI is self-documenting; lean on it before guessing.
