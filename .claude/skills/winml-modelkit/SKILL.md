---
name: winml-modelkit
description: Build, optimize, quantize, compile, and benchmark ONNX models for Windows ML using the `winml` CLI. Covers the Build-Your-Own-Model (BYOM) pipeline across NPU (Qualcomm QNN, Intel OpenVINO, AMD VitisAI), GPU, and CPU execution providers. Use this skill whenever the user wants to run a Hugging Face or ONNX model on a Windows AI PC, target an NPU, prepare a model for on-device inference, benchmark latency on Snapdragon X Elite / Intel Core Ultra / AMD Ryzen AI, or troubleshoot operator/EP compatibility — even when they don't say "ModelKit" or "winml" by name. If a user mentions running models on Windows hardware, NPU acceleration, or low-latency on-device inference, this skill applies.
---

# WinML ModelKit

ModelKit ships a CLI called `winml` that turns a source model — a Hugging Face ID or a local ONNX file — into a portable, performant artifact that runs on any Windows execution provider. This skill teaches you the *shape* of that workflow. The CLI is the source of truth for current commands and flags.

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

You don't have to run every stage. Enter wherever the user's input lives (already have an ONNX file? skip `export`) and exit when you have what you need (just want a latency number? stop at `perf`). Several stages are EP- or hardware-sensitive — `compile` in particular targets a specific NPU and can't be run without one.

Sitting on top of the primitives are two **shortcut commands** that wrap the whole pipeline:

- A **config** command auto-detects every setting the pipeline needs and writes a JSON file.
- A **build** command reads that config and runs the stages in order.

Together they replace the seven primitives with two.

The names above (`inspect`, `export`, `analyze`, `optimize`, `quantize`, `compile`, `perf`, plus the config/build pair) are stable concepts — they map to subcommands of `winml`. Confirm exact spelling and current flags via `winml --help` before you write any command.

## The golden rule: inspect first

Before any other command, run the inspect subcommand on the user's model. It reads only the model config — no weights, no GPU spin-up — and prints an **Overall Support** verdict plus the loader, exporter, and IO that the pipeline will use.

Verdicts come in three flavors:

- **`Supported`** — green-light, the toolkit has explicit support for this architecture.
- **`Default`** — the toolkit will use TasksManager defaults. May or may not survive the full pipeline. Combine with the scope rule below before deciding.
- **`Unsupported`** — hard stop. Don't push further.

**Inspect's verdict is not the whole answer.** A generative LLM like Phi-3 or LLaMA inspects as `Default` (it has a known model type and a TasksManager default), but it's still out of scope — see the scope section. The scope rule overrides the inspect verdict. So: read inspect's output, but cross-check against scope before recommending a build.

Skipping inspect and jumping to export or build is the most common cause of confusing failures three stages in, because the cost of finding out a model is unsupported climbs at every later stage.

## Choosing a path

Once inspect passes, pick one of two paths based on what the user is trying to do. Default to **config + build** unless the user explicitly wants to fiddle with a single stage.

**Primitive commands — one stage at a time.** Right when the user is exploring, debugging a specific stage, or tweaking settings between runs. They get fine-grained control at the cost of running more commands.

**Config + build — two commands for the whole thing.** Right when the user wants a clean, reproducible, end-to-end build for production, CI, or sharing with a teammate. The generated config is the single source of truth — they edit it to override defaults, version-control it, and replay deterministically.

If the user is unsure, default to config + build unless they say "I want to try different settings" or "something failed and I need to debug a specific stage."

## Hardware and execution providers

The right execution provider depends on the user's machine. The mapping below is **silicon-vendor knowledge** — it doesn't change when ModelKit ships new flags:

| Hardware | Execution provider |
|---|---|
| Qualcomm NPU (Snapdragon X Elite) | QNN |
| Intel NPU (Meteor Lake / Lunar Lake / Core Ultra) | OpenVINO |
| AMD NPU (Ryzen AI: Phoenix / Hawk Point / Strix) | VitisAI |
| NVIDIA discrete GPU | NvTensorRTRTX |
| AMD discrete GPU | MIGraphX |
| Hardware-agnostic GPU | DirectML (Dml) |
| CPU | CPU EP (always available) |

For the **current flag spelling, supported status, and device-selection options** (including any auto-pick mode), consult `winml <command> --help` and `winml sys`. Don't hardcode flag values from this skill into your suggestions — read them live.

If you don't know what hardware the user has, ask, or run `winml sys` and read the output.

## Common patterns

**"Just benchmark this model on my hardware."** A single perf invocation against the source model is enough — ModelKit will download, export, and optimize as needed before timing. Look for a hardware-utilization flag in `winml perf --help` if the user wants live CPU/RAM/NPU monitoring.

**"What's the latency on NPU vs CPU?"** Build once, then run perf twice — once against the EP-compiled artifact on the NPU, once against the optimized (pre-compile) artifact on CPU. Compiled artifacts are EP-locked, so don't try to run a QNN-compiled model on CPU; use the optimized intermediate instead.

**"Will this model work with my hardware?"** Inspect, then analyze. The analyzer's linter classifies every operator as supported / partial / unsupported per EP — that's the cheapest way to find out a build will succeed before paying the full export cost.

**"My optimize/quantize step just blew up."** Most operator-pattern failures land at these stages even when export succeeded. Re-run analyze against the exported ONNX; the linter will usually name the offending op pattern. Don't hand-edit the ONNX graph — try a different optim or quantization configuration to dodge the unsupported pattern, or escalate to "this model isn't a fit for this EP."

## Scope — what's in and what's out

**In scope.** Classic deep learning models — CNNs, encoders, vision transformers, NLP classifiers, NER, object detection, segmentation. Concretely: ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, Table Transformer, SegFormer families. If the user passes one of these, the pipeline is designed to handle it.

**Out of scope.** Generative and decoder-only architectures: GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, any seq2seq generator. If a user asks ModelKit to handle one of these, **stop and say so** — the pipeline will fail mid-way and the error won't always make the cause obvious. LLM support (with LoRA) is on the public roadmap for late 2026; don't pretend it works today.

If you're genuinely unsure whether a model is in scope, the inspect command is the source of truth. Trust its verdict over your guess.

## Things that catch people out

- **Compile validation requires the target EP to actually be registered on the machine.** This is the one place the CLI doesn't fail loudly: if you ask for `--ep qnn` on a machine without QNN registered, `winml compile` silently substitutes another available EP (e.g., OpenVINO) and exits 0. The output file name keeps the `qnn` you asked for, but the embedded artifact is for the substituted EP — the user thinks they have a QNN context binary and they don't. Always verify after compile by reading the EPContext node's `source` attribute on the output ONNX, or check `winml sys --list-ep` first to confirm the requested EP is registered.
- **Compile produces an EPContext-wrapped stub plus a separate cache blob** — the `.onnx` output is tiny (~1 KB) and references a co-located `.blob` (tens of MB) that holds the real compiled graph. If you move the artifact, move the blob with it.
- **The config file is the source of truth on the build path.** Edits to the JSON between config and build are how you override defaults; don't try to layer on conflicting flags at build time.
- **Output paths are explicit.** Every command takes an output flag — there's no implicit "current directory" convention. Tell the user where files will land.
- **EP-compiled models are EP-locked.** Running a QNN-compiled model on CPU EP (or vice versa) gives nonsense results. If perf numbers look wildly wrong, check the EP matches the artifact.
- **Don't fabricate flags.** If a flag isn't in `winml <command> --help`, it doesn't exist. Find a real one or change approach.

## When things go sideways

Read the error before suggesting a next step. ModelKit error messages are usually specific (op name, EP, stage). When you don't know what to do:

1. `winml <failing-command> --help` to confirm you used real flags.
2. `winml sys --list-ep` to confirm the EP is actually registered on this machine.
3. `winml inspect` and `winml analyze` to confirm the model is supported and the operator pattern is buildable.

The CLI is self-documenting; lean on it before guessing.
