# WinML ModelKit — Bug Bash

## What this skill does

`winml-modelkit` teaches a coding-assistant agent (Claude Code, GitHub Copilot, Cursor, etc.) how to drive the `winml` CLI to build, optimize, quantize, compile, and benchmark ONNX models for Windows AI PCs — across NPU (Qualcomm QNN, Intel OpenVINO, AMD VitisAI), GPU (NV TRT-RTX, DirectML), and CPU execution providers. Users describe what they want in plain language ("benchmark resnet on my Snapdragon NPU", "build a QNN-compiled model for CI"); the agent reads SKILL.md and drives the CLI without the user looking up flags.

Skill is at `C:/repo/WinML-ModelKit/.claude/skills/winml-modelkit/SKILL.md`. Eval baseline at [`reports/20260520-135348.md`](reports/20260520-135348.md).

## How to file bugs

For each scenario you try, note:
- **What you typed** (the prompt — paste it verbatim if you adapted)
- **What the agent did** (commands it ran, response shape, did it refuse?)
- **What was wrong** (specific failure, not just "didn't work")
- **Your hardware** (Snapdragon X Elite / Intel Core Ultra / AMD Ryzen AI / CPU-only)

Open issues at the repo's GitHub Issues tab with label `bug-bash` and the hardware tag.

---

## Scenarios to test

Five categories. Pick scenarios that match your hardware — anything marked **(any)** runs on CPU-only too. Scenarios marked **(NPU)** require the specified hardware.

### A. Setup & install

| # | Hardware | Prompt | What to verify |
|---|---|---|---|
| A1 | any (fresh machine) | "i just got my snapdragon x elite dev box and i don't have winml installed yet. python 3.10 is already on the system. give me the actual install commands i should run." | Agent should call out Python 3.10 won't work → recommend `uv venv --python 3.11`. Should install from PyPI via `uv pip install winml-cli`. Should include `winml --help` / `winml sys --list-ep` verify step. |
| A2 | any | "is winml on my path? what version do I have?" | Agent should run `winml --version` (or `--help`) and report. Should not hallucinate a version. |

### B. Happy-path NPU builds

| # | Hardware | Prompt | What to verify |
|---|---|---|---|
| B1 | Snapdragon X Elite | "i just got a snapdragon x elite dev box and i want to run microsoft/resnet-50 on the npu. can you walk me through getting it built and benchmarked? give me actual commands i can copy paste." | Agent identifies **QNN** as the EP (not OpenVINO/VitisAI). Recommends `winml inspect` first. Walks `config → build → perf` (the shortcut path, not 5 separate primitives). Final benchmark step uses `--device npu --ep qnn`. |
| B2 | Intel Core Ultra | "i want to deploy `google/vit-base-patch16-224` to my Intel Core Ultra NPU for image classification. walk me through it end-to-end." | Agent identifies **OpenVINO** as the EP. Same shape as B1. |
| B3 | AMD Ryzen AI | "i have a Ryzen AI laptop. what's the fastest way to see how fast `facebook/convnext-tiny-224` runs on my NPU? i don't care about the artifact, just the number." | Agent identifies **VitisAI** EP. Recommends a **single** `winml perf` invocation (`winml perf -m ... --device npu --ep vitisai --ignore-cache`), not chained primitives — the user asked for the number, not a saved artifact. |
| B4 | any | "build a CPU-optimized version of microsoft/resnet-50 using winml. CPU EP only, no NPU needed." | Agent should use `winml config -m ... --device cpu` then `winml build`. Should NOT compile (compile is NPU-only). Confirm a `.onnx` artifact lands in the output dir. |

### C. Pipeline knowledge

| # | Hardware | Prompt | What to verify |
|---|---|---|---|
| C1 | any | "i have a ViT model (google/vit-base-patch16-224). how do i decide whether to ship the NPU version or just use CPU? running on a snapdragon x elite. cleanest way to get apples-to-apples numbers." | Agent recommends **build once, perf twice**: NPU run uses the QNN-compiled artifact, CPU run uses the **pre-compile optimized** ONNX (not the QNN-compiled one — that's EP-locked). |
| C2 | any | "before i commit to building `nvidia/segformer-b1-finetuned-ade-512-512` end-to-end on my Intel NPU, is there a way to know if it'll make it through the full pipeline? burned half a day last time on a model that exported fine but died at the optimize stage." | Agent recommends **`winml inspect`** then **`winml analyze`** as a pre-flight (not "build it and see"). Explains analyze's role: linter classifies ops as supported / partial / unsupported per EP. |
| C3 | any | "i already have a resnet50.onnx file from a previous export. how do i benchmark it on my Intel NPU without re-downloading from huggingface?" | Agent recognizes `winml perf` accepts a local `.onnx` directly via `-m`. Should NOT walk through `winml export` again. |
| C4 | Snapdragon X Elite | "we want a CI job that rebuilds an optimized resnet50 (targeting qnn) every time the model config changes. one json config in the repo, reproducible build step." | Agent recommends `winml config` → check JSON into git → CI runs `winml build -c <json>`. CI yaml should include winml-cli install step. Should NOT claim "config overrides CLI" (CLI overrides config). |

### D. Out-of-scope refusal (this is where bugs often hide)

The skill must **refuse** generative models cleanly — not walk the user through a pipeline that will fail mid-way. Try these with different models on different hardware.

| # | Hardware | Prompt | What to verify |
|---|---|---|---|
| D1 | any | "can you help me convert microsoft/Phi-3-mini-4k-instruct to run on my Intel NPU? I want to use winml." | Agent says **"out of scope"** clearly. Does NOT run `winml build / compile / quantize / export`. Suggests alternatives (ONNX Runtime GenAI, OpenVINO GenAI, Foundry Local). Mentions late-2026 roadmap. |
| D2 | any | "i want to run openai/whisper-base on my snapdragon x elite NPU for live transcription. how do i optimize it through the winml pipeline?" | Same as D1 but for Whisper (seq2seq audio). Refusal should generalize — not just "Whisper specifically" but "seq2seq encoder-decoder generators are out of scope." |
| D3 | any | "i want to deploy `salesforce/codet5p-220m` on my snapdragon NPU for code summarization in a VSCode extension. walk me through the winml pipeline?" | **This model is not in the description's explicit blacklist** — tests whether the body's scope rule generalizes. Agent must recognize CodeT5+ as T5-family encoder-decoder seq2seq and refuse for that *general* reason. |
| D4 | any | "set up stable diffusion v1.5 to run on my NPU via winml" | Generative model — refuse, point at SD-specific runtimes. |

### E. Trigger boundary — the skill should NOT load for these

The skill's description is the only thing the host agent sees when deciding whether to load. Try these and verify the agent **does not pull in the winml skill body** — it should give whatever default response a non-winml-aware Claude would.

| # | Prompt | What to verify |
|---|---|---|
| E1 | "how do i call windows ml from a c# app? i want to load my onnx model and run inference from a WinUI 3 app." | Skill should NOT load. This is the WinML C#/WinRT SDK, not the winml CLI. |
| E2 | "Olive's QNN quantization keeps crashing at the calibration step. how do i debug?" | Skill should NOT load. Olive is a different optimization tool. |
| E3 | "how do i register a custom onnx model with windows ai foundry so copilot runtime can call it from a c# app?" | Skill should NOT load. Windows AI Foundry is a different integration layer. |
| E4 | "i'm using onnxruntime-genai to run microsoft/phi-3-mini-4k-instruct on my snapdragon. how do i benchmark this?" | Skill should NOT load — ORT-GenAI is a different runtime. (If it does load, that's an over-trigger bug.) |
| E5 | "how do i call Phi Silica from my WinUI 3 app to summarize text?" | Skill should NOT load. Phi Silica is an OS-bundled model, not BYOM. |

If you're unsure whether the skill loaded, ask the agent "did you read the winml-modelkit skill?" — it should be able to tell you.

---

## What to look for (bug categories)

**Wrong EP suggestion.** Snapdragon → QNN, Intel NPU → OpenVINO, Ryzen AI → VitisAI. If the agent suggests OpenVINO for Snapdragon, that's a hard bug.

**Fabricated flags.** Anything like `--preset fast`, `--mode npu`, `--profile production` — these don't exist in `winml`'s `--help`. The agent should run `winml <cmd> --help` to verify before quoting flags.

**Missing install step on fresh machines.** If the user signals they just got a dev box / fresh setup and the agent jumps straight to `winml inspect` without prereqs, that's a regression.

**Pipeline reversed: claiming "config overrides CLI".** Every `winml` primitive's `--help` says "config provides defaults; CLI takes precedence." If the agent says it the other way around, that's a bug.

**Out-of-scope walk-through.** If you ask for Phi-3 / LLaMA / Whisper / CodeT5+ / Stable Diffusion and the agent starts running `winml export -m phi-3` instead of refusing — that's a hard bug. The pipeline will fail mid-way with confusing errors.

**Silent EP fallback unflagged.** On a CPU-only machine, if the agent quotes `--ep qnn --device npu` and doesn't warn about needing QNN registered, that's a bug — the artifact wouldn't be a real NPU artifact.

**Refusal hedging.** "Out of scope" should be a clear refusal, not "might work, try and see." Check D-series scenarios for this.

**Over-trigger.** If you type any E-series query and the agent loads the winml-modelkit skill body or recommends `winml` commands, that's an over-trigger bug — the description failed to filter.

---

## Optional: deeper digging

If you have time after the basics:

- **Edit SKILL.md** to test fragility. Add a typo to the scope section, see if the agent still refuses Phi-3 correctly. Revert your edit before submitting bugs.
- **Try mixed languages** — the skill is English-only but the agent might handle other languages. Try prompting in Chinese / Japanese; verify the agent still routes correctly even if its response language differs.
- **Try ambiguous hardware** — "i have an AI PC, what can i do with winml?" — see how the agent disambiguates.
- **Try a model the skill doesn't list** — neither blacklisted nor explicitly listed (e.g., DETR, DeiT, MobileViT). The agent should recommend `winml inspect` to verify.

## Eval baseline

For reference, the automated eval baseline as of this bugbash:

- Trigger: 29/30 = 97% (one K=1 LLM-judge flake)
- Response: 56/56 = 100% on with-skill responses; baseline 51.8%; **delta +48.2pp**
- E2E (CPU subset): 3/3 cases Pass@3; 3 NPU cases SKIP (need hardware to close)

Bugs you find above and beyond this baseline are exactly what we want to surface.
