# eval-llm-out-of-scope

## Prompt

> Can you help me convert microsoft/Phi-3-mini-4k-instruct to run on my Intel NPU? I want to use winml or whatever the windows ml toolkit is.

## with_skill — 5/5

### Response

# Phi-3-mini on Intel NPU via WinML ModelKit — not today

Short version: **WinML ModelKit can't do this yet.** Phi-3 is a generative decoder-only LLM, and that whole family (GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, seq2seq generators) is explicitly out of scope for the current pipeline. If we push `microsoft/Phi-3-mini-4k-instruct` through the toolkit, it will likely fail somewhere mid-pipeline (export, optimize, or quantize), and the error usually won't make the root cause obvious — so I'd rather flag it now than burn an hour of your time.

LLM support (with LoRA) is on the public roadmap for **late 2026**, but it isn't shipping today.

## What the toolkit is for

WinML ModelKit's `winml` CLI is built around classic deep-learning models — the kind of thing you'd target with ONNX Runtime on a Windows AI PC for low-latency, on-device inference. Concretely, that means:

- **Vision:** ResNet, ViT, Swin, ConvNeXT, SegFormer, Table Transformer, object detection
- **NLP (encoder-style):** BERT, RoBERTa, NER, classifiers
- **Speech, segmentation, etc.** in the same encoder/CNN/ViT shape

For those, the pipeline (`inspect → export → analyze → optimize → quantize → compile → perf`, or just `config` + `build` end-to-end) handles Intel NPU via the **OpenVINO** execution provider, Qualcomm via **QNN**, AMD via **VitisAI**, plus GPU/CPU fallbacks.

Phi-3 just isn't in that shape — it's a 3.8B-parameter autoregressive transformer with KV cache, dynamic sequence lengths, and quantization needs (per-channel weight-only INT4, group-wise) that the current ModelKit optimize/quantize stages aren't designed for. The compile stage for Intel NPU in particular wouldn't produce a usable artifact.

If you want, I can run `winml inspect -m microsoft/Phi-3-mini-4k-instruct` to show you what the toolkit itself says — but I'd expect a `Default` verdict (it has a known model type with TasksManager defaults), which is *not* a green light. The scope rule overrides the inspect verdict for generative LLMs, and that's by design: inspect only reads config, not the full graph.

## What actually works for Phi-3 on Intel NPU today

Since you're on Intel hardware, the realistic options for running Phi-3-mini on the NPU right now are:

1. **Intel's OpenVINO GenAI** — `optimum-intel` has direct Phi-3 support and can export to OpenVINO IR with INT4 weight compression targeting NPU. This is the path Intel themselves recommend for Core Ultra NPUs. Search for `OVModelForCausalLM` and the OpenVINO NPU plugin docs.
2. **ONNX Runtime GenAI** (`onnxruntime-genai`) — Microsoft's runtime specifically for generative models. There are pre-built Phi-3-mini ONNX variants on Hugging Face (`microsoft/Phi-3-mini-4k-instruct-onnx`) with DirectML and CPU configurations. NPU support via OpenVINO EP is improving but less mature.
3. **DirectML via ONNX Runtime** — runs on the iGPU rather than the NPU, but on Core Ultra the Arc iGPU is often the strongest target for LLMs anyway.

None of those go through `winml`. They're separate toolchains.

## If you want me to do something concrete

A few directions I can help with:

- **Pick a non-LLM model that *does* work on your Intel NPU through ModelKit** — e.g. a vision or BERT-family model — and walk through the full inspect → config → build → perf flow end to end. Good way to learn the toolkit without hitting the scope wall.
- **Get you started on the Intel OpenVINO GenAI path for Phi-3 outside of ModelKit** — that's the right tool for this job today, even though it's a different stack.
- **Confirm what hardware/EPs are registered** on your machine with `winml sys --list-device --list-ep` so we know what your NPU actually advertises.

Which of those would be most useful? My default would be option 2 — get you onto a path that will actually run Phi-3 on your NPU — unless you specifically need the ModelKit workflow, in which case option 1 with a different model is the way.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies Phi-3 as a decoder-only LLM / generative model. | PASS | Quote: 'Phi-3 is a generative decoder-only LLM, and that whole family (GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, seq2seq generators) is explicitly out of scope.' |
| ✓ | Clearly states this model is NOT supported / out of scope for the winml pipeline today (not hedged). | PASS | Quote: 'Short version: WinML ModelKit can not do this yet.' + 'explicitly out of scope for the current pipeline.' |
| ✓ | Does NOT walk the user through export/optimize/quantize/compile/build. | PASS | No winml build pipeline commands recommended. |
| ✓ | Mentions LLM support is on the roadmap OR suggests a legitimate alternative. | PASS | Both: 'late 2026' roadmap + OpenVINO GenAI / ONNX Runtime GenAI / DirectML alternatives. Notes 'None of those go through winml.' |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 4/5 ⚠

### Response

Happy to help you think through this. Quick heads-up before we dive in: getting Phi-3-mini onto an Intel NPU via Windows ML is doable, but there are a few sharp edges. Let me lay out the landscape and the most realistic path.

## The toolchain you actually want

"Windows ML" is now a bit of an overloaded term. For your scenario you have two main options:

1. **Windows ML (the new runtime, formerly Windows AI Foundry / WinML)** — the modern ONNX Runtime-based stack that ships execution providers (EPs) for CPU, GPU (DirectML), and NPU (vendor EPs). For Intel NPUs (Meteor Lake / Lunar Lake / Arrow Lake "Core Ultra"), this means the **OpenVINO Execution Provider** is what actually targets the NPU under the hood.
2. **ONNX Runtime directly with the OpenVINO EP** — essentially the same thing without the WinML wrapper. Often easier to debug.

Either way, the model has to be **ONNX**, and for an LLM like Phi-3 it really wants to be **quantized to INT4 (or at least INT8)** to fit and run well on the NPU.

## The realistic path

**Step 1: Get an ONNX version of Phi-3-mini.**
Don't export it yourself if you can avoid it. Microsoft already publishes ONNX builds:

- `microsoft/Phi-3-mini-4k-instruct-onnx` on Hugging Face — has CPU, CUDA, and DirectML variants, with INT4 weight-only quantization (RTN) prebuilt.
- There's also a `Phi-3-mini-4k-instruct-onnx-directml` repo specifically.

For Intel NPU, neither of those is a perfect drop-in — they're tuned for DirectML/CUDA. But the **FP16 or INT4 ONNX graph** from those repos is a good starting point.

**Step 2: Convert / optimize for the Intel NPU.**
Use Intel's **OpenVINO Model Optimizer** or the newer `optimum-intel` flow:

```
pip install optimum[openvino] optimum-intel
optimum-cli export openvino \
  --model microsoft/Phi-3-mini-4k-instruct \
  --weight-format int4 \
  --task text-generation-with-past \
  phi3-mini-ov-int4
```

This produces an OpenVINO IR (`.xml` + `.bin`) which is what the NPU plugin actually runs best. INT4 weight compression is pretty much required to fit Phi-3-mini in NPU memory comfortably.

**Step 3: Run it.**
Easiest first run is straight OpenVINO GenAI:

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("phi3-mini-ov-int4", "NPU")
print(pipe.generate("Hello", max_new_tokens=100))
```

If `device="NPU"` errors out (it often does for LLMs depending on driver/OpenVINO version), fall back to `"GPU"` or `"CPU"` to sanity-check the model, then revisit NPU.

If you specifically need the WinML / ONNX Runtime path instead, install `onnxruntime-openvino` and set the provider to `OpenVINOExecutionProvider` with `device_type="NPU"`.

## Caveats you should know up front

- **NPU support for LLMs is still maturing.** You need a recent Intel NPU driver (check Intel's site — the LLM-on-NPU path needs fairly new drivers) and OpenVINO 2024.4+.
- **Context length matters.** NPU often has stricter shape constraints; you may need to compile with a fixed max sequence length rather than dynamic.
- **Not everything will offload.** Some ops may fall back to CPU. That's normal; just watch perf.
- **Phi-3-mini-4k** at INT4 is roughly ~2 GB — fine for modern NPUs, but verify your NPU's memory budget.

## What I'd do first

1. Confirm your CPU is Core Ultra (Series 1 or 2) and update the NPU driver from Intel's site.
2. `pip install --upgrade openvino openvino-genai optimum[openvino] optimum-intel`
3. Run the `optimum-cli export openvino ... --weight-format int4` command above.
4. Try the 4-line `LLMPipeline` test with `"NPU"`, fall back to `"CPU"` if it complains so you know the model itself is good.

Want me to go deeper on any of these — e.g., the ONNX Runtime + OpenVINO EP route specifically, troubleshooting NPU compilation errors, or wiring it into a C#/C++ app via Windows ML?

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Identifies Phi-3 as a decoder-only LLM / generative model. | PASS | Treats Phi-3 as an LLM throughout â€” discusses INT4 weight-only quantization, KV cache shape constraints, etc. |
| ✗ | Clearly states this model is NOT supported / out of scope for the winml pipeline today (not hedged). | **FAIL** | Goes the opposite direction: 'getting Phi-3-mini onto an Intel NPU via Windows ML is doable'. Never says winml ModelKit (the toolkit named in the user's question) doesn't support LLMs. |
| ✓ | Does NOT walk the user through winml build pipeline as if Phi-3 will go through it. | PASS | Doesn't use winml-style commands; routes the user to OpenVINO GenAI / optimum-intel instead. |
| ✓ | Mentions LLM roadmap OR a legitimate alternative. | PASS | Suggests OpenVINO Model Optimizer / optimum-intel / OpenVINO GenAI / ONNX Runtime + OpenVINO EP. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
