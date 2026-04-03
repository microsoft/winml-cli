# WinML Generation Loop — Design Specification

**Date:** April 3, 2026
**Status:** Implemented, E2E validated (monolithic TrOCR + BLIP)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Design Details](#3-design-details)
4. [Class Hierarchy](#4-class-hierarchy)
5. [Static Shape Handling](#5-static-shape-handling)
6. [Token ID Resolution](#6-token-id-resolution)
7. [Tested Configurations](#7-tested-configurations)
8. [Future Extensibility](#8-future-extensibility)

---

## 1. Overview

### Goal

Implement autoregressive text generation for ONNX models (e.g. image captioning, OCR) that works with HuggingFace pipelines. The ONNX model may be a single fused file (monolithic) or separate encoder/decoder files (split).

### Usage

```python
from transformers import AutoConfig, pipeline
from winml.modelkit.models.winml import WinMLModelForImageToText

model_id = "microsoft/trocr-base-printed"
config = AutoConfig.from_pretrained(model_id)
model = WinMLModelForImageToText(onnx_path="trocr_model.onnx", config=config)

pipe = pipeline("image-to-text", model=model, tokenizer=model_id, image_processor=model_id)
result = pipe(image)
# → [{'generated_text': 'INVOICE 12345'}]
```

### Implementation size

| File | Lines | Purpose |
|---|---|---|
| `generation_mixin.py` | ~260 | `OnnxGenerativeInputs`, `DummyEncoder`, `OnnxEncoder`, `WinMLGenerationMixin` |
| `image_to_text.py` | ~110 | `WinMLModelForImageToText` — task-specific subclass |

---

## 2. Design Principles

### 2.1 Support monolithic today, open for split and KV cache

The current implementation targets monolithic ONNX models with static shapes — the most common export format for NPU deployment. The architecture is designed so that adding split encoder/decoder support or KV cache is additive (new code, not rewriting existing code).

### 2.2 Leverage existing HF capability

HF's `GenerationMixin` provides ~2000 lines of battle-tested generation code: greedy search, multinomial sampling, beam search, stopping criteria, logits processors, and output formatting. We inherit all of it rather than reimplementing the autoregressive loop ourselves.

### 2.3 No hacking into HF internals

We do not override any private HF methods (names starting with `_`). We do not depend on internal behavior of `GenerationMixin` that could change across releases. The integration is purely through public contract: `forward()`, `get_encoder()`, `can_generate()`, `generation_config`, `device`, and `config.is_encoder_decoder`.

---

## 3. Design Details

### 3.1 HF GenerationMixin — what it gives us

`GenerationMixin` is HF's autoregressive decode loop. When a model class inherits from it, calling `model.generate()` runs the full generation pipeline:

1. **Setup:** extract main input, run encoder (once), seed `decoder_input_ids` from `decoder_start_token_id`
2. **Decode loop:** at each step, call `forward()` → get logits → select next token → append → repeat
3. **Stopping:** EOS token detection, max length, custom stopping criteria
4. **Strategies:** greedy (`do_sample=False`), sampling (`do_sample=True, temperature=...`), beam search (`num_beams>1`)

The only method we need to provide is `forward()` — one decode step that takes `decoder_input_ids` + `encoder_outputs` and returns logits. HF drives everything else.

### 3.2 ONNX vs HuggingFace — why one class fits all

HuggingFace has many architecture-specific classes for image-to-text: `VisionEncoderDecoderModel` (TrOCR), `BlipForConditionalGeneration` (BLIP), `Pix2StructForConditionalGeneration`, etc. Each has different attribute names, different forward signatures, and different internal structure.

After ONNX export, all these architectures converge to the same I/O pattern:

```python
# TrOCR, BLIP, Pix2Struct — ALL monolithic exports have:
inputs:  ["pixel_values", "input_ids", "attention_mask"]
outputs: ["logits"]
```

**ONNX export is a normalizing transformation.** Complex HF class hierarchies collapse into uniform I/O patterns. This means we need **one WinML class per ONNX I/O pattern**, not per HF architecture:

| ONNX Pattern | HF Architectures | WinML Class |
|---|---|---|
| Monolithic enc-dec `(pixel_values, input_ids → logits)` | TrOCR, BLIP, Pix2Struct, Donut | `WinMLModelForImageToText` |
| Split enc-dec `(encoder.onnx + decoder.onnx)` | Same models, split export | `WinMLModelForImageToText` (same class, `encoder_path` kwarg) |
| Decoder-only `(input_ids → logits)` | GPT-2, LLaMA | `WinMLModelForCausalLM` (future) |

### 3.3 Generation flow

```
model.generate(pixel_values=...)
    │
    │  HF GenerationMixin (inherited, not overridden)
    │
    ├─ 1. get_encoder()(pixel_values=...)  →  BaseModelOutput
    │      DummyEncoder: pass through (monolithic)
    │      OnnxEncoder:  run encoder ONNX session (split)
    │
    ├─ 2. Seed decoder_input_ids from decoder_start_token_id
    │
    └─ 3. Decode loop (greedy / sample / beam search)
           │
           ├─ prepare_inputs_for_generation()        [HF default, not overridden]
           │      maps input_ids → decoder_input_ids
           │      passes through encoder_outputs
           │
           ├─ forward(decoder_input_ids, encoder_outputs, ...)   [our override]
           │      _prepare_onnx_inputs()  →  pad to static shape
           │      _run_inference()        →  ONNX session.run()
           │      slice logits            →  remove padding
           │      returns ModelOutput(logits=...)
           │
           └─ logits[:, -1, :] → next token → append → repeat
```

The flow is straightforward: HF runs the encoder once, seeds the decoder, then loops calling our `forward()` which pads inputs to static ONNX shape, runs the session, and slices the padding off the logits.

---

## 4. Class Hierarchy

```
WinMLPreTrainedModel (base.py)
│   Session creation, _format_inputs, _run_inference, io_config
│
├── OnnxEncoder
│   Split encoder — inherits session + inference from base
│
└── WinMLGenerationMixin (+ GenerationMixin)
    │   forward(), _prepare_onnx_inputs(),
    │   get_encoder(), generation_config, can_generate()
    │
    └── WinMLModelForImageToText
            main_input_name = "pixel_values"
            _resolve_inputs() → OnnxGenerativeInputs
            generation_config override (nested token IDs)
            encoder = DummyEncoder() or OnnxEncoder(encoder_path)
```

### 4.1 `OnnxGenerativeInputs` (frozen dataclass)

Typed mapping from semantic roles to ONNX input names. Resolved once at model init from `io_config`.

```python
@dataclass(frozen=True)
class OnnxGenerativeInputs:
    decoder_input_ids: str              # e.g. "input_ids"
    attention_mask: str | None          # e.g. "attention_mask"
    encoder_hidden_states: str | None   # split only
    encoder_input: str | None           # monolithic only (e.g. "pixel_values")
```

Eliminates scattered string literals — `_prepare_onnx_inputs()` uses `mapping.decoder_input_ids` instead of hardcoded `"input_ids"`.

### 4.2 `DummyEncoder`

For monolithic ONNX models. Returns input unchanged as `BaseModelOutput(last_hidden_state=pixel_values)`. No ONNX session. Has `main_input_name` because HF reads `self.encoder.main_input_name`.

### 4.3 `OnnxEncoder(WinMLPreTrainedModel)`

For split ONNX models. Inherits session creation and `_run_inference()` from base. Runs the encoder ONNX file and returns `BaseModelOutput`.

### 4.4 Encoder `forward()` signature

Both encoder classes name parameters explicitly — HF inspects `encoder.forward()` via `inspect.signature()` to decide which kwargs to forward. Only named parameters are passed; `**kwargs` would cause HF to pass irrelevant decoder kwargs:

```python
def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
            output_attentions=None, output_hidden_states=None, return_dict=None):
```

### 4.5 `WinMLGenerationMixin(WinMLPreTrainedModel, GenerationMixin)`

The generation bridge. Overrides only public HF contract methods:

| Method / Property | Purpose |
|---|---|
| `forward()` | One decode step: pad → run ONNX → slice → return logits |
| `_prepare_onnx_inputs()` | Build ONNX numpy feed dict from HF torch args |
| `get_encoder()` | Return `self.encoder` |
| `can_generate()` | Return `True` |
| `device` | `torch.device("cpu")` — HF needs `torch.device`, base returns `str` |
| `generation_config` | `GenerationConfig.from_model_config(self.config)` + `use_cache=False` |
| `_static_seq_len` | Cached decoder sequence length from `io_config` |

**`forward()` accepts all kwargs HF sends** — even unused ones like `past_key_values`, `cache_position`, `use_cache`, `return_dict` — or HF's validation rejects them. Return type is `ModelOutput` (not `SimpleNamespace`) because HF does `if "past_key_values" in outputs` which requires `__contains__`.

### 4.6 `WinMLModelForImageToText(WinMLGenerationMixin)`

Task-specific subclass. Responsibilities:

| What | How |
|---|---|
| Main input | `main_input_name = "pixel_values"` |
| Encoder-decoder flag | `config.is_encoder_decoder = True` |
| ONNX input mapping | `_resolve_inputs()` — name-based lookup in `io_config` |
| Encoder selection | `OnnxEncoder` if `encoder_path`, else `DummyEncoder` |
| Token ID resolution | Override `generation_config` for nested sub-configs |

---

## 5. Static Shape Handling

ONNX models exported for NPU use fixed sequence lengths (e.g. `[1, 512]`). HF grows `decoder_input_ids` from 1 token to `max_new_tokens` during generation. `_prepare_onnx_inputs()` bridges this gap:

```
Step 1:  decoder_input_ids = [30522]                    (1 token)
         → pad to [30522, 0, 0, ..., 0]                (512 tokens)
         → mask = [1, 0, 0, ..., 0]
         → run ONNX → logits [1, 512, vocab]
         → slice to logits[:, :1, :]

Step 5:  decoder_input_ids = [30522, 10, 20, 30, 40]   (5 tokens)
         → pad to [30522, 10, 20, 30, 40, 0, ..., 0]  (512 tokens)
         → mask = [1, 1, 1, 1, 1, 0, ..., 0]
         → run ONNX → logits [1, 512, vocab]
         → slice to logits[:, :5, :]
         → HF reads logits[:, -1, :] for next token
```

---

## 6. Token ID Resolution

HF models store token IDs inconsistently:

| Model | `bos_token_id` location |
|---|---|
| GPT-2 | `config.bos_token_id` (top level) |
| BLIP | `config.text_config.bos_token_id` (nested) |
| TrOCR | `config.decoder.decoder_start_token_id` (nested) |

`GenerationConfig.from_model_config()` only reads top-level attributes. The base class (`WinMLGenerationMixin`) calls it directly. The subclass (`WinMLModelForImageToText`) overrides `generation_config` to fill in missing token IDs from `config.text_config` and `config.decoder`.

---

## 7. Future Extensibility

### 7.1 Split encoder/decoder export

The generation loop already supports split architecture — `OnnxEncoder` runs a separate encoder session and `WinMLModelForImageToText` selects it when `encoder_path` is provided. What's missing is the **export pipeline** that produces the split ONNX files.

Today, monolithic export fuses the encoder and decoder into a single ONNX graph. To support split export, the build pipeline needs to:

1. Export the encoder as `encoder_model.onnx` (inputs: `pixel_values`, outputs: `last_hidden_state`)
2. Export the decoder as `decoder_model.onnx` (inputs: `decoder_input_ids`, `encoder_hidden_states`, `attention_mask`, outputs: `logits`)
3. Optionally export `decoder_with_past_model.onnx` for KV cache (see below)

Optimum's `main_export()` already does this for supported models. The integration point is wiring our build pipeline to call it and store both files.

### 7.2 KV Cache support

Without KV cache, the decoder re-computes attention over all previous tokens at every generation step. For a 20-token generation with sequence length 512, that's 20 full forward passes over the growing sequence. With KV cache, only the **last token** is processed at each step — the attention keys/values from previous tokens are cached and reused.

#### What changes in the ONNX model

A KV-cache-enabled decoder has additional inputs and outputs:

```
Without KV cache (current):
  inputs:  [decoder_input_ids, attention_mask, encoder_hidden_states]
  outputs: [logits]

With KV cache:
  inputs:  [decoder_input_ids, attention_mask, encoder_hidden_states,
            past_key_values.0.decoder.key, past_key_values.0.decoder.value,
            past_key_values.0.encoder.key, past_key_values.0.encoder.value,
            past_key_values.1.decoder.key, ...]   # one pair per layer
  outputs: [logits,
            present.0.decoder.key, present.0.decoder.value,
            present.0.encoder.key, present.0.encoder.value,
            present.1.decoder.key, ...]            # updated cache
```

The `present.*` outputs from step N become the `past_key_values.*` inputs for step N+1. On the first step, `past_key_values` inputs are empty (zero-filled or omitted).

#### What changes in the code

**`_prepare_onnx_inputs()`** — needs to include KV cache tensors in the feed dict. On the first step, pass zeros. On subsequent steps, pass the `present.*` outputs from the previous step.

**`forward()`** — needs to:
1. Accept `past_key_values` from HF (currently accepted but ignored)
2. Feed them to the ONNX session via `_prepare_onnx_inputs()`
3. Capture `present.*` from the ONNX output
4. Return them as `past_key_values` in `ModelOutput` so HF carries them to the next step

**`prepare_inputs_for_generation()`** — needs an override (currently using HF default). When `past_key_values` is not None, slice `decoder_input_ids` to just the last token instead of the full sequence, since the cache already contains the attention state for previous tokens.

**`OnnxGenerativeInputs`** — may need additional fields to map the KV cache input/output names, or the cache tensor names can be discovered from `io_config` at init time.

**Session selection** — some exports produce two decoder files: `decoder_model.onnx` (first step, no cache inputs) and `decoder_with_past_model.onnx` (subsequent steps, with cache inputs). The mixin would select the right session based on whether `past_key_values` is present.

#### What does NOT change

The generation loop (`generate()`), encoder handling, `DummyEncoder`/`OnnxEncoder`, token ID resolution, and the HF integration contract all stay the same. KV cache is purely additive to the decoder-side logic.

### 7.3 New tasks

Adding a new generative task (e.g. `WinMLModelForCausalLM` for decoder-only text generation) follows the same pattern:

1. Subclass `WinMLGenerationMixin`
2. Set `main_input_name = "input_ids"` and `config.is_encoder_decoder = False`
3. Implement `_resolve_inputs()` for the task's ONNX I/O pattern
4. Override `generation_config` if the model stores token IDs in non-standard locations

No changes to `WinMLGenerationMixin` are required.
