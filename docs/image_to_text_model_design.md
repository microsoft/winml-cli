# WinMLModelForImageToText — Design Specification

Inference wrapper for image-to-text ONNX models. Takes an image, produces human-readable text.

---

## 1. Overview

### Problem

We have ONNX models exported from various HuggingFace image-to-text architectures (BLIP, TrOCR, Donut, ViT-GPT2, Nougat, manga-ocr, pix2text-mfr, mgp-str). We need a single model class that:

1. Wraps any image-to-text ONNX model for inference via ONNX Runtime
2. Plugs into HF's `ImageToTextPipeline` — preprocessing (image normalization, resizing) and postprocessing (token decoding to text) are the pipeline's responsibility, not the model's
3. Generates text autoregressively — one token at a time — by leveraging HF's `GenerationMixin`

### End-to-end flow

```
User: pipeline("image-to-text", model=winml_model)(image_url)
  → Pipeline.preprocess:    image → pixel_values tensor
  → Pipeline._forward:      pixel_values → model.generate() → token IDs
  → Pipeline.postprocess:   token IDs → tokenizer.decode() → "two cats on a couch"
Output: [{"generated_text": "two cats on a couch"}]
```

### Scope

The model class is responsible for:
- Running ONNX inference (forward pass)
- Providing the GenerationMixin hooks so the autoregressive decode loop works

The model class is NOT responsible for:
- Image loading, resizing, normalization → `ImageProcessor` (pipeline.preprocess)
- Token decoding, special token removal → `Tokenizer` (pipeline.postprocess)
- Dataset iteration, metric computation → Evaluator (separate component)

---

## 2. Model Class Interface

### Public methods used by the pipeline

```python
class WinMLModelForImageToText(WinMLPreTrainedModel, GenerationMixin):
    main_input_name = "pixel_values"

    def forward(
        self,
        pixel_values: Tensor,              # [B, C, H, W] float32
        input_ids: Tensor | None,          # [B, seq_len] int64
        attention_mask: Tensor | None,     # [B, seq_len] int64
        **kwargs,
    ) -> ImageToTextModelOutput:
        """Single ONNX inference call. Returns logits [B, seq_len, vocab_size].
        Called once per generated token by the decode loop inside generate()."""

    def generate(
        self,
        pixel_values: Tensor,              # from pipeline
        **generate_kwargs,                 # max_new_tokens, num_beams, etc.
    ) -> LongTensor:
        """Autoregressive text generation. Returns token IDs [B, output_len].
        Inherited from GenerationMixin. Not overridden.
        Called once by the pipeline. Runs the full decode loop internally."""

    def can_generate(self) -> bool:
        """Returns True. Tells pipeline to call generate() instead of forward()."""
```

### Inherited from WinMLPreTrainedModel (base class)

| Method / Property | Description |
|---|---|
| `__init__(onnx_path, config, device)` | Creates `WinMLSession` for ORT operations |
| `_format_inputs(**kwargs)` | Normalizes inputs to `dict[str, np.ndarray]` |
| `_run_inference(inputs)` | Delegates to `WinMLSession.run()`, returns `dict[str, torch.Tensor]` |
| `__call__(**kwargs)` | Delegates to `forward()` |
| `io_config` | ONNX metadata: input/output names, shapes, types |
| `config` | HF `PretrainedConfig` for token IDs, model_type, etc. |
| `device` | Device string from construction (e.g., `"auto"`, `"npu"`, `"cpu"`) |
| `to()` | No-op (ORT handles device via EP policy) |

### Pipeline execution flow

```
pipeline.__call__(image)
│
├── preprocess(image)                              ← Pipeline responsibility
│   ├── load_image(image)                          # URL/path/PIL → PIL.Image
│   ├── image_processor(image)                     # resize, normalize → pixel_values
│   └── return {"pixel_values": tensor [1,3,H,W]}
│
├── _forward(model_inputs, **generate_kwargs)      ← Calls our model
│   ├── inputs = model_inputs.pop("pixel_values")
│   ├── model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
│   └── return model_outputs                       # token IDs [B, seq_len]
│
└── postprocess(model_outputs)                     ← Pipeline responsibility
    ├── for output_ids in model_outputs:
    │   └── text = tokenizer.decode(output_ids, skip_special_tokens=True)
    └── return [{"generated_text": text}]
```

### image-to-text vs image-text-to-text

Both tasks use the same model class and same ONNX model. The only difference:

| | image-to-text | image-text-to-text |
|---|---|---|
| Pipeline sends | `pixel_values` only | `pixel_values` + `input_ids` (tokenized prompt) |
| Decoder starts from | `[BOS]` (seeded by model) | Prompt tokens (from pipeline) |
| Example | Image → "two cats on a couch" | Image + "Describe:" → "Describe: two cats..." |

The model handles both: if `input_ids` is provided, use it; if not, seed with `[decoder_start_token_id]`.

---

## 3. Generation: Decoding Tokens

### ONNX vs HF PyTorch: Key Differences

ONNX export normalizes all HF architectures into a uniform I/O. The differences between ONNX and HF PyTorch drive every design decision:

| # | Aspect | HF PyTorch | Our ONNX | Design Impact |
|---|---|---|---|---|
| 1 | Inference engine | PyTorch layer computation | ORT `session.run()` on numpy arrays | `forward()` calls `_format_inputs()` + `_run_inference()` |
| 2 | Encoder/decoder | Separate callables (`self.encoder`, `self.decoder`) | Single fused graph (monolithic ONNX) | Must keep `pixel_values` alive in decode loop via `_prepare_model_inputs()` override |
| 3 | KV cache | `forward()` accepts/returns `past_key_values` | No cache I/O in ONNX graph | `_supports_cache_class=False`; output `past_key_values=None` |
| 4 | Input shapes | Dynamic (any seq_len) | May be static (fixed, e.g. 512) | Pad inputs + slice logits in `forward()` |
| 5 | Framework detection | `isinstance(model, PreTrainedModel)` → "pt" | Not an `nn.Module` | Name shim in base class |
| 6 | Generation config | Loaded from Hub (`generation_config.json`) | Not auto-loaded | Lazy property reads tokens from nested config |
| 7 | `device` property | Returns `torch.device` | Base returns string `"auto"`, `"npu"` | Override to return `torch.device("cpu")` (see note) |

**Note on `device` (row 7):** GenerationMixin uses `self.device` to create tensors: `torch.tensor(token, device=self.device)` and `torch.ones(..., device=self.device)`. These calls require a valid `torch.device`, not a string like `"npu"` or `"auto"`. We override `device` to return `torch.device("cpu")` because all tensor operations in the generation loop (token selection, concatenation) happen on CPU — the ONNX Runtime handles actual device placement via its EP policy. This override does not affect which device the ONNX model runs on.

### Why ONNX normalizes different HF model classes

In HuggingFace, each architecture has a different class:

| HF Class | Internal Structure |
|---|---|
| `VisionEncoderDecoderModel` | `self.encoder` + `self.decoder` with cross-attention |
| `BlipForConditionalGeneration` | `self.vision_model` + `self.text_decoder` |
| `MgpstrForSceneTextRecognition` | `self.mgp_str` + `self.char_head` (no decoder loop) |

When exported to ONNX by ModelKit's `HTPExporter`, all encoder-decoder architectures trace `forward()` into a single graph. The internal wiring (cross-attention, encoder type, decoder type) is baked into the graph. From the outside, they all have the same signature:

```
Inputs:  pixel_values [B, C, H, W], input_ids [B, seq_len], attention_mask [B, seq_len]
Outputs: logits [B, seq_len, vocab_size]
```

This normalization means one `WinMLModelForImageToText` class handles all architectures without model-specific code. The model class reads ONNX metadata (`io_config`) at runtime to adapt (e.g., static vs dynamic shapes).

### GenerationMixin method overrides

All generation logic (greedy, beam search, sampling) is provided by `GenerationMixin`. We do NOT override `generate()`. We only override the hooks that GenerationMixin calls:

| Override | Original (GenerationMixin) | Our Override | Why |
|---|---|---|---|
| `forward()` | PyTorch layer computation | ORT `session.run()` with static shape padding/slicing | ONNX inference instead of PyTorch; handle fixed seq_len |
| `_prepare_model_inputs()` | Pops `main_input_name` from kwargs, never returns it | Re-injects `pixel_values` into `model_kwargs`; seeds `input_ids=[BOS]` | Monolithic ONNX needs `pixel_values` at every step (no separate encoder); pipeline doesn't send `input_ids` |
| `prepare_inputs_for_generation()` | Slices `input_ids` to last token when KV cache exists; passes `past_key_values` | Passes full `pixel_values` + full `input_ids` (no slicing) | No KV cache in monolithic ONNX; no separate encoder outputs to pass |
| `can_generate()` | Returns `False` by default | Returns `True` | Tells pipeline to call `generate()` |


### Properties and class attributes

| Member | Value | Purpose |
|---|---|---|
| `main_input_name` | `"pixel_values"` | Tells pipeline which kwarg is the image |
| `_is_stateful` | `False` | Tells GenerationMixin this model has no mutable state |
| `_supports_cache_class` | `False` | Tells GenerationMixin not to pass `Cache` objects |
| `device` (property) | `torch.device("cpu")` | GenerationMixin uses this for tensor creation |
| `generation_config` (property) | Lazy `GenerationConfig` | Reads `decoder_start_token_id`, `bos_token_id`, `eos_token_id`, `pad_token_id` from config; defaults `max_new_tokens=20` |

### The greedy decode loop (default)

GenerationMixin selects decoding strategy based on `GenerationConfig`:
- `do_sample=False, num_beams=1` → greedy (default)
- `do_sample=True, num_beams=1` → sampling
- `num_beams > 1` → beam search

All strategies are provided by GenerationMixin. The greedy loop:

```
generate(pixel_values) start:
  → _prepare_model_inputs()            # seed input_ids=[BOS], keep pixel_values
  → _sample() loop:
      Step 0:
        prepare_inputs_for_generation(input_ids=[BOS], pixel_values=img)
        → forward(pixel_values=img, input_ids=[BOS])
        → logits[:, -1, :] → argmax → token_1

      Step 1:
        prepare_inputs_for_generation(input_ids=[BOS, t1], pixel_values=img)
        → forward(pixel_values=img, input_ids=[BOS, t1])
        → logits[:, -1, :] → argmax → token_2

      ...until EOS or max_new_tokens...
  → return all token IDs
```

---

## 4. KV Cache

### Why KV cache is not supported

KV cache stores computed key/value pairs so subsequent decode steps only compute K,V for the new token (linear cost instead of quadratic). Our monolithic ONNX models don't support it because:

1. **The ONNX graph was traced without cache I/O.** `HTPExporter` traces `forward(pixel_values, input_ids, attention_mask) → logits`. There are no `past_key_values` inputs or outputs in the graph.

2. **The encoder and decoder are fused.** Even if the decoder had KV cache, the encoder would re-run every step since `pixel_values` is a graph input. Caching just the decoder would require a separate encoder ONNX file.

3. **Cost.** For N generated tokens, the monolithic approach runs the full graph N times (vs. 1 encoder call + N incremental decoder calls with cache). For short generations (5–20 tokens) the cost is acceptable.

### What would be needed for KV cache support

| Component | Current | Required for KV Cache |
|---|---|---|
| ONNX export | Monolithic: 1 file, `pixel_values + input_ids → logits` | Split: `encoder.onnx` (pixel_values → features) + `decoder.onnx` (features + input_ids + past_kv → logits + new_kv) |
| `forward()` | Runs single ONNX session | Two sessions: encoder (once) + decoder (per token); pass/receive `past_key_values` tensors |
| `prepare_inputs_for_generation()` | Passes full `pixel_values` + full `input_ids` | Passes `encoder_outputs` (cached) + last token only + `past_key_values` |
| `_prepare_model_inputs()` | Re-injects `pixel_values` | Runs encoder once, stores `encoder_outputs` in `model_kwargs`; no need to re-inject `pixel_values` |
| `ImageToTextModelOutput` | `past_key_values=None` | Contains actual KV cache tensors returned from decoder ONNX |

---

## 5. forward() — ONNX Inference

### Static shape handling

Some ONNX models have fixed `input_ids` dimensions (e.g., BLIP: always 512). During generation, `input_ids` grows from 1 to N tokens — always less than the static size. The solution:

1. **Pad** `input_ids` to the static size with zeros
2. **Create** `attention_mask` with 1s for real tokens, 0s for padding
3. **Run** ONNX inference
4. **Slice** logits back to the real sequence length

Without the slice, GenerationMixin reads `logits[:, -1, :]` at the wrong position (position 511 instead of position 2 at step 2).

Dynamic-shape models (where `_get_expected_seq_len()` returns `None`) skip padding entirely.

### Output compatibility

`ImageToTextModelOutput` implements `__contains__` and `__getitem__` because `GenerationMixin._update_model_kwargs_for_generation()` does:
```python
if "past_key_values" in outputs:       # calls __contains__
    model_kwargs["past_key_values"] = outputs["past_key_values"]
```

Our `__contains__` returns `False` for `past_key_values` (it's `None`), so the cache logic is cleanly skipped.

---

## 6. Class Diagram

```
WinMLPreTrainedModel (base.py)
│   • __init__(onnx_path, config, device)
│   • _format_inputs() / _run_inference()
│   • io_config, config, device, dtype
│
└── WinMLModelForImageToText (image_to_text.py)
    │   Inherits: WinMLPreTrainedModel + GenerationMixin
    │
    │   Overrides:
    │   • forward()
    │   • can_generate()
    │   • _prepare_model_inputs()
    │   • prepare_inputs_for_generation()
    │
    │   Properties:
    │   • device → torch.device("cpu")
    │   • generation_config → lazy GenerationConfig
    │
    │   Helpers:
    │   • _get_expected_seq_len()
    │   • _resolve_config_attr()
    │
    │   Inherited from GenerationMixin (NOT overridden):
    │   • generate()
    │   • _sample() / _beam_search() / _assisted_decoding()
    │
    └── ImageToTextModelOutput (dataclass)
        • logits: Tensor
        • past_key_values: None
        • __contains__() / __getitem__()
```

---

## 7. Task Registration

```python
TASK_TO_WINML_CLASS = {
    "image-to-text": "WinMLModelForImageToText",
    "image-text-to-text": "WinMLModelForImageToText",
}
```

---

## 8. Supported Models

| Model | model_type | ONNX Inputs | Generative? |
|---|---|---|---|
| `microsoft/trocr-*` (4 variants) | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `naver-clova-ix/donut-base` | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `nlpconnect/vit-gpt2-image-captioning` | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `kha-white/manga-ocr-base` | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `facebook/nougat-base` | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `breezedeus/pix2text-mfr` | `vision-encoder-decoder` | pixel_values + input_ids + attention_mask | Yes |
| `Salesforce/blip-image-captioning-base` | `blip` | pixel_values + input_ids + attention_mask | Yes |
| `alibaba-damo/mgp-str-base` | `mgp-str` | pixel_values only | No (single-pass) |

For `mgp-str` (the only non-generative model), a `generate()` override detects the absence of `input_ids` in the ONNX model's inputs and short-circuits to a single forward pass + argmax. This is detected from ONNX metadata — no model-type hardcoding.

---

## 9. Construction Path

```
WinMLAutoModel.from_pretrained("Salesforce/blip-image-captioning-base", task="image-to-text")
│
├── Load config:     PretrainedConfig.from_pretrained(model_id)
├── Build pipeline:  export → optimize → [quantize] → [compile] → model.onnx
├── Select class:    get_winml_class("blip", "image-to-text") → WinMLModelForImageToText
└── Construct:       WinMLModelForImageToText(onnx_path="model.onnx", config=config, device="auto")
```

Usage with HF pipeline:

```python
pipe = pipeline("image-to-text", model=model, tokenizer=tokenizer, image_processor=processor)
result = pipe("https://example.com/cats.jpg")
# → [{"generated_text": "two cats sleeping on a couch"}]
```
