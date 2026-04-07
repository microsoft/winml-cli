# Multi-Model Pipeline Design

## Problem

`WinMLAutoModel.from_pretrained` builds ONE ONNX model. Multi-component
architectures (T5 encoder+decoder, SD text_encoder+unet+vae) need multiple
ONNX models composed together.

## Class Hierarchy

```
WinMLPipelineModel(PreTrainedModel)            — multi-component base
  └─ WinMLEncoderDecoderModel(GenerationMixin) — encoder-decoder with StaticCache
       └─ WinMLT5Model                         — T5 tasks + generation config
```

- **WinMLPipelineModel**: `_SUB_MODEL_CONFIG` mapping, `from_pretrained` builds
  each component via `WinMLAutoModel`, provides `device`/`to`/`dtype`.
- **WinMLEncoderDecoderModel**: `forward()` with StaticCache KV management,
  `_EncoderWithInputPadding` wrapper, `get_encoder()`, `prepare_inputs_for_generation()`.
  Auto-pads undersized inputs to ONNX expected shapes via `_pad_inputs`.
- **WinMLT5Model**: declares `_SUB_MODEL_CONFIG` and `generation_config` only.

## Registry

`@register_pipeline_model(model_type, task)` registers a pipeline class.
`winml config` checks the registry to generate per-component configs.

```python
@register_pipeline_model("t5", "translation")
class WinMLT5Model(WinMLEncoderDecoderModel):
    _SUB_MODEL_CONFIG = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }
```

## ONNX Export

Each component is exported independently via the existing pipeline
(export → optimize → compile). Export wrappers in `models/hf/t5.py`:

| Component | Class | Description |
|---|---|---|
| Encoder | `T5EncoderWrapper` | `forward(input_ids, attention_mask) → encoder_hidden_states` |
| Decoder | `T5DecoderWrapper` | StaticCache + EncoderDecoderCache from flat KV inputs, extracts new token KV via `gather` |
| Decoder IO | `T5DecoderIOConfig` | OnnxConfig with custom DummyInputGenerators for KV cache tensors |

### Decoder ONNX I/O (all static shapes)

```
Inputs:
  decoder_input_ids        [1, 1]
  encoder_hidden_states    [1, enc_seq, d_model]
  attention_mask           [1, enc_seq]
  decoder_attention_mask   [1, max_decode]
  cache_position           [1]
  past_{i}_key             [1, heads, max_decode, d_kv]    # i=0..num_layers-1
  past_{i}_value           [1, heads, max_decode, d_kv]

Outputs:
  logits                   [1, 1, vocab_size]
  present_{i}_key          [1, heads, 1, d_kv]             # new token only
  present_{i}_value        [1, heads, 1, d_kv]
```

Cross-attention KV is always recomputed from `encoder_hidden_states`
(empty cross-attention cache → `is_updated=False` → never constant-folded).

## KV Cache Design

Uses HF `StaticCache` for both export and inference:

- **Export**: `StaticCache.update()` uses `index_copy_` which traces correctly
  in `torch.onnx.export`. `KV_index = sequence_position` always holds, so T5's
  relative position bias computes correct distances.
- **Inference**: Same `StaticCache` object persists across generation steps,
  mutated in-place via `cache.update()`. `get_seq_length()` counts non-zero
  positions automatically.
- **GenerationMixin integration**: `StaticCache` flows through the generate loop
  via `Seq2SeqLMOutput.past_key_values`. GenerationMixin may wrap it in an
  `EncoderDecoderCache`; `forward()` unwraps to find the `StaticCache`.

Known limitation: OpenVINO EP does not support ScatterElements, requires CPU
EP fallback for decoder inference.

## Usage

### 1. Generate configs (one per component)

```
winml config -m google-t5/t5-small --task translation --device cpu -o t5.json
```

Produces two files:
- `t5_encoder.json` — task `feature-extraction`
- `t5_decoder.json` — task `text2text-generation`

### 2. Build ONNX models independently

```
winml build -c t5_encoder.json -m google-t5/t5-small -o output/encoder
winml build -c t5_decoder.json -m google-t5/t5-small -o output/decoder
```

### 3. Run translation pipeline

```python
from winml.modelkit.models.winml.seq2seq import WinMLT5Model
from transformers import AutoTokenizer, pipeline

model = WinMLT5Model.from_pretrained("google-t5/t5-small")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
result = pipe("Hello, how are you?", num_beams=1)
print(result[0]["translation_text"])
# Bonjour, comment êtes-vous ?
```

`from_pretrained` builds both ONNX sub-models via `WinMLAutoModel`, wraps
them in `WinMLT5Model`, which plugs into HF `transformers.pipeline` as a
drop-in replacement for `T5ForConditionalGeneration`.

## Open Questions

- Manage KV cache and attention mask jointly in same cache class?
- Update KV in numpy to avoid pytorch tensor <-> numpy array round trip?
- Handle quantized cache (channel-wise quantization for accuracy)?
- EP-specific KV cache management to avoid ORT <-> EP round trip?
- Beam search support (requires dynamic batch)?
- Is it possible/better to use a shared model class for both export and inference?
