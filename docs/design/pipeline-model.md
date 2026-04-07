# WinMLPipelineModel: Multi-Component Model Design

## Problem

`WinMLAutoModel.from_pretrained` builds ONE ONNX model. Multi-component
architectures (T5 encoder+decoder, SD text_encoder+unet+vae) need multiple
ONNX models composed together.

## Class Hierarchy

```
WinMLPipelineModel(PreTrainedModel)     — multi-component base
  └─ WinMLEncoderDecoderModel(GenerationMixin) — encoder-decoder with StaticCache
       └─ WinMLT5Model                     — T5 sub-model tasks + generation config
```

- **WinMLPipelineModel**: `_SUB_MODEL_CONFIG` mapping, `from_pretrained` builds
  each component via `WinMLAutoModel`, provides `device`/`to`/`dtype`.
- **WinMLEncoderDecoderModel**: `forward()` with StaticCache KV management,
  `_EncoderWithInputPadding` wrapper, `get_encoder()`, `prepare_inputs_for_generation()`.
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
print(pipe("Hello, how are you?", num_beams=1)[0]["translation_text"])
# Bonjour, comment êtes-vous ?
```

`from_pretrained` builds both ONNX sub-models via `WinMLAutoModel`, wraps
them in `WinMLT5Model`, which plugs into HF's `transformers.pipeline` as a
drop-in replacement for `T5ForConditionalGeneration`.
