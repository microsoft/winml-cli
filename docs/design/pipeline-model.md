# WinMLPipelineModel: Multi-Component Model Base

## Problem

`WinMLAutoModel.from_pretrained` builds ONE ONNX model and returns a `WinMLPreTrainedModel(onnx_path, config, device)`. Multi-component architectures (T5 encoder+decoder, SD text_encoder+unet+vae) don't fit this contract.

## Design

New base class `WinMLPipelineModel` for multi-component models. Each component is a `WinMLAutoModel` instance built independently through the existing pipeline.

```python
class WinMLPipelineModel:
    """Base for multi-component models (seq2seq, stable diffusion, etc.)."""

    COMPONENTS: dict[str, str] = {}  # name → task for WinMLAutoModel

    def __init__(self, components: dict[str, WinMLAutoModel], config):
        self._components = components
        self.config = config

    @classmethod
    def from_pretrained(cls, model_id, *, device="cpu", **kwargs):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        components = {
            name: WinMLAutoModel.from_pretrained(model_id, task=task, device=device, **kwargs)
            for name, task in cls.COMPONENTS.items()
        }
        return cls(components=components, config=config)
```

Subclasses declare components and add task-specific logic:

```python
class WinMLModelForSeq2SeqLM(WinMLPipelineModel, GenerationMixin):
    COMPONENTS = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    def __init__(self, components, config):
        super().__init__(components, config)
        self._encoder = components["encoder"]
        self._decoder = components["decoder"]
        # read shapes from ONNX I/O, set up generation_config, etc.
```

## Why not extend WinMLAutoModel

| Concern | WinMLAutoModel approach | WinMLPipelineModel approach |
|---|---|---|
| `auto.py` contract | Must hack `winml_class(onnx_path, config, device)` to handle multiple paths | Unchanged — each component built independently |
| Generalization | Special-casing per architecture | Declarative `COMPONENTS` dict |
| SD support | Another special case | Same pattern: `{"text_encoder": "...", "unet": "...", "vae_decoder": "..."}` |
| Entry point | Confusing — `WinMLAutoModel` returning a multi-model pipeline | Clear — `WinMLModelForSeq2SeqLM.from_pretrained(...)` |

## Registry

Separate from single-model `TASK_TO_WINML_CLASS`:

```python
PIPELINE_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("t5", "translation"): WinMLModelForSeq2SeqLM,
}
```
