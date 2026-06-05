# Python API

winml-cli can be used as a Python library for programmatic model building and
inference. This page documents the public API surface.

---

## Quick Example

```python
from winml.modelkit import WinMLAutoModel

# Build and load in one call
model = WinMLAutoModel.from_pretrained("microsoft/resnet-50", device="npu")
output = model(pixel_values=images)

# From a local ONNX file
model = WinMLAutoModel.from_onnx("model.onnx", task="image-classification")
```

---

## `WinMLAutoModel`

Factory class for automatic model building and loading. Not instantiable directly —
use the class methods.

### `from_pretrained()`

Build and load a model from a HuggingFace ID or local path. Runs the full
pipeline: config → export → optimize → quantize → compile → load.

```python
WinMLAutoModel.from_pretrained(
    model_id_or_path: str | Path,
    *,
    task: str | None = None,
    config: WinMLBuildConfig | None = None,
    device: str = "auto",
    precision: str = "auto",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    trust_remote_code: bool = False,
    shape_config: dict | None = None,
    no_compile: bool = False,
) -> WinMLPreTrainedModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id_or_path` | `str \| Path` | required | HuggingFace model ID or path to local model. |
| `task` | `str \| None` | `None` | Task name. Auto-detected if omitted. |
| `config` | `WinMLBuildConfig \| None` | `None` | Custom build config. Auto-generated if omitted. |
| `device` | `str` | `"auto"` | Target device: `"auto"`, `"npu"`, `"gpu"`, `"cpu"`. |
| `precision` | `str` | `"auto"` | Precision: `"auto"`, `"fp32"`, `"fp16"`, `"w8a8"`, etc. |
| `cache_dir` | `str \| Path \| None` | `None` | Cache directory for built artifacts. |
| `use_cache` | `bool` | `True` | Reuse cached build if available. |
| `force_rebuild` | `bool` | `False` | Force rebuild even if cache exists. |
| `trust_remote_code` | `bool` | `False` | Trust remote code from HuggingFace. |
| `no_compile` | `bool` | `False` | Skip the compilation stage. |

**Returns:** A task-specific `WinMLPreTrainedModel` subclass.

---

### `from_onnx()`

Build from a pre-exported ONNX file. Runs: optimize → quantize → compile → load.

```python
WinMLAutoModel.from_onnx(
    onnx_path: str | Path | dict[str, str | Path],
    *,
    task: str | None = None,
    config: WinMLBuildConfig | None = None,
    device: str = "auto",
    precision: str = "auto",
    ep: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    skip_build: bool = False,
    session_options: Any | None = None,
    hf_config: PretrainedConfig | None = None,
    **kwargs: Any,
) -> WinMLPreTrainedModel | WinMLCompositeModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `onnx_path` | `str \| Path \| dict` | required | ONNX file path, or dict of submodel paths for composite models. |
| `skip_build` | `bool` | `False` | Load ONNX directly without running optimize/quantize/compile. |
| `hf_config` | `PretrainedConfig \| None` | `None` | Required for composite models (dict inputs). |

---

### `supported_tasks()`

```python
WinMLAutoModel.supported_tasks() -> list[str]
```

Returns all task strings with dedicated inference classes (16 tasks).

---

## Build Pipeline Functions

Lower-level functions for fine-grained control over the pipeline.

### `build_hf_model()`

```python
from winml.modelkit.build import build_hf_model

result = build_hf_model(
    config: WinMLBuildConfig,
    output_dir: Path,
    *,
    model_id: str | None = None,
    pytorch_model: nn.Module | None = None,
    rebuild: bool = False,
    trust_remote_code: bool = False,
    random_init: bool = False,
    cache_key: str | None = None,
    ep: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> BuildResult
```

Runs the full pipeline (export → optimize → analyze → quantize → compile) and
writes all artifacts to `output_dir`.

### `build_onnx_model()`

```python
from winml.modelkit.build import build_onnx_model

result = build_onnx_model(
    onnx_path: Path | str,
    *,
    config: WinMLBuildConfig,
    output_dir: Path | str,
    rebuild: bool = False,
    ep: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> BuildResult
```

Builds from an existing ONNX file (skips export).

### `BuildResult`

```python
@dataclass
class BuildResult:
    output_dir: Path           # Directory containing all artifacts
    final_onnx_path: Path      # Path to final model.onnx
    config_path: Path          # Path to winml_build_config.json
    stages_completed: list[str]  # e.g., ["export", "optimize", "quantize"]
    stages_skipped: list[str]
    stage_timings: dict[str, float]  # Per-stage seconds
    elapsed: float             # Total build time (seconds)
    reused: bool               # True if cache hit, no build ran
    manifest_path: Path | None # Path to build_manifest.json
```

---

## Config Generation

### `generate_build_config()`

```python
from winml.modelkit.config import generate_build_config

config = generate_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: str | None = None,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
    onnx_path: str | Path | None = None,
) -> WinMLBuildConfig | list[WinMLBuildConfig]
```

Auto-generates a complete build config by probing the model's `config.json`
(does not download weights). Equivalent to what `winml config` produces.
Returns a list when `module` is specified (one config per submodule).

---

## Inference Model Classes

All inference models inherit from `WinMLPreTrainedModel` and are HuggingFace
pipeline-compatible.

### `WinMLPreTrainedModel` (Base)

```python
class WinMLPreTrainedModel:
    def __call__(self, **kwargs) -> Any: ...
    def perf(self, warmup: int = 0) -> ContextManager: ...

    @property
    def device(self) -> str: ...
    @property
    def ep_name(self) -> str | None: ...
    @property
    def io_config(self) -> dict: ...
    @property
    def task(self) -> str | None: ...
```

### Task-Specific Classes

| Class | Task |
|-------|------|
| `WinMLModelForImageClassification` | `image-classification` |
| `WinMLModelForSequenceClassification` | `text-classification` |
| `WinMLModelForImageSegmentation` | `image-segmentation` |
| `WinMLModelForSemanticSegmentation` | `semantic-segmentation` |
| `WinMLModelForObjectDetection` | `object-detection` |
| `WinMLModelForFeatureExtraction` | `feature-extraction` |
| `WinMLModelForQuestionAnswering` | `question-answering` |
| `WinMLModelForZeroShotImageClassification` | `zero-shot-image-classification` |
| `WinMLModelForGenericTask` | fallback (raw outputs) |

### Performance Tracking

```python
model = WinMLAutoModel.from_pretrained("microsoft/resnet-50", device="npu")

with model.perf(warmup=5) as stats:
    for img in test_images:
        model(pixel_values=img)

print(f"P99 latency: {stats.p99_ms:.2f} ms")
```

---

## See also

- [Reference — Build Configuration Schema](index.md) — full config field reference
- [winml build](../commands/build.md) — CLI equivalent
- [How winml-cli Works](../concepts/how-it-works.md) — pipeline overview
