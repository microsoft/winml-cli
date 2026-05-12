# Inspect & Config Command Improvement Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all known bugs in `wmk inspect` and `wmk config`, remove MUST-rule violations from resolver.py, consolidate duplicated I/O extraction logic, and add missing features (`--list-tasks`, local ONNX support).

**Architecture:** The inspect command's resolver.py is refactored to reuse config's battle-tested `export/io.py:resolve_io_specs()` for I/O extraction (eliminating ~100 lines of duplicated code). The config command's default EP is changed from hardcoded `"qnn"` to hardware-detected. All 5 MUST-rule violations (D-1 through D-5) in resolver.py are eliminated by replacing hardcoded patterns with data-driven approaches.

**Tech Stack:** Python 3.10+, pytest, click, rich, transformers, optimum, huggingface_hub

**GitHub Issues Addressed:** #247 (MUST-rule violations), #412 (config device bug), #354 (partial — ONNX inspect groundwork)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `modelkit/inspect/resolver.py` | Remove `_extract_tensor_specs_from_onnx_config()`, reuse `resolve_io_specs()`. Fix D-1..D-5. Fix processor resolution. |
| Modify | `modelkit/inspect/types.py` | Add `value_range` field to `TensorInfo`. Extend `IOConfigInfo` for `hidden_sizes`. |
| Modify | `modelkit/inspect/__init__.py` | Pass `model_id` through to resolver for image size resolution. |
| Modify | `modelkit/inspect/formatter.py` | Display value_range. Handle `hidden_sizes` list. |
| Modify | `modelkit/commands/inspect.py` | Add `--list-tasks` flag. |
| Modify | `modelkit/compiler/configs.py` | Change `EPConfig.provider` default from `"qnn"` to `None`. |
| Modify | `modelkit/config/build.py` | Always call `resolve_device()` to populate compile config. |
| Modify | `modelkit/config/precision.py` | Handle `compile_provider=None` in no-op path. |
| Create | `tests/inspect/test_resolver.py` | Unit tests for all resolver functions. |
| Modify | `tests/commands/test_inspect_cli.py` | Add `--list-tasks` CLI test. |
| Modify | `tests/commands/test_config_cli.py` | Add device detection test. |

---

## Chunk 1: Fix I/O Extraction (Consolidation + Image Size Bug)

### Task 1: Add `value_range` to TensorInfo and extend IOConfigInfo

**Files:**
- Modify: `modelkit/inspect/types.py:18-26` (TensorInfo)
- Modify: `modelkit/inspect/types.py:92-108` (IOConfigInfo)
- Test: `tests/inspect/test_resolver.py` (new)

- [ ] **Step 1: Write failing test for TensorInfo.value_range**

```python
# tests/inspect/test_resolver.py
"""Tests for inspect resolver module."""
from modelkit.inspect.types import TensorInfo, IOConfigInfo


class TestTensorInfo:
    def test_value_range_field_exists(self):
        t = TensorInfo(name="pixel_values", dtype="float32", value_range=(0.0, 1.0))
        assert t.value_range == (0.0, 1.0)

    def test_value_range_default_none(self):
        t = TensorInfo(name="x")
        assert t.value_range is None


class TestIOConfigInfo:
    def test_hidden_sizes_field(self):
        io = IOConfigInfo(hidden_sizes=[256, 512, 1024, 2048])
        assert io.hidden_sizes == [256, 512, 1024, 2048]

    def test_hidden_sizes_default_none(self):
        io = IOConfigInfo()
        assert io.hidden_sizes is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/inspect/test_resolver.py::TestTensorInfo -v`
Expected: FAIL — `TensorInfo.__init__() got an unexpected keyword argument 'value_range'`

- [ ] **Step 3: Add value_range to TensorInfo and hidden_sizes to IOConfigInfo**

In `modelkit/inspect/types.py`, add to `TensorInfo`:
```python
@dataclass
class TensorInfo:
    """Information about a tensor."""
    name: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    shape_desc: str | None = None
    dynamic_axes: dict[int, str] | None = None
    value_range: tuple[float, float] | None = None  # ADD THIS
```

In `modelkit/inspect/types.py`, add to `IOConfigInfo`:
```python
@dataclass
class IOConfigInfo:
    # ... existing fields ...
    hidden_size: int | None = None
    hidden_sizes: list[int] | None = None  # ADD THIS (for ResNet-like models)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/inspect/test_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelkit/inspect/types.py tests/inspect/test_resolver.py
git commit -m "feat(inspect): add value_range to TensorInfo and hidden_sizes to IOConfigInfo"
```

### Task 2: Replace `_extract_tensor_specs_from_onnx_config` with `resolve_io_specs`

This is the core consolidation. Inspect's duplicated I/O extraction (~100 lines) is replaced by calling config's battle-tested `export/io.py:resolve_io_specs()`.

**Files:**
- Modify: `modelkit/inspect/resolver.py:175-277` (delete _extract_tensor_specs_from_onnx_config)
- Modify: `modelkit/inspect/resolver.py:280-384` (update resolve_exporter)
- Modify: `modelkit/inspect/__init__.py:67-195` (pass model_id)
- Test: `tests/inspect/test_resolver.py`

- [ ] **Step 1: Write failing test for resolve_exporter with correct image size**

```python
# Add to tests/inspect/test_resolver.py
from unittest.mock import patch, MagicMock
from modelkit.inspect.resolver import resolve_exporter


class TestResolveExporter:
    def test_resnet_gets_224_not_64(self):
        """Verify ResNet inspect shows 224x224, not Optimum's 64x64 fallback."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained("microsoft/resnet-50")
        info = resolve_exporter(
            "resnet", "image-classification",
            hf_config=hf_config,
            model_id="microsoft/resnet-50",
        )
        # Should have pixel_values input
        assert len(info.input_tensors) > 0
        pv = info.input_tensors[0]
        assert pv.name == "pixel_values"
        # Shape should contain 224, NOT 64
        assert "224" in (pv.shape_desc or ""), (
            f"Expected 224 in shape_desc, got {pv.shape_desc}"
        )

    def test_resnet_input_has_value_range(self):
        """Verify value_range is captured for vision models."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained("microsoft/resnet-50")
        info = resolve_exporter(
            "resnet", "image-classification",
            hf_config=hf_config,
            model_id="microsoft/resnet-50",
        )
        pv = info.input_tensors[0]
        assert pv.value_range is not None, "value_range should be captured"

    def test_output_tensors_have_dtype(self):
        """Verify output tensors get dtype from dummy forward pass."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained("microsoft/resnet-50")
        info = resolve_exporter(
            "resnet", "image-classification",
            hf_config=hf_config,
            model_id="microsoft/resnet-50",
        )
        # logits output should exist
        assert len(info.output_tensors) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/inspect/test_resolver.py::TestResolveExporter::test_resnet_gets_224_not_64 -v`
Expected: FAIL — `resolve_exporter() got an unexpected keyword argument 'model_id'` (old signature)

- [ ] **Step 3: Rewrite resolve_exporter to use resolve_io_specs**

In `modelkit/inspect/resolver.py`:

1. **Delete** the entire `_extract_tensor_specs_from_onnx_config()` function (lines 175-277).

2. **Add** a new internal helper `_build_tensor_infos_from_io_specs()`:

```python
def _build_tensor_infos_from_io_specs(
    io_specs: dict,
) -> tuple[list[TensorInfo], list[TensorInfo]]:
    """Convert resolve_io_specs() output to TensorInfo lists.

    Args:
        io_specs: Dict from export/io.py resolve_io_specs()

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    input_tensors: list[TensorInfo] = []
    output_tensors: list[TensorInfo] = []

    input_names = io_specs.get("input_names", [])
    input_shapes = io_specs.get("input_shapes", [])
    input_dtypes = io_specs.get("input_dtypes", [])
    inputs_axes = io_specs.get("inputs", {})
    value_ranges = io_specs.get("value_ranges", {})

    for i, name in enumerate(input_names):
        shape = input_shapes[i] if i < len(input_shapes) else None
        dtype = input_dtypes[i] if i < len(input_dtypes) else None
        axes = inputs_axes.get(name, {})
        vr = value_ranges.get(name)

        shape_desc = _shape_to_desc(shape, axes) if shape else None

        input_tensors.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                shape_desc=shape_desc,
                dynamic_axes=dict(axes) if axes else None,
                value_range=vr,
            )
        )

    output_names = io_specs.get("output_names", [])
    outputs_axes = io_specs.get("outputs", {})

    for name in output_names:
        axes = outputs_axes.get(name, {})
        shape_desc = _shape_to_desc(None, axes) if axes else None
        output_tensors.append(
            TensorInfo(
                name=name,
                shape_desc=shape_desc,
                dynamic_axes=dict(axes) if axes else None,
            )
        )

    return input_tensors, output_tensors


def _shape_to_desc(
    shape: tuple | list | None, dynamic_axes: dict[int, str]
) -> str:
    """Convert tensor shape to human-readable string with dynamic markers.

    Uses dynamic_axes values directly as labels (no hardcoded abbreviations).
    Falls back to dimension index labels when no axis names available.
    """
    if shape is None:
        parts = []
        for _idx, axis_name in sorted(dynamic_axes.items()):
            parts.append(axis_name)
        return f"[{', '.join(parts)}]" if parts else "[]"

    parts = []
    for i, dim in enumerate(shape):
        if i in dynamic_axes:
            axis_name = dynamic_axes[i]
            # Use the axis name for truly dynamic dims (batch),
            # use actual value for spatial dims
            if "batch" in axis_name.lower():
                parts.append("B")
            else:
                parts.append(str(dim))
        else:
            parts.append(str(dim))
    return f"[{', '.join(parts)}]"
```

3. **Update** `resolve_exporter()` signature and TasksManager path:

```python
def resolve_exporter(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig | None = None,
    *,
    model_id: str | None = None,
) -> ExporterInfo:
    """Resolve exporter configuration for a model.

    Uses MODEL_BUILD_CONFIGS registry, then falls back to
    export/io.py resolve_io_specs() for I/O extraction.
    """
    model_type_normalized = model_type.lower().replace("_", "-")

    # Check MODEL_BUILD_CONFIGS for predefined config (unchanged)
    if model_type_normalized in MODEL_BUILD_CONFIGS:
        # ... existing MODEL_BUILD_CONFIGS path (unchanged) ...

    # Check if TasksManager supports this model_type
    try:
        import optimum.exporters.onnx.model_configs  # noqa: F401
        from optimum.exporters.tasks import TasksManager

        onnx_config_cls = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type=model_type,
            task=task,
            library_name="transformers",
        )
        if onnx_config_cls:
            import functools
            config_name = (
                onnx_config_cls.func.__name__
                if isinstance(onnx_config_cls, functools.partial)
                else onnx_config_cls.__name__
            )

            # NEW: Use resolve_io_specs instead of _extract_tensor_specs_from_onnx_config
            input_tensors: list[TensorInfo] = []
            output_tensors: list[TensorInfo] = []

            if hf_config is not None:
                try:
                    from ..export.io import resolve_io_specs

                    io_specs = resolve_io_specs(
                        model_type=model_type,
                        task=task,
                        hf_config=hf_config,
                        model_id=model_id,
                    )
                    input_tensors, output_tensors = _build_tensor_infos_from_io_specs(
                        io_specs
                    )
                except Exception as e:
                    logger.debug("resolve_io_specs failed: %s", e)

            return ExporterInfo(
                onnx_config_class=config_name,
                onnx_config_source="TasksManager",
                support_level=SupportLevel.DEFAULT,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                opset_version=17,
            )
    except Exception as e:
        logger.debug("TasksManager lookup failed for %s/%s: %s", model_type, task, e)

    # Unsupported (unchanged)
    return ExporterInfo(...)
```

4. **Update** `inspect_model()` in `__init__.py` to pass `model_id`:

```python
# Step 4: Resolve exporter configuration (pass model_id for image size)
exporter_info = resolve_exporter(model_type, task, hf_config=hf_config, model_id=model_id)
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/inspect/test_resolver.py -v`
Expected: ALL PASS — ResNet shows 224, value_range captured

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `uv run pytest tests/inspect/ tests/commands/test_inspect_cli.py -v`

- [ ] **Step 6: Lint**

Run: `uv run ruff check modelkit/inspect/ --fix`

- [ ] **Step 7: Commit**

```bash
git add modelkit/inspect/resolver.py modelkit/inspect/__init__.py tests/inspect/test_resolver.py
git commit -m "refactor(inspect): consolidate I/O extraction via resolve_io_specs

Replaces _extract_tensor_specs_from_onnx_config (~100 lines) with
export/io.py resolve_io_specs(). Fixes 64x64 image size bug for
ResNet (now correctly reads preprocessor_config.json for 224x224).
Adds value_range capture to TensorInfo."
```

### Task 3: Fix IOConfigInfo to handle hidden_sizes and image_size from preprocessor

**Files:**
- Modify: `modelkit/inspect/resolver.py:605-673` (resolve_io_config)
- Modify: `modelkit/inspect/formatter.py:81-134` (_output_io_config_table)
- Test: `tests/inspect/test_resolver.py`

- [ ] **Step 1: Write failing test**

```python
class TestResolveIOConfig:
    def test_resnet_hidden_sizes(self):
        """ResNet has hidden_sizes (list), not hidden_size (scalar)."""
        from transformers import AutoConfig
        from modelkit.inspect.resolver import resolve_io_config

        config = AutoConfig.from_pretrained("microsoft/resnet-50")
        io = resolve_io_config(config)
        assert io.hidden_sizes == [256, 512, 1024, 2048]

    def test_resnet_image_size_from_preprocessor(self):
        """ResNet config lacks image_size; should get it from preprocessor."""
        from transformers import AutoConfig
        from modelkit.inspect.resolver import resolve_io_config

        config = AutoConfig.from_pretrained("microsoft/resnet-50")
        io = resolve_io_config(config, model_id="microsoft/resnet-50")
        assert io.image_size == 224 or io.image_size == (224, 224)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `resolve_io_config() got an unexpected keyword argument 'model_id'`

- [ ] **Step 3: Implement fixes**

Update `resolve_io_config()` in resolver.py:

```python
def resolve_io_config(
    config: PretrainedConfig,
    *,
    model_id: str | None = None,
) -> IOConfigInfo:
    """Extract IO configuration from HuggingFace config.

    For vision models where image_size is missing from config (e.g., ResNet),
    falls back to reading preprocessor_config.json via model_id.
    """
    # ... existing get_config_attr helper (unchanged) ...

    # Existing lookups (unchanged)
    max_position_embeddings = get_config_attr("max_position_embeddings", ["text_config"])
    vocab_size = get_config_attr("vocab_size", ["text_config"])
    image_size = get_config_attr("image_size", ["vision_config"])
    patch_size = get_config_attr("patch_size", ["vision_config"])
    num_channels = get_config_attr("num_channels", ["vision_config"])
    sampling_rate = get_config_attr("sampling_rate", ["audio_config"])
    hidden_size = get_config_attr("hidden_size", ["text_config", "vision_config"])

    # NEW: hidden_sizes (for ResNet-like models with per-stage hidden dims)
    hidden_sizes = get_config_attr("hidden_sizes")

    # NEW: Fallback to preprocessor_config.json for image_size
    if image_size is None and model_id is not None:
        try:
            from ..export.io import _populate_image_size_from_preprocessor
            shape_kwargs: dict = {}
            _populate_image_size_from_preprocessor(model_id, shape_kwargs)
            if "height" in shape_kwargs:
                h, w = shape_kwargs["height"], shape_kwargs["width"]
                image_size = h if h == w else (h, w)
        except Exception as e:
            logger.debug("Failed to get image_size from preprocessor: %s", e)

    return IOConfigInfo(
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        hidden_size=hidden_size,
        hidden_sizes=hidden_sizes,
    )
```

Update `inspect_model()` in `__init__.py`:
```python
# Step 10: Extract IO config from HF config
io_config_info = resolve_io_config(hf_config, model_id=model_id)
```

Update `_output_io_config_table()` in formatter.py to display hidden_sizes:
```python
# After the hidden_size block, add:
if io_config.hidden_sizes is not None:
    sizes_str = " → ".join(str(s) for s in io_config.hidden_sizes)
    io_table.add_row("Hidden Sizes", sizes_str)
    has_content = True
```

Update `output_json()` in formatter.py to include hidden_sizes:
```python
# In the io_config dict:
"hidden_sizes": io_config.hidden_sizes,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/inspect/test_resolver.py -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check modelkit/inspect/ --fix
git add modelkit/inspect/resolver.py modelkit/inspect/__init__.py modelkit/inspect/types.py modelkit/inspect/formatter.py tests/inspect/test_resolver.py
git commit -m "fix(inspect): resolve image_size from preprocessor, add hidden_sizes"
```

---

## Chunk 2: Fix Config Device Detection Bug (#412)

### Task 4: Change EPConfig default from "qnn" to None

**Files:**
- Modify: `modelkit/compiler/configs.py` (EPConfig.provider default)
- Modify: `modelkit/config/build.py` (always resolve device)
- Test: `tests/inspect/test_resolver.py` or `tests/config/test_device_default.py`

- [ ] **Step 1: Write failing test**

```python
# tests/config/test_device_default.py
"""Tests for config device detection defaults."""


class TestConfigDeviceDefault:
    def test_default_config_no_qnn_without_hardware(self):
        """Config should not default to QNN without NPU hardware."""
        from modelkit.compiler.configs import EPConfig
        ep = EPConfig()
        # Default should be None (detect from hardware), not "qnn"
        assert ep.provider is None or ep.provider != "qnn", (
            "EPConfig should not default to 'qnn' without hardware detection"
        )

    def test_generate_config_detects_device(self):
        """generate_build_config should detect hardware even with device=auto."""
        from unittest.mock import patch
        from modelkit.config import generate_build_config

        # Mock resolve_device to return cpu (no NPU)
        with patch("modelkit.config.build.resolve_device", return_value=("cpu", ["cpu"])):
            config = generate_build_config("microsoft/resnet-50")
            # compile config should NOT have qnn
            if config.compile and config.compile.ep_config:
                assert config.compile.ep_config.provider != "qnn"
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `EPConfig().provider` is `"qnn"`

- [ ] **Step 3: Fix EPConfig default**

In `modelkit/compiler/configs.py`, change:
```python
@dataclass
class EPConfig:
    provider: str | None = None  # Changed from "qnn" — detected from hardware
    # ... rest unchanged
```

- [ ] **Step 4: Update generate_build_config to always detect device**

In `modelkit/config/build.py`, change the device detection block (around line 466-479):

```python
# STEP 4.5: Apply device/precision policy (affects quant + compile only)
from .precision import resolve_precision
from ..sysinfo import resolve_device

# ALWAYS detect hardware — don't skip when both are "auto"
resolved_device, available_devices = resolve_device(device=device)
logger.info(
    "Device resolved: %s (available: %s)",
    resolved_device, ", ".join(available_devices),
)

policy = resolve_precision(
    device=resolved_device,
    precision=precision,
    ep=ep,
    available_devices=available_devices,
    task=parent_config.loader.task,
)

# Apply policy — always set compile provider from detected hardware
if policy.compile_provider is not None:
    parent_config.compile = WinMLCompileConfig.for_provider(
        policy.compile_provider,
    )

if policy.weight_type is not None:
    if parent_config.quant is None:
        parent_config.quant = WinMLQuantizationConfig()
    parent_config.quant.weight_type = policy.weight_type
    parent_config.quant.activation_type = policy.activation_type
elif policy.device != "auto" and policy.weight_type is None:
    parent_config.quant = None
```

- [ ] **Step 5: Update resolve_precision for auto+auto to still return device-based provider**

In `modelkit/config/precision.py`, update the both-auto path:

```python
# When both are auto, still return the hardware-detected provider
if device == "auto" and resolved_precision == "auto":
    return PrecisionPolicy(
        device="auto",
        precision="auto",
        weight_type=None,
        activation_type=None,
        compile_provider=None,  # Let caller handle — device already resolved
    )
```

Actually, the cleaner fix is in `build.py` only: always call `resolve_device()` and use the result to set compile provider when it's None. The precision module stays the same.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/config/test_device_default.py -v`
Run: `uv run pytest tests/ -k "config" --ignore=tests/integration --ignore=tests/e2e -v`

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check modelkit/compiler/ modelkit/config/ --fix
git add modelkit/compiler/configs.py modelkit/config/build.py tests/config/test_device_default.py
git commit -m "fix(config): detect hardware instead of defaulting to QNN (#412)

EPConfig.provider now defaults to None instead of 'qnn'.
generate_build_config() always calls resolve_device() to detect
available hardware, even when device='auto'."
```

---

## Chunk 3: Fix MUST-Rule Violations (#247)

### Task 5: Fix D-1 — Remove hardcoded dtype inference from name patterns

The current `infer_dtype()` hardcodes `"ids"→int64`, `"mask"→int64`. With the consolidation in Task 2, this function is deleted (resolve_io_specs captures actual dtypes from dummy inputs). **This is already fixed by Task 2.** Verify only.

- [ ] **Step 1: Write verification test**

```python
class TestNoHardcodedDtypeInference:
    def test_bert_dtypes_from_dummy_inputs(self):
        """Dtypes should come from actual tensors, not name pattern matching."""
        from transformers import AutoConfig
        from modelkit.inspect.resolver import resolve_exporter

        hf_config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
        info = resolve_exporter("bert", "fill-mask", hf_config=hf_config)
        for t in info.input_tensors:
            assert t.dtype is not None, f"Tensor {t.name} missing dtype"
```

- [ ] **Step 2: Run test — should PASS (already fixed by Task 2)**

Run: `uv run pytest tests/inspect/test_resolver.py::TestNoHardcodedDtypeInference -v`

- [ ] **Step 3: Commit verification test**

```bash
git add tests/inspect/test_resolver.py
git commit -m "test(inspect): verify D-1 fix — dtypes from dummy inputs, not name patterns"
```

### Task 6: Fix D-2 — Remove hardcoded nested config names

`resolve_io_config()` hardcodes `["text_config"]`, `["vision_config"]`, `["audio_config"]`. Replace with dynamic discovery of all nested `PretrainedConfig` objects.

**Files:**
- Modify: `modelkit/inspect/resolver.py:605-673`
- Test: `tests/inspect/test_resolver.py`

- [ ] **Step 1: Write failing test**

```python
class TestNoHardcodedNestedConfigs:
    def test_discovers_nested_configs_dynamically(self):
        """Should find nested configs without hardcoding names."""
        from transformers import AutoConfig
        from modelkit.inspect.resolver import resolve_io_config

        # CLIP has text_config and vision_config
        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        io = resolve_io_config(config)
        # Should find vocab_size from text_config
        assert io.vocab_size is not None
        # Should find image_size from vision_config
        assert io.image_size is not None
```

- [ ] **Step 2: Implement dynamic nested config discovery**

Replace the hardcoded nested config names with:

```python
def _find_nested_configs(config: PretrainedConfig) -> list[PretrainedConfig]:
    """Discover all nested PretrainedConfig objects dynamically."""
    from transformers import PretrainedConfig as _PC
    nested = []
    for attr_name in dir(config):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(config, attr_name)
            if isinstance(val, _PC):
                nested.append(val)
        except Exception:
            continue
    return nested
```

Then update `get_config_attr` to use this instead of hardcoded names:

```python
def get_config_attr(attr_name: str) -> int | tuple[int, int] | list | None:
    value = getattr(config, attr_name, None)
    if value is not None:
        return value
    for nested in nested_configs:
        value = getattr(nested, attr_name, None)
        if value is not None:
            return value
    return None

nested_configs = _find_nested_configs(config)
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "fix(inspect): D-2 — dynamic nested config discovery, no hardcoded names"
```

### Task 7: Fix D-3 — Remove hardcoded axis abbreviations from shape_to_desc

Already partially fixed in Task 2's `_shape_to_desc()` — it only hardcodes "B" for batch. The old version hardcoded "S" for sequence, special-cased "height"/"width". The new version uses axis names directly from OnnxConfig.

- [ ] **Step 1: Write verification test**

```python
class TestShapeToDesc:
    def test_uses_axis_names_not_abbreviations(self):
        """shape_to_desc should not hardcode 'S' for sequence."""
        from modelkit.inspect.resolver import _shape_to_desc

        # Dynamic axes with sequence
        axes = {0: "batch_size", 1: "sequence_length"}
        desc = _shape_to_desc((1, 128), axes)
        assert desc == "[B, 128]"
        # batch → B, sequence uses actual value
```

- [ ] **Step 2: Run test — should PASS**
- [ ] **Step 3: Commit**

### Task 8: Fix D-4 — Remove hardcoded JSON keys from processor resolution

`_resolve_processor_from_hub_configs()` hardcodes JSON keys like `"processor_class"`, `"image_processor_type"`, `"feature_extractor_type"`, `"tokenizer_class"`. These are standard HF config keys — they are part of HF's API contract, NOT model-specific hardcoding. **D-4 is a false positive** — these keys are universal HF conventions, not model-specific patterns.

However, the processor resolution has a real bug: ResNet shows `ConvNextImageProcessorFast`. This is because `preprocessor_config.json` says `"feature_extractor_type": "ConvNextFeatureExtractor"` and `AutoProcessor` returns `ConvNextImageProcessorFast`. **This is actually correct HF behavior** — ResNet's processor config on HuggingFace Hub genuinely points to ConvNext processors (they share the same preprocessing pipeline).

- [ ] **Step 1: Document D-4 as false positive**

Add a comment in resolver.py explaining this is universal HF API, not model-specific:

```python
# NOTE: These JSON keys (processor_class, image_processor_type, etc.) are
# standard HuggingFace config conventions, not model-specific hardcoding.
# See: https://huggingface.co/docs/transformers/preprocessing
```

- [ ] **Step 2: Add model_type to ProcessorInfo for transparency**

The real improvement is showing WHERE the processor class comes from, so users understand why ResNet says ConvNext. This is a formatter improvement, not a logic fix.

- [ ] **Step 3: Commit**

### Task 9: Fix D-5 — Remove modality assumptions in attribute grouping

`resolve_io_config()` groups attributes by modality (text, vision, audio). After Task 6's dynamic discovery, the grouping is implicit — `get_config_attr()` searches all nested configs without assuming modality. **D-5 is resolved by Task 6.**

- [ ] **Step 1: Verify with test**
- [ ] **Step 2: Commit verification**

---

## Chunk 4: Missing Features (M-1, B-1)

### Task 10: Add --list-tasks flag (M-1)

**Files:**
- Modify: `modelkit/commands/inspect.py`
- Modify: `modelkit/inspect/resolver.py` (expose _get_known_tasks)
- Test: `tests/commands/test_inspect_cli.py`

- [ ] **Step 1: Write failing test**

```python
class TestListTasksFlag:
    def test_list_tasks_outputs_tasks(self, runner):
        from modelkit.commands.inspect import inspect
        result = runner.invoke(inspect, ["--list-tasks"], obj={})
        assert result.exit_code == 0
        assert "image-classification" in result.output
        assert "fill-mask" in result.output
```

- [ ] **Step 2: Add --list-tasks flag to CLI**

```python
@click.option(
    "--list-tasks",
    is_flag=True,
    default=False,
    help="List all known tasks and exit",
)
```

In the command body, before the main logic:
```python
if list_tasks:
    from ..inspect.resolver import get_known_tasks
    tasks = sorted(get_known_tasks())
    for t in tasks:
        click.echo(t)
    return
```

Rename `_get_known_tasks` to `get_known_tasks` (make public).

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(inspect): add --list-tasks flag (M-1 from #247)"
```

### Task 11: Support local ONNX file input (B-1 groundwork)

This is groundwork for #354 — full ONNX inspect is a separate feature. For now, add basic detection and a helpful error message.

**Files:**
- Modify: `modelkit/commands/inspect.py`
- Test: `tests/commands/test_inspect_cli.py`

- [ ] **Step 1: Write test**

```python
class TestOnnxInput:
    def test_onnx_file_gives_helpful_message(self, runner, tmp_path):
        from modelkit.commands.inspect import inspect
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")
        result = runner.invoke(inspect, ["-m", str(onnx_file)], obj={})
        assert "ONNX" in result.output
```

- [ ] **Step 2: Add ONNX detection**

```python
# Before the main try block:
if model.endswith(".onnx") and Path(model).exists():
    raise click.ClickException(
        "ONNX file inspection is not yet supported. "
        "Use 'wmk config -m model.onnx' for ONNX build config. "
        "See issue #354 for progress on ONNX inspect."
    )
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(inspect): B-1 groundwork — helpful message for ONNX file input"
```

---

## Chunk 5: Processor Bug Investigation & Fix

### Task 12: Investigate and fix processor identification

ResNet-50 showing `ConvNextImageProcessorFast` is actually **correct HF behavior** — the Hub repo genuinely has `ConvNextFeatureExtractor` in its config. However, inspect should be more transparent about this.

**Files:**
- Modify: `modelkit/inspect/formatter.py` (_output_processor_table)

- [ ] **Step 1: Add source attribution to processor display**

Show the source of each processor class so users understand WHY:

```python
# In _output_processor_table, for each processor class:
# Show: "ConvNextImageProcessorFast (from preprocessor_config.json)"
```

This requires passing source info through ProcessorInfo. Add optional source fields:

```python
@dataclass
class ProcessorInfo:
    processor_class: str | None = None
    tokenizer_class: str | None = None
    image_processor_class: str | None = None
    feature_extractor_class: str | None = None
    # Source tracking
    processor_source: str | None = None  # "hub_config" | "auto_class"
    image_processor_source: str | None = None
```

- [ ] **Step 2: Update resolver to track sources**
- [ ] **Step 3: Update formatter to display sources**
- [ ] **Step 4: Test and commit**

```bash
git commit -m "fix(inspect): add source attribution to processor identification"
```

---

## Chunk 6: Final Verification & Cleanup

### Task 13: End-to-end verification

- [ ] **Step 1: Run `wmk inspect -m microsoft/resnet-50` and verify output**

Expected changes:
- Input shape: `[B, 3, 224, 224]` (was 64x64)
- Output logits: should show dtype
- IO Config: shows Image Size 224, Hidden Sizes 256→512→1024→2048, Channels 3
- Value range shown for pixel_values

- [ ] **Step 2: Run `wmk config -m microsoft/resnet-50` and verify output**

Expected changes:
- `execution_provider` should reflect actual hardware (not hardcoded "qnn")

- [ ] **Step 3: Run `wmk inspect --list-tasks` and verify output**

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/inspect/ tests/commands/test_inspect_cli.py tests/commands/test_config_cli.py -v`

- [ ] **Step 5: Lint all modified files**

Run: `uv run ruff check modelkit/inspect/ modelkit/compiler/ modelkit/config/ --fix`

- [ ] **Step 6: Final commit with any remaining fixes**
