# ONNX Persistence API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create universal `load_onnx`, `save_onnx`, `cleanup_onnx` functions in `modelkit/onnx/persistence.py` and migrate all 25 load + 14 save + 5 check_model call sites across the codebase.

**Architecture:** Three functions in a new `persistence.py` module. `load_onnx` always uses path-based validation (safe for any size). `save_onnx` uses `ByteSize()` vs threshold to auto-decide external data. `cleanup_onnx` reads external data references from graph-only load and deletes all associated files. Migration is mechanical: replace `onnx.load(...)` → `load_onnx(...)`, `onnx.save(...)` → `save_onnx(...)`.

**Tech Stack:** Python, onnx, pytest

**Design Doc:** `docs/design/onnx/3_design_persistence.md`

---

### Task 1: Create `persistence.py` with `load_onnx`

**Files:**
- Create: `modelkit/onnx/persistence.py`
- Test: `tests/onnx/test_persistence.py`

**Step 1: Write failing tests for `load_onnx`**

```python
# tests/onnx/test_persistence.py
"""Tests for modelkit.onnx.persistence — load_onnx, save_onnx, cleanup_onnx."""

from __future__ import annotations

import pytest
import onnx
import numpy as np
from onnx import TensorProto, helper
from pathlib import Path

from modelkit.onnx.persistence import load_onnx


def _make_tiny_model() -> onnx.ModelProto:
    """Create minimal valid ONNX model."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestLoadOnnx:
    def test_load_basic(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        loaded = load_onnx(path)
        assert len(loaded.graph.node) == 1
        assert loaded.graph.node[0].op_type == "Relu"

    def test_load_validates_by_default(self, tmp_path):
        path = tmp_path / "bad.onnx"
        path.write_bytes(b"not an onnx model")
        with pytest.raises(Exception):
            load_onnx(path)

    def test_load_skip_validation(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        loaded = load_onnx(path, validate=False)
        assert len(loaded.graph.node) == 1

    def test_load_weights_false(self, tmp_path):
        model = _make_tiny_model()
        w = onnx.numpy_helper.from_array(
            np.zeros((3,), dtype=np.float32), name="W"
        )
        model.graph.initializer.append(w)
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        loaded = load_onnx(path, load_weights=False, validate=False)
        assert len(loaded.graph.initializer) == 1

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_onnx("/nonexistent/model.onnx")

    def test_load_path_object(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        loaded = load_onnx(Path(path))
        assert loaded is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/onnx/test_persistence.py -v`
Expected: ImportError (module doesn't exist yet)

**Step 3: Implement `load_onnx`**

```python
# modelkit/onnx/persistence.py
"""Universal ONNX model persistence: load, save, cleanup.

Replaces scattered onnx.load() / onnx.save() calls with consistent
external data handling, path-based validation, and deterministic naming.

See docs/design/onnx/3_design_persistence.md for design rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
import onnx.checker

logger = logging.getLogger(__name__)


def load_onnx(
    path: str | Path,
    *,
    load_weights: bool = True,
    validate: bool = True,
) -> onnx.ModelProto:
    """Load an ONNX model with external data awareness and path-based validation.

    Args:
        path: Path to .onnx file.
        load_weights: Load tensor weights into memory. False = graph structure
            only (fast, for inspection/detection).
        validate: Run onnx.checker.check_model(str(path)) — validates graph
            structure, opset, and external data file references. Works for
            any model size (no 2GiB protobuf limit).

    Returns:
        Loaded ModelProto.

    Raises:
        FileNotFoundError: If path does not exist.
        onnx.checker.ValidationError: If model fails validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX model not found: {p}")

    logger.debug("Loading ONNX model: %s (load_weights=%s)", p, load_weights)
    model = onnx.load(str(p), load_external_data=load_weights)

    if validate:
        logger.debug("Validating ONNX model (path-based): %s", p)
        onnx.checker.check_model(str(p))

    return model
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/onnx/test_persistence.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add modelkit/onnx/persistence.py tests/onnx/test_persistence.py
git commit -m "feat(onnx): add load_onnx with path-based validation"
```

---

### Task 2: Add `save_onnx` to `persistence.py`

**Files:**
- Modify: `modelkit/onnx/persistence.py`
- Test: `tests/onnx/test_persistence.py`

**Step 1: Write failing tests**

```python
from modelkit.onnx.persistence import save_onnx

class TestSaveOnnx:
    def test_save_basic(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "out.onnx"
        save_onnx(model, path)
        assert path.exists()
        loaded = onnx.load(str(path))
        assert len(loaded.graph.node) == 1

    def test_save_small_model_inline(self, tmp_path):
        """Model under threshold stays inline (no .data file)."""
        model = _make_tiny_model()
        path = tmp_path / "small.onnx"
        save_onnx(model, path)
        assert not (tmp_path / "small.onnx.data").exists()

    def test_save_large_model_external(self, tmp_path):
        """Model over threshold gets external data."""
        model = _make_tiny_model()
        big = onnx.numpy_helper.from_array(
            np.random.randn(30_000_000).astype(np.float32), name="W"
        )
        model.graph.initializer.append(big)
        path = tmp_path / "big.onnx"
        save_onnx(model, path, threshold_size=100 * 1024 * 1024)
        assert (tmp_path / "big.onnx.data").exists()

    def test_save_threshold_zero_always_external(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "forced.onnx"
        save_onnx(model, path, threshold_size=0)
        assert (tmp_path / "forced.onnx.data").exists()

    def test_save_external_data_false(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "inline.onnx"
        save_onnx(model, path, use_external_data=False)
        assert not (tmp_path / "inline.onnx.data").exists()

    def test_save_creates_parent_dir(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "subdir" / "nested" / "model.onnx"
        save_onnx(model, path)
        assert path.exists()

    def test_save_custom_location(self, tmp_path):
        model = _make_tiny_model()
        big = onnx.numpy_helper.from_array(
            np.random.randn(30_000_000).astype(np.float32), name="W"
        )
        model.graph.initializer.append(big)
        path = tmp_path / "model.onnx"
        save_onnx(model, path, threshold_size=0, location="weights.bin")
        assert (tmp_path / "weights.bin").exists()

    def test_save_respects_existing_external(self, tmp_path):
        """Model with data_location=EXTERNAL always saves as external."""
        model = _make_tiny_model()
        t = onnx.numpy_helper.from_array(np.zeros((3,), dtype=np.float32), "W")
        model.graph.initializer.append(t)
        # Save with external data first
        path1 = tmp_path / "ext.onnx"
        save_onnx(model, path1, threshold_size=0)
        # Reload graph-only (preserves external markers)
        reloaded = load_onnx(path1, load_weights=False, validate=False)
        # Re-save with use_external_data=False — should still be external
        path2 = tmp_path / "ext2.onnx"
        save_onnx(reloaded, path2, use_external_data=False)
        assert (tmp_path / "ext2.onnx.data").exists()
```

**Step 2: Implement `save_onnx`**

```python
from onnx.external_data_helper import _get_all_tensors, uses_external_data

_EXTERNAL_DATA_THRESHOLD = 100 * 1024 * 1024  # 100 MiB

def save_onnx(
    model: onnx.ModelProto,
    path: str | Path,
    *,
    use_external_data: bool = True,
    threshold_size: int = _EXTERNAL_DATA_THRESHOLD,
    location: str | None = None,
) -> None:
    """Save an ONNX model with smart external data handling.

    Args:
        model: ONNX model to save.
        path: Output .onnx file path.
        use_external_data: Enable external data. False = always inline.
        threshold_size: ByteSize threshold to trigger external data.
            <= 0 means always use external data.
        location: External data filename. None defaults to "{filename}.data".
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Check if model already has external data markers
    has_existing_external = any(
        uses_external_data(t) for t in _get_all_tensors(model)
    )

    if has_existing_external:
        use_ext = True
    elif not use_external_data:
        use_ext = False
    elif threshold_size <= 0:
        use_ext = True
    else:
        use_ext = model.ByteSize() >= threshold_size

    if use_ext:
        ext_location = location or f"{p.name}.data"
        logger.debug("Saving ONNX model with external data: %s + %s", p, ext_location)
        onnx.save(
            model,
            str(p),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_location,
            size_threshold=1024,
        )
    else:
        logger.debug("Saving ONNX model inline: %s", p)
        onnx.save(model, str(p))
```

**Step 3: Run tests**

Run: `uv run pytest tests/onnx/test_persistence.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add modelkit/onnx/persistence.py tests/onnx/test_persistence.py
git commit -m "feat(onnx): add save_onnx with smart external data"
```

---

### Task 3: Add `cleanup_onnx` to `persistence.py`

**Files:**
- Modify: `modelkit/onnx/persistence.py`
- Test: `tests/onnx/test_persistence.py`

**Step 1: Write failing tests**

```python
from modelkit.onnx.persistence import cleanup_onnx

class TestCleanupOnnx:
    def test_cleanup_inline_model(self, tmp_path):
        model = _make_tiny_model()
        path = tmp_path / "model.onnx"
        save_onnx(model, path, use_external_data=False)
        deleted = cleanup_onnx(path)
        assert not path.exists()
        assert path in deleted

    def test_cleanup_external_data(self, tmp_path):
        model = _make_tiny_model()
        big = onnx.numpy_helper.from_array(
            np.random.randn(30_000_000).astype(np.float32), name="W"
        )
        model.graph.initializer.append(big)
        path = tmp_path / "model.onnx"
        save_onnx(model, path, threshold_size=0)
        data_path = tmp_path / "model.onnx.data"
        assert data_path.exists()
        deleted = cleanup_onnx(path)
        assert not path.exists()
        assert not data_path.exists()
        assert len(deleted) == 2

    def test_cleanup_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            cleanup_onnx("/nonexistent.onnx")

    def test_cleanup_missing_data_file_skipped(self, tmp_path):
        """If .data file was already deleted, cleanup doesn't crash."""
        model = _make_tiny_model()
        big = onnx.numpy_helper.from_array(
            np.random.randn(30_000_000).astype(np.float32), name="W"
        )
        model.graph.initializer.append(big)
        path = tmp_path / "model.onnx"
        save_onnx(model, path, threshold_size=0)
        # Pre-delete the data file
        (tmp_path / "model.onnx.data").unlink()
        # cleanup should not crash
        deleted = cleanup_onnx(path)
        assert path not in [d for d in deleted if d.exists()]
```

**Step 2: Implement `cleanup_onnx`**

```python
def cleanup_onnx(path: str | Path) -> list[Path]:
    """Delete an ONNX model and all associated external data files.

    Loads graph-only to discover external data references, then deletes
    each referenced file and the .onnx file itself.

    Args:
        path: Path to .onnx file to delete.

    Returns:
        List of all deleted file paths.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX model not found: {p}")

    deleted: list[Path] = []

    # Load graph-only to find external data references
    model = onnx.load(str(p), load_external_data=False)
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            for entry in tensor.external_data:
                if entry.key == "location":
                    data_file = p.parent / entry.value
                    if data_file.exists():
                        data_file.unlink()
                        deleted.append(data_file)
                        logger.debug("Deleted external data: %s", data_file)

    # Delete the .onnx file
    p.unlink()
    deleted.append(p)
    logger.debug("Deleted ONNX model: %s", p)

    return deleted
```

**Step 3: Run tests, commit**

Run: `uv run pytest tests/onnx/test_persistence.py -v`

```bash
git add modelkit/onnx/persistence.py tests/onnx/test_persistence.py
git commit -m "feat(onnx): add cleanup_onnx for temp file handling"
```

---

### Task 4: Export from `modelkit/onnx/__init__.py`

**Files:**
- Modify: `modelkit/onnx/__init__.py`

**Step 1: Add exports**

```python
from .persistence import cleanup_onnx, load_onnx, save_onnx
```

Add to `__all__` if it exists.

**Step 2: Verify import works**

Run: `uv run python -c "from modelkit.onnx import load_onnx, save_onnx, cleanup_onnx; print('OK')"`

**Step 3: Commit**

```bash
git add modelkit/onnx/__init__.py
git commit -m "feat(onnx): export persistence functions from package"
```

---

### Task 5: Migrate `optim/` module (unblocks Qwen3)

**Files:**
- Modify: `modelkit/optim/api.py`
- Modify: `modelkit/optim/optimizer.py`
- Modify: `modelkit/optim/pipes/graph.py`
- Modify: `modelkit/optim/pipes/fusion.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `api.py:61` | `onnx.load(str(model_path))` | `load_onnx(model_path)` |
| `api.py:264-267` | `onnx.save(..., save_as_external_data=...)` | `save_onnx(optimized_model, output_path)` |
| `optimizer.py:96-102` | `onnx.checker.check_model(model)` | Remove (load_onnx validates) |
| `optimizer.py:165` | `onnx.checker.check_model(model)` | Remove or keep as optional post-optimization check |
| `pipes/graph.py:529` | `onnx.save(model, str(input_file))` | `save_onnx(model, input_file)` |
| `pipes/graph.py:585` | `onnx.load(str(output_file))` | `load_onnx(output_file)` |
| `pipes/fusion.py:260` | `onnx.save(model, input_path)` | `save_onnx(model, input_path)` |

**Step 1: Apply changes, run existing optim tests**

Run: `uv run pytest tests/optim/ -x -q`

**Step 2: Run ruff lint**

Run: `uv run ruff check modelkit/optim/`

**Step 3: Commit**

```bash
git commit -m "refactor(optim): migrate to load_onnx/save_onnx — unblocks >2GiB models"
```

---

### Task 6: Migrate `commands/optimize.py`

**Files:**
- Modify: `modelkit/commands/optimize.py`

**Changes:**

| Line | Current | New |
|------|---------|-----|
| 411 | `onnx.load(str(model))` | `load_onnx(model)` |
| 420 | `onnx.save(optimized_model, str(output))` | `save_onnx(optimized_model, output)` |

**Step 1: Apply, lint, test**

Run: `uv run ruff check modelkit/commands/optimize.py`

**Step 2: Commit**

```bash
git commit -m "refactor(commands): migrate optimize CLI to persistence API"
```

---

### Task 7: Migrate `core/` module

**Files:**
- Modify: `modelkit/core/onnx_utils.py`
- Modify: `modelkit/core/tag_utils.py`
- Modify: `modelkit/core/universal_hierarchy_exporter.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `onnx_utils.py:41-42` | `onnx.load()` + `check_model()` | `load_onnx(path)` |
| `onnx_utils.py:473` | `onnx.load(str(model_or_path))` | `load_onnx(model_or_path, validate=False)` |
| `tag_utils.py:25` | `onnx.load(onnx_path)` | `load_onnx(onnx_path, validate=False)` |
| `universal_hierarchy_exporter.py:393` | `onnx.load(onnx_path)` | `load_onnx(onnx_path, validate=False)` |
| `universal_hierarchy_exporter.py:694` | `onnx.load(output_path)` | `load_onnx(output_path, validate=False)` |

**Note**: `onnx_utils.py:41` is inside `ONNXUtils.load_and_validate()` — deprecate this method by making it a thin wrapper around `load_onnx`.

**Step 1: Apply, lint, run core tests**

Run: `uv run pytest tests/core/ -x -q`

**Step 2: Commit**

```bash
git commit -m "refactor(core): migrate to load_onnx, deprecate ONNXUtils.load_and_validate"
```

---

### Task 8: Migrate `compiler/` module

**Files:**
- Modify: `modelkit/compiler/cli.py`
- Modify: `modelkit/compiler/utils.py`
- Modify: `modelkit/compiler/stages/optimize.py`
- Modify: `modelkit/compiler/stages/compile.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `cli.py:345` | `onnx.load(str(model_path))` | `load_onnx(model_path, validate=False)` |
| `utils.py:24` | `onnx.load(str(model_path))` | `load_onnx(model_path, load_weights=False, validate=False)` |
| `stages/optimize.py:34` | `onnx.load(str(context.model_path))` | `load_onnx(context.model_path, validate=False)` |
| `stages/optimize.py:42` | `onnx.save(model, str(output_path))` | `save_onnx(model, output_path)` |
| `stages/compile.py:137` | `onnx.save(context.model, str(model_path))` | `save_onnx(context.model, model_path)` |
| `stages/compile.py:265` | `onnx.load(str(final_ctx_path))` | `load_onnx(final_ctx_path, validate=False)` |
| `stages/compile.py:277` | `onnx.save(model, str(final_ctx_path))` | `save_onnx(model, final_ctx_path)` |

**Step 1: Apply, lint, run compiler tests**

Run: `uv run pytest tests/compiler/ -x -q`

**Step 2: Commit**

```bash
git commit -m "refactor(compiler): migrate to load_onnx/save_onnx"
```

---

### Task 9: Migrate `quant/`, `export/`, `data/`, `session/`

**Files:**
- Modify: `modelkit/quant/quantizer.py`
- Modify: `modelkit/export/htp/exporter.py`
- Modify: `modelkit/data/random_dataset.py`
- Modify: `modelkit/session/session.py`
- Modify: `modelkit/session/qairt/compile_qairt_bin.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `quantizer.py:160` | `onnx.load(..., load_external_data=False)` | `load_onnx(path, load_weights=False, validate=False)` |
| `quantizer.py:180` | `onnx.load(str(output_path))` | `load_onnx(output_path, validate=False)` |
| `quantizer.py:208` | `onnx.save(...)` | `save_onnx(quantized_model, output_path)` |
| `exporter.py:262` | `onnx.load(output_path)` | `load_onnx(output_path, validate=False)` |
| `exporter.py:366-367` | `onnx.load()` + `check_model()` | `load_onnx(output_path)` |
| `exporter.py:586` | `onnx.save(...)` | `save_onnx(onnx_model, output_path)` |
| `random_dataset.py:44` | `onnx.load(self.model_path)` | `load_onnx(self.model_path, load_weights=False, validate=False)` |
| `session.py:582` | `onnx.load(str(self._onnx_path))` | `load_onnx(self._onnx_path, load_weights=False, validate=False)` |
| `qairt/compile_qairt_bin.py:48` | `onnx.load(str(model_path))` | `load_onnx(model_path, validate=False)` |

**Step 1: Apply all, lint, run relevant tests**

Run: `uv run pytest tests/export/ tests/quant/ -x -q`

**Step 2: Commit**

```bash
git commit -m "refactor(quant,export,data,session): migrate to persistence API"
```

---

### Task 10: Migrate `onnx/` module internal calls

**Files:**
- Modify: `modelkit/onnx/io.py`
- Modify: `modelkit/onnx/detection.py`
- Modify: `modelkit/onnx/external_data.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `io.py:216` | `onnx.load(model_path, load_external_data=False)` | `load_onnx(model_path, load_weights=False, validate=False)` |
| `detection.py:28` | `onnx.load(path_str, load_external_data=False)` | `load_onnx(path_str, load_weights=False, validate=False)` |
| `external_data.py:39` | `onnx.load(..., load_external_data=False)` | `load_onnx(path, load_weights=False, validate=False)` |
| `external_data.py:113` | `onnx.load(..., load_external_data=False)` | `load_onnx(path, load_weights=False, validate=False)` |
| `external_data.py:126` | `onnx.save_model(model, str(dst))` | `save_onnx(model, dst, use_external_data=False)` |
| `external_data.py:136-144` | `onnx.load(str(src))` + `onnx.save_model(...)` | `load_onnx(src)` + `save_onnx(model, dst, threshold_size=0, ...)` |

**Note**: Be careful with `external_data.py` — its existing patterns are intentional. Preserve the same behavior with new API.

**Step 1: Apply, lint, run onnx tests**

Run: `uv run pytest tests/onnx/ -x -q`

**Step 2: Commit**

```bash
git commit -m "refactor(onnx): migrate internal calls to persistence API"
```

---

### Task 11: Migrate `utils/` and skip `static_analyzer/`

**Files:**
- Modify: `modelkit/utils/hub_utils.py`
- Modify: `modelkit/utils/optimum_loader.py`

**Changes:**

| File:Line | Current | New |
|-----------|---------|-----|
| `hub_utils.py:268` | `onnx.load(onnx_path)` | `load_onnx(onnx_path, validate=False)` |
| `optimum_loader.py:77` | `onnx.load(onnx_path)` | `load_onnx(onnx_path, validate=False)` |

**Skip `static_analyzer/`**: The static analyzer has its own `ONNXLoader` abstraction, uses `load_external_data=True` intentionally, and has `check_model` calls embedded in pattern matching logic. These are specialized and should be migrated separately with domain expertise. Add a TODO comment at each call site.

**Step 1: Apply, lint**

Run: `uv run ruff check modelkit/utils/`

**Step 2: Commit**

```bash
git commit -m "refactor(utils): migrate to load_onnx, skip static_analyzer (separate PR)"
```

---

### Task 12: Final verification + squash

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q --timeout=120`
Expected: All existing tests pass (no regressions)

**Step 2: Verify Qwen3 build works**

Run: `uv run wmk build -c temp/qwen3_0.6/config.json -m Qwen/Qwen3-0.6B -o temp/qwen3_0.6/ --no-quant --no-compile`
Expected: Export + optimize succeed (no >2GiB crash)

**Step 3: Lint all changed files**

Run: `uv run ruff check modelkit/`

**Step 4: Squash into single commit, push, create PR**
