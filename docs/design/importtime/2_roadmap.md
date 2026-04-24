# Import Time Optimization — Roadmap

**Prereq**: `1_prd.md` (problem/measurements), `1_plan.md` (approach/design)

## Phase 1: Lazy CLI Command Discovery (independent, low risk)

### 1.1 Implement `LazyGroup` in `cli.py`
- [ ] Add `LazyGroup(click.Group)` class with `list_commands()` and `get_command()`
- [ ] `list_commands()`: return command names from filesystem (glob `*.py`, skip `_` prefix)
- [ ] `get_command()`: `import_module()` only when command is invoked
- [ ] `get_command()`: error handling — catch `ImportError` (warning) + `Exception` (error)
- [ ] `get_command()`: prefer `click.Group` over `click.Command` (match current behavior)
- [ ] Change `@click.group()` to `@click.group(cls=LazyGroup)`
- [ ] Remove `_discover_commands()` function and its module-level call

### 1.2 Verify Phase 1
- [ ] `wmk --help` lists all commands correctly
- [ ] `wmk sys --format compact` works
- [ ] `wmk build --help` works (command imported on demand)
- [ ] Broken/missing command module → graceful warning, not crash
- [ ] Measure: `time uv run wmk --help` (expect improvement if `__init__.py` still eager)

### 1.3 Commit Phase 1
- [ ] Ruff lint `modelkit/cli.py`
- [ ] Run `uv run pytest tests/ -x -q -o "required_plugins="`
- [ ] Commit: `perf: lazy CLI command discovery via LazyGroup`

---

## Phase 2: Lazy `modelkit/__init__.py` + Registration Guard (atomic pair)

### 2.1 Make `modelkit/__init__.py` lazy
- [ ] Remove eager imports: `from .config import WinMLBuildConfig` (line 30)
- [ ] Remove eager imports: `from .models import ...` (lines 31-35)
- [ ] Add `__getattr__(name)` with lazy imports + `globals()` caching
- [ ] Add `__dir__()` returning `__all__` (for debugger/IPython compatibility)
- [ ] Keep `from . import _warnings` as eager (line 29)
- [ ] Keep `__version__` assignment as eager
- [ ] Keep `__all__` listing all public names

### 2.2 Verify `__init__.py` in isolation
- [ ] `from modelkit.cli import main` does NOT trigger torch/transformers/optimum
  ```bash
  uv run python -X importtime -c "from modelkit.cli import main" 2>&1 | grep -cE "torch|transformers|optimum"
  # Should output: 0
  ```
- [ ] `from modelkit import __version__` works and is fast
- [ ] `from modelkit import WinMLAutoModel` works (triggers lazy load)
- [ ] `from modelkit import WinMLBuildConfig` works
- [ ] `from modelkit import WinMLPreTrainedModel` works
- [ ] `from modelkit import WinMLModelForImageClassification` works
- [ ] `dir(modelkit)` includes all `__all__` names

### 2.3 Add ONNX config registration guard
- [ ] Add `ensure_hf_models_registered()` in `modelkit/export/io.py`
- [ ] Call in `_get_onnx_config_constructor()` (before `TasksManager.get_exporter_config_constructor()`)
- [ ] Defensively call in `config/build.py:generate_build_config()` (already safe, belt-and-suspenders)
- [ ] Verify: `wmk export` finds custom ONNX configs (BERT, CLIP, DETR, SAM)
- [ ] Verify: `wmk config -m microsoft/resnet-50 --device npu --precision int8`
- [ ] Verify: `wmk inspect -m google-bert/bert-base-uncased`

### 2.4 Check `export/__init__.py` exposure
- [ ] Verify no CLI path reaches `modelkit.export` package import (only `modelkit.export.config`, `modelkit.export.io` submodules)
- [ ] If exposed: make `export/__init__.py` lazy too
- [ ] If not exposed: document as known landmine for future devs

### 2.5 Verify Phase 2 end-to-end
- [ ] `wmk --help` < 2s
- [ ] `wmk --version` < 1s
- [ ] `wmk sys --format compact` < 3s
- [ ] `wmk export -m prajjwal1/bert-tiny -o temp/test_export` works
- [ ] `wmk config -m microsoft/resnet-50 --device npu --precision int8` works
- [ ] `wmk inspect -m google-bert/bert-base-uncased` works
- [ ] `wmk build` works (with existing test configs)
- [ ] `wmk perf` works

### 2.6 Run full test suite
- [ ] `uv run pytest tests/ -x -q -o "required_plugins="`
- [ ] No regressions vs baseline (808 passed, 1 pre-existing failure)

### 2.7 Commit Phase 2
- [ ] Ruff lint all changed files
- [ ] Commit: `perf: lazy __init__.py + ONNX config registration guard`

---

## Phase 3: Regression Prevention & Cleanup

### 3.1 Add import time regression test
- [ ] Add `tests/test_import_time.py`:
  ```python
  def test_cli_import_no_heavy_deps():
      """Importing the CLI must not pull in torch/transformers/optimum."""
      result = subprocess.run(
          [sys.executable, "-c",
           "import sys; from modelkit.cli import main; "
           "heavy = [m for m in sys.modules if m.startswith(('torch', 'transformers', 'optimum'))]; "
           "assert not heavy, f'Heavy modules loaded: {heavy}'"],
          capture_output=True, text=True,
      )
      assert result.returncode == 0, result.stderr
  ```
- [ ] Verify test passes

### 3.2 Final measurements
- [ ] Record before/after timing table
- [ ] Update `1_prd.md` status to "Complete" with final measurements

### 3.3 Commit Phase 3
- [ ] Commit: `test: add import time regression test`

---

## Summary

| Phase | Files Modified | Risk | Dependency |
|-------|---------------|------|------------|
| 1: LazyGroup | `cli.py` | Low | None |
| 2: Lazy init + guard | `__init__.py`, `export/io.py`, maybe `export/__init__.py` | Medium | Phase 1 recommended first |
| 3: Regression test | `tests/test_import_time.py` | None | Phase 2 |

**Total files changed**: 3-4 (+ 1 new test file)
