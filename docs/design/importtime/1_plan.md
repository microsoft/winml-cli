# Import Time Optimization — Implementation Plan

**Prereq**: Read `1_prd.md` for problem analysis and measurements.

## Approach: Lazy Loading via PEP 562

Three changes addressing the three heavy import chains identified in the PRD.
**Steps 1 and 3 must ship together** (Step 3 guards registrations broken by Step 1).
Step 2 is independent and low-risk.

## Step 1: Lazy `modelkit/__init__.py`

**Addresses**: All three chains — the entire 30s root cost.

**What**: Replace eager top-level imports with PEP 562 `__getattr__`:

```python
# modelkit/__init__.py — AFTER

from importlib.metadata import PackageNotFoundError, version

from . import _warnings  # Must stay eager: warning filters before any heavy import

try:
    __version__ = version("winml-modelkit")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"

__all__ = [
    "WinMLAutoModel",
    "WinMLBuildConfig",
    "WinMLModelForImageClassification",
    "WinMLPreTrainedModel",
    "__version__",
]

def __getattr__(name: str):
    if name == "WinMLBuildConfig":
        from .config import WinMLBuildConfig
        globals()["WinMLBuildConfig"] = WinMLBuildConfig  # Cache after first access
        return WinMLBuildConfig

    if name in ("WinMLAutoModel", "WinMLPreTrainedModel", "WinMLModelForImageClassification"):
        from .models import (
            WinMLAutoModel,
            WinMLModelForImageClassification,
            WinMLPreTrainedModel,
        )
        globals().update({
            "WinMLAutoModel": WinMLAutoModel,
            "WinMLPreTrainedModel": WinMLPreTrainedModel,
            "WinMLModelForImageClassification": WinMLModelForImageClassification,
        })
        return globals()[name]

    raise AttributeError(f"module 'modelkit' has no attribute {name!r}")

def __dir__():
    return __all__
```

**Key mechanism**: `cli.py` does `from . import __version__` which executes
`modelkit/__init__.py` — but with the lazy version, only `_warnings` + version
detection run at module scope. The heavy imports in `__getattr__` are never triggered
by the CLI path. This is the **primary mechanism** that makes the CLI fast.

**Why `globals()` caching**: Without it, `__getattr__` fires on every attribute access.
Caching into `globals()` means the import only happens once.

**Why `__dir__`**: Without it, `dir(modelkit)` would not include lazy attributes.
Required for debugger/IPython tab-completion.

**Verification**:
```bash
# Should NOT show transformers/torch/optimum in the trace
uv run python -X importtime -c "from modelkit.cli import main" 2>&1 | grep -E "torch|transformers|optimum"
```

## Step 2: Lazy CLI Command Discovery

**Addresses**: `_discover_commands()` importing all command modules at load time.

**What**: Replace `_discover_commands()` + `@click.group()` with a `LazyGroup` that
only imports command modules when a specific command is invoked:

```python
class LazyGroup(click.Group):
    """Click group that defers command imports until invoked."""

    _commands_dir = Path(__file__).parent / "commands"

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return command names from filesystem — no imports."""
        if not self._commands_dir.exists():
            return []
        return sorted(
            p.stem for p in self._commands_dir.glob("*.py")
            if not p.name.startswith("_")
        )

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Import command module only when the command is actually invoked."""
        try:
            module = import_module(f".commands.{cmd_name}", package=__package__)
        except ImportError as e:
            logger.warning("Failed to import command module %s: %s", cmd_name, e)
            return None
        except Exception as e:
            logger.error("Error loading command %s: %s", cmd_name, e)
            return None

        # Find Click command in module (prefer Group over Command)
        discovered = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, click.Group):
                return attr
            if isinstance(attr, click.Command) and discovered is None:
                discovered = attr
        return discovered

@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="wmk")
# ... rest of main() unchanged
```

Remove the old `_discover_commands()` function and the `_discover_commands()` call
at module scope.

**Error handling**: Both `ImportError` and generic `Exception` are caught with
appropriate log levels, matching the current `_discover_commands()` behavior
(lines 121-124 of `cli.py`).

**Verification**:
```bash
time uv run wmk --help
# Should list commands without importing any command module
```

## Step 3: ONNX Config Registration Guard

**Addresses**: Lazy loading breaks `@register_onnx_overwrite` decorator side effects.

**MUST ship with Step 1** — without this guard, `wmk export` would fail to find
custom ONNX configs for models like BERT, CLIP, DETR, SAM.

**Problem**: With lazy loading, HF model files are never imported until explicitly
accessed. Their `@register_onnx_overwrite` decorators never fire. These files also
have direct optimum imports (e.g., `bert.py:15-19` imports `BertOnnxConfig`), so
the entire `models/hf/` package must be treated as a unit.

**What**: Add an idempotent trigger in `export/io.py` (near TasksManager usage):

```python
# modelkit/export/io.py

_hf_models_registered = False

def ensure_hf_models_registered() -> None:
    """Trigger HF model ONNX config registrations. Idempotent."""
    global _hf_models_registered
    if _hf_models_registered:
        return
    from modelkit.models import hf as _hf  # noqa: F401 — triggers decorators
    _hf_models_registered = True
```

**Call sites** — only the genuinely unguarded path needs the explicit guard:
- `export/io.py` → `_get_onnx_config_constructor()` — **REQUIRED**, this is the only
  path that reaches `TasksManager.get_exporter_config_constructor()` without going
  through `models/__init__.py` first.

The following paths are **already safe** through transitive imports but get the guard
defensively:
- `config/build.py` → `generate_build_config()` — already imports
  `MODEL_BUILD_CONFIGS` at line 413 which triggers `models/hf/__init__.py`
- `inspect/resolver.py` — imports from `..models` at top level (lines 16-21)
  which triggers `models/__init__.py` → `models/hf/__init__.py`

**Verification**:
```bash
# Export must still find custom ONNX configs
uv run wmk export -m prajjwal1/bert-tiny -o temp/test_export
# Config generation must still work
uv run wmk config -m microsoft/resnet-50 --device npu --precision int8
# Inspect must still work
uv run wmk inspect -m google-bert/bert-base-uncased
```

## Implementation Order

```
Step 2 (independent, low risk) → measure
    ↓
Steps 1 + 3 (atomic pair) → measure → full test suite
```

Steps 1 and 3 MUST ship together. Step 2 is truly independent and can be done first.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Circular imports from changed load order | Test incrementally; existing workarounds documented in PRD |
| `@register_onnx_overwrite` not called | Step 3 — explicit guard; verify with `wmk export` |
| `_warnings.py` runs too late | Kept as eager import in `__init__.py` |
| `export/__init__.py` eagerly imports `io.py` → optimum | Verify no CLI path reaches `export/__init__.py`; make lazy if needed |
| `dir(modelkit)` incomplete | `__dir__` override returns `__all__` |
| Broken command modules crash CLI | `LazyGroup.get_command()` catches both ImportError and Exception |
| Future dev re-introduces eager import | Add regression test (see verification criteria) |

## Verification Criteria

1. `wmk --help` < 2s
2. `wmk sys --format compact` < 3s
3. `from modelkit import WinMLAutoModel` still works
4. `wmk export` finds custom ONNX configs (BERT, CLIP, DETR, SAM)
5. `wmk config -m microsoft/resnet-50 --device npu --precision int8` produces correct config
6. No test regressions
7. `wmk build`, `wmk inspect`, `wmk perf` work normally
8. Automated regression test: `import modelkit` does NOT pull torch/transformers into `sys.modules`
