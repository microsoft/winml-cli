# Import Time Optimization — PRD

**Status**: Analysis complete, implementation pending
**Date**: 2026-03-18
**Branch**: `mvp`

## Problem Statement

`wmk --help` takes ~30 seconds. Every CLI invocation — including lightweight commands
like `wmk sys`, `wmk --version`, and `wmk --help` — pays the full import cost of
torch, transformers, optimum, diffusers, and sklearn, even though these libraries are
only needed for model export/build/inference.

This is unacceptable UX for a CLI tool. Users expect sub-second response for help and
system info commands.

## Current State (Baseline Measurements)

### Per-Package Import Cost

Measured in isolated subprocesses (`uv run python -c "import X"` per package):

| Rank | Package | Import Time | Used Directly? |
|------|---------|-------------|----------------|
| 1 | `optimum.exporters.tasks` | **6.38s** | Yes (TasksManager) |
| 2 | `diffusers` | **3.34s** | No (pulled in by optimum) |
| 3 | `transformers` | **3.01s** | Yes (loader, export) |
| 4 | `torchvision` | **2.67s** | No (pulled in by transformers) |
| 5 | `torch` | **1.51s** | Yes (inference, export) |
| 6 | `sklearn` | **1.28s** | No (pulled in by transformers) |
| 7 | `onnx` | **0.22s** | Yes (session, compiler) |
| 8 | `onnxruntime` | **0.19s** | Yes (session) |
| 9 | `scipy` | **0.11s** | No (pulled in by sklearn) |
| 10 | `numpy` | **0.08s** | Yes (everywhere) |

**Total deferrable cost**: ~18.2s from top 6 packages.
Packages 7-10 are fast enough (~0.6s combined) — not worth optimizing.

### Import Chain (Why Everything Loads)

```
modelkit/__init__.py  (~30s)
  │
  ├── _warnings.py                           (~0s, fast)
  │
  ├── from .config import WinMLBuildConfig   (~8s)  ─── CHAIN B
  │     └── config/build.py
  │          └── export/config.py → compiler/configs.py → ...
  │
  └── from .models import WinMLAutoModel     (~22s) ─── CHAIN A + C
        └── models/__init__.py
             │
             ├── models/hf/__init__.py                         CHAIN A (22s)
             │    └── bert.py, clip.py, detr.py, sam.py
             │         └── export/io.py line 33:
             │              from optimum.exporters.tasks import TasksManager
             │              (6.4s self + transitively loads everything above)
             │
             └── models/winml/__init__.py                      CHAIN C (2s)
                  └── winml/base.py
                       ├── import torch         (1.5s)
                       ├── import numpy          (0.1s)
                       └── session/session.py
                            ├── import onnx          (0.2s)
                            └── import onnxruntime    (0.2s)
```

**Two import pathways into optimum from HF model files**:
1. `export/io.py` line 33 imports `TasksManager` at module scope because line 54
   needs it immediately: `register_onnx_overwrite = TasksManager.create_register(...)`.
   HF model files import `register_onnx_overwrite` for their decorators.
2. HF model files also **directly** import from `optimum` at top level:
   - `bert.py:15-19`: `from optimum.exporters.onnx.model_configs import BertOnnxConfig, ...`
   - `clip.py:26-30`: `from optimum.exporters.onnx.model_configs import CLIPOnnxConfig, ...`
   - `detr.py:22-23`: `from optimum.exporters.onnx.model_configs import ...`
   - `sam.py:36-41`: `from optimum.exporters.onnx import OnnxConfig, ...`

Even if `export/io.py` were deferred, the HF model files themselves pull in optimum.
The entire `models/hf/` package must be deferred as a unit.

### Additional: CLI Command Discovery

`cli.py` line 128 calls `_discover_commands()` at module load time, which imports
every command module. Some commands have heavy top-level imports:
- `commands/optimize.py`: `import onnx` (0.2s)
- `commands/perf.py`: `import numpy` (0.1s)
- `commands/sys.py`: `from ..sysinfo import OS` (hardware detection)

Note: command discovery cost overlaps with `__init__.py` cost when measured in the
same process. The ~2-4s estimate may be inflated due to double-counting. With a lazy
`__init__.py`, command discovery's incremental cost is likely ~0.5-1s.

## Requirements

### R1: Lightweight CLI Commands Must Be Fast

| Command | Current | Target |
|---------|---------|--------|
| `wmk --help` | ~30s | < 2s |
| `wmk --version` | ~30s | < 1s |
| `wmk sys --format compact` | ~30s | < 3s |

### R2: Heavy Commands Are Unaffected

`wmk export`, `wmk build`, `wmk inspect`, `wmk perf`, `wmk config` — these need
torch/transformers/optimum and will continue to pay the import cost. No regression.

### R3: Library API Unchanged

All public exports must still work via lazy loading on first access:
- `from modelkit import WinMLAutoModel`
- `from modelkit import WinMLBuildConfig`
- `from modelkit import WinMLPreTrainedModel`
- `from modelkit import WinMLModelForImageClassification`

Acceptable since library users need the heavy deps anyway.

### R4: Zero Behavior Change

- No test regressions (baseline: 808 passed, 1 pre-existing failure in static_analyzer)
- ONNX config registrations (`@register_onnx_overwrite`) still work
- Warning filters (`_warnings.py`) still apply before heavy imports
- Existing circular import workarounds remain valid

### R5: No New Dependencies

Use Python stdlib mechanisms only (PEP 562 `__getattr__`). No third-party lazy
import libraries.

## Constraints

- torch, transformers, optimum imports **cannot be removed** — they are required
  for core functionality (export, build, inference)
- `export/io.py` line 54 (`register_onnx_overwrite = TasksManager.create_register(...)`)
  requires `TasksManager` at module scope — this is the registration factory
- HF model files use `@register_onnx_overwrite` decorators at top level — these
  registrations must happen before any `TasksManager.get_exporter_config_constructor()`
  call
- HF model files also **directly import from optimum** at top level (model configs,
  normalized configs, dummy generators) — deferring only `export/io.py` is insufficient;
  the entire `models/hf/` package must be deferred as a unit
- `export/__init__.py` line 16 eagerly imports `from .io import resolve_io_specs` —
  any `from modelkit.export import ...` triggers the full optimum chain
- `_warnings.py` must execute before any heavy package import to suppress noisy
  warnings from transformers/torch/diffusers

## Known Circular Import Workarounds (Must Survive)

| Location | Pattern | Cycle |
|----------|---------|-------|
| `models/__init__.py:50-56` | `__getattr__` for WinMLAutoModel | models → loader → models |
| `config/build.py:413` | Lazy import of MODEL_BUILD_CONFIGS inside `generate_build_config()` | config → models.hf → config |
| `loader/task.py:154,238,314,383,482` | TasksManager imported inside functions (5 call sites) | loader → optimum → heavy deps |

## Out of Scope

- Optimizing upstream package import times (torch, transformers, etc.)
- Changing the public API surface (what users import from `modelkit`)
- Deferring onnx/onnxruntime/numpy imports (fast enough at <0.3s each)

Note: internal module restructuring (e.g., making `__init__.py` files lazy) IS in
scope — only the user-facing import API is preserved unchanged.
