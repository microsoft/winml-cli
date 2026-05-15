# EP / Device Taxonomy Consolidation Plan

**Date:** 2026-05-13
**Branch:** `feat/op-tracing-refactor`
**Base:** `db39b80d`
**Driver:** Code review summary + taxonomy sweep findings (47 distinct touch points across `src/` and `tests/`).

## Goal

Consolidate all EP / device taxonomy into a single home (`session/ep_device.py`), expose it through the `session/` package facade, eliminate duplicates and inline literals, fix the `winml compile` private-import violation, and make `--ep` / `--device` flags consistently optional across CLIs via a widened `resolve_device()` that deduces partial inputs.

## Decisions (locked)

### Decision A — Placement and import shape

- **Implementation lives at:** `src/winml/modelkit/session/ep_device.py` (existing file, expanded with the moved tables).
- **Public surface exposed via:** `src/winml/modelkit/session/__init__.py` re-exports.
- **Import shape for source code:** `from ..session import EPDevice, resolve_device, ...`
- **Import shape for tests:** `from winml.modelkit.session import EPDevice, resolve_device, ...`
- **Never:** `from ..session.ep_device import ...` (drills past the facade).

### Decision B — `resolve_device` accepts partials

```python
def resolve_device(
    ep: str | None = None,
    device: str | None = None,
) -> EPDevice:
    """Deduction matrix:
        both given      -> validate + return
        ep only         -> _EP_TO_DEVICE[ep] gives device
        device only     -> _DEVICE_TO_PROVIDER[device] gives ep
        neither (None)  -> sysinfo auto-detect: pick the strongest device,
                           then the default EP for it
    """
```

The deduction happens **once, at the boundary** (CLI option resolver or top-level API). Downstream code (`WinMLSession`, `add_provider_for_devices`, `compile()`) receives a concrete `EPDevice` and never infers.

### Decision C — `winml compile` CLI

- Both `--ep` and `--device` are optional (default `None`).
- The CLI passes the values through `resolve_device(ep, device)`.
- No more cross-package private imports inside `compile.py`.

---

## Phase 1 — Taxonomy consolidation (steps 1–6)

Single agent. One commit at the end. Verification gate: `uv run pytest tests/unit/session/ tests/unit/architecture/` green.

### Step 1: Move taxonomy tables into `session/ep_device.py`

From `src/winml/modelkit/config/precision.py` move:

```python
_EP_TO_DEVICE: dict[str, str] = {
    "qnn": "npu", "vitisai": "npu",
    "dml": "gpu", "migraphx": "gpu", "tensorrt": "gpu", "cuda": "gpu",
    "openvino": "gpu",
    "cpu": "cpu",
}
_DEVICE_TO_PROVIDER: dict[str, str | None] = {
    "npu": "qnn", "gpu": "dml", "cpu": None,
}
VALID_EPS = frozenset(_EP_TO_DEVICE.keys())
_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})

def get_provider_for_device(device: str) -> str | None: ...
```

Append all five to `session/ep_device.py`. Mark `_EP_TO_DEVICE` and `_DEVICE_TO_PROVIDER` as module-private (leading underscore). `VALID_EPS`, `_VALID_DEVICES`, `get_provider_for_device` are public.

In `config/precision.py` replace the definitions with:
```python
from ..session import VALID_EPS, _VALID_DEVICES, get_provider_for_device, _EP_TO_DEVICE
```
(The leading underscore on `_EP_TO_DEVICE` becomes acceptable here because `precision.py` is part of the same logical package family that ships the taxonomy — but it's a code smell we accept rather than re-implement validation. Alternative: write a tiny `is_known_ep(name) -> bool` and `is_known_device(name) -> bool` to keep `_EP_TO_DEVICE` truly private. See Step 6 below.)

### Step 2: Widen `resolve_device()` signature

`resolve_device(ep: str | None = None, device: str | None = None) -> EPDevice`

Deduction rules per the matrix in Decision B.

- For `ep only`: lookup `_EP_TO_DEVICE[ep_canonical_or_short]`. Raise `ValueError` if unknown EP.
- For `device only`: lookup `_DEVICE_TO_PROVIDER[device]`. Raise if no default (e.g. `_DEVICE_TO_PROVIDER["cpu"] is None` → use built-in CPU EP).
- For `both None`: call into `sysinfo.resolve_device_category()` to pick the strongest available device, then fall through to `device only` path.

The bundled-CPU EP fallback (commit `9cce0163`) already handles `register_ep("CPUExecutionProvider")` against the catalog.

### Step 3: Re-exports in `session/__init__.py`

Add to the `__all__` and corresponding imports:

```python
from .ep_device import (
    EPDevice,
    EPNotDiscovered,
    EPRegistrationFailed,
    DeviceNotFound,
    AmbiguousMatch,
    EPMonitorMismatch,
    resolve_device,
    expand_ep_name,
    short_ep_name,
    canonicalize_ep_name,
    VALID_EPS,
    _VALID_DEVICES,
    get_provider_for_device,
    # _EP_TO_DEVICE and _DEVICE_TO_PROVIDER stay private — not re-exported.
)
```

### Step 4: Rewrite all internal imports

Affected files (from the taxonomy sweep + earlier reviews):

```
src/winml/modelkit/commands/perf.py                 # from ..session import ...
src/winml/modelkit/commands/eval.py
src/winml/modelkit/commands/config.py
src/winml/modelkit/compiler/stages/compile.py       # was: from ...config.precision import _EP_TO_DEVICE
src/winml/modelkit/config/precision.py              # was: locally defined; now imports
src/winml/modelkit/config/build.py
src/winml/modelkit/eval/evaluate.py
src/winml/modelkit/models/auto.py
src/winml/modelkit/models/winml/base.py
src/winml/modelkit/session/session.py               # from .ep_device → leave as-is OR switch to package-relative
src/winml/modelkit/session/qairt/qairt_session.py
src/winml/modelkit/session/ep_registry.py
src/winml/modelkit/sysinfo/__init__.py              # only if it uses anything from ep_device
```

All non-`session/` imports become `from ..session import …` (single ascent through the facade).

### Step 5: Delete `utils/constants.py` taxonomy entries

Per sweep findings:

```python
# DELETE:
EP_ALIASES = { "ov": "openvino", "vitis": "vitisai", ... }   # 5 entries
SUPPORTED_EPS = ["QNNExecutionProvider", ...]                # 3 entries
SUPPORTED_DEVICES = ["CPU", "GPU", "NPU"]                    # uppercase BUG
```

These conflict with `_SHORT_TO_CANONICAL` (different alias entries) and `_VALID_DEVICES` (uppercase vs lowercase). Replace all callers with the `..session` imports. Specifically:

- `src/winml/modelkit/utils/cli.py:82` — fix the case bug as a free side effect.
- `src/winml/modelkit/commands/analyze.py` — `--device` Click option becomes lowercase consistent with the rest of the codebase.

### Step 6: Eliminate inline duplicates

Four duplicate inline copies of `{"cpu":"cpu","npu":"qnn","gpu":"dml"}`:

```
commands/perf.py:472   → resolve_device(None, device).ep
commands/perf.py:1552  → resolve_device(None, device).ep
eval/evaluate.py:138   → resolve_device(None, device).ep
```

(The `_DEVICE_TO_PROVIDER` itself in `precision.py` is moved by Step 1 — that's the 4th copy and it becomes the single home.)

The `--ep` flag's `click.Choice([…])` lists in 8 commands → consolidate via a constant `_EP_CHOICES = sorted(VALID_EPS)` to be shared.

### Phase 1 verification gate

```
uv run ruff check src/ tests/
uv run pytest tests/unit/session/ tests/unit/architecture/ -x
# Manual smoke (no commit yet):
uv run python -c "from winml.modelkit.session import EPDevice, resolve_device, VALID_EPS, _VALID_DEVICES; print(VALID_EPS); print(resolve_device('qnn', 'npu'))"
uv run python -c "from winml.modelkit.session import resolve_device; print(resolve_device('qnn'))"            # device deduced
uv run python -c "from winml.modelkit.session import resolve_device; print(resolve_device(None, 'npu'))"     # ep deduced
```

If all pass → commit Phase 1.

---

## Phase 2 — CLI integration + cleanup (steps 7–11)

Single agent. One commit at the end. Runs AFTER Phase 1 is committed (so its imports resolve correctly).

### Step 7: `winml compile` CLI flags

`src/winml/modelkit/compiler/cli.py`:

- Add `@click.option("--device", type=click.Choice(["cpu", "gpu", "npu"]), default=None)`
- Make `--ep` optional (`default=None`).
- Inside the command handler, call `resolve_device(ep, device)` once → store the resulting `EPDevice` in `WinMLCompileConfig.ep_device`.

`src/winml/modelkit/compiler/configs.py`:

- Add `ep_device: EPDevice | None = None` field to `WinMLCompileConfig`.
- `to_dict` / `from_dict` serialize/deserialize via `EPDevice.to_dict()` / `EPDevice.from_dict()`.

### Step 8: `CompileStage.process()` reads `ep_device`

`src/winml/modelkit/compiler/stages/compile.py`:

- Drop the `from ...config.precision import _EP_TO_DEVICE` import.
- Read `ep_device` from `CompileContext` (which now carries it via `WinMLCompileConfig.from_dict(context.config).ep_device`).
- Pass `ep_device=ep_device` to `session_cls(...)`.

The `CompileContext.execution_provider` property stays (for backward compat with other stages like `optimize.py` that key off the EP name string), but reads `ep_device.ep` via `short_ep_name(ep_device.ep)`.

### Step 9: Architecture regression test

Add `tests/unit/architecture/test_ep_device_import_rule.py`:

```python
"""Architecture regression: nobody outside session/ may import directly from ep_device.py."""
# Walks src/ AST; flags any `from ...session.ep_device import …` or `import …session.ep_device`
# from files OUTSIDE src/winml/modelkit/session/. session.py itself is the sole exception
# (within-package sibling-relative is allowed).
```

### Step 10: Update tests

- `tests/unit/session/test_ep_device.py` — already at the right path. Adjust imports to `from winml.modelkit.session import ...`.
- `tests/unit/session/test_ep_registry.py` — same.
- `tests/unit/session/test_build_session_options.py` — same.
- `tests/unit/commands/test_perf_cli.py` — mock targets follow the new import paths (`patch("winml.modelkit.commands.perf.resolve_device", ...)` not `…ep_device.resolve_device`).
- `tests/unit/eval/test_eval.py` — same mock-path update.
- `tests/unit/models/auto/test_auto_onnx.py` — same.

### Step 11: Phase 2 verification gate

```
uv run ruff check --fix src/ tests/
uv run pytest tests/unit/ -x          # full unit suite must be green or only have known hardware skips
uv run pytest tests/unit/architecture/    # new regression test must pass
```

E2E confirmation (manual, not gating the commit):

```
uv run winml perf -m <fp32-onnx> --ep qnn --device npu --iterations 3 --warmup 1   # known good
uv run winml perf -m <fp32-onnx> --ep qnn --iterations 3 --warmup 1                # NEW: device deduced
uv run winml perf -m <fp32-onnx> --device npu --iterations 3 --warmup 1            # NEW: ep deduced
uv run winml compile -m <some.onnx> --ep qnn --device npu                          # known good after fix
uv run winml compile -m <some.onnx> --ep qnn                                       # NEW: device deduced
```

If all pass → commit Phase 2.

---

## What this plan does NOT fix

Out of scope (separate PR / commit):

1. **Audit Gap #1 fallout in `models/auto.py`** — the `device=` string callers. Likely the `winml perf -m microsoft/resnet-50` exit 127 has its root cause here. Needs its own diagnostic + commit.
2. **Audit Gap #3 — legacy `WinMLSession._build_session_options` instance method.** Causes `winml compile` exit 1 (no output file). Needs separate refactor pass.
3. **Native QNN HTP AOT crashes** on QDQ-quantized graphs (the cmd 5 / compile crash). Out of this codebase's control — upstream QNN SDK issue.
4. **Analyze command's slow probing** when rule zip is missing (cmd 2 hang). Tracked separately.

---

## Risks

- **Circular imports.** Putting taxonomy in `session/ep_device.py` means `config/precision.py` now imports from `..session`. If `..session` (via `session/__init__.py`) transitively imports `..config.precision` for any reason, we get a cycle. Mitigation: keep `session/__init__.py` minimal; never have it import from `..config`.
- **Stale `_EP_TO_DEVICE` references** in test fixtures we don't catch. Mitigation: run full pytest at the end of each phase; ruff lint catches unused imports.
- **CLI behavior surprises** if `resolve_device` deduction picks an unexpected default. Mitigation: log the deduced `EPDevice` at INFO level in CLI handlers ("Resolved to: EPDevice(ep='qnn', device='npu')") so users see what got picked.

## Rollback plan

Each phase commits separately. If Phase 2 breaks something, `git revert HEAD` reverts only the CLI/compile changes; Phase 1's taxonomy consolidation stays intact (it's strictly an improvement).
