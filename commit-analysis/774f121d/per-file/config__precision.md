# src/winml/modelkit/config/precision.py

## TL;DR

The module is purged of its own EP/device taxonomy tables and helper, in favour of the new `session/ep_device.py` single source of truth.

**Deleted:** `_DEVICE_TO_PROVIDER` dict, `get_provider_for_device()` function, `_EP_TO_DEVICE` dict, the locally-defined `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())`, and `_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})`.

**Added:** a top-level `from ..session import VALID_DEVICES, VALID_EPS, default_ep_for_device, ep_to_device, short_ep_name`.

The pure-logic precision-resolution flow is unchanged — every reference to the old privates is swapped for the equivalent session helper, plus the same explicit `"cpu" → None` post-mapping seen in `config/build.py` so that `compile_provider=None` keeps meaning "no compile stage" even though `default_ep_for_device("cpu")` returns `"CPUExecutionProvider"`. **This file strips out the parallel device-resolution path that previously lived alongside `session/ep_device.py`**, making the catalog the canonical source.

## Diff metrics

- Lines changed: 73 total (per `git show --stat`): -46 deletions / +27 additions.
- Net removal: ~40 LOC (the three local taxonomy blocks, comments included).
- Additions: 12 LOC of imports (with a multi-line `from ..session import ...`), a 6-line replacement around the EP→provider resolution, and two narrowed re-uses (`device not in VALID_DEVICES` and `ep_to_device(ep)`).
- `PrecisionPolicy` dataclass unchanged in shape; one docstring word (`"qnn", "dml", or None` → `Short EP name (e.g. "qnn", "dml") or None for CPU`) updated.

## Role before vs after

- **Before (parent `7a66c024`):** Authoritative owner of two parallel taxonomy maps (`_DEVICE_TO_PROVIDER`, `_EP_TO_DEVICE`) plus a small set of validation primitives (`VALID_EPS`, `_VALID_DEVICES`). External callers (notably `config/build.py`) imported `get_provider_for_device` from here. The module mixed pure precision-resolution logic with this taxonomy. The taxonomy was *also* duplicated in `session/ep_device.py`'s `EP_DEVICE_SPECS` catalog — two sources of truth living side-by-side.
- **After:** Pure precision-resolution module. Taxonomy is *consumed* via the session facade. The deletion is explicit ("not re-exported from `config/__init__.py`; callers must use `from ..session import ...`") per the module-level docstring comment block. **The "parallel device-resolution path" framing is correct:** the local `_EP_TO_DEVICE` / `_DEVICE_TO_PROVIDER` dicts encoded the same information as the `EP_DEVICE_SPECS` catalog in `session/ep_device.py`, just in a different shape. With those dicts gone, `EP_DEVICE_SPECS` is now the only place EP↔device facts live.

## Symbol-level changes

- **Deleted (file-private but externally imported in one case):**
  - `_DEVICE_TO_PROVIDER: dict[str, str | None]` — the `{"npu": "qnn", "gpu": "dml", "cpu": None}` direct map.
  - `def get_provider_for_device(device: str) -> str | None` — wrapper over the above. Was imported by `config/build.py` inline.
  - `_EP_TO_DEVICE: dict[str, str]` — the `{"qnn": "npu", "vitisai": "npu", "dml": "gpu", ...}` reverse map. Pre-commit this was the canonical home; the commit-body directive explicitly lists `_EP_TO_DEVICE` as "do not import outside session/ep_device.py".
  - `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())` — replaced by the session re-export of the same name (derived structurally from `EP_DEVICE_SPECS`).
  - `_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})` — replaced by the session re-export `VALID_DEVICES`.

- **New top-level import block (l.17-27):**
  ```python
  # EP / device taxonomy — single source of truth lives in session/ep_device.py,
  # exposed through the session/ facade.  These names are used within this
  # module's logic; they are NOT re-exported from config/__init__.py
  # (callers must use `from ..session import ...`).
  from ..session import (
      VALID_DEVICES,
      VALID_EPS,
      default_ep_for_device,
      ep_to_device,
      short_ep_name,
  )
  ```
  The docstring-grade comment block is load-bearing: it documents the SSOT directive and the re-export rule. Five names imported; none re-exported.

- **`resolve_precision()` body, EP-validity check (l.224-231):** `if ep not in VALID_EPS: raise ValueError(...)` — same logic but `VALID_EPS` is now the session-exported frozenset. Auto-deduction `device = _EP_TO_DEVICE[ep]` → `device = ep_to_device(ep)`. The local error message that listed `_VALID_DEVICES` is updated to `VALID_DEVICES`.

- **`resolve_precision()` body, device-validity check (l.244-247):** `if device not in _VALID_DEVICES: ...` → `if device not in VALID_DEVICES: ...`. Same control flow.

- **`resolve_precision()` body, compile_provider derivation (l.268-277):**
  ```python
  # was
  compile_provider = ep if ep else _DEVICE_TO_PROVIDER.get(resolved_device)

  # now
  # EP override takes precedence over device→provider mapping.
  # For CPU, default_ep_for_device returns "CPUExecutionProvider" → short name "cpu".
  # compile_provider=None means "no compile stage"; CPUExecutionProvider has no
  # compile step, so map "cpu" (the short name) to None explicitly.
  if ep:
      compile_provider: str | None = ep
  else:
      _canonical = default_ep_for_device(resolved_device)
      _short = short_ep_name(_canonical) if _canonical is not None else None
      compile_provider = _short if _short != "cpu" else None
  ```
  The four-line dance replaces a one-liner. Cost: visual complexity. Benefit: a single SSOT-backed chain instead of two parallel dicts.

- **`PrecisionPolicy.compile_provider` docstring** widened to `Short EP name (e.g. "qnn", "dml") or None for CPU`.

- **`resolve_precision()` `available_devices` docstring** updated from "from `resolve_device()`" to "from `sysinfo.get_available_devices()`". The `available_devices: list[str]` parameter shape is unchanged; only the documented producer changed (callers in `config/build.py` now build the list directly via `get_available_devices()`).

## Behavior / contract changes

- **Return-value contracts of `resolve_precision` / `resolve_quant_types` / `is_quantized_precision` / `_pick_device_for_precision` are unchanged.** All public behaviour preserved.
- **`PrecisionPolicy.compile_provider`'s set of values is structurally identical to before**, by virtue of:
  - QDQ NPU path: `ep_to_device(ep)` returns `"npu"`, then `default_ep_for_device("npu")` returns `"QNNExecutionProvider"`, then `short_ep_name(...)` returns `"qnn"` ⇒ matches pre-commit `_DEVICE_TO_PROVIDER["npu"] = "qnn"`.
  - GPU path: yields `"dml"` ⇒ matches pre-commit.
  - CPU path: yields `None` (via explicit guard) ⇒ matches pre-commit.
  - Explicit `ep` parameter: passed through verbatim ⇒ matches pre-commit.
- **Validation error messages now list the catalog-derived sorted EP/device names**, which may include more EPs than the old `_EP_TO_DEVICE` had (e.g. `nv_tensorrt_rtx`, `migraphx`). Cosmetic UX change.
- **Module import side-effect.** `from ..session import ...` at top level now means importing `precision.py` pulls in the session module (and transitively onnxruntime). Previously this module was pure-Python with no third-party deps. Tests that mock onnxruntime or have lightweight precision-only test files will now pay the import cost. Acceptable tradeoff given the SSOT win, but worth flagging.
- **`default_ep_for_device` can return `None` on headless servers** (per the commit body's "catch RuntimeError from EP_CATALOG.is_compatible" fix). This means `compile_provider` can land on `None` for a non-CPU resolved device — which the downstream `WinMLCompileConfig.for_provider(None)` must handle. If `for_provider` raises on `None`, that's a latent crash on headless builders. (This file does not change `for_provider`; it just exposes a new None-source.)

## Cross-file impact

- `config/build.py` drops its `from .precision import get_provider_for_device` import in this same commit, replacing it with the structural derivation via the session facade. The deletion of `get_provider_for_device` here is the symmetric removal.
- `config/build.py` uses the same "cpu → None" guard pattern — duplication noted in the per-file report for that module.
- The `from ..session import ...` at module top-level becomes a permanent dependency from `config/` to `session/`. Any future refactor that moves `session` to depend on `config` (it doesn't today) would create a cycle.
- The five imported public names (`VALID_DEVICES`, `VALID_EPS`, `default_ep_for_device`, `ep_to_device`, `short_ep_name`) all appear in `winml.modelkit.session.__init__.__all__` (l.42, 43, 68, 69, 74), satisfying the project's "no reaching into internal submodules" rule from `CLAUDE.md`.
- A grep for any external import of `config.precision.get_provider_for_device` or `config.precision._EP_TO_DEVICE` (the only externally-touched private) should now be empty. If any other module reaches into these, the diff would leave a `ImportError` — none seen in the commit, but this is a check worth re-running across the codebase if precision-related tests start failing.

## Risks / subtleties

- **The duplicated `"cpu" → None` translation** between this file and `config/build.py` is a smell. If a future EP gets a short name like `"cpu_arm"` or similar, this string compare won't catch it. Probably a `default_ep_for_device` should grow an `or_none_for_cpu: bool = False` parameter — or the spec for CPU could be marked "no-compile" structurally (a `EPDeviceSpec.no_compile: bool` field).
- **`ep_to_device(ep)` now raises `ValueError` for unknown EPs** (replacing the old `_EP_TO_DEVICE[ep]` `KeyError`). The earlier validation `if ep not in VALID_EPS: raise ValueError(...)` should catch this case first, so the swap is safe — but it depends on `VALID_EPS == set(ep_to_device's domain)`, which is true today by construction (`VALID_EPS = frozenset({short_ep_name(s.ep) for s in EP_DEVICE_SPECS})`) but not guaranteed by any test in this diff.
- **Annotation on `compile_provider: str | None = ep`** inside an `if`-branch is a PEP 526 form that creates a new local binding; the `else` branch's `compile_provider = _short if ...` has no annotation. Mypy/ruff should accept both; minor stylistic note.
- **`available_devices=` docstring update is documentation-only.** The function still consumes a `list[str]` and doesn't care whether it came from the new or old name. Aliasing risk is minimal.
- **`default_ep_for_device(resolved_device)` for `resolved_device == "cpu"` returns `"CPUExecutionProvider"` (not `None`), then `short_ep_name(...)` returns `"cpu"`, then the explicit guard maps to `None`.** This is the load-bearing detail. If anyone "simplifies" the chain to drop the `if _short != "cpu" else None` guard, every CPU build will start invoking `WinMLCompileConfig.for_provider("cpu")` — which produces a non-None compile stage where none was previously emitted.
- **`default_ep_for_device` lazy-imports `WinMLEPRegistry.instance()`** (per `session/ep_device.py:362`), so the first call from `resolve_precision` triggers registry initialization — which may scan plugin EPs on disk. This is invisible from `precision.py` but means a "pure decision logic" function now has a side-effect at first use. The module docstring still claims "No I/O", which is now slightly aspirational.
- **`resolve_precision` no longer raises for "no EP registered for device".** Previously, a CPU-only host calling `resolve_precision(device="npu", ...)` would have produced `compile_provider="qnn"` and let `WinMLCompileConfig.for_provider("qnn")` succeed/fail downstream. Post-commit, `default_ep_for_device("npu")` returns `None` (no registered QNN EP), so `compile_provider=None`. Subtle behaviour shift on misconfigured hosts.

## Open questions / TODOs surfaced

- Should `compile_provider` be promoted to an `EPDevice` (or even a canonical full EP name) instead of a short string? The current type carries an implicit "None means skip compile" sentinel — a cleaner contract would use a tagged enum.
- Should `default_ep_for_device("cpu")` return `None` so that consumers don't have to special-case `"cpu"`? Today this string compare is duplicated in two files; either both should call a shared helper or the underlying catalog should distinguish "no-compile EP" structurally.
- The `_LLM_TASKS`, `_AUTO_PRECISION`, `_WEIGHT_TYPE`, `_ACTIVATION_TYPE`, `_BITS_TO_*` tables remain locally defined here. They're pure-precision concerns and rightly belong here, but consistency with the EP-catalog-as-SSOT principle suggests a similar audit for the precision tables — out of scope for this commit.
- The module docstring says "no I/O, no sysinfo dependency" but `default_ep_for_device` now hits the EP registry (potentially a plugin scan). The docstring is now slightly stale.

## Simplification opportunities

- **The `_canonical → _short → compile_provider` three-step dance** is duplicated between this file (l.275-277) and `config/build.py` (l.610-612). A single `compile_provider_for_device(device: str) -> str | None` helper in the session facade — with the CPU guard built in — would collapse both call sites to one line and remove the duplicated string-compare. Risk: such a helper bakes in the `cpu → None` semantic, which may not be appropriate everywhere.
- **`_pick_device_for_precision` is single-use and 25 LOC** (l.294-319). Inline it into `resolve_precision`. The `for d in available_devices: if d == "npu": return d` patterns reduce to `"npu" in available_devices` checks. With those substitutions the function shrinks to ~5 LOC.
- **Two `for d in available_devices: if d == "<dev>": return d` blocks** (l.309-316) are O(n) lookups against what is typically a 1-3 element list. The membership test `if "<dev>" in available_devices` reads cleaner and is the canonical Pythonic form.
- **Annotation asymmetry** between the two `compile_provider` branches (one annotated `str | None`, one not). Drop the PEP 526 annotation or apply it consistently.
- **The `resolve_precision` body is long enough (~80 LOC)** that the natural decomposition into `_validate_ep(...) -> str | None`, `_validate_device(...) -> str`, `_derive_compile_provider(...) -> str | None` would help readability. The function does five distinct things (validate, infer, pick device, derive compile_provider, resolve quant types) in one block.
- **`resolved_precision = precision.lower() if precision != "auto" else "auto"`** (l.214) is a fancy way to write `precision.lower()` — `"auto".lower() == "auto"`. The conditional is dead weight unless `precision` could be `None`, which the type annotation forbids.
- **`if ep is not None:` then immediately `ep = ep.lower()`** (l.224-225) could be collapsed by accepting only the lowercase form upstream (click already normalises) — but this is a defensive normalisation worth keeping.
- **The deleted `get_provider_for_device` was a one-liner** (`return _DEVICE_TO_PROVIDER.get(device)`). It existed only because `config/build.py` needed it. With both call sites now using `default_ep_for_device + short_ep_name`, the helper rightly vanishes — a textbook over-abstraction removal.
