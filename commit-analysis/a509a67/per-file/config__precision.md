# src/winml/modelkit/config/precision.py

## TL;DR
The module is purged of its own EP/device taxonomy tables and helper, in favour of the new `session/ep_device.py` single source of truth. Deleted: `_DEVICE_TO_PROVIDER` dict, `get_provider_for_device()` function, `_EP_TO_DEVICE` dict, the locally-defined `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())`, and `_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})`. Added: a top-level `from ..session import VALID_DEVICES, VALID_EPS, default_ep_for_device, ep_to_device, short_ep_name`. The pure-logic precision-resolution flow is unchanged — every reference to the old privates is swapped for the equivalent session helper, plus the same explicit `"cpu" → None` post-mapping seen in `config/build.py` so that `compile_provider=None` keeps meaning "no compile stage" even though `default_ep_for_device("cpu")` returns `"CPUExecutionProvider"`.

## Diff metrics
- Net removal: ~40 LOC (the three local taxonomy blocks, comments included).
- Additions: 12 LOC of imports (with a multi-line `from ..session import ...`), a 6-line replacement around the EP→provider resolution, and two narrowed re-uses (`device not in VALID_DEVICES` and `ep_to_device(ep)`).
- `PrecisionPolicy` dataclass unchanged in shape; one docstring word ("`qnn`, `dml`, or None" → "Short EP name (e.g. `qnn`, `dml`) or None for CPU") updated.

## Role before vs after
- **Before:** Authoritative owner of two parallel taxonomy maps (`_DEVICE_TO_PROVIDER`, `_EP_TO_DEVICE`) plus a small set of validation primitives (`VALID_EPS`, `_VALID_DEVICES`). External callers (notably `config/build.py`) imported `get_provider_for_device` from here. The module mixed pure precision-resolution logic with this taxonomy.
- **After:** Pure precision-resolution module. Taxonomy is *consumed* via the session facade. The deletion is explicit ("not re-exported from `config/__init__.py`; callers must use `from ..session import ...`") per the docstring comment block.

## Symbol-level changes
- **Deleted (file-private but externally imported in one case):**
  - `_DEVICE_TO_PROVIDER: dict[str, str | None]` — the `{"npu": "qnn", "gpu": "dml", "cpu": None}` direct map.
  - `def get_provider_for_device(device: str) -> str | None` — wrapper over the above. Was imported by `config/build.py` inline.
  - `_EP_TO_DEVICE: dict[str, str]` — the `{"qnn": "npu", "vitisai": "npu", "dml": "gpu", ...}` reverse map. Pre-commit this was the canonical home; the commit-body directive explicitly lists `_EP_TO_DEVICE` as "do not import outside session/ep_device.py".
  - `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())` — replaced by the session re-export of the same name (derived structurally from `EP_DEVICE_SPECS`).
  - `_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})` — replaced by the session re-export `VALID_DEVICES`.
- **New top-level import block (l.17-27):**
  ```
  from ..session import (
      VALID_DEVICES,
      VALID_EPS,
      default_ep_for_device,
      ep_to_device,
      short_ep_name,
  )
  ```
  With a docstring-grade comment block stating "single source of truth lives in session/ep_device.py" and "NOT re-exported from config/__init__.py".
- **`resolve_precision()` body, EP-validity check (~l.225-231):** `if ep not in VALID_EPS: raise ValueError(...)` — same logic but `VALID_EPS` is now the session-exported frozenset. Auto-deduction `device = _EP_TO_DEVICE[ep]` → `device = ep_to_device(ep)`. The local error message that listed `_VALID_DEVICES` is updated to `VALID_DEVICES`.
- **`resolve_precision()` body, device-validity check (~l.244-247):** `if device not in _VALID_DEVICES: ...` → `if device not in VALID_DEVICES: ...`. Same control flow.
- **`resolve_precision()` body, compile_provider derivation (~l.268-277):**
  ```
  # was
  compile_provider = ep if ep else _DEVICE_TO_PROVIDER.get(resolved_device)

  # now
  if ep:
      compile_provider: str | None = ep
  else:
      _canonical = default_ep_for_device(resolved_device)
      _short = short_ep_name(_canonical) if _canonical is not None else None
      compile_provider = _short if _short != "cpu" else None
  ```
  With an inline comment explaining the "cpu → None" rewrite (CPUExecutionProvider has no compile step; `compile_provider=None` is the historical sentinel for "skip compile").
- **`PrecisionPolicy.compile_provider` docstring** widened to "Short EP name (e.g. `qnn`, `dml`) or None for CPU".
- **`resolve_precision()` `available_devices` docstring** updated from "from `resolve_device()`" to "from `sysinfo.get_available_devices()`". The `available_devices: list[str]` parameter shape is unchanged; only the documented producer changed (callers in `config/build.py` now build the list directly via `get_available_devices()`).

## Behavior / contract changes
- **Return-value contracts of `resolve_precision` / `resolve_quant_types` / `is_quantized_precision` / `_pick_device_for_precision` are unchanged.** All public behaviour preserved.
- **`PrecisionPolicy.compile_provider`'s set of values is structurally identical to before**, by virtue of:
  - QDQ NPU path: `ep_to_device(ep)` returns `"npu"`, then `default_ep_for_device("npu")` returns `"QNNExecutionProvider"`, then `short_ep_name(...)` returns `"qnn"` ⇒ matches pre-commit `_DEVICE_TO_PROVIDER["npu"] = "qnn"`.
  - GPU path: yields `"dml"` ⇒ matches pre-commit.
  - CPU path: yields `None` (via explicit guard) ⇒ matches pre-commit.
  - Explicit `ep` parameter: passed through verbatim ⇒ matches pre-commit.
- **Validation error messages now list the catalog-derived sorted EP/device names**, which may include more EPs than the old `_EP_TO_DEVICE` had (e.g. `nv_tensorrt_rtx`). Cosmetic UX change.
- **Module import side-effect.** `from ..session import ...` at top level now means importing `precision.py` pulls in the session module (and transitively onnxruntime). Previously this module was pure-Python with no third-party deps. Tests that mock onnxruntime or have lightweight precision-only test files will now pay the import cost.

## Cross-file impact
- `config/build.py` already drops its `from .precision import get_provider_for_device` import in this same commit, replacing it with the structural derivation via the session facade. The deletion of `get_provider_for_device` here is the symmetric removal.
- `config/build.py` uses the same "cpu → None" guard pattern — duplication noted in the per-file report for that module.
- The `from ..session import ...` at module top-level becomes a permanent dependency from `config/` to `session/`. Any future refactor that moves `session` to depend on `config` (it doesn't today) would create a cycle.
- The four imported public names (`VALID_DEVICES`, `VALID_EPS`, `default_ep_for_device`, `ep_to_device`, `short_ep_name`) all appear in `winml.modelkit.session.__init__.__all__`, satisfying the project's "no reaching into internal submodules" rule from `CLAUDE.md`.

## Risks / subtleties
- **The duplicated `"cpu" → None` translation** between this file and `config/build.py` is a smell. If a future EP gets a short name like `"cpu_arm"` or similar, this string compare won't catch it. Probably a `default_ep_for_device` should grow an `or_none_for_cpu: bool = False` parameter — or the spec for CPU could be marked "no-compile" structurally.
- **`ep_to_device(ep)` now raises `ValueError` for unknown EPs** (replacing the old `_EP_TO_DEVICE[ep]` `KeyError`). The earlier validation `if ep not in VALID_EPS: raise ValueError(...)` should catch this case first, so the swap is safe — but it depends on `VALID_EPS == set(ep_to_device's domain)`, which is true today by construction but not guaranteed by any test in this diff.
- **Annotation on `compile_provider: str | None = ep`** inside an `if`-branch is a PEP 526 form that creates a new local binding; pre-commit it was a single-expression assignment without annotation. Mypy/ruff should accept both; minor stylistic note.
- **The `available_devices=` docstring update is documentation-only.** The function still consumes a `list[str]` and doesn't care whether it came from the new or old name. Aliasing risk is minimal.

## Open questions / TODOs surfaced
- Should `compile_provider` be promoted to an `EPDevice` (or even a canonical full EP name) instead of a short string? The current type carries an implicit "None means skip compile" sentinel — a cleaner contract would use a tagged enum.
- Should `default_ep_for_device("cpu")` return `None` so that consumers don't have to special-case `"cpu"`? Today this string compare is duplicated in two files; either both should call a shared helper or the underlying catalog should distinguish "no-compile EP" structurally.
- The `_LLM_TASKS`, `_AUTO_PRECISION`, `_WEIGHT_TYPE`, `_ACTIVATION_TYPE`, `_BITS_TO_*` tables remain locally defined here. They're pure-precision concerns and rightly belong here, but consistency with the EP-catalog-as-SSOT principle suggests a similar audit for the precision tables — out of scope for this commit.
