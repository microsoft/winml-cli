# Final Taxonomy Cleanup Plan — 2026-05-13

Closes the 4 remaining audit items in
`docs/design/session/2026-05-13-post-refactor-taxonomy-audit.md`.

---

## 1. Decisions

| # | Decision |
|---|---|
| D1 | **DELETE `get_provider_for_device` + the embedded `_compile_provider` dict** — no backward-compat wrapper; migrate all callers to `short_ep_name(default_ep_for_device(device))`. |
| D2 | **Reorder `EP_DEVICE_SPECS`** so the primary EP per device comes first: DML before OpenVINO for GPU; `CPUExecutionProvider` before QNN-CPU for CPU. |
| D3 | **Add `eps_for_device(device)` helper** — returns canonical EP names targeting a device; replaces the hardcoded `candidate_eps` list in `commands/build.py`. |
| D4 | **Rename `_VALID_DEVICES` → `VALID_DEVICES`** (drop underscore) — it is already used cross-package; the leading underscore implies package-private which is false. |
| D5 | **Replace `sysinfo/device.py:61` duplicate** with an import from the session facade. |
| D6 | **Fix `compiler/cli.py:53` stale `click.Choice`** — replace with `sorted(VALID_EPS)`. |
| D7 | **Single atomic commit** for all six changes. |

---

## 2. Catalog Reordering — Before / After

### Rationale

`default_ep_for_device(device)` uses first-match semantics over `EP_DEVICE_SPECS`.
The current order gives:
- `default_ep_for_device("gpu")` → `"OpenVINOExecutionProvider"` (first GPU in catalog)
- `default_ep_for_device("cpu")` → `"OpenVINOExecutionProvider"` (first CPU in catalog)

Both are wrong for the Windows-ML compile path:
- GPU default on Windows is **DML** (DirectML, the Windows-native GPU EP).
- CPU default should be the **bundled `CPUExecutionProvider`**, not the QNN-CPU shim or OpenVINO-CPU.

The reorder makes first-match semantics align with what every compile-path caller historically expected from `get_provider_for_device`. After reordering, `get_provider_for_device` becomes a one-liner
(`short_ep_name(default_ep_for_device(device))`) and can be deleted.

### Full Before / After Table (all 13 entries)

| Position | Before (current) | After (proposed) | Notes |
|---|---|---|---|
| 0 | `QNNExecutionProvider / npu` | `QNNExecutionProvider / npu` | Unchanged — primary NPU |
| 1 | `OpenVINOExecutionProvider / gpu` | **`DmlExecutionProvider / gpu`** | Moved up — primary Windows GPU |
| 2 | `OpenVINOExecutionProvider / npu` | **`CPUExecutionProvider / cpu`** | Moved up — bundled CPU always available |
| 3 | `OpenVINOExecutionProvider / cpu` | `OpenVINOExecutionProvider / gpu` | Demoted to secondary GPU |
| 4 | `VitisAIExecutionProvider / npu` | `OpenVINOExecutionProvider / npu` | Secondary NPU |
| 5 | `DmlExecutionProvider / gpu` | `OpenVINOExecutionProvider / cpu` | Secondary CPU |
| 6 | `MIGraphXExecutionProvider / gpu` | `VitisAIExecutionProvider / npu` | Secondary NPU |
| 7 | `TensorrtExecutionProvider / gpu` | `MIGraphXExecutionProvider / gpu` | Secondary GPU |
| 8 | `CUDAExecutionProvider / gpu` | `TensorrtExecutionProvider / gpu` | Secondary GPU |
| 9 | `NvTensorRtRtxExecutionProvider / gpu` | `CUDAExecutionProvider / gpu` | Secondary GPU |
| 10 | `QNNExecutionProvider / gpu` | `NvTensorRtRtxExecutionProvider / gpu` | Secondary GPU |
| 11 | `QNNExecutionProvider / cpu` | `QNNExecutionProvider / gpu` | QNN secondary |
| 12 | `CPUExecutionProvider / cpu` | `QNNExecutionProvider / cpu` | QNN secondary |

### Expected Lookup Results AFTER Reordering

| Call | Before | After | Change? |
|---|---|---|---|
| `default_ep_for_device("npu")` | `"QNNExecutionProvider"` | `"QNNExecutionProvider"` | No |
| `default_ep_for_device("gpu")` | `"OpenVINOExecutionProvider"` | **`"DmlExecutionProvider"`** | **YES** |
| `default_ep_for_device("cpu")` | `"OpenVINOExecutionProvider"` | **`"CPUExecutionProvider"`** | **YES** |
| `default_device_for_ep("qnn")` | `"npu"` | `"npu"` | No |
| `default_device_for_ep("openvino")` | `"gpu"` | `"gpu"` | No |
| `default_device_for_ep("dml")` | `"gpu"` | `"gpu"` | No |
| `default_device_for_ep("cpu")` | `"cpu"` | `"cpu"` | No |
| `eps_for_device("npu")` (new helper) | n/a | `{"QNNExecutionProvider","OpenVINOExecutionProvider","VitisAIExecutionProvider"}` | New |

**Important:** `default_device_for_ep` results are unchanged because the reorder
only moves entries across device groups, never reorders the first appearance of an
EP. QNN's first appearance stays at position 0 (npu), DML at position 1 (gpu),
OpenVINO at position 3 (gpu), CPU at position 2 (cpu).

---

## 3. New Helpers

### 3a. `eps_for_device(device: str) -> frozenset[str]`

```python
def eps_for_device(device: str) -> frozenset[str]:
    """All canonical EP names in the catalog that target the given device.

    Replaces the inline ``candidate_eps`` list in ``commands/build.py``.
    Returns canonical (full) names — callers needing short names use
    ``short_ep_name(ep)`` per element.

    Args:
        device: Device category (``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Frozenset of canonical EP names for that device.
        Returns an empty frozenset for unknown devices (no raise).
    """
    return frozenset(s.ep for s in EP_DEVICE_SPECS if s.device == device)
```

**Location:** `src/winml/modelkit/session/ep_device.py`, immediately after
`default_ep_for_device`.

### 3b. `VALID_DEVICES` (renamed from `_VALID_DEVICES`)

```python
# Was:
_VALID_DEVICES: Final[frozenset[str]] = frozenset({s.device for s in EP_DEVICE_SPECS})

# Becomes:
VALID_DEVICES: Final[frozenset[str]] = frozenset({s.device for s in EP_DEVICE_SPECS})
```

**Rationale:** The name `_VALID_DEVICES` is already imported cross-package by
`config/precision.py` and `utils/cli.py`. A leading underscore in Python signals
"do not import this outside the module." Since the convention is already violated
and the name is useful public API, promote it to public by dropping the underscore.

### 3c. `session/__init__.py` — updated re-exports

Add `eps_for_device` and `VALID_DEVICES` to the import and `__all__`.
Remove `get_provider_for_device` and `_VALID_DEVICES` from both.

---

## 4. Per-File Before / After

---

### File A — `src/winml/modelkit/session/ep_device.py`

**Change 1: DELETE `get_provider_for_device` (lines 265–291)**

Before (exact current code):
```python
def get_provider_for_device(device: str) -> str | None:
    """Get the default *compile* provider short name for a resolved device.

    Preserved for backward compatibility.  The compile-provider mapping is a
    subset of the full EP catalog and intentionally differs from the
    :func:`default_ep_for_device` ordering:

    * ``"npu"``  → ``"qnn"``   (first NPU EP in catalog)
    * ``"gpu"``  → ``"dml"``   (Windows-native GPU compile target; preserved
                                from the old ``_DEVICE_TO_PROVIDER`` mapping)
    * ``"cpu"``  → ``None``    (built-in CPU EP; no provider needed)

    Callers that want the first-in-catalog EP for a device should use
    :func:`default_ep_for_device` directly.

    Args:
        device: Resolved device name (``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Provider short name (e.g. ``"qnn"``, ``"dml"``) or ``None`` for CPU.
    """
    _compile_provider: dict[str, str | None] = {
        "npu": "qnn",
        "gpu": "dml",
        "cpu": None,
    }
    return _compile_provider.get(device)
```

After:
```python
# Function DELETED. _compile_provider shadow dict DELETED.
# Callers now use:
#     short_ep_name(default_ep_for_device(device))   → "qnn" | "dml" | "cpu"
# CPUExecutionProvider maps to short name "cpu", not None.
# Callers that want None for CPU must check: provider if provider != "cpu" else None
# (see migration notes for precision.py below)
```

**Change 2: Reorder `EP_DEVICE_SPECS` (lines 166–203)**

Before (current order, abbreviated):
```python
EP_DEVICE_SPECS: Final[tuple[EPDeviceSpec, ...]] = (
    # Order encodes deduction preference per device category:
    #   npu-first: QNNExecutionProvider (highest-throughput NPU EP on Snapdragon)
    #   gpu-first: OpenVINOExecutionProvider (cross-platform, multi-vendor GPU)
    #   cpu-first: QNNExecutionProvider (QNN CPU backend, when available)
    #
    EPDeviceSpec(ep="QNNExecutionProvider", device="npu", default_provider_options={...}),
    # ---- OpenVINO family ----
    # OpenVINO GPU comes before QNN GPU so default_ep_for_device("gpu") == OpenVINO.
    # OpenVINO CPU comes before QNN CPU so default_ep_for_device("cpu") == OpenVINO
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="cpu"),
    # ---- Single-device EPs ----
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu"),
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="MIGraphXExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="TensorrtExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="CUDAExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),
    # ---- QNN secondary devices ----
    EPDeviceSpec(ep="QNNExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="QNNExecutionProvider", device="cpu"),
    # ---- Bundled CPU ----
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu"),
)
```

After (proposed order):
```python
EP_DEVICE_SPECS: Final[tuple[EPDeviceSpec, ...]] = (
    # Order encodes first-match deduction preference per device:
    #   npu-first:  QNNExecutionProvider   (Snapdragon HTP — highest-throughput)
    #   gpu-first:  DmlExecutionProvider   (Windows-native; compile-path default)
    #   cpu-first:  CPUExecutionProvider   (bundled with ORT — always available)
    # Secondary entries follow their primary within each device group.
    #
    # ---- Primary per-device (positions 0-2) ----
    EPDeviceSpec(
        ep="QNNExecutionProvider",
        device="npu",
        default_provider_options={
            # Verified 2026-05-13: +3x throughput on ResNet-50 vs default mode
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    ),
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu"),          # primary GPU
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu"),          # primary CPU
    # ---- OpenVINO family ----
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="cpu"),
    # ---- Other NPU EPs ----
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu"),
    # ---- Other GPU EPs ----
    EPDeviceSpec(ep="MIGraphXExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="TensorrtExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="CUDAExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),
    # ---- QNN secondary devices ----
    EPDeviceSpec(ep="QNNExecutionProvider", device="gpu"),  # TODO: measure
    EPDeviceSpec(ep="QNNExecutionProvider", device="cpu"),
)
```

**Change 3: Rename `_VALID_DEVICES` → `VALID_DEVICES` (line 212)**

Before:
```python
_VALID_DEVICES: Final[frozenset[str]] = frozenset({s.device for s in EP_DEVICE_SPECS})
```

After:
```python
VALID_DEVICES: Final[frozenset[str]] = frozenset({s.device for s in EP_DEVICE_SPECS})
```

All downstream references in this file (`resolve_device` at lines 379, 381) must also change from `_VALID_DEVICES` to `VALID_DEVICES`.

**Change 4: Add `eps_for_device` helper** (insert after `default_ep_for_device`):

```python
def eps_for_device(device: str) -> frozenset[str]:
    """All canonical EP names in the catalog that target the given device.

    Replaces the inline ``candidate_eps`` list in ``commands/build.py``.

    Args:
        device: Device category (``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Frozenset of canonical EP names for that device.  Returns an empty
        frozenset for unknown devices (no raise — callers can check membership).
    """
    return frozenset(s.ep for s in EP_DEVICE_SPECS if s.device == device)
```

---

### File B — `src/winml/modelkit/session/__init__.py`

Before (relevant lines):
```python
from .ep_device import (
    _VALID_DEVICES,
    EP_DEVICE_SPECS,
    VALID_EPS,
    ...
    get_provider_for_device,
    ...
)

__all__ = [
    "EP_DEVICE_SPECS",
    "VALID_EPS",
    "_VALID_DEVICES",
    ...
    "get_provider_for_device",
    ...
]
```

After:
```python
from .ep_device import (
    VALID_DEVICES,          # was _VALID_DEVICES
    EP_DEVICE_SPECS,
    VALID_EPS,
    ...
    eps_for_device,         # NEW
    # get_provider_for_device — DELETED
    ...
)

__all__ = [
    "EP_DEVICE_SPECS",
    "VALID_EPS",
    "VALID_DEVICES",        # was "_VALID_DEVICES"
    ...
    "eps_for_device",       # NEW
    # "get_provider_for_device" — REMOVED
    ...
]
```

---

### File C — `src/winml/modelkit/config/precision.py`

**Caller 1: import line (line 25)**

Before:
```python
from ..session import (
    _VALID_DEVICES,
    VALID_EPS,
    ep_to_device,
    get_provider_for_device,
)
```

After:
```python
from ..session import (
    VALID_DEVICES,
    VALID_EPS,
    default_ep_for_device,
    ep_to_device,
    short_ep_name,
)
```

**Caller 2: call site (line 270)**

Before:
```python
    # EP override takes precedence over device→provider mapping
    compile_provider = ep if ep else get_provider_for_device(resolved_device)
```

After:
```python
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

**Note on semantics:** The old `get_provider_for_device("cpu") == None`. After the
catalog reorder, `default_ep_for_device("cpu") == "CPUExecutionProvider"`, which
maps to short name `"cpu"`. The CPU case must still produce `None` for
`compile_provider` because `WinMLCompileConfig.for_provider(None)` returns `None`
(no compile stage). The two-step conversion above preserves this.

**Caller 3: `_VALID_DEVICES` at lines 244–246**

Before:
```python
        if device not in _VALID_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. Expected one of: {sorted(_VALID_DEVICES)}"
            )
```

After:
```python
        if device not in VALID_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. Expected one of: {sorted(VALID_DEVICES)}"
            )
```

**Docstring update (line 166):**

Before:
```
    compile_provider: "qnn", "dml", or None.
```

After:
```
    compile_provider: Short EP name (e.g. "qnn", "dml") or None for CPU.
```

---

### File D — `src/winml/modelkit/config/build.py`

**Caller (lines 603–614) — the `else` branch of the auto/auto policy:**

Before:
```python
    else:
        # Even in auto/auto mode, set compile provider from detected hardware
        # instead of preserving the hardcoded EPConfig default (#412).
        from .precision import get_provider_for_device

        hw_provider = get_provider_for_device(resolved_device)
        if hw_provider is not None:
            parent_config.compile = WinMLCompileConfig.for_provider(
                hw_provider,
            )
        # When hw_provider is None (CPU-only), keep the default compile config
        # so the pipeline still has a valid compile section.
```

After:
```python
    else:
        # Even in auto/auto mode, set compile provider from detected hardware
        # instead of preserving the hardcoded EPConfig default (#412).
        from ..session import default_ep_for_device, short_ep_name

        _canonical = default_ep_for_device(resolved_device)
        _short = short_ep_name(_canonical) if _canonical is not None else None
        hw_provider = _short if _short != "cpu" else None
        if hw_provider is not None:
            parent_config.compile = WinMLCompileConfig.for_provider(
                hw_provider,
            )
        # When hw_provider is None (CPU-only), keep the default compile config
        # so the pipeline still has a valid compile section.
```

**Note:** `from .precision import get_provider_for_device` becomes a direct import
from `session` — no intermediate re-export through `precision` needed.

---

### File E — `src/winml/modelkit/sysinfo/device.py`

**Change: Delete duplicate `_VALID_DEVICES` definition (line 61)**

Before:
```python
# Valid explicit device values
_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})
```

After:
```python
# Valid explicit device values — canonical set lives in session/ep_device.py.
from ..session import VALID_DEVICES as _VALID_DEVICES  # noqa: PLC0414 (re-export alias)
```

**Rationale for alias:** `sysinfo/device.py` internally references `_VALID_DEVICES`
at line 160:
```python
    if device != "auto" and device not in _VALID_DEVICES:
```
Rather than rename the local reference, import under the old alias to keep the
diff minimal. The `noqa` suppresses the "imported-but-unused-alias" lint warning.

**Alternative:** Rename to `VALID_DEVICES` throughout `sysinfo/device.py` (2 additional
occurrences at lines 160, 162). This is clean but slightly larger diff. Either is
acceptable — recommend the rename for consistency.

The `_EP_DEVICE_MAP` and `_DEVICE_EP_MAP` in `sysinfo/device.py` are NOT touched.
They serve runtime availability detection (a different problem from taxonomy), and
the OpenVINO entry `"npu/gpu/cpu"` is intentionally multi-device for that purpose.

---

### File F — `src/winml/modelkit/commands/build.py`

**Change: Replace hardcoded `candidate_eps` list (lines 369–373)**

Before:
```python
        candidate_eps = [
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
        ]
        for candidate_ep in candidate_eps:
            if registry.is_ep_available(candidate_ep):
                ep = candidate_ep
                logger.info("EP unspecified for build, auto-selecting: %s", ep)
                break
```

After:
```python
        from ..session import eps_for_device

        # Walk NPU-capable EPs in catalog order (QNN first, then OpenVINO, VitisAI).
        # eps_for_device returns a frozenset; sort by catalog position for determinism.
        _npu_eps = sorted(
            eps_for_device("npu"),
            key=lambda e: next(i for i, s in enumerate(EP_DEVICE_SPECS) if s.ep == e),
        )
        for candidate_ep in _npu_eps:
            if registry.is_ep_available(candidate_ep):
                ep = candidate_ep
                logger.info("EP unspecified for build, auto-selecting: %s", ep)
                break
```

**Context note:** The `ep` variable at this site is a **canonical full EP name**
(e.g. `"QNNExecutionProvider"`) — that is what `candidate_ep` assigns and what
downstream callers of `generate_build_config(ep=ep)` expect. `eps_for_device("npu")`
returns canonical names, so this matches without conversion.

**Import addition at top of file:** Add `EP_DEVICE_SPECS` to the existing session
import or import it locally as shown above. Using a local import avoids touching
the module-level import block.

---

### File G — `src/winml/modelkit/compiler/cli.py`

**Change: Replace stale `click.Choice` (line 53)**

Before:
```python
@click.option(
    "--ep",
    "--execution-provider",
    "execution_provider",
    type=click.Choice(["qnn", "cpu", "cuda", "dml"]),
    default="qnn",
    help="Target execution provider",
    show_default=True,
)
```

After:
```python
@click.option(
    "--ep",
    "--execution-provider",
    "execution_provider",
    type=click.Choice(sorted(VALID_EPS)),
    default="qnn",
    help="Target execution provider",
    show_default=True,
)
```

**Import addition at top of file (after existing imports):**
```python
from ..session import VALID_EPS
```

`VALID_EPS` is a frozenset of short names derived from `EP_DEVICE_SPECS`. Using
`sorted(VALID_EPS)` at module load time is idiomatic and matches how
`commands/compile.py:61` already does it.

---

### File H — `src/winml/modelkit/utils/cli.py`

**Change: Rename `_VALID_DEVICES` → `VALID_DEVICES` in import (line 11)**

Before:
```python
from ..session import _VALID_DEVICES, VALID_EPS
```

After:
```python
from ..session import VALID_DEVICES, VALID_EPS
```

The downstream usage at line 16 must also be updated:

Before:
```python
_DEVICE_CHOICES = sorted(_VALID_DEVICES)
```

After:
```python
_DEVICE_CHOICES = sorted(VALID_DEVICES)
```

---

## 5. Tests That Need Updating

All tests in `tests/unit/config/test_precision.py` that assert on
`compile_provider == "dml"` (i.e., that GPU maps to DML) must be **verified, not
changed** — they should still pass because the migration preserves that semantic.
The old `get_provider_for_device("gpu")` returned `"dml"`. After the catalog
reorder, `short_ep_name(default_ep_for_device("gpu"))` also returns `"dml"` (DML
is now primary GPU). No assertion value changes needed.

Similarly, `test_ep_overrides_default_dml` at line 184–190:
```python
    def test_ep_overrides_default_dml(self) -> None:
        """Without ep, gpu maps to dml. With ep='tensorrt', should be tensorrt."""
        default = resolve_precision(device="gpu")
        assert default.compile_provider == "dml"   # still passes after reorder
        override = resolve_precision(device="gpu", ep="tensorrt")
        assert override.compile_provider == "tensorrt"
```
This continues to pass because after the catalog reorder, `default_ep_for_device("gpu")`
→ `"DmlExecutionProvider"` → `short_ep_name` → `"dml"`. No change needed.

The tests that assert `compile_provider == "qnn"` for NPU likewise remain
unaffected — QNN-NPU is still first.

The tests that assert `compile_provider is None` for CPU remain correct — the
migration logic maps `short_ep_name("CPUExecutionProvider") == "cpu"` → `None`.

### Tests referencing `get_provider_for_device` directly: NONE

The audit found **0 test callers** of `get_provider_for_device` (section 2a of
audit doc). No test mocks or patches this function. No test-side changes required
for the deletion.

### Tests referencing `_VALID_DEVICES`: 0 direct references found in tests.

The rename `_VALID_DEVICES` → `VALID_DEVICES` is transparent to tests that import
via the facade (`from winml.modelkit.session import ...`).

### Summary of test files to examine (grep pass before commit)

| File | Check | Expected result |
|---|---|---|
| `tests/unit/config/test_precision.py` | All `compile_provider == "dml"` assertions (lines 50, 187, 188) | Still pass — semantics preserved by catalog reorder |
| `tests/unit/config/test_precision.py` | All `compile_provider == "qnn"` assertions (lines 42–48, 208, 239, 461) | Still pass — QNN-NPU unchanged |
| `tests/unit/config/test_precision.py` | `compile_provider is None` for CPU (lines 56–60, 116) | Still pass — CPU → None mapping preserved |
| `tests/unit/config/test_build.py` | `expect_compile_provider == "dml"` rows (lines 1788, 1790) | Still pass |
| `tests/unit/config/test_build.py` | `expect_compile_provider == "qnn"` rows (lines 1785, 1787, 1796–1798) | Still pass |
| `tests/unit/session/test_ep_device.py` | Any test asserting `default_ep_for_device("gpu")` | **Must update**: old assertion `== "OpenVINOExecutionProvider"` changes to `== "DmlExecutionProvider"` |
| `tests/unit/session/test_ep_device.py` | Any test asserting `default_ep_for_device("cpu")` | **Must update**: old assertion `== "OpenVINOExecutionProvider"` changes to `== "CPUExecutionProvider"` |
| `tests/unit/architecture/test_ep_device_import_rule.py` | String literal `"get_provider_for_device"` used as a negative-test detector | Must add to the "deleted names" detector list |

> **Action before coding:** run `grep -n "default_ep_for_device" tests/unit/session/test_ep_device.py`
> and inspect every assertion against `"gpu"` or `"cpu"` — those will need value updates.

---

## 6. Risks

### R1 — Behavior change for `default_ep_for_device("gpu")` and `("cpu")`

Any code that called `default_ep_for_device` directly (not through
`get_provider_for_device`) will see a different return value after the catalog
reorder. Grep to find all call sites:

```bash
grep -rn "default_ep_for_device" src/ tests/
```

Expected hits (non-exhaustive):
- `session/ep_device.py` — internal (resolver) — uses `device_only` deduction;
  after reorder, `device="gpu"` with no ep deduces `ep="dml"` rather than
  `ep="openvino"`. This is the desired behavior change.
- `session/ep_device.py:385` — `resolve_device` device-only branch — same effect.
- Any test that asserts the return value against `"OpenVINOExecutionProvider"` for gpu/cpu.

### R2 — `sysinfo/device.py` uses `_VALID_DEVICES` locally

After the import-alias change, any linter configured to flag `noqa: PLC0414` must
be checked. The alternative (full rename throughout the file) avoids this at the
cost of 2 extra line changes.

### R3 — `compiler/cli.py` now accepts all VALID_EPS

The old choice list was `["qnn", "cpu", "cuda", "dml"]`. After the fix it includes
`"openvino"`, `"vitisai"`, `"migraphx"`, `"tensorrt"`, `"nv_tensorrt_rtx"`.
The `compiler/cli.py` compile path may not support all EPs. This is acceptable —
the CLI validates input at the click level, but the actual compiler stages
determine what is runnable. The expanded list is more honest than the stale subset.

---

## 7. Verification Gate

```bash
# 1. Lint
uv run ruff check --fix src/ tests/

# 2. Full unit suite — must be green
uv run pytest tests/unit/ --tb=short -q

# 3. Catalog introspection smoke test
uv run python -c "
from winml.modelkit.session import (
    EP_DEVICE_SPECS, VALID_DEVICES, VALID_EPS,
    default_ep_for_device, default_device_for_ep, eps_for_device, short_ep_name,
)
print('Total variants:', len(EP_DEVICE_SPECS))
print('VALID_DEVICES:               ', VALID_DEVICES)
print('default_ep_for_device(npu):  ', default_ep_for_device('npu'))
print('default_ep_for_device(gpu):  ', default_ep_for_device('gpu'))   # must be DmlExecutionProvider
print('default_ep_for_device(cpu):  ', default_ep_for_device('cpu'))   # must be CPUExecutionProvider
print('eps_for_device(npu):         ', sorted(eps_for_device('npu')))  # 3 entries
print('eps_for_device(gpu):         ', sorted(eps_for_device('gpu')))  # 7 entries
"
# Expected:
#   default_ep_for_device(gpu):   DmlExecutionProvider
#   default_ep_for_device(cpu):   CPUExecutionProvider
#   eps_for_device(npu):          ['OpenVINOExecutionProvider', 'QNNExecutionProvider', 'VitisAIExecutionProvider']

# 4. Compile short-name sanity
uv run python -c "
from winml.modelkit.session import default_ep_for_device, short_ep_name
for dev in ('npu', 'gpu', 'cpu'):
    full = default_ep_for_device(dev)
    short = short_ep_name(full)
    provider = short if short != 'cpu' else None
    print(f'  {dev}: full={full!r}, short={short!r}, compile_provider={provider!r}')
"
# Expected:
#   npu: full='QNNExecutionProvider',  short='qnn', compile_provider='qnn'
#   gpu: full='DmlExecutionProvider',  short='dml', compile_provider='dml'
#   cpu: full='CPUExecutionProvider',  short='cpu', compile_provider=None

# 5. Deleted name check — must be 0 hits
grep -rn "get_provider_for_device" src/ tests/
grep -rn "_compile_provider" src/ tests/
grep -rn "_VALID_DEVICES" src/ tests/   # must be 0 (only VALID_DEVICES henceforth)

# 6. Functional smoke (requires hardware or mocked session)
uv run winml perf -m <fp32>.onnx --ep qnn --device npu --iterations 100 --warmup 10
# Expected: Avg ~1.90 ms, Throughput ~525 samp/s (same as Phase 1)

uv run winml compile -m <fp32>.onnx --ep qnn --device npu --output-dir test_post_cleanup
# Expected: produces *_ctx.onnx + *_qnn.bin
```

---

## 8. Questions for User

**Q1 — Catalog reordering (required for D1 to work)**
> Confirm the proposed catalog order: DML as primary GPU (position 1) and
> `CPUExecutionProvider` as primary CPU (position 2). Consequence: `default_ep_for_device("gpu")`
> → `"DmlExecutionProvider"` (was `"OpenVINOExecutionProvider"`), and
> `default_ep_for_device("cpu")` → `"CPUExecutionProvider"` (was `"OpenVINOExecutionProvider"`).
> Any test currently asserting the old values for gpu/cpu must be updated.

**Q2 — `_VALID_DEVICES` → `VALID_DEVICES` rename**
> The underscore convention signals "do not import outside this module" but the
> name is already used by `config/precision.py` and `utils/cli.py`. Confirm the
> rename to `VALID_DEVICES` (no underscore) as a public API name.

**Q3 — `compiler/cli.py` EP choice expansion**
> After the fix, `--ep` in the legacy `compiler/cli.py` will accept all 9 short
> EP names (not just the 4 currently hardcoded). The `compiler/` module may not
> support all of them. Confirm this is acceptable (reject at runtime, not at parse
> time), or specify a curated subset that the compiler module actually supports.

---

## 9. Out of Scope

- Adding `is_compile_target: bool` or `is_npu_capable: bool` fields to
  `EPDeviceSpec` — deferred per design doc §11.
- Replacing the 4 hardcoded `["auto","npu","gpu","cpu"]` device `click.Choice`
  lists in `commands/*.py` with `["auto"] + sorted(VALID_DEVICES)` — a separate
  cosmetic improvement.
- `analyze/runtime_checker/check_ops.py:330-335` hardcoded EP list — accepted as
  intentionally curated for the runtime-checker subprocess boundary.
- Speculative `default_provider_options` for unverified variants (OpenVINO, TensorRT,
  CUDA, etc.) — requires hardware measurement.

---

## 10. Affected File Summary

| File | Change type | Lines affected |
|---|---|---|
| `src/winml/modelkit/session/ep_device.py` | DELETE function, reorder tuple, rename const, add helper | ~265–291 (delete), 166–203 (reorder), 212 (rename), insert after 262 (add) |
| `src/winml/modelkit/session/__init__.py` | Update imports + `__all__` | lines 7–71 |
| `src/winml/modelkit/config/precision.py` | Update import, rewrite line 270, rename `_VALID_DEVICES` refs | lines 21–26, 244–246, 270–271 |
| `src/winml/modelkit/config/build.py` | Replace import + 6-line block | lines 603–614 |
| `src/winml/modelkit/sysinfo/device.py` | Replace line 61 with import | line 61 |
| `src/winml/modelkit/commands/build.py` | Replace 6-line `candidate_eps` block | lines 369–378 |
| `src/winml/modelkit/compiler/cli.py` | Add import, update `click.Choice` | lines 1–19, 53 |
| `src/winml/modelkit/utils/cli.py` | Rename `_VALID_DEVICES` → `VALID_DEVICES` | line 11, 16 |
| `tests/unit/session/test_ep_device.py` | Update assertions for `default_ep_for_device("gpu")` and `("cpu")` | TBD (grep required) |
| `tests/unit/architecture/test_ep_device_import_rule.py` | Add `"get_provider_for_device"` to deleted-names detector | TBD |

**Total: 8 src files + 2 test files = 10 files.**
