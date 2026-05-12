# EP+Device Selection Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace WinMLSession's non-deterministic `_find_ep_device(ep_name)` first-match selection with explicit `EPDevice` descriptors and migrate to ORT 1.23+'s register-based `add_provider_for_devices` flow. `wmk perf` on convnext with QNN+NPU must work end-to-end.

**Architecture:** Frozen `EPDevice` dataclass (pure data, JSON-serializable, no ORT dependency) carries `(ep, device, vendor_id, device_id, vendor)`. `resolve_device(ep, device)` builds it from `ort.get_ep_devices()`; `WinMLSession.__init__` accepts it; `_build_session_options` re-resolves to `OrtEpDevice` and binds via `add_provider_for_devices`. `EPMonitor` integrates via `perf(monitor=...)` (not the ctor) — today's save/restore lifecycle preserved.

**Tech Stack:** Python 3.11+, onnxruntime 1.23+, pytest, ruff.

---

## File Structure

**Create:**
- `src/winml/modelkit/session/ep_device.py` — `EPDevice` dataclass, `resolve_device`, `expand_ep_name`, `canonicalize_ep_name` stub, exception types.
- `tests/unit/session/test_ep_device.py` — EPDevice, resolve_device, expand_ep_name tests.
- `tests/unit/session/test_build_session_options.py` — `_build_session_options` + `_build_provider_options` tests.
- `tests/unit/architecture/test_winml_session_ctor.py` — regression test for hard-break signature.

**Modify:**
- `src/winml/modelkit/session/ep_registry.py` — ADD `register_ep(name) -> list[OrtEpDevice]` method (additive).
- `src/winml/modelkit/session/session.py` — rewrite `WinMLSession.__init__` (hard break), add `_build_session_options` / `_build_provider_options` / `_ep_defaults` free functions, refactor `perf()` to call them while preserving save/restore.
- `src/winml/modelkit/sysinfo/device.py` — rename `resolve_device` → `resolve_device_category`; update callers.
- `src/winml/modelkit/cli/*.py` — every `WinMLSession(...)` callsite updated to pass `ep_device=`.
- `tests/unit/session/test_session.py` — fixtures updated for new ctor.
- `tests/**/test_*.py` — sweep for legacy `device="auto"` / `ep="qnn"` ctor patterns; replace with `EPDevice` or `resolve_device(...)`.

**Migration note:** `feat/update-pkg-deps` is NOT yet merged. `canonicalize_ep_name` is stubbed locally in `ep_device.py` with a `# MIGRATION:` comment. After the other PR lands, replace the stub with `from .ep_path import canonicalize_ep_name` in a one-line follow-up.

---

## Task 1: Exceptions + `EPDevice` dataclass

**Files:** Create `src/winml/modelkit/session/ep_device.py` and `tests/unit/session/test_ep_device.py`.

### Steps

- [ ] **1.1** Write failing tests in `tests/unit/session/test_ep_device.py`:

```python
# tests/unit/session/test_ep_device.py
"""Unit tests for EPDevice descriptor and resolution helpers."""

import pytest

from winml.modelkit.session.ep_device import EPDevice


def test_ep_device_round_trip() -> None:
    """EPDevice -> to_dict -> from_dict yields an equal instance."""
    original = EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )
    rehydrated = EPDevice.from_dict(original.to_dict())
    assert rehydrated == original
    assert rehydrated.ep == "QNNExecutionProvider"
    assert rehydrated.device == "npu"
    assert rehydrated.vendor_id == 0x4D4F
    assert rehydrated.device_id == 0x0001
    assert rehydrated.vendor == "Qualcomm"


def test_ep_device_lowercase_invariant() -> None:
    """`device` field is forced to lowercase by __post_init__."""
    ep_device = EPDevice(
        ep="QNNExecutionProvider",
        device="NPU",
        vendor_id=0x4D4F,
        device_id=0x0001,
    )
    assert ep_device.device == "npu"
```

- [ ] **1.2** Run `uv run pytest tests/unit/session/test_ep_device.py -v`. Expected: ImportError on `from winml.modelkit.session.ep_device import EPDevice`.

- [ ] **1.3** Implement minimal `src/winml/modelkit/session/ep_device.py`:

```python
# src/winml/modelkit/session/ep_device.py
"""EPDevice descriptor + resolution helpers + exception taxonomy.

EPDevice is a pure-data identifier for one (EP, hardware-device) target.
It is frozen, JSON-serializable, and has no runtime dependency on ORT.
Construction is performed by resolve_device(...) or rehydrated via
from_dict(...). The OrtEpDevice handle is re-derived inside session.py
at session-build time and never stored on EPDevice itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# --- exceptions ------------------------------------------------------------


class EPNotDiscovered(Exception):
    """EP plugin is not in the catalog or MODELKIT_EP_PATH."""


class EPRegistrationFailed(Exception):
    """ort.register_execution_provider_library raised."""


class DeviceNotFound(Exception):
    """EP registered, but no OrtEpDevice matches the descriptor."""


class AmbiguousMatch(Exception):
    """Multiple OrtEpDevices match the descriptor after dedup (bug signal)."""


class EPMonitorMismatch(Exception):
    """Monitor.ep_name does not agree with EPDevice.ep."""


# --- dataclass -------------------------------------------------------------


@dataclass(frozen=True)
class EPDevice:
    """Pure-data identifier of one (EP, hardware-device) binding target."""

    ep: str
    device: str
    vendor_id: int
    device_id: int
    vendor: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass — must use object.__setattr__ to mutate.
        if self.device != self.device.lower():
            object.__setattr__(self, "device", self.device.lower())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EPDevice:
        return cls(
            ep=d["ep"],
            device=d["device"],
            vendor_id=d["vendor_id"],
            device_id=d["device_id"],
            vendor=d.get("vendor", ""),
        )
```

- [ ] **1.4** Run `uv run pytest tests/unit/session/test_ep_device.py -v` — both tests PASS.

- [ ] **1.5** Run `uv run ruff check --fix src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py`.

- [ ] **1.6** Commit:

```bash
git add src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py
git commit -m "$(cat <<'EOF'
feat(session): add EPDevice descriptor + exception types

Frozen dataclass for the (EP, hardware-device) binding target plus
the five-exception taxonomy used downstream by resolve_device and
_build_session_options.

Constraint: EPDevice is pure data — no ORT runtime in __init__
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 2: `canonicalize_ep_name` stub + `expand_ep_name`

**Files:** Modify `src/winml/modelkit/session/ep_device.py` and `tests/unit/session/test_ep_device.py`.

### Steps

- [ ] **2.1** Append failing tests to `tests/unit/session/test_ep_device.py`:

```python
from winml.modelkit.session.ep_device import expand_ep_name


def test_expand_ep_name_short_form() -> None:
    assert expand_ep_name("qnn") == "QNNExecutionProvider"
    assert expand_ep_name("openvino") == "OpenVINOExecutionProvider"
    assert expand_ep_name("vitisai") == "VitisAIExecutionProvider"
    assert expand_ep_name("migraphx") == "MIGraphXExecutionProvider"
    assert expand_ep_name("nv_tensorrt_rtx") == "NvTensorRtRtxExecutionProvider"
    assert expand_ep_name("dml") == "DmlExecutionProvider"
    assert expand_ep_name("cpu") == "CPUExecutionProvider"


def test_expand_ep_name_passthrough() -> None:
    """Already-canonical names flow through unchanged."""
    assert expand_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"
    assert expand_ep_name("CPUExecutionProvider") == "CPUExecutionProvider"


def test_expand_ep_name_alias_casing() -> None:
    """Mixed-case canonical aliases are normalized."""
    assert expand_ep_name("NvTensorRTRTXExecutionProvider") == "NvTensorRtRtxExecutionProvider"
```

- [ ] **2.2** Run `uv run pytest tests/unit/session/test_ep_device.py -v`. Expected: ImportError on `expand_ep_name`.

- [ ] **2.3** Append to `src/winml/modelkit/session/ep_device.py`:

```python
from typing import Final


# --- canonicalization ------------------------------------------------------

# MIGRATION: After feat/update-pkg-deps merges, replace this stub with
#     from .ep_path import canonicalize_ep_name
# and delete _EP_NAME_ALIASES below. This stub is only the casing-fix
# slice required to keep this PR self-contained.
_EP_NAME_ALIASES: Final[dict[str, str]] = {
    "nvtensorrtrtxexecutionprovider": "NvTensorRtRtxExecutionProvider",
}


def canonicalize_ep_name(name: str) -> str:
    """Normalize a canonical EP name's casing via the alias table."""
    return _EP_NAME_ALIASES.get(name.lower(), name)


_SHORT_TO_CANONICAL: Final[dict[str, str]] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRtRtxExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def expand_ep_name(name: str) -> str:
    """Expand a short EP name to canonical; passthrough if already canonical.

    "xxx" is the short form of "xxxExecutionProvider" (case-folded for
    lookup). Already-canonical names flow through canonicalize_ep_name()
    for casing fixes (e.g. NvTensorRTRTX -> NvTensorRtRtx).
    """
    canonical = _SHORT_TO_CANONICAL.get(name.lower())
    if canonical is not None:
        return canonical
    return canonicalize_ep_name(name)
```

- [ ] **2.4** Run `uv run pytest tests/unit/session/test_ep_device.py -v` — all 5 tests PASS.

- [ ] **2.5** Run `uv run ruff check --fix src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py`.

- [ ] **2.6** Commit:

```bash
git add src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py
git commit -m "$(cat <<'EOF'
feat(session): add expand_ep_name + canonicalize_ep_name stub

Short-form expansion via _SHORT_TO_CANONICAL with passthrough to
canonicalize_ep_name() for already-canonical aliases. The stub
handles only NvTensorRtRtx casing for now.

Directive: Replace canonicalize_ep_name stub with import from .ep_path after feat/update-pkg-deps merges
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 3: `WinMLEPRegistry.register_ep(name)` (additive)

**Files:** Modify `src/winml/modelkit/session/ep_registry.py`. Create or extend `tests/unit/session/test_ep_registry.py`.

### Steps

- [ ] **3.1** Write failing tests in `tests/unit/session/test_ep_registry.py`:

```python
# tests/unit/session/test_ep_registry.py
"""Unit tests for WinMLEPRegistry.register_ep selective registration."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session.ep_device import EPNotDiscovered, EPRegistrationFailed
from winml.modelkit.session.ep_registry import WinMLEPRegistry


@pytest.fixture
def fresh_registry() -> WinMLEPRegistry:
    """Singleton with stubbed catalog + cleared registered set."""
    reg = WinMLEPRegistry.get_instance()
    reg._ep_paths = {"QNNExecutionProvider": "C:/fake/qnn.dll"}
    reg._registered_eps = set()
    return reg


def _fake_ep_device(ep_name: str, dev_type: str) -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice shape used downstream."""
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = dev_type
    d.device.vendor_id = 0x4D4F
    d.device.device_id = 0x0001
    return d


def test_register_ep_happy_path(fresh_registry: WinMLEPRegistry) -> None:
    qnn_devs = [
        _fake_ep_device("QNNExecutionProvider", "NPU"),
        _fake_ep_device("QNNExecutionProvider", "GPU"),
        _fake_ep_device("QNNExecutionProvider", "GPU"),
        _fake_ep_device("QNNExecutionProvider", "CPU"),
    ]
    other = _fake_ep_device("CPUExecutionProvider", "CPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [*qnn_devs, other]
        mock_ort.register_execution_provider_library = MagicMock()
        result = fresh_registry.register_ep("QNNExecutionProvider")
    mock_ort.register_execution_provider_library.assert_called_once_with(
        "QNNExecutionProvider", "C:/fake/qnn.dll"
    )
    assert result == qnn_devs


def test_register_ep_unknown_raises(fresh_registry: WinMLEPRegistry) -> None:
    with pytest.raises(EPNotDiscovered):
        fresh_registry.register_ep("MysteryExecutionProvider")


def test_register_ep_idempotent(fresh_registry: WinMLEPRegistry) -> None:
    qnn = _fake_ep_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [qnn]
        mock_ort.register_execution_provider_library = MagicMock()
        fresh_registry.register_ep("QNNExecutionProvider")
        fresh_registry.register_ep("QNNExecutionProvider")
    assert mock_ort.register_execution_provider_library.call_count == 1


def test_register_ep_failure_wraps(fresh_registry: WinMLEPRegistry) -> None:
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library.side_effect = RuntimeError("dll boom")
        mock_ort.get_ep_devices.return_value = []
        with pytest.raises(EPRegistrationFailed):
            fresh_registry.register_ep("QNNExecutionProvider")
```

- [ ] **3.2** Run `uv run pytest tests/unit/session/test_ep_registry.py -v` — fails on missing `register_ep`.

- [ ] **3.3** Add to `src/winml/modelkit/session/ep_registry.py` (imports at top, method on the class):

```python
# at top of file, alongside existing imports:
import onnxruntime as ort

from .ep_device import EPNotDiscovered, EPRegistrationFailed
```

```python
# new method on class WinMLEPRegistry:
    def register_ep(self, ep_name: str) -> list["ort.OrtEpDevice"]:
        """Register a single discovered EP and return its claimed devices.

        Idempotent: if already registered, returns the current device list
        without re-loading the DLL. Callers must pass canonicalize_ep_name(...)
        on user-supplied names first.

        Raises:
            EPNotDiscovered:      ep_name absent from self._ep_paths.
            EPRegistrationFailed: ort.register_execution_provider_library
                                  raised (original exception chained).
        """
        if ep_name not in self._ep_paths:
            raise EPNotDiscovered(
                f"EP {ep_name!r} not in discovered catalog. "
                f"Known: {sorted(self._ep_paths)}. "
                f"Hint: install the plugin or set MODELKIT_EP_PATH."
            )
        if ep_name not in self._registered_eps:
            dll_path = self._ep_paths[ep_name]
            try:
                ort.register_execution_provider_library(ep_name, dll_path)
            except Exception as exc:  # noqa: BLE001 — wrap and chain
                raise EPRegistrationFailed(
                    f"ort.register_execution_provider_library({ep_name!r}, "
                    f"{dll_path!r}) failed: {exc}"
                ) from exc
            self._registered_eps.add(ep_name)
        return [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
```

- [ ] **3.4** Run `uv run pytest tests/unit/session/test_ep_registry.py -v` — all 4 tests PASS.

- [ ] **3.5** Run `uv run ruff check --fix src/winml/modelkit/session/ep_registry.py tests/unit/session/test_ep_registry.py`.

- [ ] **3.6** Commit:

```bash
git add src/winml/modelkit/session/ep_registry.py tests/unit/session/test_ep_registry.py
git commit -m "$(cat <<'EOF'
feat(session): add WinMLEPRegistry.register_ep for selective registration

Additive method that registers exactly one discovered EP and returns the
OrtEpDevice list it claims. Idempotent. register_to_ort() is unchanged.

Constraint: must not modify feat/update-pkg-deps surface beyond this one method
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 4: `resolve_device(ep, device)`

**Files:** Modify `src/winml/modelkit/session/ep_device.py` and `tests/unit/session/test_ep_device.py`.

### Steps

- [ ] **4.1** Append failing tests to `tests/unit/session/test_ep_device.py`:

```python
from unittest.mock import MagicMock, patch

from winml.modelkit.session.ep_device import (
    AmbiguousMatch,
    DeviceNotFound,
    resolve_device,
)


def _fake_ort_dev(dev_type: str, vendor_id: int, device_id: int) -> MagicMock:
    d = MagicMock()
    d.device.type.name = dev_type
    d.device.vendor_id = vendor_id
    d.device.device_id = device_id
    d.device.vendor = "Qualcomm"
    return d


def test_resolve_device_qnn_npu() -> None:
    devices = [
        _fake_ort_dev("NPU", 0x4D4F, 0x0001),
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("CPU", 0x4D4F, 0x0003),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "npu")
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.vendor_id == 0x4D4F
    assert result.device_id == 0x0001
    assert result.vendor == "Qualcomm"


def test_resolve_device_dedup_qnn_gpu() -> None:
    """Two OrtEpDevices with identical (vendor_id, device_id) collapse to one."""
    devices = [
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "gpu")
    assert result.device == "gpu"
    assert result.device_id == 0x0002


def test_resolve_device_device_not_found_raises() -> None:
    devices = [_fake_ort_dev("NPU", 0x4D4F, 0x0001)]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        with pytest.raises(DeviceNotFound):
            resolve_device("qnn", "gpu")


def test_resolve_device_ambiguous_raises() -> None:
    """Two distinct GPU entries (different device_id) cannot be auto-resolved."""
    devices = [
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("GPU", 0x4D4F, 0x0003),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        with pytest.raises(AmbiguousMatch):
            resolve_device("qnn", "gpu")
```

- [ ] **4.2** Run `uv run pytest tests/unit/session/test_ep_device.py -v` — new tests fail on missing `resolve_device`.

- [ ] **4.3** Append to `src/winml/modelkit/session/ep_device.py`:

```python
def resolve_device(ep: str, device: str) -> EPDevice:
    """Resolve a (user-friendly EP name, device kind) pair to an EPDevice.

    Args:
        ep: User-supplied EP name. Short forms (e.g. "qnn") are expanded
            via expand_ep_name().
        device: "cpu" | "gpu" | "npu" (case-insensitive).

    Raises:
        EPNotDiscovered:      EP plugin not in catalog or MODELKIT_EP_PATH.
        EPRegistrationFailed: ort.register_execution_provider_library raised.
        DeviceNotFound:       EP registered, but no matching OrtEpDevice.
        AmbiguousMatch:       multiple OrtEpDevice match after dedup.
    """
    # Imported lazily to keep the module free of session-time imports.
    from .ep_registry import WinMLEPRegistry  # noqa: PLC0415

    ep_canonical = expand_ep_name(ep)
    device_lower = device.lower()
    devices = WinMLEPRegistry.get_instance().register_ep(ep_canonical)

    matching = [d for d in devices if d.device.type.name.lower() == device_lower]

    # Dedup by (vendor_id, device_id) — handles QNN's duplicate-GPU rows.
    seen: set[tuple[int, int]] = set()
    deduped: list[Any] = []
    for d in matching:
        key = (d.device.vendor_id, d.device.device_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    if not deduped:
        available = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id))
            for d in devices
        ]
        raise DeviceNotFound(
            f"No OrtEpDevice for {ep_canonical} matches device={device_lower!r}. "
            f"Available: {available}"
        )
    if len(deduped) > 1:
        conflicting = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id))
            for d in deduped
        ]
        raise AmbiguousMatch(
            f"Multiple OrtEpDevice match {ep_canonical}+{device_lower} after "
            f"dedup: {conflicting}. This is a registry bug; not a user error."
        )

    chosen = deduped[0]
    return EPDevice(
        ep=ep_canonical,
        device=device_lower,
        vendor_id=chosen.device.vendor_id,
        device_id=chosen.device.device_id,
        vendor=getattr(chosen.device, "vendor", "") or "",
    )
```

Note: the `WinMLEPRegistry` import is deferred. The tests patch `winml.modelkit.session.ep_device.WinMLEPRegistry`, so re-export it at module scope to satisfy the patch target:

```python
# Below the lazy import inside resolve_device, also re-export at module level
# for monkeypatching in tests:
from .ep_registry import WinMLEPRegistry  # noqa: E402, F401  (re-export for tests)
```

Place that re-export at the bottom of the module to avoid a circular-import problem at startup (ep_registry imports from ep_device).

- [ ] **4.4** Run `uv run pytest tests/unit/session/test_ep_device.py -v` — all tests PASS.

- [ ] **4.5** Run `uv run ruff check --fix src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py`.

- [ ] **4.6** Commit:

```bash
git add src/winml/modelkit/session/ep_device.py tests/unit/session/test_ep_device.py
git commit -m "$(cat <<'EOF'
feat(session): add resolve_device(ep, device) -> EPDevice

Single deterministic resolution: expand_ep_name -> register_ep ->
filter by device type -> dedup (vendor_id, device_id) -> EPDevice.
Raises DeviceNotFound on miss, AmbiguousMatch on >1 after dedup.

Constraint: strict 4-tuple matching everywhere, including CPU
Rejected: first-match-by-name | non-deterministic across hosts
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 5: `_ep_defaults` + `_build_provider_options` (three-layer merge)

**Files:** Modify `src/winml/modelkit/session/session.py`. Create `tests/unit/session/test_build_session_options.py`.

### Steps

- [ ] **5.1** Write failing tests in `tests/unit/session/test_build_session_options.py`:

```python
# tests/unit/session/test_build_session_options.py
"""Unit tests for _build_session_options / _build_provider_options."""

from unittest.mock import MagicMock

import pytest

from winml.modelkit.session.ep_device import EPDevice
from winml.modelkit.session.session import (
    _build_provider_options,
    _ep_defaults,
)


@pytest.fixture
def qnn_npu() -> EPDevice:
    return EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )


@pytest.fixture
def cpu_ep() -> EPDevice:
    return EPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=0x8086,
        device_id=0x0000,
    )


def _stub_monitor(prov: dict[str, str], sess: dict[str, str] | None = None) -> MagicMock:
    m = MagicMock()
    m.get_provider_options.return_value = prov
    m.get_session_options.return_value = sess or {}
    return m


def test_build_provider_options_qnn_defaults_only(qnn_npu: EPDevice) -> None:
    """No config, no monitor -> just the EP defaults."""
    opts = _build_provider_options(qnn_npu, ep_config=None, ep_monitor=None)
    assert opts == {"backend_type": "htp"}


def test_build_provider_options_user_overrides_defaults(qnn_npu: EPDevice) -> None:
    """ep_config.provider_options overrides EP defaults."""
    ep_config = MagicMock()
    ep_config.provider_options = {"backend_type": "gpu", "custom_key": "custom_val"}
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=None)
    assert opts["backend_type"] == "gpu"
    assert opts["custom_key"] == "custom_val"


def test_build_provider_options_monitor_overrides_user(qnn_npu: EPDevice) -> None:
    """Monitor wins last — tracing correctness invariant."""
    ep_config = MagicMock()
    ep_config.provider_options = {"profiling_level": "off"}
    monitor = _stub_monitor({"profiling_level": "detailed", "profiling_file_path": "/tmp/x"})
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=monitor)
    assert opts["profiling_level"] == "detailed"
    assert opts["profiling_file_path"] == "/tmp/x"
    assert opts["backend_type"] == "htp"  # default still present


def test_ep_defaults_unknown_ep_returns_empty(cpu_ep: EPDevice) -> None:
    assert _ep_defaults(cpu_ep) == {}
```

- [ ] **5.2** Run `uv run pytest tests/unit/session/test_build_session_options.py -v` — fails on missing imports.

- [ ] **5.3** Add to the top of `src/winml/modelkit/session/session.py` (after existing imports):

```python
from typing import Final

from .ep_device import (
    AmbiguousMatch,
    DeviceNotFound,
    EPDevice,
    EPMonitorMismatch,
    expand_ep_name,
)
from .ep_registry import WinMLEPRegistry
```

Then append the three free functions at module scope (above `class WinMLSession`):

```python
_QNN_BACKEND: Final[dict[str, str]] = {"npu": "htp", "gpu": "gpu", "cpu": "cpu"}


def _ep_defaults(ep_device: EPDevice) -> dict[str, str]:
    """EP-specific defaults driven by ep_device.device.

    Most EPs return {} — they pick up settings via ep_config.provider_options
    and ep_monitor.get_provider_options(). Only EPs that must signal a
    backend/device kind at registration time appear here.
    """
    match ep_device.ep:
        case "QNNExecutionProvider":
            return {"backend_type": _QNN_BACKEND[ep_device.device]}
        case _:
            return {}


def _build_provider_options(
    ep_device: EPDevice,
    ep_config: "EPConfig | None",
    ep_monitor: "EPMonitor | None",
) -> dict[str, str]:
    """Flat provider_options for add_provider_for_devices().

    Three layers, each overrides the previous:
      1. EP-specific defaults from ep_device (e.g. QNN backend_type).
      2. User overrides from ep_config.provider_options.
      3. EPMonitor-required options (e.g. QNN profiling_level).

    Monitor wins last because tracing correctness depends on its options
    actually reaching the EP. Callers who want to disable tracing should
    drop the monitor, not override its keys.
    """
    options: dict[str, str] = _ep_defaults(ep_device)
    if ep_config is not None and getattr(ep_config, "provider_options", None):
        options.update(ep_config.provider_options)
    if ep_monitor is not None:
        options.update(ep_monitor.get_provider_options())
    return options
```

Use forward-reference strings (`"EPConfig | None"`, `"EPMonitor | None"`) so this task doesn't depend on whichever import order session.py already has for `EPConfig` / `EPMonitor`.

- [ ] **5.4** Run `uv run pytest tests/unit/session/test_build_session_options.py -v` — all 4 tests PASS.

- [ ] **5.5** Run `uv run ruff check --fix src/winml/modelkit/session/session.py tests/unit/session/test_build_session_options.py`.

- [ ] **5.6** Commit:

```bash
git add src/winml/modelkit/session/session.py tests/unit/session/test_build_session_options.py
git commit -m "$(cat <<'EOF'
feat(session): add _build_provider_options three-layer merge

EP defaults -> user overrides -> monitor wins last. _ep_defaults
emits QNN backend_type today; every other EP returns {}. Pure
functions, unit-testable without instantiating WinMLSession.

Constraint: monitor wins last — tracing correctness depends on it
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 6: `_build_session_options` (free function)

**Files:** Modify `src/winml/modelkit/session/session.py` and `tests/unit/session/test_build_session_options.py`.

### Steps

- [ ] **6.1** Append failing tests to `tests/unit/session/test_build_session_options.py`:

```python
from unittest.mock import patch

from winml.modelkit.session.ep_device import AmbiguousMatch, DeviceNotFound
from winml.modelkit.session.session import _build_session_options


def _ort_dev(name: str, vid: int, did: int) -> MagicMock:
    d = MagicMock()
    d.device.type.name = name
    d.device.vendor_id = vid
    d.device.device_id = did
    return d


def test_build_session_options_no_monitor_qnn_npu(qnn_npu: EPDevice) -> None:
    chosen = _ort_dev("NPU", 0x4D4F, 0x0001)
    sibling = _ort_dev("GPU", 0x4D4F, 0x0002)
    fake_so = MagicMock()
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so):
        mock_reg.get_instance.return_value.register_ep.return_value = [chosen, sibling]
        result = _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)
    assert result is fake_so
    fake_so.add_provider_for_devices.assert_called_once_with(
        [chosen], {"backend_type": "htp"}
    )
    fake_so.add_session_config_entry.assert_not_called()


def test_build_session_options_monitor_plumbs_session_options(qnn_npu: EPDevice) -> None:
    chosen = _ort_dev("NPU", 0x4D4F, 0x0001)
    monitor = _stub_monitor(
        prov={"profiling_level": "detailed"},
        sess={"session.disable_cpu_ep_fallback": "1"},
    )
    fake_so = MagicMock()
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so):
        mock_reg.get_instance.return_value.register_ep.return_value = [chosen]
        _build_session_options(qnn_npu, ep_config=None, ep_monitor=monitor)
    fake_so.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback", "1"
    )
    fake_so.add_provider_for_devices.assert_called_once()
    args, _ = fake_so.add_provider_for_devices.call_args
    assert args[1]["profiling_level"] == "detailed"


def test_build_session_options_device_not_found_raises(qnn_npu: EPDevice) -> None:
    """Registry returns only a GPU — npu request raises DeviceNotFound."""
    only_gpu = _ort_dev("GPU", 0x4D4F, 0x0002)
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()):
        mock_reg.get_instance.return_value.register_ep.return_value = [only_gpu]
        with pytest.raises(DeviceNotFound):
            _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)


def test_build_session_options_ambiguous_match_raises(qnn_npu: EPDevice) -> None:
    a = _ort_dev("NPU", 0x4D4F, 0x0001)
    b = _ort_dev("NPU", 0x4D4F, 0x0001)  # exact same ids — but list still has 2 entries
    # Pre-deduped EPDevice was supposed to be unique; two raw matches with same ids
    # collapse via the comparison loop. We force ambiguity by using different device_id
    # while keeping the descriptor pointing at one of them.
    a.device.device_id = 0x0001
    b.device.device_id = 0x0001
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()):
        mock_reg.get_instance.return_value.register_ep.return_value = [a, b]
        with pytest.raises(AmbiguousMatch):
            _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)
```

- [ ] **6.2** Run `uv run pytest tests/unit/session/test_build_session_options.py -v` — new tests fail on missing `_build_session_options`.

- [ ] **6.3** Append to `src/winml/modelkit/session/session.py` directly above `class WinMLSession`:

```python
def _build_session_options(
    ep_device: EPDevice,
    ep_config: "EPConfig | None" = None,
    ep_monitor: "EPMonitor | None" = None,
    base_session_options: "ort.SessionOptions | None" = None,
) -> "ort.SessionOptions":
    """Build a fully-bound ort.SessionOptions for one EPDevice target.

    Free function (not a method): pure inputs -> pure outputs.
    Bridges the EPDevice descriptor to an OrtEpDevice handle inline —
    no _select_one / to_ort_ep_device helper.
    """
    so = base_session_options if base_session_options is not None else ort.SessionOptions()

    if ep_monitor is not None:
        for key, value in ep_monitor.get_session_options().items():
            so.add_session_config_entry(key, value)

    devices = WinMLEPRegistry.get_instance().register_ep(ep_device.ep)
    matching = [
        d
        for d in devices
        if d.device.type.name.lower() == ep_device.device
        and d.device.vendor_id == ep_device.vendor_id
        and d.device.device_id == ep_device.device_id
    ]
    if not matching:
        available = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id))
            for d in devices
        ]
        raise DeviceNotFound(
            f"No OrtEpDevice for {ep_device.ep} matches device="
            f"{ep_device.device}, vendor_id=0x{ep_device.vendor_id:x}, "
            f"device_id=0x{ep_device.device_id:x}. Available: {available}"
        )
    if len(matching) > 1:
        raise AmbiguousMatch(
            f"Multiple OrtEpDevices match {ep_device!r} after dedup — "
            f"registry bug. Matched count: {len(matching)}."
        )

    options = _build_provider_options(ep_device, ep_config, ep_monitor)
    so.add_provider_for_devices([matching[0]], options)
    return so
```

- [ ] **6.4** Run `uv run pytest tests/unit/session/test_build_session_options.py -v` — all 8 tests PASS.

- [ ] **6.5** Run `uv run ruff check --fix src/winml/modelkit/session/session.py tests/unit/session/test_build_session_options.py`.

- [ ] **6.6** Commit:

```bash
git add src/winml/modelkit/session/session.py tests/unit/session/test_build_session_options.py
git commit -m "$(cat <<'EOF'
feat(session): add _build_session_options orchestrator

Pure free function. Builds SessionOptions, plumbs monitor's session-level
keys, bridges EPDevice -> OrtEpDevice via register_ep + 4-tuple match,
binds via add_provider_for_devices with the layered provider_options.

Constraint: descriptor -> handle bridge stays inlined (no _select_one helper)
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 7: Rewrite `WinMLSession.__init__` (hard break)

**Files:** Modify `src/winml/modelkit/session/session.py` and `tests/unit/session/test_session.py`.

### Steps

- [ ] **7.1** Write failing tests in `tests/unit/session/test_session.py` (append, don't overwrite existing):

```python
def test_winml_session_accepts_ep_device(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """WinMLSession compiles with an explicit EPDevice and rejects legacy kwargs."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")  # minimal placeholder; ORT is mocked
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.InferenceSession") as mock_sess, \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
    mock_sess.assert_called_once()
    assert sess._ep_device == qnn_npu_ep_device


def test_winml_session_rejects_legacy_ep_kwarg(tmp_path, qnn_npu_ep_device) -> None:
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, ep="qnn")  # type: ignore[call-arg]


def test_winml_session_rejects_legacy_device_kwarg(tmp_path, qnn_npu_ep_device) -> None:
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, device="auto")  # type: ignore[call-arg]
```

Add the fixtures used above near the top of the test file (or in `conftest.py`):

```python
@pytest.fixture
def qnn_npu_ep_device() -> EPDevice:
    return EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )


@pytest.fixture
def fake_ort_npu() -> MagicMock:
    d = MagicMock()
    d.device.type.name = "NPU"
    d.device.vendor_id = 0x4D4F
    d.device.device_id = 0x0001
    return d
```

- [ ] **7.2** Run `uv run pytest tests/unit/session/test_session.py -v` — new tests fail because old ctor accepts `ep=`/`device=`.

- [ ] **7.3** Rewrite `WinMLSession.__init__` in `src/winml/modelkit/session/session.py`. Replace the existing signature and body (the one currently containing `_find_ep_device`, `_EP_NAME_MAP`, `DEVICE_POLICY_MAP`) with:

```python
class WinMLSession:
    """ONNX Runtime session bound to one explicit (EP, device) target."""

    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: EPDevice,
        *,
        ep_config: "EPConfig | None" = None,
        base_session_options: "ort.SessionOptions | None" = None,
    ) -> None:
        self._onnx_path = str(onnx_path)
        self._ep_device = ep_device
        self._ep_config = ep_config
        self._base_session_options = base_session_options

        # Snapshots preserved across perf() entry/exit (see perf()).
        self._provider_options: dict[str, str] = _build_provider_options(
            ep_device, ep_config, None
        )
        self._active_session_option_entries: dict[str, str] = {}
        self._ep: str = ep_device.ep  # legacy alias used elsewhere in the class

        so = _build_session_options(
            self._ep_device,
            self._ep_config,
            None,
            self._base_session_options,
        )
        self._session = ort.InferenceSession(self._onnx_path, sess_options=so)
```

Delete the now-dead members:

- `_EP_NAME_MAP` (class-level dict at module top)
- `DEVICE_POLICY_MAP`
- `_find_ep_device`

(They should disappear in the same edit so the class no longer references them.)

- [ ] **7.4** Run `uv run pytest tests/unit/session/test_session.py -v` — three new tests PASS. **Other session tests will likely break — leave them red for now; Task 11 sweeps fixtures.**

- [ ] **7.5** Run `uv run ruff check --fix src/winml/modelkit/session/session.py tests/unit/session/test_session.py`.

- [ ] **7.6** Commit (test suite will be partially red until Task 11):

```bash
git add src/winml/modelkit/session/session.py tests/unit/session/test_session.py
git commit -m "$(cat <<'EOF'
refactor(session): WinMLSession.__init__ hard break — EPDevice required

Drops device="auto", ep="qnn" kwargs, _EP_NAME_MAP, DEVICE_POLICY_MAP,
_find_ep_device. ep_device is positional-required. Legacy ep= / device=
now raise TypeError. Constructor delegates to _build_session_options.

Note: test suite is partially red after this commit until Tasks 10-11
sweep CLI callsites and test fixtures.

Constraint: hard break (Option A) — no shims, no deprecation period
Rejected: keep ep=/device= kwargs as soft deprecation | doubles surface area
Directive: every WinMLSession(...) callsite MUST pass ep_device= explicitly
Confidence: high
Scope-risk: broad
EOF
)"
```

---

## Task 8: Refactor `WinMLSession.perf()` — new flow + validation, preserve save/restore

**Files:** Modify `src/winml/modelkit/session/session.py` and `tests/unit/session/test_session.py`.

### Steps

- [ ] **8.1** Append failing tests:

```python
def test_perf_validates_monitor_ep_name_match(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Monitor for QNN against an OpenVINO EPDevice -> EPMonitorMismatch."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    openvino_ep = EPDevice(
        ep="OpenVINOExecutionProvider",
        device="npu",
        vendor_id=0x8086,
        device_id=0x0BD0,
    )
    fake_ov = MagicMock()
    fake_ov.device.type.name = "NPU"
    fake_ov.device.vendor_id = 0x8086
    fake_ov.device.device_id = 0x0BD0
    qnn_monitor = MagicMock()
    qnn_monitor.ep_name = "qnn"
    qnn_monitor.get_provider_options.return_value = {}
    qnn_monitor.get_session_options.return_value = {}
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.InferenceSession"), \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ov]
        sess = WinMLSession(onnx_path, ep_device=openvino_ep)
        with pytest.raises(EPMonitorMismatch):
            sess.perf(monitor=qnn_monitor)


def test_perf_preserves_save_restore(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Mid-perf raise must restore _provider_options snapshot."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    bad_monitor = MagicMock()
    bad_monitor.ep_name = "qnn"
    bad_monitor.get_provider_options.return_value = {"oops": "x"}
    bad_monitor.get_session_options.return_value = {}
    bad_monitor.__enter__.side_effect = RuntimeError("boom")
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg, \
         patch("winml.modelkit.session.session.ort.InferenceSession"), \
         patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        snapshot = dict(sess._provider_options)
        with pytest.raises(RuntimeError):
            sess.perf(monitor=bad_monitor)
        assert sess._provider_options == snapshot
        assert sess._ep == "QNNExecutionProvider"
```

- [ ] **8.2** Run `uv run pytest tests/unit/session/test_session.py::test_perf_validates_monitor_ep_name_match tests/unit/session/test_session.py::test_perf_preserves_save_restore -v` — fail (old `perf()` does not validate / uses old merge).

- [ ] **8.3** Locate the existing `WinMLSession.perf(...)` method in `session.py` (around `session.py:670-707` in today's code; the save/restore block is at the heart of it). Replace its body with:

```python
    def perf(self, monitor: "EPMonitor | None" = None, *args, **kwargs):
        """Run a perf window, optionally with an EPMonitor attached.

        Save/restore lifecycle around `self._provider_options`,
        `self._active_session_option_entries`, and `self._ep` is preserved
        from the previous implementation — still needed to protect re-entry
        while the session is rebuilt with monitor options.
        """
        if monitor is not None and getattr(monitor, "ep_name", None):
            if expand_ep_name(monitor.ep_name) != self._ep_device.ep:
                raise EPMonitorMismatch(
                    f"Monitor ep_name={monitor.ep_name!r} expands to "
                    f"{expand_ep_name(monitor.ep_name)!r}, but session is bound "
                    f"to {self._ep_device.ep!r}. Monitor and session must agree."
                )

        saved_sess_entries = dict(self._active_session_option_entries)
        saved_prov = dict(self._provider_options)
        saved_ep = self._ep
        try:
            so = _build_session_options(
                self._ep_device,
                self._ep_config,
                monitor,
                self._base_session_options,
            )
            self._provider_options = _build_provider_options(
                self._ep_device, self._ep_config, monitor
            )
            self._session = ort.InferenceSession(self._onnx_path, sess_options=so)
            if monitor is not None:
                with monitor:
                    return self._run_perf_window(*args, **kwargs)
            return self._run_perf_window(*args, **kwargs)
        finally:
            self._active_session_option_entries = saved_sess_entries
            self._provider_options = saved_prov
            self._ep = saved_ep
            # Rebuild a bare session so post-perf state is clean.
            self._session = ort.InferenceSession(
                self._onnx_path,
                sess_options=_build_session_options(
                    self._ep_device,
                    self._ep_config,
                    None,
                    self._base_session_options,
                ),
            )
```

If the previous `perf()` implementation called an inline benchmark loop rather than a `_run_perf_window` helper, extract that loop into a helper of that name (the body unchanged) or rename the helper above to match whatever the existing internal benchmark entry point is in this file. The point is: the new `perf()` does **only** validation + so-build + save/restore + delegation.

- [ ] **8.4** Run `uv run pytest tests/unit/session/test_session.py -v` — both new perf tests PASS.

- [ ] **8.5** Run `uv run ruff check --fix src/winml/modelkit/session/session.py tests/unit/session/test_session.py`.

- [ ] **8.6** Commit:

```bash
git add src/winml/modelkit/session/session.py tests/unit/session/test_session.py
git commit -m "$(cat <<'EOF'
refactor(session): perf() routes through _build_session_options, preserves save/restore

Validates monitor.ep_name against self._ep_device.ep at entry. Builds
SessionOptions once via _build_session_options (monitor passed inline).
Save/restore lifecycle (saved_sess_entries, saved_prov, saved_ep)
preserved — still needed for re-entry safety while session is rebuilt
with monitor-aware options. finally rebuilds bare session without monitor.

Constraint: save/restore lifecycle (session.py:670-707) is load-bearing for re-entry safety
Rejected: kill save/restore | breaks re-entry if perf() raises mid-window
Confidence: high
Scope-risk: moderate
EOF
)"
```

---

## Task 9: Rename `sysinfo.device.resolve_device` → `resolve_device_category`

**Files:** Modify `src/winml/modelkit/sysinfo/device.py`. Sweep callers.

### Steps

- [ ] **9.1** Write failing test in `tests/unit/sysinfo/test_device.py`:

```python
# tests/unit/sysinfo/test_device.py
"""Unit test for the renamed resolve_device_category helper."""

from winml.modelkit.sysinfo.device import resolve_device_category


def test_resolve_device_category_returns_category_and_eps() -> None:
    """Smoke: function still returns a (category, list) tuple under new name."""
    category, eps = resolve_device_category("auto")
    assert isinstance(category, str)
    assert isinstance(eps, list)
```

- [ ] **9.2** Run `uv run pytest tests/unit/sysinfo/test_device.py -v` — fails on ImportError.

- [ ] **9.3** Edit `src/winml/modelkit/sysinfo/device.py`:

- Find `def resolve_device(device="auto")` near the `:146` mark and rename to `def resolve_device_category(device="auto")`.
- Update its docstring's first line to `"""Resolve a device hint to (category, candidate EP names)."""`.

- [ ] **9.4** Find every caller. Use the Grep tool (not bash grep):

  - Search pattern: `from winml.modelkit.sysinfo.device import resolve_device` (and the alias-import form `from winml.modelkit.sysinfo import device`).
  - Search pattern: `sysinfo.device.resolve_device\b` to catch attribute-style calls.

- [ ] **9.5** For each match returned by the Grep results, open the file and rename the call site to `resolve_device_category`. Be careful: do **not** rename calls to the new `winml.modelkit.session.ep_device.resolve_device` — those are a different function.

- [ ] **9.6** Run `uv run pytest tests/unit/sysinfo/ -v` and `uv run pytest tests/unit/cli/ -v` to confirm callers compile.

- [ ] **9.7** Run `uv run ruff check --fix src/winml/modelkit/sysinfo/device.py tests/unit/sysinfo/test_device.py`.

- [ ] **9.8** Commit:

```bash
git add src/winml/modelkit/sysinfo/device.py tests/unit/sysinfo/test_device.py src/
git commit -m "$(cat <<'EOF'
refactor(sysinfo): rename resolve_device -> resolve_device_category

The unqualified resolve_device name is now claimed by
winml.modelkit.session.ep_device.resolve_device (returns EPDevice).
The sysinfo helper returns (category, ep_names) and is renamed for
disambiguation. All call sites in src/ updated.

Constraint: one unambiguous resolve_device per namespace
Confidence: high
Scope-risk: moderate
EOF
)"
```

---

## Task 10: Update CLI callsites

**Files:** Modify `src/winml/modelkit/cli/*.py` (sweep).

### Steps

- [ ] **10.1** Use the Grep tool with pattern `WinMLSession\(` and `path src/winml/modelkit/cli`. Note every match with file + line.

- [ ] **10.2** For each CLI file containing a `WinMLSession(...)` call:

  - At the top of the file, add: `from winml.modelkit.session.ep_device import resolve_device`.
  - At each call site, identify the existing `ep` and `device` CLI args (today they come from Click/Typer options). Replace:

    ```python
    sess = WinMLSession(model_path, ep=ep_arg, device=device_arg, ep_config=cfg)
    ```

    with:

    ```python
    ep_device = resolve_device(ep_arg, device_arg)
    sess = WinMLSession(model_path, ep_device=ep_device, ep_config=cfg)
    ```

  - If the CLI today only passes `ep=`, supply a sensible default device (`"auto"` is no longer valid). Add a `--device` option to the CLI with default `"npu"` for QNN / VitisAI and `"gpu"` for DML / Nv. Wire it through.

- [ ] **10.3** If a CLI command receives a serialized config containing an `ep_device` dict, prefer `EPDevice.from_dict(cfg["ep_device"])` over `resolve_device(...)` and skip the resolution call entirely.

- [ ] **10.4** Run `uv run pytest tests/unit/cli/ -v`. Expect all tests passing (or, if some CLI tests still hit legacy fixtures, mark them for Task 11).

- [ ] **10.5** Run `uv run ruff check --fix src/winml/modelkit/cli/`.

- [ ] **10.6** Commit:

```bash
git add src/winml/modelkit/cli/
git commit -m "$(cat <<'EOF'
refactor(cli): wire WinMLSession via EPDevice across all callsites

Every wmk command now resolves (ep, device) to an EPDevice at the CLI
boundary and passes ep_device= to WinMLSession. Serialized configs
containing an ep_device dict are rehydrated via EPDevice.from_dict.

Confidence: high
Scope-risk: moderate
EOF
)"
```

---

## Task 11: Test fixture sweep

**Files:** Modify `tests/**/test_*.py` and any `conftest.py` files.

### Steps

- [ ] **11.1** Use the Grep tool to find legacy patterns. Run these searches (each in a separate Grep call):

  - Pattern: `WinMLSession\([^)]*ep="`, path `tests/`, output_mode `content`.
  - Pattern: `WinMLSession\([^)]*ep='`, path `tests/`.
  - Pattern: `WinMLSession\([^)]*device="auto"`, path `tests/`.
  - Pattern: `WinMLSession\([^)]*device='auto'`, path `tests/`.

- [ ] **11.2** For each match, replace the call. Two replacement shapes:

  - **Live-resolution path (slow, requires a real EP):**

    ```python
    from winml.modelkit.session.ep_device import resolve_device
    sess = WinMLSession(path, ep_device=resolve_device("qnn", "npu"), ep_config=cfg)
    ```

  - **Pure-data path (fast, preferred in unit tests):**

    ```python
    from winml.modelkit.session.ep_device import EPDevice
    qnn_npu = EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
    )
    sess = WinMLSession(path, ep_device=qnn_npu, ep_config=cfg)
    ```

    Prefer the pure-data shape. Patching `WinMLEPRegistry.get_instance().register_ep` is the same pattern used in Tasks 6-8.

- [ ] **11.3** If any test was previously calling `session.perf(...)` without a monitor and relying on the legacy ctor: keep the call signature unchanged — `perf()` keeps its existing `monitor=None` default.

- [ ] **11.4** Centralize the QNN-NPU EPDevice fixture in `tests/conftest.py` (or the nearest package-level `conftest.py`):

```python
# tests/conftest.py
import pytest

from winml.modelkit.session.ep_device import EPDevice


@pytest.fixture
def qnn_npu_ep_device() -> EPDevice:
    return EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )
```

- [ ] **11.5** Run `uv run pytest tests/ -v` — confirm all tests pass or are skipped via documented hardware markers.

- [ ] **11.6** Run `uv run ruff check --fix tests/`.

- [ ] **11.7** Commit:

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test(session): update fixtures for EPDevice-based ctor

Legacy device="auto" / ep="qnn" kwargs swept across the test tree;
each call site now passes ep_device= built either via resolve_device
(when a real EP is needed) or an EPDevice literal (preferred for unit
tests, paired with a patched WinMLEPRegistry).

Confidence: high
Scope-risk: moderate
EOF
)"
```

---

## Task 12: Architecture regression test

**Files:** Create `tests/unit/architecture/test_winml_session_ctor.py`.

### Steps

- [ ] **12.1** Write the regression test:

```python
# tests/unit/architecture/test_winml_session_ctor.py
"""Architecture regression — WinMLSession.__init__ stays explicit.

This test guards against accidental revival of the autoep / policy paths.
If a future edit re-adds device="auto" or string-typed ep= kwargs, this
test fails immediately.
"""

from pathlib import Path

import pytest

from winml.modelkit.session.session import WinMLSession


def test_init_requires_ep_device(tmp_path: Path) -> None:
    """Positional ep_device is required."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path)  # type: ignore[call-arg]


def test_init_rejects_legacy_ep_kwarg(tmp_path: Path) -> None:
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, ep="qnn")  # type: ignore[call-arg]


def test_init_rejects_legacy_device_kwarg(tmp_path: Path) -> None:
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, device="auto")  # type: ignore[call-arg]
```

Create the `tests/unit/architecture/__init__.py` (empty) if the directory doesn't already exist.

- [ ] **12.2** Run `uv run pytest tests/unit/architecture/test_winml_session_ctor.py -v` — all 3 tests PASS (because Task 7 already removed those kwargs).

- [ ] **12.3** Run `uv run ruff check --fix tests/unit/architecture/`.

- [ ] **12.4** Commit:

```bash
git add tests/unit/architecture/
git commit -m "$(cat <<'EOF'
test(architecture): regression — WinMLSession rejects legacy ep=/device= kwargs

Guard test that fires immediately if a future edit re-adds the
device="auto" / ep="qnn" surface area. Cheap to maintain, prevents
silent drift back into the two-path constructor that motivated this PR.

Directive: Do not loosen these assertions without revisiting the design spec
Confidence: high
Scope-risk: narrow
EOF
)"
```

---

## Task 13: Full pytest + ruff gate

### Steps

- [ ] **13.1** Run `uv run ruff check --fix src/ tests/`. Expect zero violations after the fix pass. If anything remains, address the diagnostic before continuing.

- [ ] **13.2** Run the full suite:

```bash
uv run pytest tests/ -v 2>&1 | tee /tmp/pytest.out
```

PowerShell equivalent (this repo runs on Windows):

```powershell
uv run pytest tests/ -v *>&1 | Tee-Object -FilePath "$env:TEMP\pytest.out"
```

Expect all green except for known hardware-skip markers (CUDA, DirectML, QNN-NPU hardware) per `CLAUDE.md`.

- [ ] **13.3** Triage any failures:

  - **Missed CLI callsite** — search with the Grep tool, pattern `WinMLSession\([^)]*ep="`, path `src/`.
  - **Missed test fixture** — Grep `WinMLSession\([^)]*ep="`, path `tests/`.
  - **Wrong `_run_perf_window` extraction in Task 8** — read `session.py` around the original `perf()` to verify the benchmark loop body wasn't truncated.
  - **`canonicalize_ep_name` stub mismatch** — confirm `_EP_NAME_ALIASES` only handles the casing-fix slice, not full canonicalization.

- [ ] **13.4** Apply minimal fixes. If any cleanup is needed:

```bash
git add -p
git commit -m "$(cat <<'EOF'
fix(session): residual callsite/fixture cleanup from full-suite sweep

Triaged failures uncovered after Task 12. Pure mechanical follow-ups
(missed call sites, fixture polish). No design changes.

Confidence: high
Scope-risk: narrow
EOF
)"
```

- [ ] **13.5** Re-run `uv run pytest tests/ -v` until all non-hardware-skipped tests are green.

---

## Task 14: E2E gate — `wmk perf` convnext + QNN + NPU

### Steps

- [ ] **14.1** Verify CLI is on PATH:

```bash
uv run wmk --version
```

- [ ] **14.2** Identify a convnext ONNX file. Likely paths in this repo: `temp/convnext/*.onnx`, the artifact produced by `wmk export` on a small convnext model, or the file referenced by the existing op-tracing tests. If no local artifact exists, document the fetch step the user should take (e.g. `uv run wmk export --hf-model facebook/convnext-tiny-224 --task image-classification --out temp/convnext`) — do not gate the plan on producing the artifact.

- [ ] **14.3** Capture today's baseline (do this **before** making destructive changes; if Task 14 is run after the implementation, compare against the committed reference output instead):

```bash
uv run wmk perf temp/convnext/model.onnx --ep qnn --device npu --top-k 5 > /tmp/perf_baseline.txt
```

- [ ] **14.4** Run the new path:

```bash
uv run wmk perf temp/convnext/model.onnx --ep qnn --device npu --top-k 5
```

Expected output:

  - Per-op timings table (top-5 by latency).
  - Hardware (CPU/RAM/NPU-util/NPU-mem) chart.
  - No tracebacks, no `_find_ep_device` references, no `DeviceNotFound`.
  - Same structural shape as `/tmp/perf_baseline.txt` (column order, headers, op-trace format).

- [ ] **14.5** If output differs structurally:

  - Verify Task 8's `perf()` actually calls `_run_perf_window` (the benchmark inner loop). A missing call yields an empty op-trace table.
  - Verify Task 5/6's monitor merge order — provider_options leaking `backend_type` instead of `htp` indicates `_QNN_BACKEND` wasn't keyed by `device`.
  - Verify Task 10's CLI wiring — if the CLI defaults `device=None` and never calls `resolve_device`, the ctor will reject the call.

- [ ] **14.6** No commit needed for verification. Document the outcome in the PR description, including:

  - Top-5 op latencies before / after.
  - Confirmation that the device is now selected deterministically (mention vendor_id / device_id of the chosen `OrtEpDevice`).
  - Reference to `docs/design/session/2026-05-11-ep-device-refactor.md` §6 (Verification plan) for cross-check.

---

## Self-review checklist before saving

Spec coverage:

- §3.1 (EPDevice) — Task 1.
- §3.2 (resolve_device, expand_ep_name) — Tasks 2, 4.
- §3.3 (WinMLSession.__init__ hard break) — Task 7.
- §3.4 (_build_session_options, _build_provider_options, _ep_defaults, perf()) — Tasks 5, 6, 8.
- §3.5 (WinMLEPRegistry.register_ep) — Task 3.
- §3.6 (layering vs feat/update-pkg-deps) — file-structure section + migration note.
- §4 (error taxonomy: EPNotDiscovered, EPRegistrationFailed, DeviceNotFound, AmbiguousMatch, EPMonitorMismatch) — Tasks 1, 3, 4, 6, 8.
- §5 (migration plan / task ordering) — task ordering in this plan.
- §6 (verification plan: E2E + unit + lint + architecture regression + roundtrip + provider-options layering + mismatch + expand + save/restore) — Tasks 5, 6, 7, 8, 12, 13, 14.

Invariants:

- `EPDevice.ep` is canonical (set by `resolve_device` via `expand_ep_name`).
- `EPDevice.device` is lowercase (enforced by `__post_init__`).
- 4-tuple match `(ep, device.type.lower(), vendor_id, device_id)` everywhere.
- `register_to_ort()` is **not** modified.
- `canonicalize_ep_name` stub is replaceable in one line once `feat/update-pkg-deps` merges.
- Save/restore lifecycle in `perf()` is preserved.

No "TBD" / "similar to" placeholders. Every code block is complete and runnable as written (modulo paths and existing scaffolding that the spec assumes is already in place).
