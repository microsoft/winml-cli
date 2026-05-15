# Review: `src/winml/modelkit/sysinfo/__init__.py`

**Status:** modified (export rename only)
**Lines added/removed:** 2+ / 2-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/sysinfo/__init__.py`

---

## 1. Purpose of this file

`sysinfo/__init__.py` is the public surface of the `sysinfo` package. It re-exports the package's public symbols (`get_ep_device_map`, `resolve_device_category`, `CPU`, `GPU`, `NPU`, `OS`, `SysInfo`) so external code can import from `winml.modelkit.sysinfo` without knowing the internal module structure. This change updates the export of `resolve_device` to `resolve_device_category` to match the rename in `device.py`.

---

## 2. Changes summary

- Line 5: `from .device import get_ep_device_map, resolve_device` → `from .device import get_ep_device_map, resolve_device_category`.
- Line 18: `__all__` updated: `"resolve_device"` → `"resolve_device_category"`.
- No other changes.

---

## 3. Per-symbol review

### `__all__` and import line

- **Role:** Package public API declaration. Ensures `from winml.modelkit.sysinfo import *` and `from winml.modelkit.sysinfo import resolve_device_category` both work.
- **Behavior:** Both the import line (line 5) and `__all__` (line 18) are updated. This is the correct two-place update for a rename — updating only one of the two would create an inconsistency between `import *` behavior and direct named imports.
- **Invariants:** `resolve_device` is no longer exported from this package. Any code using `from winml.modelkit.sysinfo import resolve_device` will get an `ImportError` immediately.
- **Risks / concerns:** None. The rename is consistent between the import and `__all__`. Any missed call site will fail loudly with `ImportError` or `AttributeError`, making the migration straightforward to verify.
- **Tests:** Covered by the sysinfo test suite (`tests/unit/sysinfo/test_device.py`) and indirectly by any test that imports from `winml.modelkit.sysinfo`.

---

## 4. Cross-cutting concerns

**Spec drift:** None. Spec §3.2 says "Rename it to `resolve_device_category` in this PR" — done.

**Deferred work:** None.

**Dependencies:** `device.py` (the source of the renamed symbol).

---

## 5. Confidence level

**High.**

Two-line rename; no logic; correctly updates both the `from .device import` line and `__all__`.

---

## 6. Verbatim risk inventory

No issues.
