# EP+Device Refactor — Implementation Status (2026-05-12)

**Spec:** `docs/design/session/2026-05-11-ep-device-refactor.md` v1.2
**Plan:** `docs/plans/2026-05-11-ep-device-refactor-plan.md`
**Branch:** `feat/op-tracing-refactor`
**Status:** Tasks 1-10 landed (with one notable deviation in Task 8/10 reconciled by `54cb6e81`); Task 11 fixture sweep in flight; Tasks 12-14 pending. The core new API surface is complete and unit-tested, but **three pre-existing call sites still pass the legacy `device=` string** to constructors that now require `ep_device`, which will crash at runtime once exercised. See §3 / §7.

---

## 1. Spec coverage matrix

| Spec § | Requirement | Status | Commit | Evidence |
|---|---|---|---|---|
| §3.1 | `EPDevice` frozen dataclass + `__post_init__` lowercase + `to_dict`/`from_dict` | DONE | `40c68991` | `src/winml/modelkit/session/ep_device.py:49-77` |
| §3.2 | `resolve_device(ep, device)` + `expand_ep_name` + `_SHORT_TO_CANONICAL` + `canonicalize_ep_name` stub | DONE | `98b33dcb`, `b12b0eec`, `0b20f22c` | `ep_device.py:86-117` (helpers), `:140-195` (resolver) |
| §3.2 | `sysinfo/device.py::resolve_device` renamed to `resolve_device_category` (namespace collision) | DONE | `d9ddf71e`, `e033571a` | `src/winml/modelkit/sysinfo/device.py:146`, `sysinfo/__init__.py:5,18` |
| §3.3 | `WinMLSession.__init__` hard-break — `ep_device: EPDevice` required, no `device=`/`ep=`/policy/auto | DONE | `5b86e8a9`, `d8200837`, **`54cb6e81`** (shim removal) | `session/session.py:211-269` |
| §3.3 | Removed `_EP_NAME_MAP`, `DEVICE_POLICY_MAP`, `_find_ep_device` | DONE | `5b86e8a9` | verified absent in src tree (grep) |
| §3.4 | `_build_session_options` private **free function** with inlined descriptor→handle bridge | DONE | `ce6d4db0` | `session.py:162-205` |
| §3.4 | `_build_provider_options` three-layer merge (defaults → user → monitor wins last) | DONE | `ee22fe00` | `session.py:98-119` |
| §3.4 | `_ep_defaults` per-EP, QNN `backend_type` | DONE | `ee22fe00` | `session.py:84-95`, `_QNN_BACKEND: session.py:81` |
| §3.4 | Strict 4-tuple match `(ep, device.type, vendor_id, device_id)` everywhere | DONE | `ce6d4db0` | `session.py:181-187`; also in `resolve_device` via dedup `ep_device.py:159-169` |
| §3.4 | `perf()` validates monitor.ep_name via `expand_ep_name` and raises `EPMonitorMismatch` | DONE | `833971e2`, `6f27bd7b`, `08d4b119` | `session.py:680-689` |
| §3.4 | `perf()` save/restore lifecycle preserved | DONE | `833971e2`, `08d4b119` | `session.py:704-706, 776-778` (save/restore of `_active_session_option_entries`, `_provider_options`, `_ep`) |
| §3.4 | `perf()` rebuilds bare session in `finally` (clean post-perf state) | DONE | `08d4b119` | `session.py:786-795` |
| §3.4 | `_build_session_options(monitor=None)` called from `__init__`; `(monitor=monitor)` from `perf()` | DONE | `5b86e8a9`, `08d4b119` | `session.py:263-269` (ctor), `session.py:722-728` (perf rebuild branch) |
| §3.5 | `WinMLEPRegistry.register_ep(name)` additive, idempotent, raises typed errors | DONE | `8c158702` | `session/ep_registry.py:150-178` |
| §3.5 | `register_to_ort` unchanged | DONE | `8c158702` | `ep_registry.py:118-148` (unchanged shape) |
| §3.6 | Layering: this PR adds only `register_ep`; consumes `canonicalize_ep_name` (stubbed locally pending other PR) | DONE | `98b33dcb` | `ep_device.py:82-93` carries `MIGRATION:` marker |
| §4 | Five exception types: `EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch` | DONE | `40c68991` | `ep_device.py:26-43` |
| §5 step 1 | Wait for `feat/update-pkg-deps`, rebase | DEFERRED | n/a | `canonicalize_ep_name` is stubbed locally with `MIGRATION:` marker (`ep_device.py:82-93`); rebase happens later |
| §5 steps 2-9 | register_ep / ep_device.py / rename / __init__ rewrite / build_session_options / build_provider_options / perf() refactor / CLI sweep | DONE | (various) | See per-§ rows above |
| §5 step 10 | Test fixture sweep | IN FLIGHT | — | Task 11 parallel subagent in progress per task brief |
| §6 | E2E `wmk perf <convnext> --ep qnn --device npu` deterministic | PENDING | — | Task 14 — not yet run |
| §6 | Unit tests pass / ruff clean | PARTIAL | various | Tests for new API exist (see §5); fixture sweep / full suite gate pending (Task 11/13) |
| §6 | Architecture regression test (rejects legacy ep=/device=) | NOT STARTED | — | Spec calls for `tests/unit/architecture/test_winml_session_ctor.py`; the directory exists but contains only `test_qnn_imports.py`. Three equivalent tests are however present in `test_winml_session.py:649-662`. |
| §6 | Roundtrip / layering / mismatch / expand / save-restore tests | DONE | various | `test_ep_device.py:22-69` (roundtrip + expand), `test_build_session_options.py:56-145` (layering), `test_winml_session.py:670-721` (mismatch + save-restore) |
| §7 | All open decisions encoded (class name, frozen, strict tuple, hard break, free fns, three-layer, EP-monitor at `perf()`, short forms, additive register_ep, layered consumer, rename `resolve_device_category`) | DONE | various | See per-row rows above |

### 1.1 perf() contextmanager deviation

This is the only structural drift from spec/plan as written; it is documented explicitly here because it is intentional and worth reconciling in spec v1.3.

- **Spec §3.3** said `EPMonitor` integrates via `perf(monitor=...)`, **not** the ctor. Preserved (`session/session.py:631-634`).
- **Spec §3.4** said `_build_session_options` is called from `__init__` (no monitor) AND from `perf()` (with monitor) as a pure free function. **Partly preserved** — the new free function `_build_session_options` (`session.py:162-205`) is wired into both `__init__` (`session.py:263`) and `perf()` (`session.py:722`). However, a **separate legacy instance method** `WinMLSession._build_session_options(self, device: str)` (`session.py:462-485`) still exists and is consumed by `compile()` (`session.py:317, 338`), `is_compatible()` (`session.py:954`), and `WinMLQairtSession._create_inference_session()` (`session/qairt/qairt_session.py:237`). The legacy method still calls `set_provider_selection_policy()` — the very autoep mechanism the spec said to delete. It is marked `TODO Task 8 [bridge]` in code but was not removed.
- **Spec §3.4 / Plan Task 8** chose a `perf()` shape that has `def perf(self, monitor=None, *args, **kwargs): … return self._run_perf_window(...)` — i.e. a regular method that internally delegates. Commit `08d4b119` (Task 10) reverted to `@contextmanager` per `session.py:630`, with the original yield-based shape (`yield ctx` at `session.py:757`). The commit message ("Chose Option A for wmk perf: restored perf() as a @contextmanager") owns the deviation. `54cb6e81` did NOT touch this — the contextmanager shape is intentional and stays.
- **Spec did NOT explicitly require regular method vs contextmanager.** Acceptable per save/restore preservation. Should be documented in spec v1.3 as the explicit chosen shape.

---

## 2. Implementation surface inventory

Symbols added or substantially modified in the refactor commit range `d0aa3419..HEAD`. Read from the actual source after `54cb6e81` (current HEAD).

### 2.1 `src/winml/modelkit/session/ep_device.py` (new file)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `EPNotDiscovered` | EP plugin absent from catalog/MODELKIT_EP_PATH | public exception | §4 |
| `EPRegistrationFailed` | `ort.register_execution_provider_library` raised | public exception | §4 |
| `DeviceNotFound` | EP registered but no matching `OrtEpDevice` | public exception | §4 |
| `AmbiguousMatch` | Multiple OrtEpDevices match after dedup | public exception | §4 |
| `EPMonitorMismatch` | monitor.ep_name disagrees with EPDevice.ep | public exception | §4 |
| `EPDevice` | frozen dataclass descriptor with `__post_init__` lowercase | public | §3.1; `ep_device.py:49-77` |
| `EPDevice.to_dict` / `from_dict` | JSON-serializable round-trip | public | `ep_device.py:64-77` |
| `_EP_NAME_ALIASES` | local stub alias table (NvTensorRt casing) | private constant | carries `MIGRATION:` marker `ep_device.py:82-88` |
| `canonicalize_ep_name(name)` | stub for alias-casing fix | public | §3.2; `ep_device.py:91-93` (one-line replacement target on rebase) |
| `_SHORT_TO_CANONICAL` | short-form lookup table | private constant | `ep_device.py:96-104` |
| `expand_ep_name(name)` | short → canonical with passthrough through `canonicalize_ep_name` | public | §3.2; `ep_device.py:107-117` |
| `WinMLEPRegistry` (module-level binding) | lazy-init shim to avoid circular import | public binding | `ep_device.py:128`; populated on first call to `_get_ep_registry` |
| `_get_ep_registry()` | lazy `importlib.import_module(".ep_registry", ...)` of registry | private | `ep_device.py:131-137`; **documented circular-import deviation**, not in spec but justified inline |
| `resolve_device(ep, device)` | EPDevice from (ep, device) strings; dedup; strict raises | public | §3.2; `ep_device.py:140-195` |

### 2.2 `src/winml/modelkit/session/ep_registry.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `WinMLEPRegistry.register_ep(ep_name)` | per-EP selective registration, idempotent, returns `list[OrtEpDevice]` | public method | §3.5; `ep_registry.py:150-178`; **additive** to existing class |
| (import) `from .ep_device import EPNotDiscovered, EPRegistrationFailed` | wire typed errors | top-level | `ep_registry.py:18` |
| (existing) `WinMLEPRegistry.register_to_ort` | unchanged (bulk registration for `wmk sys --list-ep`) | public | spec §3.5 promise honoured |

### 2.3 `src/winml/modelkit/session/session.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `_QNN_BACKEND` | EP defaults table (`{"npu": "htp", "gpu": "gpu", "cpu": "cpu"}`) | private constant | §3.4; `session.py:81` |
| `_ep_defaults(ep_device)` | per-EP defaults dict | private free fn | §3.4; `session.py:84-95` (match-statement, returns `{}` for non-QNN) |
| `_build_provider_options(ep_device, ep_config, ep_monitor)` | three-layer merge | private free fn | §3.4; `session.py:98-119` |
| `_build_session_options(ep_device, ep_config, ep_monitor, base)` | descriptor → handle bridge + `add_provider_for_devices` | private free fn | §3.4; `session.py:162-205` |
| `WinMLSession.__init__` | session ctor — `ep_device: EPDevice` required, positional, no defaults | public | §3.3; `session.py:211-269`; hard break confirmed |
| `WinMLSession.perf` | `@contextmanager` — validates monitor / save & restore / rebuild | public | §3.4 + deviation §1.1; `session.py:630-799` |
| `WinMLSession._build_session_options` (instance method) | legacy bridge; uses `set_provider_selection_policy(PREFER_NPU)` | private method | **TECH DEBT** — `session.py:462-485`; carries `TODO Task 8 [bridge]`; still used by `compile()` and `is_compatible()` |
| `WinMLSession._build_op_type_map` | static helper for ONNX node→op_type map (op-tracing wiring) | private static | unchanged in this refactor — already present from earlier op-tracing work |
| `PerfContext` | frozen dataclass yielded by `perf()` | public | unchanged shape; `session.py:68-78` |
| (removed) `_EP_NAME_MAP`, `DEVICE_POLICY_MAP`, `_find_ep_device` | — | — | confirmed absent in src tree |

### 2.4 `src/winml/modelkit/session/qairt/qairt_session.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `WinMLQairtSession.__init__` | ctor — `ep_device: EPDevice \| None = None`, defaults `resolve_device("qnn","npu")` | public | `qairt_session.py:52-73`; default differs slightly from `WinMLSession`'s positional-required ep_device but is intentional (qairt is QNN-NPU specific) |
| (existing) `WinMLQairtSession._create_inference_session` | reuses legacy `self._build_session_options` instance method | private | `qairt_session.py:237` — inherits legacy bridge path |

### 2.5 `src/winml/modelkit/models/winml/base.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `WinMLPreTrainedModel.__init__` | now `(onnx_path, ep_device: EPDevice, config=None)` — required | public | `base.py:64-89`; matches hard-break; passes `ep_device=ep_device` to `WinMLSession` |

### 2.6 `src/winml/modelkit/sysinfo/device.py` + `__init__.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `resolve_device_category(device="auto") -> tuple[str, list[str]]` | renamed from `resolve_device`; returns (category, available_devices) | public | §3.2 namespace fix; `sysinfo/device.py:146-191` |
| `get_ep_device_map() -> dict[str, str]` | accessor for `_EP_DEVICE_MAP` | public | unchanged shape; `sysinfo/device.py:64-74` |
| `sysinfo/__init__.py` | re-exports `resolve_device_category` (NOT `resolve_device`) | public surface | `sysinfo/__init__.py:5, 18` |

### 2.7 `src/winml/modelkit/commands/perf.py` (modified)

| Symbol | Responsibility | Visibility | Notes |
|---|---|---|---|
| `_run_onnx_benchmark` | signature changed from `device=`/`ep=` strings to `ep_device: EPDevice` | private (module fn) | `commands/perf.py:1086-1102` |
| ONNX-direct CLI flow | resolves `(ep, device)` to EPDevice at CLI boundary | private flow | `commands/perf.py:1540-1548` |
| per-module path (`--module` flag) | uses `resolve_device("cpu","cpu")` for CPU sniff | private flow | `commands/perf.py:764-770` |

### 2.8 `src/winml/modelkit/commands/eval.py`, `commands/config.py`, `config/build.py`, `config/precision.py` (modified)

All updated to call `resolve_device_category` instead of `resolve_device`. See `git diff d0aa3419..HEAD -- src/winml/modelkit/config/build.py src/winml/modelkit/config/precision.py src/winml/modelkit/commands/eval.py src/winml/modelkit/commands/config.py`.

---

## 3. Integration points needing revision

This is the gap list — pre-existing callsites that the refactor missed or imperfectly updated.

| File:line | Current state | Required revision | Priority |
|---|---|---|---|
| `src/winml/modelkit/commands/perf.py:1102` | `WinMLSession(onnx_path=onnx_path, ep_device=ep_device)` | already correct (Task 10) | done |
| `src/winml/modelkit/commands/perf.py:767-770` | `WinMLSession(...ep_device=resolve_device("cpu","cpu"))` | already correct (Task 10 fix) | done |
| `src/winml/modelkit/models/winml/base.py:86-89` | `WinMLSession(onnx_path=..., ep_device=ep_device)` | already correct (Task 10 fix `54cb6e81`) | done |
| `src/winml/modelkit/session/qairt/qairt_session.py:62` | `super().__init__(onnx_path, ep_device, ep_config=ep_config)` | already correct (Task 10 fix `54cb6e81`) | done |
| **`src/winml/modelkit/models/auto.py:163-168`** | `winml_class(onnx_path=..., config=None, device=device)` (skip-build path in `from_onnx`) | `WinMLPreTrainedModel.__init__` now requires `ep_device: EPDevice`, NOT `device: str`. **Will TypeError at runtime.** Must compute `ep_device = resolve_device(ep_arg, device_arg)` here and pass it. | **CRITICAL** |
| **`src/winml/modelkit/models/auto.py:196-203`** | `winml_class(onnx_path=..., config=None, device=device)` (post-build path in `from_onnx`) | same as above | **CRITICAL** |
| **`src/winml/modelkit/models/auto.py:355-362`** | `winml_class(onnx_path=..., config=hf_config, device=device)` (HF runtime phase in `from_pretrained`) | same as above; this is the **headline `wmk perf <hf-model> --ep qnn --device npu`** path; will TypeError before reaching `session.perf()` | **CRITICAL — gates §6 E2E gate / Task 14** |
| **`tests/e2e/test_ep_monitor.py:62`** | `WinMLSession(str(simple_onnx_model), device="npu")` | legacy `device=` kwarg now raises `TypeError`. Replace with `ep_device=resolve_device("qnn","npu")` and patch registry, or skip when QNN absent. | Task 11 fixture sweep |
| **`tests/e2e/test_session.py:107-110`** | `WinMLSession(onnx_path=..., device=device)` (parametrised over EPs) | same as above | Task 11 fixture sweep |
| `tests/unit/session/test_winml_session.py:315-393` | five tests in the `@pytest.mark.skip(reason="Re-batching not yet implemented")` class still use `device="cpu"` ctor kwarg | guarded by class-level skip; will fail if class skip is removed | Low — already skipped, document as Task 11 cleanup |
| `tests/unit/commands/test_compile_quantize_flags.py:35` | docstring references removed `DEVICE_POLICY_MAP` | rename in comment | Trivial |
| **CLI argparse/click → resolve_device** | `wmk perf` Click options `--ep` (`perf.py:1253-1260`) and `--device` (`perf.py:1239-1244`) both parse user input; the ONNX-direct branch calls `resolve_device(ep_str, resolved_device)` at `perf.py:1548`; the HF branch routes through `_load_model` → `WinMLAutoModel.from_pretrained(..., device=device, ep=ep)` and never reaches `resolve_device` because `models/auto.py:361` passes the device string into `winml_class()` directly | **Critical wiring gap** — the HF code path never builds an `EPDevice` at all; the CLI handler in `perf.py` only constructs `ep_device` on the ONNX-direct branch (`perf.py:1548`). This is the root cause of the §3 CRITICAL entries above. | **CRITICAL — gates Task 14** |
| `src/winml/modelkit/session/session.py:550` (was) → now `session.py:462-485` | legacy `WinMLSession._build_session_options(self, device)` method survives, uses `set_provider_selection_policy(PREFER_NPU)`, called from `compile()` (`:317, :338`), `is_compatible()` (`:954`), and `WinMLQairtSession._create_inference_session()` (`:237`) | Task 8/11-bridge follow-up — out of scope this PR per spec deviation note, but should be tracked | Tech debt |
| `src/winml/modelkit/session/session.py:286-289` | `compile()` still has `if target_device == "auto": target_device = self._detect_best_device()`; `_detect_best_device` (`:571-582`) returns `"auto"` and logs `"using PREFER_NPU policy"` | Dead-ish — `ep_device.device` is always concrete (`"cpu"/"gpu"/"npu"`), never `"auto"`, after the hard break; this branch is unreachable now. Cleanup, not a bug. | Cleanup |
| `src/winml/modelkit/session/session.py:368-374` | `compile()` post-build block reads `self._device == "auto"` to back-resolve from `actual_providers` | Same as above — unreachable; can be removed | Cleanup |

**Search results used to derive this list:**

- `grep "WinMLSession\(" src/`: 3 callsites (`commands/perf.py:767`, `commands/perf.py:1102`, `models/winml/base.py:86`) — all pass `ep_device=` correctly.
- `grep 'device=device\b' src/`: 38 hits. Most are unrelated (passing through to `build_*`/`analyze_*` APIs). The three critical ones are in `models/auto.py:163-168, 196-203, 355-362` where `winml_class()` calls `WinMLPreTrainedModel.__init__`.
- `grep "set_provider_selection_policy\|_find_ep_device\|_EP_NAME_MAP\|DEVICE_POLICY_MAP" src/`: one residual — `session/session.py:480` inside the legacy `_build_session_options` instance method.
- `grep "WinMLSession\b" tests/`: two e2e files (`test_ep_monitor.py:62`, `test_session.py:107`) still use the legacy `device=` kwarg.

---

## 4. Code quality review across the refactor

### Tasks 1-10 commit-by-commit

| Task | Commit(s) | What landed | Code-quality re-review? | Lingering markers |
|---|---|---|---|---|
| 1 | `40c68991` | `EPDevice` + 5 exception types | first-pass | none |
| 2 | `98b33dcb` | `expand_ep_name` + `canonicalize_ep_name` stub | first-pass | `MIGRATION:` marker (intentional, awaiting `feat/update-pkg-deps`) — `ep_device.py:82-85` |
| 3 | `8c158702` | `WinMLEPRegistry.register_ep` | first-pass | none |
| 4 | `b12b0eec` + `0b20f22c` | `resolve_device` | **YES** (`0b20f22c` re-review) | module-level `WinMLEPRegistry: Any = None` sentinel + `_get_ep_registry()` lazy importer (`ep_device.py:122-137`) — documented circular-import deviation |
| 5 | `ee22fe00` | `_ep_defaults` + `_build_provider_options` | first-pass | none |
| 6 | `ce6d4db0` | `_build_session_options` free function | first-pass | none |
| 7 | `5b86e8a9` + `d8200837` | `WinMLSession.__init__` hard break | **YES** (`d8200837` re-review) | `TODO Task 10` on `self._ep` legacy alias `session.py:236`; `TODO Task 8/11` on `self._session_options` legacy storage `session.py:242-246` |
| 8 | `833971e2` + `6f27bd7b` | `perf()` refactor | **YES** (`6f27bd7b` re-review) | none (re-review tightened `getattr(monitor, "ep_name", None)` to `is not None`, raised on no-op `_run_perf_window`) |
| 9 | `d9ddf71e` + `e033571a` | rename `resolve_device` → `resolve_device_category` | **YES** (`e033571a` test mock-target sweep) | none |
| 10 | `08d4b119` + **`54cb6e81`** (fix) | CLI sweep | **YES** — `54cb6e81` is the fix that reverted the `device=` compat shim deviation Task 10 had introduced | `TODO Task 8 [bridge]` on `_build_session_options` instance method `session.py:465-467`; `TODO Task 10` on `self._ep` `session.py:236` |

### Markers scan

```
grep -rn "TODO Task" src/winml/modelkit/
  session/session.py:236  # legacy alias; TODO Task 10: replace consumers and remove
  session/session.py:243  # TODO Task 8/11: remove once _build_session_options is refactored.
  session/session.py:465  # TODO Task 8 [bridge]: this method is retained...

grep -rn "MIGRATION:" src/winml/modelkit/
  session/ep_device.py:82  # MIGRATION: After feat/update-pkg-deps merges, replace this stub...
```

### Subtle smells

- **Circular-import shim** (`ep_device.py:122-137`): module-level `WinMLEPRegistry: Any = None` plus `_get_ep_registry()` lazy importer is a documented intentional deviation. The reason — `ep_registry.py` imports `EPNotDiscovered`/`EPRegistrationFailed` from `ep_device.py`, so `ep_device.py` cannot import from `ep_registry.py` at module load time. The pattern is clean (test patches replace the binding directly via `patch("winml.modelkit.session.ep_device.WinMLEPRegistry")`), and the rationale is preserved inline at `ep_device.py:122-127`. Acceptable.
- **Sentinel/`None` writes to a class-level annotated binding**: the `WinMLEPRegistry: Any = None` binding is module-level (not class-level) and is rebound via `global` inside `_get_ep_registry`. Standard Python; just unusual.
- **Legacy `WinMLSession._build_session_options` method survives** with the autoep mechanism (`set_provider_selection_policy(PREFER_NPU)`) inside. This is the largest piece of debt left by the refactor — the *new* free function with the same name should eventually replace it everywhere. Tracked by three `TODO Task 8/11` markers.
- **`compile()` policy back-resolution** (`session.py:368-374`) reads `self._device == "auto"` to remap. With the hard break, `self._device` is sourced from `ep_device.device` (`session.py:239`) and `EPDevice.device` is one of `{"cpu","gpu","npu"}` — never `"auto"`. The branch is unreachable. Dead code.
- **`_detect_best_device` returns `"auto"`** (`session.py:582`) — only meaningful for the dead branch above. Dead code.
- **Plan Task 11 expected `tests/conftest.py`** to centralize the `qnn_npu_ep_device` fixture (Plan 11.4). Currently it lives in `tests/unit/session/conftest.py:247-264`. Lower scope than spec'd — fine for now, but tests outside `tests/unit/session/` cannot use it.

---

## 5. Test coverage assessment

| Symbol | Test file | Tests | Coverage gap? |
|---|---|---|---|
| `EPDevice` | `tests/unit/session/test_ep_device.py:22-48` | `test_ep_device_round_trip`, `test_ep_device_lowercase_invariant` | adequate |
| `expand_ep_name` | `tests/unit/session/test_ep_device.py:51-69` | short-form / passthrough / alias-casing | adequate |
| `resolve_device` | `tests/unit/session/test_ep_device.py:84-132` | qnn-npu happy / dedup / device-not-found / ambiguous | adequate |
| `canonicalize_ep_name` (stub) | covered indirectly via `expand_ep_name` alias-casing test | one test | adequate for stub scope |
| `WinMLEPRegistry.register_ep` | `tests/unit/session/test_ep_registry.py` | 4 tests (happy / unknown / idempotent / failure-wraps) — confirmed by Task 3 plan and current file existence (see directory listing above) | adequate |
| `_ep_defaults` | `tests/unit/session/test_build_session_options.py:82-84` | 1 test (unknown ep returns `{}`) | thin — no QNN test, but the qnn case is exercised indirectly by `test_build_provider_options_qnn_defaults_only:56-59` |
| `_build_provider_options` | `test_build_session_options.py:56-80` | defaults-only / user-overrides-defaults / monitor-overrides-user | adequate |
| `_build_session_options` (free) | `test_build_session_options.py:87-145` | no-monitor / monitor-session-opts / device-not-found / ambiguous | adequate |
| `WinMLSession.__init__` (new) | `tests/unit/session/test_winml_session.py:38-79, 634-662` | accepts EPDevice (NPU + CPU) / rejects `ep="qnn"` / rejects `device="auto"` / nonexistent ONNX | adequate |
| `WinMLSession.perf` (new validation flow) | `tests/unit/session/test_winml_session.py:670-721` | monitor.ep_name mismatch / mid-perf raise save-restore | adequate for new validation; existing PerfTracking class (`:489-626`) covers basic ctx manager behaviour |
| `WinMLQairtSession.__init__` (updated) | `tests/unit/session/test_qairt_session.py:22-53` (autouse fixture mocks `resolve_device`) | construction + SDK env / paths / compile-idempotent / subprocess-failure / JSON-wrap | adequate |
| `WinMLPreTrainedModel.__init__` (updated) | tests in `tests/unit/models/auto/test_*.py` test the factory but **do not directly exercise the new ep_device signature**; the `auto.py` callsites that pass `device=device` strings will fail once tests actually instantiate the model class | **GAP** — no test covers `WinMLPreTrainedModel(onnx_path=..., ep_device=...)` directly. The bug in §3 would be caught by such a test. |
| `_run_onnx_benchmark` (updated) | `tests/unit/commands/test_perf_cli.py` (22 lines changed per diff stat) — exact test names not enumerated here, but the file was swept | needs eyeball at Task 11 review |
| `resolve_device_category` | `tests/unit/sysinfo/test_device.py` (29 lines changed) | smoke + mock-patch sweep (`e033571a`) | adequate |
| Architecture regression (rejects legacy kwargs at ctor) | `tests/unit/architecture/test_winml_session_ctor.py` | **MISSING** — directory exists but only `test_qnn_imports.py` is present; the three required assertions are duplicated in `test_winml_session.py:649-662` instead | Plan Task 12 not yet executed |

---

## 6. `wmk perf` E2E verification plan — code path trace

Tracing `wmk perf <model> --ep qnn --device npu` from CLI entry to monitor lifecycle:

| Step | File:line | State | Status |
|---|---|---|---|
| 1. CLI entry | `wmk` executable wired through `pyproject.toml` to `winml.modelkit.cli:main` (not re-verified for this audit) | unchanged | assumed OK |
| 2. Argument parsing | `src/winml/modelkit/commands/perf.py:1239-1260` — Click options `--device` (choice CPU/GPU/NPU/auto) and `--ep` (free string) | unchanged | OK |
| 3. Branch on input type | `commands/perf.py:1515` — `is_onnx = model_path.suffix.lower() == ".onnx"` | branches into ONNX-direct vs HF | OK |
| 4a. ONNX-direct branch: EPDevice construction | `commands/perf.py:1540-1548` — `resolved_device, _ = resolve_device_category(device=config.device)`; `_default_ep_for_device = {"cpu": "cpu", "npu": "qnn", "gpu": "dml"}`; `ep_str = config.ep or _default_ep_for_device.get(resolved_device, "cpu")`; `ep_device = resolve_device(ep_str, resolved_device)` | resolves at CLI boundary | OK |
| 4b. HF branch: device propagated as string | `commands/perf.py:1564` → `PerfBenchmark(config).run()` → `_load_model` (`:456-498`) → `WinMLAutoModel.from_pretrained(..., device=resolved_device, ep=ep)` → returns a `WinMLPreTrainedModel` whose ctor still receives `device=device` (auto.py:361) NOT `ep_device=...` | **broken** — `WinMLPreTrainedModel.__init__` now requires `ep_device`. TypeError at line `auto.py:358-362`. | **CRITICAL** |
| 5. WinMLSession construction | `commands/perf.py:1102` (ONNX path) or `models/winml/base.py:86-89` (HF path) | ONNX-direct: OK. HF: never reached because step 4b raises first. | broken (downstream) |
| 6. `session.compile()` | `commands/perf.py:1109` (ONNX path) or `commands/perf.py:424` (HF path via `self._model._session.compile()`) — both go through `session.py:271-374` which calls legacy `self._build_session_options` (`session.py:317, 338`) | legacy bridge — still uses `set_provider_selection_policy(PREFER_NPU)`; works but uses the autoep path the spec disowns | tech debt (works) |
| 7. `perf()` context manager flow | `commands/perf.py:1143-1159` (ONNX) or `commands/perf.py:573-590` (HF monitored) — `with session.perf(warmup=..., monitor=ep_monitor) as ctx, hw_monitor as hw:` | `perf()` is the new `@contextmanager` (`session.py:630`); validation + save/restore + rebuild — all in place | OK |
| 8. Benchmark loop | `_run_monitored_loop` / `_run_simple_loop` in `commands/perf.py:1065-1078` | unchanged | OK |
| 9. Monitor lifecycle | `_resolve_ep_monitor` (`perf.py:116-186`) → `QNNMonitor(level, output_dir)`; entered/exited inside `session.perf()` (`session.py:736, 770-773`) | unchanged | OK |
| 10. Op-trace post-report | `commands/perf.py:1584-1630` | unchanged | OK |

**Bottom line:** the ONNX-direct path (`wmk perf model.onnx --ep qnn --device npu`) should work end-to-end. The HF path (`wmk perf microsoft/resnet-50 --ep qnn --device npu`) — the original §6 spec demo case — **will not** until `models/auto.py:163-168, 196-203, 355-362` are migrated to call `resolve_device(ep_arg, device_arg)` and pass `ep_device=` to the inference-model class.

---

## 7. Open issues + next-step recommendations

### Bugs (must fix before merge)

- **`models/auto.py:163-168, 196-203, 355-362`** pass `device=device` string to `winml_class()` (i.e. `WinMLPreTrainedModel.__init__`), which now requires `ep_device: EPDevice`. **Breaks `wmk perf <hf-model> --ep ... --device ...`**, blocks Task 14 E2E gate. Fix: resolve at the boundary, pass `ep_device=resolve_device(ep_arg, device_arg)`.
- **`tests/e2e/test_ep_monitor.py:62`** and **`tests/e2e/test_session.py:107-110`** still use `device=` ctor kwarg — will fail under `TypeError` once Task 13 runs the full suite. Must be swept by Task 11 (e2e tests are out of the documented Task 11 scope; raise to spec or include in Task 11).

### Spec drift items (reconcile in spec v1.3)

- `perf()` is implemented as `@contextmanager` (intentional, `08d4b119`); spec §3.4 / plan Task 8 stylistically implied a regular method. Spec v1.3 should explicitly say `@contextmanager`.
- Lazy circular-import shim `WinMLEPRegistry: Any = None` + `_get_ep_registry()` in `ep_device.py:122-137` is not in spec §3.2 pseudocode. Spec v1.3 should note this is the chosen mechanism (vs. moving exceptions out of `ep_device.py` into a third module).
- `WinMLSession._build_session_options` legacy instance method survives. Spec §3.4 said the *new* free function replaces it; in practice the *new* free function lives alongside the *old* method which still serves `compile()`. Spec v1.3 should either commit to the bridge or schedule the migration explicitly.
- `WinMLQairtSession.__init__` accepts `ep_device: EPDevice | None = None` (with default `resolve_device("qnn","npu")` at construction time, `qairt_session.py:55-60`). The hard-break would suggest required-positional, but in this subclass the default is reasonable (QNN+NPU is the only target). Spec v1.3 should explicitly carve out qairt.

### Cleanup tasks for follow-up PR (out of scope this PR)

- Delete legacy `WinMLSession._build_session_options(self, device)` (`session.py:462-485`) once `compile()` and `is_compatible()` are migrated to the free function.
- Delete `_detect_best_device` (`session.py:571-582`) and the dead `compile()` policy back-resolution block (`:286-289, :368-374`).
- Remove `self._ep` legacy alias (`session.py:236`) — replace all consumers with `self._ep_device.ep`.
- Remove `self._session_options` storage (`session.py:242-246`) — only used by the legacy `_build_session_options` method.
- Replace `canonicalize_ep_name` stub (`ep_device.py:86-93`) with `from .ep_path import canonicalize_ep_name` once `feat/update-pkg-deps` merges. One-line change.
- Add architecture regression test at `tests/unit/architecture/test_winml_session_ctor.py` per Plan Task 12 (the three assertions exist today only inline in `test_winml_session.py:634-662`; a dedicated file with the documented intent is the spec ask).
- Centralize `qnn_npu_ep_device` fixture in top-level `tests/conftest.py` (per Plan 11.4); today it's at `tests/unit/session/conftest.py:247-264`.

---

## 8. Verification gates remaining

- [ ] **Task 11** — fixture sweep (in flight in parallel subagent per task brief). Includes pulling the legacy `device=` ctor kwargs out of `tests/e2e/test_ep_monitor.py:62` and `tests/e2e/test_session.py:107-110` (or scoping them out of Task 11 explicitly and into a follow-up).
- [ ] **Task 12** — architecture regression test file `tests/unit/architecture/test_winml_session_ctor.py` does not exist; three equivalent assertions are inlined in `tests/unit/session/test_winml_session.py:634-662`. Spec ask is the dedicated file.
- [ ] **Task 13** — full pytest + ruff gate. Will surface the `models/auto.py` breakage as soon as any HF-path test fires (`test_perf_cli.py` or `test_eval.py`).
- [ ] **Task 14** — E2E: `uv run wmk perf <convnext-tiny-224> --ep qnn --device npu`. **Blocked** by the `models/auto.py` `device=device` → `ep_device=` migration (HF path). The ONNX-direct path (`wmk perf path/to/model.onnx --ep qnn --device npu`) should work today and could be used as a partial E2E demo while the HF path is fixed.
- [ ] **Spec v1.3** — reconcile (1) `perf()` is `@contextmanager`, (2) the `_get_ep_registry` circular-import shim, (3) legacy `_build_session_options` method bridge tech-debt schedule, (4) `WinMLQairtSession` default-EP behaviour.
- [ ] **Cleanup** — remove legacy `WinMLSession._build_session_options` method once `compile()` and `is_compatible()` are migrated. Out of scope this PR; track as follow-up.

### Top 3 gap findings (severity-ordered)

1. **`models/auto.py:355-362` HF-path breakage.** Three `winml_class(..., device=device)` callsites in `from_pretrained` (line 361) and `from_onnx` (lines 167, 202) feed a string device into `WinMLPreTrainedModel.__init__` which now requires `ep_device: EPDevice`. Will raise `TypeError`. This is the headline §6 E2E scenario (`wmk perf microsoft/resnet-50 --ep qnn --device npu`).
2. **Two e2e tests still use legacy `device=` kwarg** (`tests/e2e/test_ep_monitor.py:62`, `tests/e2e/test_session.py:107-110`). Will surface as `TypeError` in Task 13 full-suite run; Task 11 fixture sweep should include them or explicitly defer.
3. **Legacy `WinMLSession._build_session_options` method survives** and still uses `set_provider_selection_policy(PREFER_NPU)` — the very autoep path the spec promised to delete. Used by `compile()`, `is_compatible()`, and `WinMLQairtSession._create_inference_session()`. Three `TODO Task 8/11` markers document the debt.

### Top 3 deviations from spec

1. **`perf()` kept as `@contextmanager`.** Plan Task 8 chose regular method + `_run_perf_window` delegation. `08d4b119` reverted to context manager with intentional message ("Chose Option A for wmk perf"). Acceptable but undocumented in spec.
2. **Circular-import shim** (`WinMLEPRegistry: Any = None` + `_get_ep_registry()`) in `ep_device.py:122-137`. Spec §3.2 implied a direct import of the registry. The shim is justified inline but is not in the spec pseudocode.
3. **`models/auto.py` not migrated** as part of Task 10. Task 10's brief said "every `wmk` command now resolves `(ep, device)` to an `EPDevice` at the CLI boundary"; `models/auto.py` sits between the CLI handler (`perf.py:1564`) and the `WinMLPreTrainedModel` ctor and was not touched. The `54cb6e81` fix swept the immediately-broken sites but did not push the migration up through `from_pretrained` / `from_onnx`. This is the proximate root cause of gap #1.
