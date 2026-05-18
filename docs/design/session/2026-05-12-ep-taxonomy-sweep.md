# EP / Device Taxonomy Sweep — 2026-05-12

Goal: identify every place in the repo where EP / device knowledge is hard-coded,
so we can consolidate under a single home (`winml.modelkit.session.ep_device`).

## Summary

- **47 findings total**
- **9 unique tables / constants** (spread across 5 files)
- **5 helper functions** (spread across 3 files)
- **9 inline literals** (production code)
- **3 CLI `click.Choice` lists** with redundancy
- **5 test fixtures / mocks** with their own independent EP maps
- **6 duplicates** — same mapping data defined in multiple places

---

## Section 1: Centralized constants (already exist)

These are the candidates for the canonical home. Both `ep_device.py` (new) and `precision.py` / `sysinfo/device.py` (old) contain overlapping but not identical tables.

| File:Line | Symbol | Content / Cardinality | Action |
|---|---|---|---|
| `session/ep_device.py:86-88` | `_EP_NAME_ALIASES` | 1-entry casing stub: `{nvtensorrtrtx…: NvTensorRtRtx…}` | KEEP as migration stub; delete when `ep_path.canonicalize_ep_name` merges |
| `session/ep_device.py:96-104` | `_SHORT_TO_CANONICAL` | 7-entry dict: `{qnn, openvino, vitisai, migraphx, nv_tensorrt_rtx, dml, cpu}→canonical` | **CANONICAL HOME** — keep here |
| `session/ep_device.py:122` | `_CANONICAL_TO_SHORT` | Derived inverse of `_SHORT_TO_CANONICAL`, 7 entries | KEEP — derived, auto-updates |
| `config/precision.py:65-69` | `_DEVICE_TO_PROVIDER` | 3-entry dict: `{npu→qnn, gpu→dml, cpu→None}` | MOVE to `ep_device.py` |
| `config/precision.py:85-94` | `_EP_TO_DEVICE` | 8-entry dict: `{qnn→npu, vitisai→npu, dml→gpu, migraphx→gpu, tensorrt→gpu, cuda→gpu, openvino→gpu, cpu→cpu}` | MOVE to `ep_device.py`; note: `tensorrt`, `cuda`, `openvino→gpu` are not in `_SHORT_TO_CANONICAL` — coverage gap |
| `config/precision.py:97` | `VALID_EPS` | `frozenset(_EP_TO_DEVICE.keys())` — 8 short names | MOVE to `ep_device.py`; derive from `_SHORT_TO_CANONICAL` keys |
| `config/precision.py:99` | `_VALID_DEVICES` | `frozenset({"npu","gpu","cpu"})` | MOVE to `ep_device.py` |
| `sysinfo/device.py:38-52` | `_EP_DEVICE_MAP` | 7-entry dict keyed by **canonical** EP names → device; includes `OpenVINO→npu/gpu/cpu` | MOVE to `ep_device.py`; this is the canonical-form variant of `_EP_TO_DEVICE` from `precision.py` — **duplicate** |
| `sysinfo/device.py:55-58` | `_DEVICE_EP_MAP` | Derived inverse of `_EP_DEVICE_MAP` (excl. multi-device), 3 entries | MOVE to `ep_device.py`; can be derived from consolidated map |
| `sysinfo/device.py:61` | `_VALID_DEVICES` | `frozenset({"npu","gpu","cpu"})` — **duplicate** of `precision.py:99` | DELETE; import from `ep_device.py` |
| `utils/constants.py:11-15` | `SUPPORTED_EPS` | 3-entry list: `[QNN, OpenVINO, VitisAI]` canonical names | MOVE to `ep_device.py`; subset of `_SHORT_TO_CANONICAL` values — **partial duplicate** |
| `utils/constants.py:18-24` | `EP_ALIASES` | 5-entry dict: `{qnn, openvino, ov, vitisai, vitis}→canonical` — **partial duplicate** of `_SHORT_TO_CANONICAL` | MOVE/MERGE into `ep_device.py._SHORT_TO_CANONICAL`; `ov` and `vitis` aliases are missing from `_SHORT_TO_CANONICAL` |
| `utils/constants.py:27` | `ALL_EP_NAMES` | `SUPPORTED_EPS + EP_ALIASES.keys()` — 8 strings | DELETE; derive from consolidated tables |
| `utils/constants.py:94-98` | `SUPPORTED_DEVICES` | `["CPU","GPU","NPU"]` uppercase strings | NORMALIZE; inconsistent casing vs lowercase everywhere else; replace with `_VALID_DEVICES` uppercased |
| `utils/constants.py:101-111` | `DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE` | ORT enum ↔ uppercase device string, 3 entries each | KEEP in `utils/constants.py` — ORT-enum knowledge belongs with the ORT import; not EP taxonomy per se |

---

## Section 2: Inline literals (production code)

| File:Line | Literal | Duplicates? | Action |
|---|---|---|---|
| `session/session.py:469-473` | `{"npu":PREFER_NPU, "gpu":PREFER_GPU, "cpu":PREFER_CPU, "auto":PREFER_NPU}` | New; policy-based, not in any table | KEEP as local; this is ORT-policy knowledge, not EP identity — but `"auto"→PREFER_NPU` is a policy decision that belongs in `ep_device.py` or removed with Task 8 |
| `commands/perf.py:472` | `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` | Inverse of `_DEVICE_TO_PROVIDER` from `precision.py` | NORMALIZE: import `_DEVICE_TO_PROVIDER` from `ep_device.py` |
| `commands/perf.py:1552` | `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` | Same as line 472 — **identical duplicate in same file** | NORMALIZE: extract to module-level constant or import |
| `eval/evaluate.py:138` | `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` | Third copy of same `_DEVICE_TO_PROVIDER` inverse | NORMALIZE: import from `ep_device.py` |
| `commands/build.py:370-372` | `["QNNExecutionProvider","OpenVINOExecutionProvider","VitisAIExecutionProvider"]` | Subset of `_SHORT_TO_CANONICAL` values; same 3 as `utils/constants.py.SUPPORTED_EPS` | NORMALIZE: derive from consolidated table |
| `optim/pipes/graph.py:572` | `providers=["CPUExecutionProvider"]` | Single-EP list for safe ONNX load | KEEP — intentional CPU-only session, not taxonomy |
| `analyze/pattern/check_patterns.py:288-289` | `{"QNNExecutionProvider":QNNNPUChecker, "OpenVINOExecutionProvider":OpenVINONPUChecker}` | EP-to-checker map, structural | KEEP — this is a polymorphism table, not EP naming taxonomy |
| `analyze/runtime_checker/check_ops.py:285-289` | `{"QNNExecutionProvider":QNNNPUChecker, …, "NvTensorRTRTXExecutionProvider":RTXChecker}` | EP-to-checker map (5 entries) | KEEP — same reason as above |
| `analyze/runtime_checker/check_ops.py:331-335` | `["QNNExecutionProvider","OpenVINOExecutionProvider","VitisAIExecutionProvider","MIGraphXExecutionProvider","NvTensorRTRTXExecutionProvider"]` | Supported EPs for analyze — superset of `SUPPORTED_EPS` | NORMALIZE: derive from or contribute to `ep_device.py` table |
| `compiler/cli.py:53` | `click.Choice(["qnn","cpu","cuda","dml"])` | Subset of `_SHORT_TO_CANONICAL` + adds `cuda`; **inconsistent** with `commands/compile.py` | NORMALIZE: replace with `sorted(VALID_EPS)` filter; align with `commands/compile.py:62` |
| `utils/optimum_loader.py:68` | `"CPUExecutionProvider" if device=="cpu" else "CUDAExecutionProvider"` | Hardcoded two-EP branching | KEEP — Optimum-specific path; `CUDAExecutionProvider` not in WinML taxonomy |

---

## Section 3: Test fixtures and mocks

| File:Line | Literal / Symbol | Category | Action |
|---|---|---|---|
| `tests/unit/session/conftest.py:45-55` | `EP_NAME_MAP = {"qnn":"QNNExecutionProvider","openvino":…,"directml":"DmlExecutionProvider","cuda":"CUDAExecutionProvider","tensorrt":"TensorrtExecutionProvider","tensorrt_rtx":"NvTensorRTRTXExecutionProvider","vitisai":"VitisAIExecutionProvider","coreml":"CoreMLExecutionProvider","rocm":"ROCMExecutionProvider"}` | Test fixture — 9-entry marker→canonical map | KEEP as test-local; includes cross-platform EPs (coreml, rocm, cuda) not in production tables |
| `tests/e2e/test_session.py:57-94` | Parametrize table: `("qnn","npu","QNNExecutionProvider"), ("openvino","npu",…), ("directml","gpu","DmlExecutionProvider"), ("cuda","gpu","CUDAExecutionProvider"), ("tensorrt","gpu","TensorrtExecutionProvider"), ("tensorrt_rtx","gpu","NvTensorRTRTXExecutionProvider"), ("vitisai","npu",…), ("rocm","gpu","ROCMExecutionProvider")` | Test data — inline short→device→canonical | KEEP as test fixture; cross-platform scope intentional |
| `tests/unit/session/conftest.py:249,296` | `EPDevice(ep="QNNExecutionProvider"…)`, `EPDevice(ep="CPUExecutionProvider"…)` | Fixture objects | KEEP — use of proper `EPDevice` constructor is correct |
| `tests/unit/commands/test_compile_quantize_flags.py:37` | Comment: `"auto is not in _DEVICE_TO_PROVIDER"` | Documentation reference | NORMALIZE: update comment when `_DEVICE_TO_PROVIDER` moves |
| `tests/unit/session/test_winml_session.py:674` | `EPDevice(ep="OpenVINOExecutionProvider"…)` | Fixture | KEEP |

---

## Section 4: Helper functions

| Function | File | Purpose | Callers | Action |
|---|---|---|---|---|
| `expand_ep_name(name)` | `session/ep_device.py:107` | short→canonical via `_SHORT_TO_CANONICAL`, falls through to `canonicalize_ep_name` | `session/ep_registry.py`, `session/session.py` | KEEP — canonical home |
| `canonicalize_ep_name(name)` | `session/ep_device.py:91` | casing fix via `_EP_NAME_ALIASES` (stub) | `expand_ep_name` | KEEP as stub; replace with `ep_path.canonicalize_ep_name` post-merge |
| `short_ep_name(canonical)` | `session/ep_device.py:125` | canonical→short, fallback to suffix-strip | Internal, tests | KEEP |
| `get_provider_for_device(device)` | `config/precision.py:72` | device→short EP via `_DEVICE_TO_PROVIDER` | `config/precision.py:296` | MOVE to `ep_device.py` once `_DEVICE_TO_PROVIDER` moves |
| `normalize_ep_name(ep)` | `utils/constants.py:30` | short/alias→canonical, None-safe | `analyze/analyzer.py` (8×), `commands/analyze.py`, `commands/config.py` | MOVE to `ep_device.py`; overlaps with `expand_ep_name` — **duplicate function**; `ov` and `vitis` alias support would need to be added to `_SHORT_TO_CANONICAL` |
| `get_ep_device_map()` | `sysinfo/device.py:64` | public accessor for `_EP_DEVICE_MAP` | (no callers found in src) | MOVE to `ep_device.py`; or DELETE if no callers |
| `infer_ihv_from_ep_name(ep_name)` | `analyze/utils/ep_utils.py:20` | EP name→IHV vendor enum (QC/INTEL/AMD) via substring match | `analyze/core/information_engine.py`, `analyze/core/output_aggregator.py` | KEEP in `analyze/utils/` — IHV typing is analyze-domain knowledge, not core EP taxonomy |

---

## Section 5: CLI option Choice() lists

| File:Line | Choice values | Device or EP? | Consistent? |
|---|---|---|---|
| `commands/compile.py:55` | `["auto","npu","gpu","cpu"]` | Device | Consistent with `config.py`, `eval.py`, `perf.py` |
| `commands/config.py:117` | `["auto","npu","gpu","cpu"]` | Device | Consistent |
| `commands/eval.py:56` | `["auto","cpu","gpu","npu"]` | Device | Consistent (order differs, functionally same) |
| `commands/perf.py:1247` | `["auto","cpu","gpu","npu"]` | Device | Consistent |
| `commands/compile.py:62` | `sorted(VALID_EPS)` | EP (short names) | Programmatic — correct approach |
| `compiler/cli.py:53` | `["qnn","cpu","cuda","dml"]` | EP (short names) | **Inconsistent** — hardcoded, missing `vitisai`/`migraphx`/`openvino`, adds `cuda` not in `VALID_EPS` |
| `utils/cli.py:56` | `ALL_EP_NAMES` from `constants.py` | EP (short + canonical names) | Mixes short and canonical — inconsistent with `commands/compile.py` which uses only short names |
| `utils/cli.py:82` | `SUPPORTED_DEVICES` from `constants.py` | Device (uppercase strings) | **Inconsistent** — uses uppercase `["CPU","GPU","NPU"]` vs all other CLI device choices use lowercase `["cpu","gpu","npu"]` |

---

## Section 6: Recommended consolidation plan

Ordered by dependency (move sources before updating callers).

1. **Add `_DEVICE_TO_PROVIDER`, `VALID_EPS`, `_VALID_DEVICES` to `ep_device.py`** — move from `config/precision.py:65-99`. Export `VALID_EPS` and `_VALID_DEVICES` as public. Callers: `precision.py` (3×), `config/__init__.py`, `compile.py` (1×). ~10 LOC moved, ~5 import lines updated.

2. **Merge `_EP_DEVICE_MAP` / `_EP_TO_DEVICE` into one canonical-keyed table in `ep_device.py`** — `precision.py._EP_TO_DEVICE` is short-keyed; `sysinfo/device.py._EP_DEVICE_MAP` is canonical-keyed. They encode the same data with different key forms. Unify as a single canonical-keyed dict; derive short-keyed view. Add `tensorrt`, `cuda`, `openvino` coverage gaps. ~20 LOC affected; callers: `compile.py`, `compiler/stages/compile.py`, `precision.py`.

3. **Absorb `EP_ALIASES` + `SUPPORTED_EPS` from `utils/constants.py` into `ep_device.py._SHORT_TO_CANONICAL`** — add `ov` and `vitis` alias entries (currently absent). Then delete `EP_ALIASES`, `SUPPORTED_EPS`, `ALL_EP_NAMES` from `constants.py`. Callers of `normalize_ep_name`: 10 sites in `analyze/` — redirect to `expand_ep_name` or add None-safe wrapper. ~30 LOC moved, ~10 import lines updated.

4. **Replace three copies of `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` with import** — `commands/perf.py:472`, `commands/perf.py:1552`, `eval/evaluate.py:138`. All three should import `_DEVICE_TO_PROVIDER` from `ep_device.py`. ~6 LOC.

5. **Fix `compiler/cli.py:53`** — replace hardcoded `["qnn","cpu","cuda","dml"]` with `sorted(VALID_EPS)` (or a filtered subset). ~1 LOC, prevents drift.

6. **Fix `utils/cli.py:82`** — replace `SUPPORTED_DEVICES` (`["CPU","GPU","NPU"]`) with lowercase-consistent `["cpu","gpu","npu"]` or derive from `_VALID_DEVICES`. ~3 LOC.

7. **Delete `sysinfo/device.py._VALID_DEVICES`** — after step 1, import from `ep_device.py`. ~1 LOC deleted.

8. **Delete `utils/constants.py.SUPPORTED_DEVICES`, `DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE` or move `DEVICE_TO/TYPE` maps to `ep_device.py`** — `DEVICE_TO_DEVICE_TYPE`/`DEVICE_TYPE_TO_DEVICE` use `ort` imports; keep near `ort` usage in `utils/constants.py` unless `ep_device.py` already imports `ort`.

9. **Move `get_provider_for_device` to `ep_device.py`** after step 1. Update `precision.py` to import and re-export for backward compat if needed.

10. **Task 8 follow-up (out of scope here)**: Delete legacy `_build_session_options` instance method in `session.py:460-483` including the `_device_policy_map` inline dict. Tracked by `TODO Task 8 [bridge]` markers.

**Estimated LOC change**: ~90 lines moved/deleted, ~40 new import lines, net reduction ~50 LOC.

---

## Section 7: Open questions

1. **`precision.py._EP_TO_DEVICE` has `"cuda"→"gpu"` and `"tensorrt"→"gpu"` but `_SHORT_TO_CANONICAL` has no entry for either.** Are these intended to be valid `--ep` values in `wmk compile`? If not, they are dead entries in `VALID_EPS`. If yes, `_SHORT_TO_CANONICAL` needs `"cuda"→"CUDAExecutionProvider"` and `"tensorrt"→"TensorrtExecutionProvider"`.

2. **`utils/constants.py.SUPPORTED_DEVICES` uses uppercase `"CPU"/"GPU"/"NPU"` while every other `_VALID_DEVICES` and `click.Choice` list uses lowercase.** Whether the fix is rename-to-lowercase or add a case-insensitive normalize step needs a decision before fixing `utils/cli.py`.

3. **`normalize_ep_name` (utils) vs `expand_ep_name` (ep_device)** — two functions with the same contract but different alias sets. `normalize_ep_name` handles `ov` and `vitis`; `expand_ep_name` does not. Decide: add missing aliases to `_SHORT_TO_CANONICAL` and redirect all callers to `expand_ep_name`, or keep `normalize_ep_name` as a None-safe wrapper delegating to `expand_ep_name`.

4. **`sysinfo/device.py.get_ep_device_map()`** — no callers found in `src/`. Confirm it is dead before deleting; may be used by external callers or tests.

5. **`commands/build.py:370-372`** hardcodes the NPU-capable EP list for auto-selection. This should eventually be derived from `_SHORT_TO_CANONICAL` values filtered by device, but the "auto-select NPU EP" logic is heuristic and may need explicit curation regardless.
