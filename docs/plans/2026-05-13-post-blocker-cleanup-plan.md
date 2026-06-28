# Post-BLOCKER Cleanup Plan

**Date:** 2026-05-13
**Branch:** `feat/op-tracing-refactor`
**Base:** `eb37f6c3` (Gap #1 fix — dual-singleton DLL register)
**Driver:** Audit + per-file review punch list, with BLOCKERs resolved.

## Status snapshot

All BLOCKERs resolved. Current branch chain since `1bea4cf`:

```
eb37f6c3  fix(session): WinMLEPRegistry defensive against dual-singleton DLL register — Gap #1
7c64ca23  fix(session): WinMLSession.compile() actually runs ORT ModelCompiler — Gap #3
e70c2a20  fix(consolidation): complete plan §C, §5, §6 — audit follow-up
62807ac9  feat(compile): winml compile --device flag + ep_device threading
0a9d422a  refactor(session): _SHORT_TO_CANONICAL -> _SHORT_TO_FULL naming
3b155784  refactor(session): consolidate EP/device taxonomy under session/ facade
db39b80d  feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor
1bea4cf8  feat: MVP v2 (base)
```

## Remaining work, bundled

### Bundle A — Monitor pipeline silent failures (IMPORTANT)

Three coupled silent failures in the QNN op-tracing data path that corrupt results when SDK behavior shifts.

#### A1. `session/monitor/qnn_monitor.py:439-441` — `int()` truncation
```python
# Current:
cycles = int(meta.get("accel_execute_cycles", 0) or 0)
us = int(meta.get("accel_execute_us", 0) or 0)
# Bug: if QNN ever returns "12345.6" as a string (legal float), int() raises
# ValueError → caught by upstream try/except → entire op record is dropped silently.
# If QNN returns "12345.6" parseable through float() but truncated by int(),
# cycle_to_us ratio is computed off-by-N → every duration_us in the report is wrong.

# Fix:
cycles = round(float(meta.get("accel_execute_cycles", 0) or 0))
us = round(float(meta.get("accel_execute_us", 0) or 0))
```

Add a unit test that feeds a float-string and confirms a sensible value rather than a crash.

#### A2. `session/monitor/qnn/_internal.py` — hard `dict[key]` × 14
Lines `311, 342-355, 402-403`. Bare `meta["time_us"]`, `meta["cycles"]`, etc. catches the `KeyError` in the outer `_try_qhas` `except Exception` → status silently becomes `"basic_fallback"` with no logged reason for which key was missing.

**Fix:** Replace bare `dict[key]` with `dict.get(key)` + explicit `KeyError` raising on missing required keys, naming the key in the message. The outer `except` then logs the named key so users can diagnose SDK schema drift.

Add a unit test that feeds a QHAS JSON missing one key and asserts a useful error message.

#### A3. `commands/perf.py:1577-1605` — JSON written before op-trace status check
The benchmark JSON file is written to disk BEFORE the `if op_tracing:` block checks `trace_result.status == "no_data"` → exits 4 with a usable JSON on disk → CI sees the exit code but the JSON artifact misleads diagnostic tools.

**Fix:** Reorder: check `trace_result.status` first; if `"no_data"`, raise and exit 4 BEFORE the JSON write. Or: include the trace status in the JSON itself so consumers know the run was partial.

### Bundle B — `_build_session_options` mutation safety (IMPORTANT)

`session/session.py:172` — the free function mutates the caller-supplied `base_session_options` in place via `add_session_config_entry`. If ORT's API isn't idempotent for repeated same-key calls, monitor session-config entries accumulate across `perf()` windows.

**Investigation step:** Use `uv run python -c "import onnxruntime as ort; help(ort.SessionOptions.add_session_config_entry)"` to confirm semantics. If it overwrites on same key, mutation in-place is fine. If it appends or raises, fix.

**Two possible fixes:**
- Copy `base_session_options` defensively before mutation (`copy.deepcopy` doesn't work on SessionOptions — need a manual copy)
- Track which entries we set in this call and clear them on cleanup

Add a regression test: run two consecutive `perf()` blocks with different monitor configs and assert the second one sees only its own session-config entries.

### Bundle C — Taxonomy gaps (MEDIUM)

#### C1. `cuda` / `tensorrt` accepted by precision, unresolvable at session

`session/ep_device.py`: `_EP_TO_DEVICE` has `cuda`/`tensorrt` keys but `_SHORT_TO_FULL` does not. `resolve_precision(ep="cuda")` succeeds; downstream `register_ep("cuda")` raises `EPNotDiscovered`.

**Fix:** Add `"cuda": "CUDAExecutionProvider"` and `"tensorrt": "TensorRTExecutionProvider"` to `_SHORT_TO_FULL`. If we don't actually support them in practice, alternatively remove from `_EP_TO_DEVICE` and `VALID_EPS`. The current schizophrenic state is worst.

#### C2. Consolidate the two EP-registration singletons

The Gap #1 fix added a defensive check in `WinMLEPRegistry.register_ep()`. The proper fix is to remove `winml.py:WinML.register_execution_providers()` calls from `_is_ep_available_locally()` (or whatever caller is duplicating registration) and let `WinMLEPRegistry` be the sole registrar.

Risk: `winml.py:WinML` may be the canonical entry point for some other code paths. Audit before deleting.

### Bundle D — Analyze command (MEDIUM)

`winml analyze -m <model>` runs the runtime-checker unconditionally when the rule zip is missing. For ConvNeXt's 667 nodes × ~11 unique op types, that's 30+ min — observed timeout in CLI verification cmd 2.

**Two mitigations:**
- Ship the rule zip with the package (or include a download step at first-use).
- Dedup probes by unique op-type — should be 11 probes not 667 → 60× speedup.

Single-agent investigation: read `commands/analyze.py` + the analyze runtime-checker. Decide which mitigation lands first.

## Execution plan

Three agents, with priorities:

| Agent | Bundle | Scope | Estimated |
|---|---|---|---|
| **Agent X** | Bundle A (monitor) | 3 files, 3 small fixes + 3 new tests | 30 min |
| **Agent Y** | Bundles B + C (session + taxonomy) | 2 files, idempotency check + 2 new mappings + 1 regression test | 20 min |
| **Agent Z** | Bundle D (analyze) | Investigation + speedup or rule-zip integration | 45-60 min (open-ended) |

Agents X and Y can run in parallel (different files). Agent Z is open-ended and may be punted to a follow-up PR if scope creeps.

After each agent commits:
- `uv run ruff check --fix src/ tests/` clean
- `uv run pytest tests/unit/` green
- For Bundle A: smoke `winml perf` with QNN+NPU still produces clean output

## What's NOT in this plan

- Native QNN HTP AOT crashes on QDQ-quantized graphs (upstream QNN SDK issue).
- Anything in `feat/update-pkg-deps` territory (separate branch, separate concerns).
- Removing the legacy `WinML.register_execution_providers()` singleton entirely (deferred to a follow-up).

## Rollback plan

Each bundle commits separately. If Bundle A or B breaks runtime, `git revert <sha>` rewinds cleanly. Bundle D is investigation-only at first; commit anything that materializes.
