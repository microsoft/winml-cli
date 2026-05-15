# Review: `src/winml/modelkit/commands/eval.py`

**Status:** modified (small touch)
**Lines added/removed:** 2+ / 2-

## 1. Purpose

`eval.py` is the CLI entry point for the `wmk eval` command. The changed lines rename
a single import in the device-resolution call from `resolve_device` (old sysinfo name)
to `resolve_device_category` (new name), keeping the evaluation command's behavior
identical while aligning with the naming refactor applied across `config.py`, `perf.py`,
and `evaluate.py`.

## 2. Changes summary

- Line 215: `from ..sysinfo import resolve_device` → `from ..sysinfo import resolve_device_category`
- Line 217: `resolved_device, _ = resolve_device(device)` → `resolved_device, _ = resolve_device_category(device)`

## 3. Per-symbol review

### `eval` function — device resolution block (lines 215–217)

- **Role:** Translate the user's `--device` CLI string into a concrete device category
  string that is stored in `WinMLEvaluationConfig.device`.
- **Signature:** (unchanged; only the internal call site updated)
- **Behavior:** `resolve_device_category(device)` returns `(category_str, ep_list)`
  where `category_str` is e.g. `"cpu"`, `"npu"`, or `"gpu"`. The result populates
  `config.device` which is then passed into `evaluate(config)`. The `_` discard is
  correct — the EP list is not needed at the eval CLI boundary.
- **Invariants:** `device` is a `click.Choice(["auto", "cpu", "gpu", "npu"])` validated
  by Click before this function body executes, so `resolve_device_category` never
  receives an unexpected string.
- **Risks / concerns:**
  - `eval.py` stores only the resolved *category* string (`resolved_device`) in
    `WinMLEvaluationConfig`, not an `EPDevice`. The actual EPDevice construction is
    deferred to `evaluate.evaluate._load_model` (see `evaluate.py` review). This
    split boundary is intentional per the refactor design but means the eval command
    has **two** resolve steps: category-resolve here and EP-resolve inside `_load_model`.
    If either diverges from the other (e.g. a future `WinMLEvaluationConfig` field
    gets an `ep` added), the hardcoded `_default_ep_for_device` map in `_load_model`
    could produce a different EP than what a hypothetical `--ep` flag would. This is
    a known gap, not introduced by this PR.
  - The `eval` command has no `--ep` option. The default EP derivation inside
    `_load_model` (`{"cpu": "cpu", "npu": "qnn", "gpu": "dml"}`) is therefore the
    sole determinant of which EP is used at eval time. Users who want `vitisai` on NPU
    cannot override this without code changes.
- **Tests:** `tests/unit/eval/test_eval.py` (config/result roundtrip, `_resolve_task`)

## 4. Cross-cutting concerns

- **Audit gap:** The eval command does not expose `--ep` and has no way to override
  the default EP derived from device. This is a pre-existing limitation not introduced
  by this PR, but the `_default_ep_for_device` duplication across `eval.py:_load_model`,
  `perf.py:_load_model`, and `perf.py:perf` is a DRY violation — all three inline the
  same `{"cpu": "cpu", "npu": "qnn", "gpu": "dml"}` dict.
- **Legacy `device=` callers:** The old `resolve_device` (sysinfo) is fully replaced.
  No legacy call sites remain in this file.
- **CLI help text:** Unaffected; no option signatures changed.

## 5. Confidence level

**High.** Two-line rename with no semantic change at the eval command boundary. The
deeper EPDevice construction inside `_load_model` is covered separately.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Medium | `eval.py:217` + `evaluate.py:138–141` | `_default_ep_for_device` is duplicated inline at 3 call sites across eval.py and perf.py. A future device addition (e.g. `"fpga"`) must be updated in all three places. Should be centralized in ep_device.py or sysinfo. |
| Low | `eval.py` (entire file) | `wmk eval` has no `--ep` option. NPU eval is always routed to QNN; VitisAI users on NPU cannot override. Pre-existing gap, not new. |
