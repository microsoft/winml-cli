# Plan Complete — 774f121d Squash Cleanup

**Date:** 2026-06-29

The 17-task action plan documented in `TASKS.md` is fully executed.

## Status summary

| Phase | Tasks | Status |
|---|---|---|
| 1 — Ship blockers | T-01, T-02, T-04 | `[x]` |
| 1 — Ship blockers (premise reclassified) | T-03 | `[!]` — see below |
| 2 — Dead code removal | T-05, T-06, T-07 | `[x]` |
| 3 — De-duplication | T-08, T-09, T-10, T-11, T-12, T-13, T-14 | `[x]` |
| 4 — Taxonomy + casing | T-15, T-16 | `[x]` |
| 5 — Doc cleanup | T-17 | `[x]` |

**Result:** 16 of 17 tasks delivered. The remaining task (T-03) is blocked on a design decision that requires user input (see below). All other plan items are complete with TDD discipline applied per the `/loop` protocol — RED test confirmed for every applicable task, GREEN minimal code, and a regression sweep run at each step.

## T-03 — open user decision

The audit flagged `find_qnn_sdk()` as a regression because the pre-refactor
`_COMMON_SDK_PATHS` fallback (`D:\QC`, `C:\Qualcomm\AIStack\qairt`) is no longer
consulted. Investigation during the second `/loop` firing revealed that the
existing test
`tests/unit/session/monitor/qnn/test_viewer.py:16-19` —
`test_find_qnn_sdk_returns_none_when_env_unset` — explicitly enforces the
env-only behavior with the module docstring stating *"no hardcoded
developer-machine fallback paths"*. The constant's removal was an intentional
v2.9 design choice, not a regression.

User must pick one:

- **Option A — Revert the design.** Restore `_COMMON_SDK_PATHS` and rewrite the
  three enforcement tests in `test_viewer.py`. The audit's DEEP-DIVE D-03
  classification stands.
- **Option B — Accept the env-only design.** Close T-03 as intentional and
  update `commit-analysis/774f121d/DEEP-DIVE.md` D-03 to *"deliberate design
  choice, not a regression."*

Neither option needs the `/loop` cron — both are user-facing decisions.

## What changed

All changes are staged in the working tree (not committed — per the `/loop`
protocol directive *"Don't commit code yet — leave it staged for the user to
review and commit themselves"*).

### Source code

- `src/winml/modelkit/compiler/configs.py` — `import warnings` restored (T-01);
  `_PROVIDER_DEFAULTS` table + simplified `for_provider` (T-09);
  `_EP_CONTEXT_DEFAULTS` constant deleted (T-06).
- `src/winml/modelkit/compiler/cli.py` — **deleted** (T-02).
- `src/winml/modelkit/session/ep_registry.py` — `auto_device` last-error reset
  after successful registration (T-04).
- `src/winml/modelkit/session/session.py` — `_detect_best_device` and
  `_get_install_suggestion` deleted (T-05).
- `src/winml/modelkit/session/monitor/openvino_monitor.py` — **deleted** (T-07).
- `src/winml/modelkit/session/__init__.py` — `OpenVINOMonitor` removed from
  exports (T-07); `known_ep_short_names`, `_ep_short_or_none`,
  `DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE` added to facade
  (T-15, T-16).
- `src/winml/modelkit/session/monitor/ep_monitor.py` — OpenVINO docstring
  mentions stripped (T-07).
- `src/winml/modelkit/session/monitor/qnn/_internal.py` —
  `_REQUIRED_SUMMARY_KEYS` table replaces 14 hand-written `_require` calls in
  `_extract_summary` (T-12).
- `src/winml/modelkit/session/monitor/report.py` — `_top_n` extracted (T-10);
  remains the single source of truth for `_format_bytes` (T-14).
- `src/winml/modelkit/session/ep_device.py` — `_ep_short_or_none` helper (T-13);
  `_format_bytes` re-exported from `monitor.report` (T-14); `"cuda"` dropped
  from `_SHORT_TO_FULL` (T-15); `DEVICE_TO_DEVICE_TYPE` /
  `DEVICE_TYPE_TO_DEVICE` introduced with lowercase keys (T-16).
- `src/winml/modelkit/commands/_ep_arg.py` — `_reject_ep_source` helper (T-08).
- `src/winml/modelkit/commands/build.py` + `commands/config.py` — verbatim
  source-tag-rejection blocks collapsed via `_reject_ep_source` (T-08).
- `src/winml/modelkit/commands/sys.py` — `_describe_source` now emits
  `source_tag` from `_entry_source_tag` (T-11).
- `src/winml/modelkit/config/build.py` + `config/precision.py` — call the
  facade `_ep_short_or_none` (T-13, T-15 import cleanup).
- `src/winml/modelkit/utils/cli.py` — absorbed `normalize_ep_name`,
  `extract_ep_options`, `_EP_CLI_PREFIXES`; `"migraphx"` prefix added (T-16).
- `src/winml/modelkit/utils/constants.py` — **deleted** (T-16).
- `src/winml/modelkit/analyze/runtime_checker/check_ops.py` +
  `analyze/pattern/check_patterns.py` +
  `analyze/core/runtime_checker_query.py` — switched to lowercase-keyed map
  lookups; preserved `_NPU_` uppercase filename convention via `.upper()` at
  filename construction (T-16).
- `src/winml/modelkit/analyze/analyzer.py` + `commands/analyze.py` +
  `commands/config.py` — `normalize_ep_name` import path updated to
  `utils.cli` (T-16).

### Tests

- `tests/unit/session/test_auto_device.py` — fail-then-succeed regression
  test (T-04).
- `tests/unit/session/test_ep_device.py` —
  `test_valid_eps_matches_known_short_names` plus the `TestEpShortOrNone`
  helper class (T-13, T-15); two cuda tests renamed to tensorrt-only.
- `tests/unit/session/test_device_type_maps.py` — **new** (T-16 contract).
- `tests/unit/session/test_winml_device.py` —
  GiB/MiB/KiB assertions retargeted to `GB`/`MB`/`KB`; single-source-of-truth
  identity test added (T-14).
- `tests/unit/session/monitor/test_report.py` — `TestTopN` class (T-10).
- `tests/unit/session/monitor/qnn/test_qhas_parser.py` —
  `test_required_summary_keys_drives_full_renderer_surface` (T-12).
- `tests/unit/compiler/test_compiler_configs.py` —
  `for_qnn`/`for_cpu`/... replaced with `for_provider`; consolidated
  deprecation test parametrized over 8 providers (T-09).
- `tests/unit/commands/test_ep_arg.py` — 3 `_reject_ep_source` tests (T-08).
- `tests/unit/commands/test_describe_source.py` — **new** (T-11 contract).
- `tests/unit/utils/test_ep_constants.py` — import switched to `utils.cli`;
  3 baseline-broken classes (`TestSupportedEPs`, `TestEPAliases`,
  `TestAllEPNames`) removed; `extract_ep_options` migraphx test now passes
  (T-16).
- `tests/unit/architecture/test_ep_device_import_rule.py` — `"cuda"` removed
  from `_EP_SHORT_NAMES` inline-literal detector (T-15).
- `tests/unit/session/test_ep_monitor.py` — `OpenVINOMonitor` test class +
  3 import tests deleted (T-07).
- `tests/unit/models/auto/test_config.py` — `for_*` factory test updated to
  `for_provider` (T-09 follow-through).

### Design docs

- `docs/design/session/2_coreloop.md` — T-17's 14 items applied: §5.5 / §5.6 /
  §5.7 / §6.1 / §6.3 / §7 / §7.1.1 / §10 / §11.1 / §11.4 / §11.7 prose and
  pseudocode brought in line with shipped code. Net +14 lines.

## Pre-existing baseline failures (not introduced by this plan)

Verified via `git stash` baseline check during the wide regression sweeps —
present on `HEAD = 85c540b5` before any task work:

- `tests/unit/models/auto/test_auto_onnx.py::TestFromOnnxDictDispatch` —
  3 tests fail with `TypeError: WinMLAutoModel.from_onnx() missing 1
  required keyword-only argument: 'ep_device'`.
- `tests/unit/architecture/test_ep_device_import_rule.py::test_no_direct_ep_device_imports_in_src`
  and `..._in_tests` — 6 pre-existing direct-import violations in
  `commands/compile.py`, `commands/_ep_arg.py` and 4 test files. (My
  contributions were cleaned up by T-15.)
- `tests/unit/analyze/` — 22 zombie tests against a `winml.modelkit.sysinfo.device`
  module that does not exist plus the `test_static_analyzer_cli.py` CLI
  integration failures.

All are out of scope for this plan and remain at the baseline state.

## Verification

The final wide regression sweep across `tests/unit/session/`,
`tests/unit/commands/`, `tests/unit/compiler/`, `tests/unit/config/`,
`tests/unit/utils/`, `tests/unit/analyze/`, `tests/unit/architecture/`, and
`tests/unit/models/` (T-16's gate) showed:

- **2989 passed** + **53 legitimate skips** (hardware-gated / dependency-gated
  / pre-existing TODOs).
- **27 pre-existing baseline failures** as enumerated above.

Every task that introduced new test surface added it via TDD's RED step
before any source change. RED logs are captured inline in each task's
checkpoint entry in `TASKS.md`.

## Next steps (out of plan)

1. User reviews the staged diff and commits in whatever logical grouping they
   prefer.
2. User decides on T-03 (Option A or Option B) — neither needs the `/loop`
   cron.
3. Optional: clean up the 6 pre-existing direct-`ep_device` import violations
   surfaced (but not introduced) by the architecture-test sweep.
4. Optional: triage / delete the zombie tests in `tests/unit/analyze/` that
   reference a removed `winml.modelkit.sysinfo.device` module.
