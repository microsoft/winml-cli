# src/winml/modelkit/inference/engine.py

## TL;DR

Pure cosmetic Python-3.11+ modernization: `from datetime import datetime, timezone` → `from datetime import UTC, datetime`, and the single call site `datetime.now(tz=timezone.utc)` → `datetime.now(tz=UTC)`. Zero behavioural change. This file has nothing to do with the v2.9 EP refactor — it appears in the commit only because the squash collapsed 45 incremental commits, one of which happened to touch the import and call site.

## Diff metrics

- Lines changed: +2 / -2 (4 total per `git show --stat`).
- Two lines actually edited:
  - l.34: import line.
  - l.519: call site inside `InferenceEngine.predict()`'s latency-tracking block.
- No functions added, removed, or renamed.
- No public surface change.

## Role before vs after

- **Before:** `from datetime import datetime, timezone` at the top, `self._last_request_at = datetime.now(tz=timezone.utc)` at the call site. Uses the Python 3.10-style `timezone.utc` constant.
- **After:** `from datetime import UTC, datetime`, `self._last_request_at = datetime.now(tz=UTC)`. Uses the Python 3.11+ `UTC` constant introduced in PEP 663-adjacent stdlib work.

`InferenceEngine`'s role as the inference-side orchestrator (pipeline mapping, latency tracking, prediction normalisation) is unchanged.

## Symbol-level changes

- **Import line (l.34):**
  ```python
  # before
  from datetime import datetime, timezone
  # after
  from datetime import UTC, datetime
  ```
  `timezone` is no longer imported because it's no longer used in this module. The two `from datetime import ...` orderings differ (`timezone` comes after `datetime`, `UTC` comes before — alphabetical) — ruff/isort enforce this.

- **Call site (l.519):**
  ```python
  # before
  self._last_request_at = datetime.now(tz=timezone.utc)
  # after
  self._last_request_at = datetime.now(tz=UTC)
  ```
  `datetime.UTC` was added in CPython 3.11 as `datetime.UTC = timezone.utc` — identical object, shorter name.

- **No other changes.** The `InferenceEngine.predict()` method body (the +500-LOC giant) is otherwise untouched.

## Behavior / contract changes

- **None.** `datetime.UTC is datetime.timezone.utc` evaluates to `True` in CPython 3.11+ — they are literally the same singleton. Wire-level behaviour (the timestamp stored on `_last_request_at`) is byte-identical.
- **Python version floor:** this change tightens the minimum supported Python to 3.11. If `pyproject.toml`'s `requires-python` was `>=3.10`, this would break on 3.10 where `datetime.UTC` does not exist. Worth a quick check of the project's declared Python version. (Not done here — out of scope for a per-file analysis.)
- **No new exceptions, no new side effects, no new sentinel values.**

## Cross-file impact

- Zero. No file in the codebase imports `inference.engine.timezone` (it was never re-exported), so the removal of the `timezone` import is local.
- `_last_request_at` is read by `InferenceEngine.health()` / `InferenceEngine.metrics()` (per nearby code patterns); both consumers use the value as a `datetime` regardless of which tz constant produced it.
- If any test patches `inference.engine.timezone`, that test will break (`AttributeError`). Unlikely — `timezone` is never patched in real code — but worth a grep.

## Risks / subtleties

- **Python 3.11 floor.** `datetime.UTC` is unavailable on 3.10. Project must declare `requires-python = ">=3.11"` for this to be safe. Worth verifying `pyproject.toml`.
- **No production behavioural risk.** This is a vocabulary swap.
- **Import-order sensitivity.** Some isort configurations sort `UTC` before `datetime` alphabetically, but `from datetime import UTC, datetime` with `UTC` first looks unusual against the convention "first-import-the-module-name". Ruff handles this correctly. If the project ever switches to a different linter that enforces "alphabetical except for module-name primary imports", a churn-only refactor would follow.

## Open questions / TODOs surfaced

- Was this change deliberate, or a drive-by from a different incremental commit during the 45-commit squash? It has no thematic connection to "v2.9 unified-source EP refactor". The commit body explicitly lists the misc bug fixes (`auto.py:411`, `auto_detect_device` RuntimeError catch, `benchmark = None` in perf.py) but does **not** mention the `engine.py` timezone swap.
- Should the rest of the codebase get the same treatment? A grep for `timezone.utc` across `src/` would surface other call sites. Out of scope for this file, but a one-line ruff/upgrade rule (`pyupgrade` UP017) automates this.

## Simplification opportunities

- **Nothing further to simplify in this 4-LOC patch.** The change *is* the simplification (a shorter, more readable name for the same object).
- **Codebase-wide opportunity:** add `ruff` rule `UP017` to enforce the `UTC` form across the codebase. A grep would identify other `timezone.utc` call sites; one-liner ruff autofix.
- **Module-level note (not from this diff):** the surrounding `InferenceEngine.predict()` method is ~150 LOC of mixed concerns (input validation, pipeline dispatch, raw-tensor fallback, latency tracking, EP-name inference). Breaking it up is out of scope for this commit but a perennial candidate for cleanup.
