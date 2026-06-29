# src/winml/modelkit/serve/cli_api.py

## TL;DR
Two-character Python 3.11 modernization: replaces `asyncio.TimeoutError` (deprecated alias) with the built-in `TimeoutError`. Two `except` clauses in `_run_with_semaphore` (one wrapping `sem.acquire()`, the other wrapping `asyncio.wait_for(loop.run_in_executor(...), ...)`). Behaviorally identical — `asyncio.TimeoutError` has been an alias for the built-in `TimeoutError` since Python 3.11.

## Diff metrics
- Lines: +2 / -2 (net 0)
- Hunks: 2 (two `except` clauses)
- Symbols touched: 0

## Role before vs after
Unchanged. The `_run_with_semaphore` helper still arbitrates between heavy/light command semaphores, raises `HTTPException(503)` when the queue is saturated past `_SEMAPHORE_TIMEOUT_SEC`, and raises `HTTPException(504)` when an individual command exceeds `_EXEC_TIMEOUT_SEC`.

## Symbol-level changes
- `except asyncio.TimeoutError as exc:` → `except TimeoutError as exc:` (two occurrences). The wrapping `HTTPException` constructors are unchanged.

## Behavior / contract changes
- None at runtime. `asyncio.TimeoutError is TimeoutError` since Python 3.11; before that it was distinct but compatible.
- DeprecationWarning suppression: ruff (`PIE808` / `UP041` rule) was likely flagging the old usage. The new form is the canonical Python 3.11+ idiom.

## Cross-file impact
- None. The exception types are interchangeable at the call site of `_run_with_semaphore`.

## Risks / subtleties
- None. If the deployment ever targets Python < 3.11 again (it shouldn't — pyproject pins ≥ 3.11), the new code still works because the built-in `TimeoutError` has been catchable everywhere; `asyncio` would raise its alias, which is the same class on 3.11+ but a distinct class on older versions. Pinning ≥ 3.11 makes this moot.

## Simplification opportunities
- The two `except TimeoutError as exc:` clauses construct similar `HTTPException` instances. A small helper (`_timeout_to_http(exc, *, status, message)`) would compress. Marginal.

## Open questions / TODOs surfaced
- None. Mechanical modernization.
