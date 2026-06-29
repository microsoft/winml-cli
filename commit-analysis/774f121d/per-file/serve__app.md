# src/winml/modelkit/serve/app.py

## TL;DR
Two-occurrence rename: `EpSwitchRequest` → `EPSwitchRequest`. Reflects the naming-convention rule that "EP" is an initialism (per `docs/naming-convention.md`). Affects one import and one type hint on the `switch_ep` POST handler. No behavior change.

## Diff metrics
- Lines: +2 / -2 (net 0)
- Hunks: 2 (one import line; one route handler signature)
- Symbols touched: 1 (`EpSwitchRequest` rename — the underlying class is renamed in `serve/schema.py`)

## Role before vs after
Unchanged. Still the FastAPI app factory; still exposes `/v1/ep` as a POST endpoint that switches the active execution provider on the model slot manager.

## Symbol-level changes
- Import: `from .schema import EpSwitchRequest, ...` → `from .schema import EPSwitchRequest, ...` (single character change inside a multi-line import).
- Route handler type annotation: `async def switch_ep(request: EpSwitchRequest) -> dict[str, Any]:` → `async def switch_ep(request: EPSwitchRequest) -> dict[str, Any]:`.

## Behavior / contract changes
- None at runtime. The class is identical Pydantic structure; only its Python name changes.
- The OpenAPI / FastAPI schema name may change as a side effect (Pydantic uses class `__name__` for the generated component name). API clients consuming `/openapi.json` will see `EPSwitchRequest` instead of `EpSwitchRequest`. Any TypeScript/Python client generator that pinned the old name will need regeneration.

## Cross-file impact
- `serve/schema.py`: class rename happens there (see `serve__schema.md`).
- Anywhere else in `src/` that imported `EpSwitchRequest`: verified zero matches outside `serve/app.py` and `serve/schema.py`.
- Tests that exercise the `/v1/ep` endpoint should be agnostic to the Python class name (they POST a JSON payload).

## Risks / subtleties
- **OpenAPI schema name change.** Any downstream consumer who generated client code from a prior `openapi.json` snapshot has a brand-new component name. If they regenerate, they get matching code. If they don't, JSON contracts still work but their TypeScript/Python types are stale.
- Reviewers should verify there isn't an outdated docs file (e.g. `docs/api-reference.md` or a Swagger-UI'd README) that still references `EpSwitchRequest`.

## Simplification opportunities
- The rename is the simplification — aligns with naming-convention guidance. No further follow-up.

## Open questions / TODOs surfaced
- None. The rename is mechanical and obviously correct given the naming convention.
