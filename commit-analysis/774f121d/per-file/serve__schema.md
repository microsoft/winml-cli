# src/winml/modelkit/serve/schema.py

## TL;DR
One-character class rename: `EpSwitchRequest` → `EPSwitchRequest`. Aligns with the project's naming convention (`docs/naming-convention.md`): "EP" is an initialism and should be uppercase. The class itself (a Pydantic model with a single `ep` field) is unchanged. Companion changes in `serve/app.py` update the one importer + one type hint.

## Diff metrics
- Lines: +1 / -1 (net 0)
- Hunks: 1 (class declaration)
- Symbols renamed: 1 (`EpSwitchRequest` → `EPSwitchRequest`)

## Role before vs after
Unchanged. Defines the request body schema for `POST /v1/ep`. One field: `ep: str` with description "EP short name: cpu, dml, qnn, openvino" (note: not a `Literal` type, so any string is accepted at parse and rejected at handler-time validation — see Open Questions).

## Symbol-level changes
- `class EpSwitchRequest(BaseModel):` → `class EPSwitchRequest(BaseModel):`. Field definitions and docstring unchanged.

## Behavior / contract changes
- **Pydantic generates the OpenAPI component name from the class `__name__`.** Anyone consuming `/openapi.json` sees `EPSwitchRequest` now; before it was `EpSwitchRequest`. JSON wire format (the `{"ep": "..."}` payload) is identical.
- TypeScript/Python client-side generators that pinned the old name will produce stale code until regenerated. The actual HTTP contract is unchanged.

## Cross-file impact
- `serve/app.py` imports the renamed class (see `serve__app.md`).
- No other in-tree consumer (verified).
- Any external documentation that referenced the class name needs update. None found in `docs/` for this specific class (`grep "EpSwitchRequest"` in `docs/` returns nothing).

## Risks / subtleties
- **OpenAPI client regeneration is silent on this kind of rename.** A consumer who didn't notice the schema change will silently lose IDE autocomplete on the old name but their runtime code (JSON serialization) still works.
- The `ep` field uses `str` not `Literal["cpu", "dml", "qnn", "openvino"]`. So Pydantic accepts any string and the actual validation happens in the route handler (`if ep not in _VALID_EPS: raise HTTPException(...)`). The description field documents the allowed values but the type doesn't enforce them. Not changed by this commit; remaining design choice.

## Simplification opportunities
- Tighten `ep` to a `Literal` type derived from `session.VALID_EPS`. Would let Pydantic reject invalid strings at parse, surfacing 422 instead of 4xx-via-handler, and would auto-update the OpenAPI schema's enum constraint. Trade-off: every catalog addition rebuilds the schema, which may not be desirable for stable API contracts.
- This file likely contains other `EpFoo` / `EpBar` names that should also be migrated. Worth a quick `grep "Ep[A-Z]"` follow-up; if found, do them in one commit.

## Open questions / TODOs surfaced
- Should `ep` be a `Literal` type rather than `str`? Today the field-level validation is descriptive only.
- Are there other Pydantic models in this file (or sibling `schema.py` files) with the same naming-convention issue? A targeted audit would be worth it.
