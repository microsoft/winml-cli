# src/winml/modelkit/analyze/core/doc_constraint_checker.py

## TL;DR
Lint hygiene only: drops the `# noqa: PERF203` suppression comment on a `try/except` inside
`DocConstraintChecker._load_*` (the loop that loads operator constraint Dataframes). Zero
behavioural change — the suppression is no longer needed because the project's ruff
configuration was loosened or the rule retired upstream. Nothing else in the file or
surrounding code is touched.

## Diff metrics
- 1 insertion / 1 deletion (net 0), one token.
- Touches one `except Exception as e:` line inside the per-op-constraint loader loop near
  line 127.
- No imports, no signatures, no logic, no tests touched.

## Role before vs after
Before: the `try/except` was annotated with `# noqa: PERF203` to silence ruff's
"try-except in loop" performance warning, signalling that the loop body's exception handler
was an intentional design choice.

After: the suppression is gone; the `except` block is now untagged. Either the rule was
muted in `pyproject.toml`/`ruff.toml` config so the local suppression became redundant, or
this is a sweep cleaning up stale `noqa` markers across the codebase.

## Symbol-level changes
- `DocConstraintChecker._load_constraints` (or sibling loader; the loop that builds
  `op_dfs[op_type]`): `except Exception as e:  # noqa: PERF203` → `except Exception as e:`.
- No other symbol changed.

## Behavior / contract changes
None. Error-logging path is identical; constraint Dataframes load with the same semantics.

## Cross-file impact
- Coupled with similar `# noqa: PERF203` deletions in `model_validator_manager.py` (this
  same commit) — suggests a project-wide sweep. If the PERF203 rule still fires on the
  current ruff version, CI lint will flag these sites again.

## Risks / subtleties
- **Lint regression risk.** If a contributor later upgrades ruff or re-enables PERF203 in
  the project config, these `try/except`-in-loop sites will flag. No test catches lint
  drift, only CI.

## Open questions / TODOs surfaced
- Was PERF203 disabled in `pyproject.toml` as part of this commit (or earlier)? Worth
  grepping the lint config to confirm — if not, this deletion is incorrect and CI will
  re-flag.

## Simplification opportunities
- The loop-with-try pattern itself is the underlying cost PERF203 was warning about; if
  per-op failures are common, batching the loads or pre-filtering invalid `op_type`s would
  outperform try/except-per-iteration. Not a v2.9 concern, but a candidate for future
  cleanup.
