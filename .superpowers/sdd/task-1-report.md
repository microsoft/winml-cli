What I implemented

- Added a pure-data export compatibility policy resolver per Task 1.
- Files added/updated:
  - src/winml/modelkit/export/policy.py (new)
  - src/winml/modelkit/export/__init__.py (re-exports)
  - tests/unit/export/test_export_policy.py (new tests)

What I tested (exact commands + results)

1) RED (expected failure after writing tests, before implementation)
Command:
  uv run pytest tests\unit\export\test_export_policy.py -q
Result (abbreviated):
  ERROR collecting tests/unit/export/test_export_policy.py
  ModuleNotFoundError: No module named 'winml.modelkit.export.policy'

2) Implemented policy.py and updated export.__init__ then ran tests
Command:
  uv run pytest tests\unit\export\test_export_policy.py -q
Result (abbreviated):
  collected 6 items
  tests\unit\export\test_export_policy.py ...... [100%]
  6 passed in 0.48s

3) Attempted ruff fix as requested:
Command:
  uv run ruff check --fix src\winml\modelkit\export\policy.py tests\unit\export\test_export_policy.py src\winml\modelkit\export\__init__.py
Result:
  error: Failed to spawn: `ruff` (program not found in PATH)

4) Committed changes (pre-commit hooks ran; ruff hooks executed and reformatted/fixed files)
Command:
  git add ... && git commit -m "feat(export): add export compatibility policy resolver" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
Result:
  Commit succeeded. short SHA f991880f

TDD evidence

- RED output (initial pytest run) shows import error for missing module (expected).
- GREEN output (post-implementation pytest) shows 6 passed.

Files changed

- Affected files committed:
  - src/winml/modelkit/export/policy.py (new)
  - src/winml/modelkit/export/__init__.py (modified to re-export policy types)
  - tests/unit/export/test_export_policy.py (new)

Self-review findings

- Implementation follows the brief and provides:
  - ExportCompatibilityConfig with serialization/deserialization checks
  - ExportPolicyTarget, ExportCompatibilityRule types
  - resolve_export_compatibility() and export_policy_targets_for_request()
  - Default rule: QNNExecutionProvider -> transformers_attention='eager'
- Used relative imports in src/ as required.
- Ensured policy module does not import heavy libs (torch/transformers).
- Type-checking imports used for "Sequence" to satisfy lint.

Issues / Concerns

- Running "uv run ruff" failed in this environment (ruff not found). Pre-commit hook run during git commit did run ruff and fixed style issues automatically. You may want to ensure ruff is available for local developer runs.
- I kept the TransformersAttentionPolicy as a Literal["eager"]; the brief specified the type alias. No model-specific logic was added.

Commits created

- f991880f feat(export): add export compatibility policy resolver

Report file path

.C:\Users\qiowu\source\repos\copilot-worktrees\winml-cli\dingmaomaobjtu-refactored-funicular\.superpowers\sdd\task-1-report.md

Controller follow-up:
- uv run --with ruff ruff check --fix src\winml\modelkit\export\policy.py tests\unit\export\test_export_policy.py src\winml\modelkit\export\__init__.py
- Result: All checks passed; no tracked file changes.
