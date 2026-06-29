# src/winml/modelkit/commands/_ep_arg.py

## TL;DR

New file (+98 lines): the `--ep <name>[@<source-tag>]` CLI argument parser. `split_ep_at_source(value)` is the standalone function; `EpAtSourceParamType` is the `click.ParamType` that wires it into click decorators. Both surface `click.UsageError` on malformed input by way of `click.ParamType.fail`, replacing the prior pattern of every command writing its own try/except + UsageError boilerplate.

## Diff metrics

- Mode: NEW FILE (+98 / -0).
- Public surface: `split_ep_at_source(value) -> tuple[str, str | None]`, `EpAtSourceParamType` (class).

## Role before vs after

**Before.** Every CLI command (`perf`, `build`, `compile`, `config`) that accepted `--ep` had to parse the `<name>@<source-tag>` form by hand. Per the commit body: "EpAtSourceParamType wires the --ep <name>[@<source-tag>] CLI syntax into click at parse time, replacing the try/except UsageError boilerplate at every command."

**After.** Commands declare `@click.option("--ep", type=EpAtSourceParamType())` and receive a pre-split `tuple[str, str | None] | None`. Validation (whitespace, multiple `@`, empty halves, unknown source tag) fires at click parse time. No more per-command boilerplate.

## Symbol-level changes

### `split_ep_at_source(value: str) -> tuple[str, str | None]` (lines 21-63)

Five validation steps:
1. Whitespace check (`any(c.isspace() for c in value)`) — raise `ValueError`.
2. Multiple-`@` check (`value.count("@") > 1`) — raise `ValueError`.
3. No-`@` shortcut: return `(value, None)`.
4. Split on first `@`: `ep, source = value.split("@", 1)`. Reject empty ep or source.
5. Lowercase source, validate against `VALID_SOURCE_TAGS` — raise `ValueError` on miss.

Returns `(ep, source)` with ep case-preserved (so full names like `"OpenVINOExecutionProvider"` survive — matches `EPDeviceTarget`'s case-sensitive full-name match) and source lowercased.

### `EpAtSourceParamType(click.ParamType)` (lines 66-99)

- `name = "ep_at_source"`.
- `convert(self, value, param, ctx) -> tuple[str, str | None] | None`:
  1. None or empty → None passthrough.
  2. If `isinstance(value, tuple)`: pre-split, return as-is (idempotency for double-convert calls).
  3. Else: `try: split_ep_at_source(value) except ValueError: self.fail(...)`.

## Behavior / contract changes

1. **Source-tag validation happens at click parse time.** A malformed `--ep` value causes click's usage error formatter to fire (showing usage hints, exiting non-zero) — much friendlier than a deep Python traceback five layers down.
2. **`split_ep_at_source` is whitespace-strict.** `"openvino @pypi"` raises; the prior hand-rolled parsers may have been lenient (depends on the implementation each command used).
3. **The EP name half is case-preserved.** Allows full-form EP names like `"OpenVINOExecutionProvider"`. The source half is always lowercased.
4. **Multiple `@`** raises — the syntax is `<ep>@<source>`, no nested forms.
5. **`convert` accepts pre-split tuples idempotently.** Documented as a click callback re-invocation defense.
6. **Empty string → None.** Matches click's "option not provided" shape.

## Cross-file impact

- `commands/perf.py`, `commands/build.py`, `commands/compile.py`, `commands/config.py` all import `EpAtSourceParamType` and use it for `--ep`.
- `commands/_ep_arg.py` imports `VALID_SOURCE_TAGS` from `..session.ep_device`. The closed set is the v2.9 canonical seven (`bundled`, `pypi`, `nuget`, `msix-microsoft`, `msix-workload`, `winml-catalog`, `directory`).
- Tests at `tests/unit/commands/test_ep_arg.py` (+236 lines) cover this file.
- The `--ep <name>` form (no `@`) still works; the splitter returns `(name, None)` and downstream `EPDeviceTarget(ep=name, device=..., source=None)` runs the Scenario A path.
- `build.py` and `config.py` reject the source-pin (no @-tag allowed) via duplicated reject blocks in each callback. `compile.py` and `perf.py` accept the source pin. The decision lives downstream of the ParamType.

## Risks / subtleties

1. **The `convert` signature uses `# type: ignore[override]`** because click's `ParamType.convert` has a return-type annotation that doesn't match the parameterized tuple. The override is intentional.
2. **`split_ep_at_source` raises `ValueError`, not `click.UsageError`.** The conversion to UsageError happens via `self.fail(str(e), param, ctx)` inside `EpAtSourceParamType.convert`. So callers that import `split_ep_at_source` directly (tests, programmatic SDK users) get ValueError; CLI users get UsageError. Consistent.
3. **The empty-string-to-None branch is BEFORE the tuple-passthrough branch** — so an empty tuple `()` would fail the `value == ""` check (since `() != ""`) and then be matched by `isinstance(value, tuple)` and returned as-is. **Defensible**: empty tuple is an unusual but valid pre-parsed shape. The bigger surprise is what happens if `split_ep_at_source` ever gets a tuple (it doesn't from this file but could from misuse) — it would raise `TypeError` inside `c.isspace()`. Low risk.
4. **The `idempotency: click may invoke convert() twice` comment** (lines 91-93) describes a real click behavior: when an option has a callback that returns the parsed value, click calls `convert` again on the callback's return. The defensive `isinstance(value, tuple)` short-circuit handles this. Correct.
5. **`split_ep_at_source` does not normalize the EP name to a short form.** A user typing `--ep QNNExecutionProvider@pypi` gets `("QNNExecutionProvider", "pypi")` returned. Downstream `EPDeviceTarget.__post_init__` validates via `_FULL_TO_SHORT` so it works — but the asymmetry between the case-preserved EP and the case-folded source is one place to remember.
6. **No EP-name validation at this layer.** A user typing `--ep qnnz` (typo) passes the splitter; the failure surfaces in `EPDeviceTarget.__post_init__` later. Acceptable layering choice (the CLI doesn't have to know the catalog) but worth confirming the downstream error message reads well.
7. **`VALID_SOURCE_TAGS` reaches into a submodule** (`from ..session.ep_device import VALID_SOURCE_TAGS`) rather than the package facade. CLAUDE.md import rules allow relative-submodule imports; this file follows them. A future cleanup could re-export `VALID_SOURCE_TAGS` from `session/__init__.py` to consolidate the import surface — currently it's NOT in the facade's `__all__`.

## Simplification opportunities

1. **A `unpack_ep_arg(ep) -> tuple[str | None, str | None]` helper** would collapse the boilerplate `ep_part, source_part = ep if ep else (None, None)` repeated in each consuming command's body.
2. **A `EpAtSourceParamType.reject_source(ep, command_name)`** companion classmethod would collapse the `build.py` / `config.py` duplicated "source pin not allowed for this command" rejection block (currently ~7 lines per command).
3. **`VALID_SOURCE_TAGS` should be re-exported from `session/__init__.py`** so this file can import from the facade. Currently the submodule import works but conflicts with the facade-only-imports posture.
4. **`EpAtSourceParamType.name = "ep_at_source"`** could be more user-friendly: `"<ep>[@<source>]"` would show in click's help as `[<ep>[@<source>]]` — more discoverable.
5. **The `convert` method could be type-annotated** as `def convert(self, value: str | tuple[str, str | None] | None, ...) -> tuple[str, str | None] | None:` without the `# type: ignore`. Would require careful click typing investigation.
6. **The `idempotency` defensive branch could `assert len(value) == 2`** for a hard fail on malformed pre-split tuples.

## Open questions / TODOs surfaced

- Should `--device <class>` get a similar `DeviceParamType` to validate against `VALID_DEVICES` at click parse time? Currently `--device` accepts any string and validation fires deep in `EPDeviceTarget.__post_init__`. Symmetry with `--ep` would be cleaner UX.
- Should `VALID_SOURCE_TAGS` move to the session package facade? Today's import path is `from ..session.ep_device import VALID_SOURCE_TAGS` — reaches past the facade. Worth a small cleanup.
- Is multi-`@` syntax (`--ep openvino@pypi@2024.0`) on the roadmap as a way to pin a specific version inside a source? Currently rejected. If yes, the splitter's `value.count("@") > 1` check would need to relax.
- A duplicated `_reject_source` block in `build.py` and `config.py` is dead weight (Simplification #2). One classmethod on `EpAtSourceParamType` would consolidate.
