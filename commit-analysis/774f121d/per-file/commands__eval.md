# src/winml/modelkit/commands/eval.py

## TL;DR
Small change — three lines of effective diff. Replaces the legacy
`sysinfo.resolve_device(device)` call with the `VALID_DEVICES`-driven
Click Choice and removes the call-site `resolved_device` indirection
(the device string is now passed verbatim to `WinMLEvaluationConfig`).
The `--device` Choice list source-of-truth swap matches the rest of
the batch.

## Diff metrics
- 8 lines changed (3 insertions / 5 deletions per `--stat`).
- Two hunks: `--device` Choice (one line), evaluate() body
  (three lines removed — `resolve_device` import + call +
  `resolved_device` assignment).

## Role before vs after
Role unchanged: `winml eval` evaluates a model's accuracy on a dataset.
The pre-state's `device` resolution at the CLI boundary is **deleted**;
the device string flows through to `WinMLEvaluationConfig(device=device, ...)`
and downstream consumers handle resolution themselves (or accept
`"auto"` directly).

## Symbol-level changes

### `--device` Click decorator
```python
type=click.Choice(["auto", *sorted(VALID_DEVICES)], case_sensitive=False),
```
Same shape change as the rest of the batch — `["auto","cpu","gpu","npu"]`
hardcoded list → `VALID_DEVICES`-driven Choice.

### Removed: pre-flight device resolution
Before:
```python
from ..sysinfo import resolve_device
resolved_device, _ = resolve_device(device)
...
WinMLEvaluationConfig(..., device=resolved_device, ...)
```

After:
```python
WinMLEvaluationConfig(..., device=device, ...)
```

The `resolved_device` indirection is dropped. The eval command now
**passes the raw `--device` value through** (lowercased by Click
Choice). Resolution moves downstream to `WinMLEvaluationConfig` /
`evaluate(config)`.

### Import
- Added top-level: `from ..session import VALID_DEVICES`.
- Removed local-scoped: `from ..sysinfo import resolve_device`.

## Behavior / contract changes

### (a) `device="auto"` is now passed downstream
Pre-state: `sysinfo.resolve_device("auto")` returned a concrete
category (e.g., `"npu"`) before passing to `WinMLEvaluationConfig`.
Post-state: `"auto"` reaches the config and `evaluate(config)` is
responsible for handling it. **This means `WinMLEvaluator` (or
whoever consumes `config.device`) must now do the deduction**, or
gracefully accept `"auto"` as a sentinel.

The downstream impact lives in `eval/evaluate.py` (per `git status`,
not modified in this commit's commands batch). If that path doesn't
handle `"auto"`, the eval command silently breaks on its own default.

### (b) Display
`display_eval_report(...)` prints `f"[dim]Device:[/dim]     {cfg.device}"`.
Pre-diff: showed the resolved string ("npu"). Post-diff: shows whatever
the user typed (potentially "auto" verbatim) since the resolution
is no longer at the CLI boundary. If `evaluate()` updates
`config.device` to the resolved value, the display still shows the
resolution. If not, the display shows "auto" — a UX regression.

### (c) Choice list source-of-truth
`VALID_DEVICES` now drives the list. New device classes added to the
catalog (e.g., CUDA) appear in the Choice automatically.

## Cross-file impact
- `..session` must export `VALID_DEVICES` (confirmed in prior commit).
- `WinMLEvaluationConfig` / `evaluate` must accept `"auto"` (or
  resolve it internally). This is the load-bearing assumption.
- `eval/evaluate.py` not in this batch; flagged as a dependency.

## Risks / subtleties
- **`device="auto"` flows downstream untouched**: if `evaluate()` or
  `WinMLEvaluator` doesn't know how to handle `"auto"`, the new
  default is broken. Worth verifying with a manual run or unit test.
- **Display regression possibility**: if downstream doesn't update
  `cfg.device` with the resolved value, the "Device: auto" line in
  the eval report is meaningless. Solvable by adding an `actual_device`
  field (mirroring `perf.py`'s pattern) but this commit doesn't do that.
- **Inconsistency vs other commands**: `compile.py` and `perf.py`
  resolve at the CLI boundary (and pass a typed `WinMLEPDevice` /
  resolved string into config). `eval.py` and `build.py` and `analyze.py`
  pass strings through. The batch is half-migrated; the eventual goal
  per the commit body is presumably full `EPDevice` propagation, but
  it's deferred for non-perf/compile commands.

## Simplification opportunities
- **`eval.py` should adopt `EpAtSourceParamType` for `--ep`**: it
  doesn't have an `--ep` flag at all (the diff confirms — only
  `--device`). Adding one would be a separate change.
- **`actual_device` field** mirroring `perf.py` would make the display
  resilient to "auto"-pass-through.
- **The two-line resolution removal** is the right shape — pre-state
  needlessly resolved at the CLI boundary just to pass downstream.
  Now the downstream owns its own resolution. Clean.

## Open questions / TODOs surfaced
- Does `evaluate()` / `WinMLEvaluationConfig` handle `"auto"`? Verify
  in the `eval/evaluate.py` review.
- Should `eval` adopt `EpAtSourceParamType` for `--ep`? It doesn't
  have an `--ep` option — should it? `winml eval` against an ONNX
  file presumably needs an EP pin to evaluate on the right device.
  Today it must rely on the EP baked into the ONNX or some implicit
  default. Worth a brainstorm.
- `actual_device` display field? See above.
