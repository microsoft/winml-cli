# src/winml/modelkit/commands/eval.py

## TL;DR
Eval CLI: the `--device` Choice is now derived from `VALID_DEVICES` (was a hard-coded `["auto","cpu","gpu","npu"]` list), and the CLI no longer pre-resolves the device at the boundary — it hands `device` (possibly `"auto"`) straight through to `WinMLEvaluationConfig`. The previous `sysinfo.resolve_device` call site is gone entirely; `evaluate()` calls the typed `session.resolve_device(device=...)` internally one layer below. No `EPDevice` is constructed in this file — it is purely the click adapter. The commit body's claim that "eval/evaluate.py migrated to ep_device at CLI boundaries" applies to `winml/modelkit/eval/evaluate.py`, not to this CLI wrapper.

## Diff metrics
- 8 lines changed (4 insertions / 4 deletions per `--stat`).
- Two hunks: one in the top imports / decorator block, one in the function body.

## Role before vs after
Role unchanged. `winml eval` still:
- Accepts a model path or HF model id (`-m / --model`, `--model-id`).
- Builds a `DatasetConfig` and `WinMLEvaluationConfig`.
- Calls `evaluate(config)` from `winml.modelkit.eval`.
- Displays a Rich eval report and optionally writes JSON.
What changed: the CLI no longer performs any device pre-resolution. The raw `device` arg (possibly `"auto"`) is passed straight through to `WinMLEvaluationConfig.device`; auto-pick is the callee's job inside `evaluate()`, which calls the typed `session.resolve_device(device=...)` internally.

## Symbol-level changes
- **Top imports**: added `from ..session import VALID_DEVICES` (placed below `click`, separated by blank lines — kept out of the `TYPE_CHECKING` block).
- **`--device` click.Choice**: was `["auto", "cpu", "gpu", "npu"]`, now `["auto", *sorted(VALID_DEVICES)]`. `default="auto"`, `show_default=True`, help text unchanged.
- **In-function import**: the previous `from ..sysinfo import resolve_device` is **removed entirely**. No pre-resolution helper is imported in this file anymore.
- **Call site**: the previous `resolved_device, _ = resolve_device(device)` line is **deleted**. The raw `device` arg flows straight into `WinMLEvaluationConfig(device=device)`, and `evaluate()` resolves it via the typed `session.resolve_device(device=...)` downstream.

## Behavior / contract changes
- **Device choices set**: derived from the catalog. As of this commit, observable set is identical (`auto`, `cpu`, `gpu`, `npu`). Future catalog edits propagate without touching this file.
- **`evaluate()` contract unchanged**: it still takes `WinMLEvaluationConfig(device=<str>)`, not an `EPDevice`. The CLI adapter does no `resolve_device(ep, device)` call — that work happens downstream in `eval/evaluate.py` (out of scope for this file). The CLI is now a pure pass-through; even the previous CLI-side device-string pre-resolution is gone.
- **No new flags**. There is no `--ep` flag on `winml eval` (in contrast with `winml perf` and `winml compile`). That is consistent with this commit's scope — eval continues to deduce the EP downstream.

## Cross-file impact
- Hard dependency on the public re-export `VALID_DEVICES` from `..session`. No device-resolution helper is imported here anymore — the CLI is a pure pass-through.
- The downstream `evaluate()` function in `winml/modelkit/eval/evaluate.py` is what actually consumes an `EPDevice` per the commit body — but that boundary is one layer below this CLI, invisible at the call site here.

## Risks / subtleties
- Click `--help` output now lists devices alphabetically (`auto, cpu, gpu, npu`) where before it was `auto, cpu, gpu, npu` (already alphabetical). Cosmetically equivalent for current catalog contents, but the rendering rule has changed.
- If the user passes a device string that survives Click's Choice validation but isn't installed (e.g. `gpu` on an NPU-only box), the failure point shifts into `evaluate()` / `WinMLEvaluationConfig`. This file does not surface the new `DeviceNotFound` / `EPNotDiscovered` exceptions; depending on `evaluate.py`'s error handling the user may or may not get the friendlier remediation hints the commit advertises.
- The deprecated `sysinfo.resolve_device` is now gone (renamed). Any third-party callers reaching into `commands.eval` for it would have broken — but nothing in this file's public surface advertised that import, so external impact is unlikely.

## Open questions / TODOs surfaced
- No `--ep` flag on `winml eval` — is that intentional, or just deferred? The commit body lists `eval/evaluate.py` as migrated to `ep_device` but does not mention the CLI gaining `--ep`. Worth confirming whether `WinMLEvaluationConfig` carries an `ep_device` field downstream and, if so, whether a CLI surface is planned.
- The two redundant imports (`from ..datasets import DatasetConfig` and `from ..eval import WinMLEvaluationConfig, evaluate` placed inside the function body) are unchanged and still load lazily — fine for CLI start-up time but worth noting as a deliberate pattern.
