# Review: `src/winml/modelkit/commands/_pre_bench.py`

**Status:** new file
**Lines added/removed:** 85+ / 0-

## 1. Purpose

`_pre_bench.py` is a new display-only module that renders the pre-benchmark identity
panel before the inference loop starts. It replaces the ad-hoc `_print_model_info`
function that was previously embedded in `perf.py`. The panel has three logical
sub-blocks: model identity (HF or raw ONNX path), a reserved surface slot
(placeholder, emits nothing), and the resolved device/EP pair.

## 2. Changes summary

- New file extracted from `perf.py` logic.
- `print_pre_bench_block` accepts the full identity payload as keyword-only args; callers
  supply `model_id` (HF path) or `onnx_file` (direct path), never both.
- `_fmt_io` is a private helper that formats a list of `(name, dtype, shape)` triples
  into a compact single-line string.
- Surface sub-block is structurally present as a comment — forward-looking slot, no
  output yet.

## 3. Per-symbol review

### `print_pre_bench_block`

- **Role:** Render the 3-sub-block pre-benchmark console panel to a caller-supplied
  `Console` instance.
- **Signature:**
  ```python
  def print_pre_bench_block(
      console: Console,
      *,
      model_id: str | None,
      task: str | None,
      opset: int | None,
      inputs: Sequence[tuple[str, str, tuple[int | str, ...]]] | None,
      outputs: Sequence[tuple[str, str, tuple[int | str, ...]]] | None,
      cached_onnx_path: str | None,
      onnx_file: str | None,
      device: str,
      ep: str,
  ) -> None
  ```
- **Behavior:** When `model_id` is set, renders a Rich `Table.grid` with model name,
  task (if set), opset (if set), inputs/outputs (if provided), and cached ONNX path
  (if any) inside a "Model" panel. When `model_id` is `None` but `onnx_file` is set,
  renders just the file path. Then always renders a "Device" panel with `device` and
  `ep` rows.
- **Invariants:** At least one of `model_id` or `onnx_file` should be set; when both
  are `None` only the Device panel is rendered with no Model panel. The function does
  not raise in that case.
- **Risks / concerns:**
  - When both `model_id` and `onnx_file` are set simultaneously (caller bug), only the
    `model_id` branch fires; `onnx_file` is silently ignored (`elif` structure at
    line 62). No assertion guards this invariant, so a future caller could pass both
    and observe confusing output. Low severity — all current callers are correct.
  - The `device` and `ep` args are plain strings, not `EPDevice` fields. `perf.py`
    passes `str(self._model.device)` and `str(self.config.ep) if self.config.ep else "auto"`,
    so the pre-bench panel may display `"auto"` for EP even after the session compiled to
    a concrete EP. This is cosmetically misleading but not a correctness bug — the
    session resolves EP at compile time, and the ep shown here is the *requested* value.
  - `Console` is only imported under `TYPE_CHECKING` (line 23). At runtime the type
    annotation is just a string, so mypy can verify it but there is no isinstance guard.
    This is intentional and idiomatic.
- **Tests:** `tests/unit/commands/test_pre_bench.py`

### `_fmt_io`

- **Role:** Format a list of `(name, dtype, shape)` triples into one readable line.
- **Signature:** `def _fmt_io(specs: Sequence[tuple[str, str, tuple[int | str, ...]]]) -> str`
- **Behavior:** Produces e.g. `"input_ids (int64, (1, 128)), attention_mask (int64, (1, 128))"`.
  Empty shapes render as `"()"` — caller in `_io_specs_from_config` already handles
  missing shapes by substituting `()`.
- **Invariants:** Pure function; no mutation.
- **Risks / concerns:** None.
- **Tests:** `tests/unit/commands/test_pre_bench.py`

## 4. Cross-cutting concerns

- **Audit gap:** `ep` shown in the Device panel always uses the config-level short name
  or `"auto"` string, not the resolved `EPDevice.ep` (canonical name). This is a known
  display inconsistency noted in the mockup spec but not a behavioral regression.
- **Legacy `device=` callers:** None in this file.
- **CLI help text:** Not applicable — this is a display helper, not a CLI option.

## 5. Confidence level

**High.** New pure-display module with full keyword-only interface, no runtime state,
and a dedicated test file. The only actionable risk (both args set) is defensive.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Low | `_pre_bench.py:62` | `elif onnx_file` branch silently wins when `model_id=None`; no guard against callers mistakenly passing both. |
| Low | `_pre_bench.py:76` | EP label shows the *requested* short name or `"auto"`, not the compiled `EPDevice.ep` canonical name. Can mislead a user who did not pass `--ep` explicitly. |
