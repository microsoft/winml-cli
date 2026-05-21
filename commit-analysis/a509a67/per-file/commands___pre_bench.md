# src/winml/modelkit/commands/_pre_bench.py

## TL;DR
New private CLI helper (85 lines, brand-new file) that renders the "pre-bench
identity block" — a 3-sub-block Rich `Panel` shown immediately before the
`winml perf` benchmark loop. Mirrors the design mockup at
`docs/design/perf/console_mockup.py`. Sub-block 2 (Surface) is a forward-looking
placeholder; only sub-blocks 1 (Model) and 3 (Device) emit content today.

## Diff metrics
- Lines added: 85
- Lines removed: 0
- Net: +85
- New / modified: **new file**

## Role before vs after
Before: did not exist. The perf command emitted no structured pre-benchmark
identity panel — model/device info was scattered across log lines or implicit.

After: provides `print_pre_bench_block(console, *, model_id, task, opset,
inputs, outputs, cached_onnx_path, onnx_file, device, ep)`. A pure Rich-render
helper called once per `winml perf` invocation just before iteration starts.
Underscore-prefixed module name signals private-to-commands status (per
project import rules — not re-exported from `commands/__init__.py`).

## Symbol-level changes
New public function:
- `print_pre_bench_block(console: Console, *, model_id: str | None, task: str
  | None, opset: int | None, inputs: Sequence[(name, dtype, shape)] | None,
  outputs: Sequence[(name, dtype, shape)] | None, cached_onnx_path: str | None,
  onnx_file: str | None, device: str, ep: str) -> None`
  - All identity-card fields are keyword-only and individually optional
    (typed `| None`) — caller can render whatever subset is available.
  - `device` and `ep` are required positional-by-name strings (resolved
    EPDevice → string).
  - Branches:
    1. If `model_id` truthy → full HF identity card (Model / Task / Opset /
       Inputs / Outputs / Cached ONNX rows, each row only added if value
       provided).
    2. Else if `onnx_file` truthy → minimal "ONNX file" row.
    3. Else → no Model panel at all.
  - Always prints the Device panel (Device + EP rows) at the bottom.
  - Surface sub-block (#2) is an inline comment placeholder — no code, no
    panel — explicitly "forward-looking" per the docstring.

New private helper:
- `_fmt_io(specs) -> str` — formats `[(name, dtype, shape_tuple), …]` into
  a single comma-separated line like
  `"input_ids (int64, (1, 128)), attention_mask (int64, (1, 128))"`.

Imports:
- `rich.panel.Panel`, `rich.table.Table` (runtime).
- `collections.abc.Sequence` and `rich.console.Console` are typing-only
  (`TYPE_CHECKING`), keeping the import cost low for the CLI hot path.

## Behavior / contract changes
- Renders **two** panels at most (Model + Device); the design's intermediate
  Surface panel is reserved and silently skipped. Callers that expect three
  panels visually need to know sub-block #2 is intentionally empty.
- All shape elements are stringified via `str(d)` so symbolic dimensions
  (e.g., `"batch_size"`) and ints both render uniformly.
- `expand=True` on both panels — they fill the console width.
- The function returns `None`; it has no side effects beyond `console.print`.
- No exception handling: any rendering error propagates (Rich rarely raises,
  but caller has to be tolerant if it does).

## Cross-file impact
- Consumed by `commands/perf.py` (commit body says "Perf console: 4-column
  basic / 10-column detail tables, pre-bench identity block"). Perf is
  expected to import this as `from ._pre_bench import print_pre_bench_block`.
- Mirrors the structure of `docs/design/perf/console_mockup.py` — that file
  is the design source of truth and any future surface-block content should
  be added in parallel.
- Underscore prefix means it is **not** re-exported through
  `commands/__init__.py`; per the project's import rules in CLAUDE.md
  ("Private (`_`-prefixed) symbols are never added to `__init__.py`"), the
  only legitimate consumers are sibling files in `commands/`.

## Risks / subtleties
- The `(device, ep)` parameters are pre-stringified — caller is responsible
  for converting `EPDevice` to display strings. There is no validation that
  the strings are consistent with the resolved device.
- `inputs`/`outputs` tuple shape `(name, dtype, shape_tuple)` is an
  implicit contract — there is no dataclass or TypedDict guarding it. If
  `perf.py` later refactors its tensor-spec representation, this helper
  must be kept in sync.
- The "Surface" sub-block comment (line 69) is dead today but reserved
  prime real estate — future quantization/precision metadata (cf. the
  DRAFT QuantSpec design mentioned in the commit body) likely lands here.
- No tests for this helper; given it is pure-rendering, that is consistent
  with project norms.

## Open questions / TODOs surfaced
- What does the Surface sub-block render once the QuantSpec design lands?
  (Per commit body: "DRAFT QuantSpec design — follow-on for per-variant
  quantization attached to EPDevice".)
- Should the I/O tuple be replaced by a typed dataclass to make the contract
  explicit and discoverable?
- No `__all__`; both public and private symbols are reachable via `from
  ._pre_bench import *`. Probably intentional for a private helper, but
  worth noting.
