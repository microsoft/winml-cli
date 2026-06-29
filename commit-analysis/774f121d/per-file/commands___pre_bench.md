# src/winml/modelkit/commands/_pre_bench.py

## TL;DR
**NEW file (+85)**. Single-function helper that renders the pre-benchmark
identity block — a 3-sub-block Rich panel (Model identity, Surface
placeholder, Device) shown before the benchmark loop starts. Mirrors the
mockup in `docs/design/perf/console_mockup.py`. Consumer is `perf.py`
(two call sites: `PerfBenchmark.run` for the HF path, `_run_onnx_benchmark`
for the ONNX-direct path).

(Note: per the brief's batch index this is listed as "NEW FILE +85". The
git diff confirms it's a new file in this commit — it did not exist
at `7a66c024`.)

## Diff metrics
- 85 lines added, 0 deleted (NEW file).
- One public function (`print_pre_bench_block`), one private helper
  (`_fmt_io`).

## Role
Replaces the legacy `_print_model_info(io_config, *, task, device)`
helper that lived inside `perf.py` and rendered a single free-form
device/task/io-tensors block. The new shape is a structured 3-panel
"identity card" that mirrors the design mockup in
`docs/design/perf/console_mockup.py`, intended to be reused by other
benchmark-style commands in future (currently `perf` only).

## Symbol-level shape

### `print_pre_bench_block(console, *, model_id, task, opset, inputs,
outputs, cached_onnx_path, onnx_file, device, ep) -> None`

Keyword-only signature with eight optional content fields. Two render
branches:

- **HF identity card**: when `model_id` is set, builds a Table.grid
  with rows for Model, Task, Opset, Inputs, Outputs, Cached ONNX
  (each shown only if its value is provided). Wrapped in a Panel
  titled "Model".
- **ONNX-file card**: when `onnx_file` is set (and no `model_id`),
  builds a single-row Table.grid with the ONNX file path. Wrapped
  in a Panel titled "Model".

The "Surface" sub-block (the design's middle pane) is a documented
placeholder — `# 2. Surface (placeholder; forward-looking — no content
emitted)`. No panel is rendered. The doc comment is explicit so a future
reader doesn't think they're looking at incomplete code.

The Device sub-block always renders: a 2-row Table.grid with `Device:`
and `EP:`, wrapped in a Panel titled "Device".

### `_fmt_io(specs: Sequence[tuple[name, dtype, shape]]) -> str`

Joins a sequence of `(name, dtype, shape)` triples into a single
comma-separated string of `name (dtype, (d1, d2, ...))` entries.
Shape elements can be `int | str` (the `int | str` Union accommodates
dynamic-axis sentinels — `perf.py`'s `_io_specs_from_config` renders
dynamic dims as `"?"` instead of `0`).

## Behavior / contract

- **No I/O fallback**: if `inputs` / `outputs` are `None`, the rows
  are silently dropped. `perf.py`'s `_io_specs_from_config` returns
  `None` when names are missing, so this is the right shape.
- **`opset is None`** suppresses the Opset row (per the inline
  comment in `perf.py`: *"opset is not currently extracted on this
  path; pass None."*).
- **Panel titles fixed strings** ("Model", "Device"). Not parameterized
  — fine for the current single consumer.

## Cross-file impact
- Consumer: `perf.py` only (two call sites). The signature shape
  (kw-only args with str | None) was designed around the call site's
  available data.
- No tests visible in the diff. The visual output is the contract;
  unit-testing the exact Rich rendering would be brittle.

## Risks / subtleties
- **`_fmt_io` is a tiny helper that could be inlined** — it's called
  twice (once per `inputs` / `outputs` row) per call site, so the
  per-call cost is trivial. Keeping it private at module scope is
  fine.
- **The `Surface` placeholder comment** is appropriate per the
  design — but a future reader might add a placeholder Panel
  without realizing it's intentionally blank. The comment helps but
  a `# TODO(perf-surface):` tag would make the link to the design
  doc explicit.
- **No `from __future__ import annotations`-aware Sequence**: the
  hint uses `Sequence[tuple[str, str, tuple[int | str, ...]]]` —
  fine on Python 3.10+. Pre-3.10 callers would have issues, but
  the project's `pyproject.toml` likely pins 3.10+.
- **The Panel title strings are not Rich-markup-aware**: `Panel(...,
  title="Model")` shows literal "Model". If localization is ever
  needed, this is a leaf-level i18n hotspot. Not relevant short-term.
- **`expand=True`** on both Panels means they stretch to console
  width. Combined with `_live_chart.py`'s default chart_width=120,
  the visual baseline is "wide terminals". On 80-col terminals the
  Panels word-wrap.

## Simplification opportunities
- **Surface sub-block could just not be in the code at all**: it's a
  documented placeholder. Removing it entirely (and adding when
  there's content) would shrink the file. Trade-off: the comment
  preserves the design's intent and shows the reader the structure
  they're following.
- **`_fmt_io` consolidation with `BenchmarkResult.input_shapes`
  rendering**: same data, two stringifiers. Could share a single
  shape-formatter. Tangential.
- **Title parameterization**: `Panel(..., title="Model")` and
  `Panel(..., title="Device")` could take a title string parameter
  so consumers (e.g., `winml eval`'s pre-eval block, if added later)
  reuse the renderer. Trivial.

## Open questions / TODOs surfaced
- **`opset` data source**: the call site passes `None`; the build
  pipeline knows the opset. Threading it through would surface a
  useful diagnostic for op-tracing investigations.
- **Surface sub-block content**: what's the design intent? The
  mockup likely shows EP-bound surfaces (NPU memory budget,
  capability flags); when the data is available, this is where
  it renders.
- **`device` and `ep` rendering**: both are stringified by the
  caller. If the caller has a `WinMLEPDevice`, passing it directly
  and letting the renderer call `.device.device_type` / `.ep.ep_name`
  would tighten the contract and surface a cleaner abstraction.
  Today the caller does `str(self._model.device)` and
  `str(self.config.ep) if self.config.ep else "auto"`, which is
  fine but loses the typed information.
