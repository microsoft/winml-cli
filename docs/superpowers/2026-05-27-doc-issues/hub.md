# Issues: docs/commands/hub.md

Source verified against: `src/winml/modelkit/commands/catalog.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **The command documented is `winml hub` but the source registers it as `winml catalog`.** Source line 362 is `@click.command()` with no `name=` argument; the function is named `catalog` (line 387), and the CLI is wired to `winml catalog` per the docstring (lines 6–17). Every invocation example in the doc uses `winml hub` (e.g. `$ winml hub`, `$ winml hub --model-type bert`) — these will all fail unless there is an alias registered elsewhere. The doc must either be renamed to `catalog.md` and updated throughout, or the alias must be verified.

- **`--model` / `-m` flag for detail view does not exist in source.** The doc table lists `--model / -m` as "Show detailed latency and accuracy benchmarks for a specific model ID" (doc line 23). The source `catalog` command (lines 362–429) has no `--model` option. The source accepts only `--model-type / -t`, `--task / -k`, `--ep`, `--device`, and `--output`. There is no per-model detail view in the source at all. Any user running `winml hub --model ProsusAI/finbert` will get an "unrecognized option" error.

- **`--ep` and `--device` flags are absent from the doc flag table entirely.** Source lines 377–385 add `ep_option(required=False)` and `device_option(required=False, default=None)`. The doc only lists four flags and makes no mention of `--ep` or `--device`. These are functional filters that change output — omitting them is a content gap that will confuse users trying to filter by EP or device.

## Important (misleading or stale)

- **"How it works" describes per-EP latency stats (avg, P50, P90, P95, P99, min, max, QPS) and accuracy verdicts (PASS/AT_RISK/REGRESSION)** — the source `catalog.py` makes no reference to these fields. The catalog data source is `hub_models.json` (line 64) and the rendering code (lines 276–306) shows columns: Model, Task, Size, Model Type, and optionally Devices or EPs. No latency stats or accuracy verdict columns appear in the rendered output. The "How it works" section describes functionality that either does not exist in this command or belongs to a different one (e.g., `winml perf`).

- **Accuracy verdict description (`drop_pct`) in "How it works"** is not supported by any code in `catalog.py`. The `See also` section points to `quantization.md` to explain `drop_pct`, but this doc is describing `winml catalog` which has no such output.

- **Example output shows "winml-cli Catalog"** (doc line 50) but source line 301 renders `"WinML CLI Catalog"`. Minor discrepancy.

- **Pitfall says `--model` performs substring matching** (doc line 90–92) — this flag does not exist in source. The entire pitfall is based on a non-existent feature.

- **Pitfall "no flag to dump entire catalog"** (doc line 97–99) says "omit all filters and add `--output`" — the source does support `--output` with no filters (lines 428–429), so this pitfall hint is correct, but the surrounding text refers to `--model` which does not exist.

## Minor (polish)

- The synopsis `$ winml hub [options]` uses the wrong command name; should be `$ winml catalog [options]`.
- Cross-reference at doc line 108 reads `hub.md` in `sys.md` which will be a broken link if this doc is renamed.
- The `--task` short flag warning pitfall ("use `-k`, not `-t`") is correct → source line 373 confirms `-k`.

## Verified correct (key claims checked)

- `--model-type / -t` filter exists, case-insensitive → source lines 363–369.
- `--task / -k` filter exists, case-insensitive → source lines 370–376.
- `--output / -o` saves JSON → source lines 428–429 via `cli_utils.output_option`.
- Catalog loaded from local package data (no network) → source lines 53–65.
- `_filter_models` applies exact case-insensitive equality on `model_type` and `task` → source lines 68–88.
