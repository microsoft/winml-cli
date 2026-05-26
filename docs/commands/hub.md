# winml hub

> Browse the curated winml-cli catalog of validated models and benchmarks.

## When to use this

Use `winml hub` to discover which HuggingFace models have been validated end-to-end
by the winml-cli team — exported, quantized, compiled, and benchmarked on real Windows
ML devices. It is the starting point when you want a model that is known to work
before investing time in a custom build.

## Synopsis

```bash
$ winml hub [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model-type` | `-t` | string | `null` | Filter the catalog by model architecture (case-insensitive). Examples: `bert`, `roberta`, `vit`. |
| `--task` | `-k` | string | `null` | Filter by HuggingFace task (case-insensitive). Examples: `text-classification`, `image-segmentation`. |
| `--model` | `-m` | string | `null` | Show detailed latency and accuracy benchmarks for a specific model ID. Accepts exact ID or an unambiguous substring. |
| `--output` | `-o` | path | `null` | Save the displayed results to a JSON file. Works for both list and detail views. |
| `--help` | `-h` | flag | — | Show help and exit. |

> `winml hub` reads a local catalog bundled with the package — no network access is
> required. It does not accept `--device`, `--ep`, or `--precision`.

## How it works

The catalog is stored in `winml/modelkit/data/hub_models.json` and is loaded
directly from the installed package data without any network call. Each catalog
entry records the model ID, task, architecture type, per-EP latency statistics
(avg, P50, P90, P95, P99, min, max, QPS), and per-EP accuracy results compared
against a floating-point FP32 baseline. The accuracy verdict uses three levels:
`PASS` (drop within tolerance), `AT_RISK` (borderline), and `REGRESSION` (exceeds
threshold). When `--output` is provided, the displayed data — whether a filtered
list or a single model's detail — is written as indented JSON to the specified path.

## Examples

```bash
# List all validated models in the catalog
$ winml hub
```

```text
╭─── winml-cli Catalog  |  12 validated model(s) ───────────────────────────╮
│  Model                             Task                    Model Type     │
│ ├ microsoft/resnet-50              image-classification    resnet         │
│ ├ bert-base-uncased                fill-mask               bert           │
│ ├ ProsusAI/finbert                 text-classification     bert           │
│ └ ...                                                                     │
╰────────────────────────────────────────────────────────────────────────────╯
Use  winml hub --model <id>  to see perf and accuracy details.
```

```bash
# Filter to BERT-family models only
$ winml hub --model-type bert
```

```bash
# Filter by task — show only text-classification models
$ winml hub --task text-classification
```

```bash
# Combine filters — BERT models for text classification
$ winml hub --model-type bert --task text-classification
```

```bash
# Show latency and accuracy details for a specific model
$ winml hub --model ProsusAI/finbert
```

```bash
# Save filtered results to JSON for offline review
$ winml hub --task image-classification --output results/image_catalog.json
```

## Common pitfalls

- **`--task` short flag is `-k`, not `-t`.** The `-t` short flag is taken by
  `--model-type`. Using `-t text-classification` will set the architecture filter,
  not the task filter. Use `-k` or the full `--task` flag.
- **`--model` performs substring matching when no exact match exists.** If the
  substring matches more than one catalog entry, the command raises an error and
  lists the candidates. Use the full model ID to avoid ambiguity.
- **The catalog reflects a point-in-time snapshot.** Models listed in the catalog
  were validated against a specific version of winml-cli, ONNX Runtime, and the
  relevant EP driver. Accuracy and latency may differ on your hardware or with
  updated drivers.
- **`--output` only saves what was displayed.** Combining `--model` with `--output`
  saves the single model's detail dict. Combining a filter with `--output` saves the
  filtered list. There is no flag to dump the entire catalog in one call — omit all
  filters and add `--output` to do so.
- **A model not in the hub can still be used with winml-cli.** The catalog covers
  tested models; `winml inspect` and `winml export` work with any HuggingFace model
  that has a supported architecture, whether or not it appears in the hub.

## See also

- [inspect.md](inspect.md) — check loader, exporter, and task detection for any
  HuggingFace model ID
- [sys.md](sys.md) — verify your environment and EP availability before building
- [How winml-cli Works](../concepts/how-it-works.md) — pipeline overview from export
  to benchmark
- [Quantization & QDQ](../concepts/quantization.md) — understand accuracy verdicts
  and what `drop_pct` measures
