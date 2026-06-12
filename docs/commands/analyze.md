# winml analyze

> Verify an ONNX model is compatible with a target execution provider before deployment.

## When to use this

Use `winml analyze` before running the full build pipeline to confirm that your ONNX model's operators are supported by the intended execution provider and device. It surfaces operator gaps and actionable recommendations early, saving time that would otherwise be spent on a failed compile or quantize run.

## Synopsis

```bash
$ winml analyze [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `PATH` | *(required)* | Path to the ONNX model file to analyze. |
| `--ep` | | choice | `auto` | Target execution provider. Accepts full names (e.g., `QNNExecutionProvider`) or short aliases (`qnn`, `openvino`, `vitisai`, `cpu`, `cuda`, `dml`, `nvtensorrtrtx`, `migraphx`). Use `all` for every rule-data-backed EP, or `auto` to infer from local availability. |
| `--device` | | `cpu\|gpu\|npu\|all\|auto` | `auto` | Target device type. `auto` infers from local availability; `all` evaluates all rule-data-backed devices. |
| `--verbose` | `-v` | flag | off | Enable verbose output. |
| `--quiet` | `-q` | flag | off | Suppress non-essential output. |
| `--config` | `-c` | `PATH` | *(none)* | Build configuration file (YAML/JSON). |
| `--output` | | `PATH` | *(none)* | Save the full JSON result to a file in addition to printing the console summary. |
| `--information` / `--no-information` | | flag | enabled | Include detailed per-operator recommendations and remediation hints in the output. Pass `--no-information` for a compact pass/fail summary. |
| `--htp-metadata` | | `PATH` | *(none)* | Path to an HTP metadata JSON file (produced by `winml export`). Enriches subgraph pattern extraction by mapping nodes back to their source module hierarchy. Benefits all target EPs. |
| `--run-unknown-op` / `--no-run-unknown-op` | | flag | disabled | For operators not in the rule database, build a minimal ONNX graph and run it on the target EP locally to determine support. Enable when local EP libraries are available. |
| `--save-node` | | `partial\|unsupported` | *(none)* | Save partial or unsupported node subgraphs to disk for further investigation. Can be specified multiple times: `--save-node partial --save-node unsupported`. |
| `--optim-config` | | `PATH` | *(none)* | Save the auto-discovered optimization config (merged across all analyzed EPs) to a JSON file. |

## How it works

`winml analyze` loads the ONNX model and runs a static analysis pass via `ONNXStaticAnalyzer`. For each operator (and recognized subgraph pattern), the analyzer consults the target EP's rule database. For operators not in the database, it can optionally probe them locally when `--run-unknown-op` is enabled. The combined answer classifies each node as supported, partial, unsupported, or unknown (see [Analyze and optimize](../concepts/analyze-and-optimize.md) for definitions).

The analysis always produces a **lint** result — the pass/fail verdict. When `--information` is enabled (the default), it additionally produces an **autoconf** result: a set of fusion-flag suggestions that, if applied in the optimize stage, would resolve partial or unsupported patterns. Pass `--no-information` to skip autoconf and get just the lint verdict.

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | All operators are fully supported on the target EP. |
| `1`  | At least one operator is unsupported, partially supported, or unknown. |
| `2`  | Input or configuration error (bad path, unknown EP, etc.). |

Exit codes make `winml analyze` safe to use as a CI gate with `set -e` or `$?` checks.

## Examples

Analyze using auto-detected EP and device:

```bash
$ winml analyze --model microsoft/resnet-50.onnx
```

The output shows a live progress table per EP followed by an `ANALYSIS SUMMARY` section. Each EP line displays support counts in `S/P/U/Unk` format (Supported / Partial / Unsupported / Unknown) with color-coded indicators.

Check QNN NPU support using the short alias:

```bash
$ winml analyze --model bert-base-uncased.onnx --ep qnn --device NPU
```

Check Intel OpenVINO GPU support and print operator-level recommendations:

```bash
$ winml analyze --model bert-base-uncased.onnx --ep openvino --device GPU --information
```

Save the full JSON result for offline inspection while still printing the console summary:

```bash
$ winml analyze --model facebook/convnext-tiny-224.onnx --output results.json
```

Use HTP metadata for enhanced subgraph pattern extraction:

```bash
$ winml analyze --model bert-base-uncased.onnx \
    --ep qnn --device NPU \
    --htp-metadata bert-base-uncased_htp_metadata.json
```

Run a lint-only pass (no recommendations) for a CI gate:

```bash
$ winml analyze --model model.onnx --ep qnn --device NPU --no-information
echo "Exit code: $?"  # 0 = clean, 1 = issues, 2 = input error
```

Dump unsupported subgraphs to disk for debugging:

```bash
$ winml analyze --model model.onnx --ep qnn \
    --save-node partial --save-node unsupported \
    --output result.json
```

Enable local execution for operators not in the rule database:

```bash
$ winml analyze --model model.onnx --ep qnn --device NPU --run-unknown-op
```

## Common pitfalls

- **Omitting `--ep` uses `auto` (inferred from local availability)** — to analyze every EP regardless of what is installed, pass `--ep all`. Specify `--ep <name>` when you know your target hardware.
- **Exit code 1 is not a hard failure** — it means at least one operator is unsupported, not that the model cannot run at all. Many EPs fall back unsupported nodes to the CPU EP automatically; review the recommendations before deciding to restructure the model.
- **`--htp-metadata` is EP-agnostic** — HTP metadata enriches pattern extraction before any EP-specific checks, so it benefits all target EPs equally. You do not need separate metadata files per EP.
- **`--run-unknown-op` is disabled by default** — operators not covered by the rule database are classified as `UNKNOWN` (not unsupported) unless you explicitly pass `--run-unknown-op` to probe them locally. Enable it only when the target EP's libraries are available on the local machine.
- **The model path must point to an existing `.onnx` file** — symbolic HuggingFace model IDs are not accepted; export the model first with `winml export`.

## See also

- [Analyze and optimize](../concepts/analyze-and-optimize.md) — conceptual deep dive on classifications, lint vs autoconf, and the analyzer/optimizer loop
- [eps-and-devices.md](../concepts/eps-and-devices.md) — background on ONNX operators and execution providers
- [export.md](export.md) — convert a HuggingFace model to ONNX before analyzing
- [compile.md](compile.md) — compile the model for the target EP after analysis passes
- [sys.md](sys.md) — list EPs available on the current machine
