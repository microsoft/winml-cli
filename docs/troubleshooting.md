# Troubleshooting

Common issues and solutions when working with winml-cli.

---

## Compile

### Cannot enable compilation: no compile section

```text
UsageError: Cannot enable compilation: no compile section found in the config file
```

**Cause:** Compilation is **off by default** in `winml build`. You passed `--compile` to explicitly enable it, but the config JSON has no `"compile"` section (it's `null`). This happens when the config was generated without a device target that supports EPContext (e.g., `--device cpu` or `--device auto` on a machine without NPU).

**Solution:** Regenerate the config targeting a device that supports compilation (NPU or GPU with an EP that produces EPContext):

```bash
uv run winml config -m <model> -d npu --compile -o output/
```

!!! note
    By default `winml build` skips the compile stage unless `--compile` is passed or the config contains a non-null `"compile"` section. To include compilation in the generated config, specify a device that maps to an EPContext-capable EP (e.g., `-d npu`).

---

### Already a compiled EPContext model

```text
ClickException: model_ctx.onnx is already a compiled EPContext model and cannot be re-compiled
```

**Cause:** You're trying to compile a model that is already an EPContext artifact (the `_ctx.onnx` output).

**Solution:** Run compilation on the original (pre-compiled) ONNX file instead:

```bash
uv run winml compile -m model.onnx -d npu -o output/
```

---

### Provider does not support EPContext compilation

```text
ClickException: Provider 'DmlExecutionProvider' does not support EPContext compilation
```

**Cause:** Not all EPs produce EPContext format. DML and CPU do not support pre-compilation.

**Solution:** EPContext is supported by QNN, OpenVINO, TensorRT, and Vitis AI. For DML/CPU, skip the compile step — the runtime compiles on first load automatically:

```bash
uv run winml build -c config.json -m model -o output/ --no-compile
```

---

## Analyze

### Unsupported nodes persist after analysis

```text
RuntimeError: Unsupported nodes persist after analysis
```

**Cause:** The model contains operators that the selected EP cannot dispatch natively.

**Solution:** Run `winml analyze` with `--optim-config` to identify problematic operators and get recommended graph optimizations:

```bash
# Analyze and output optimization recommendations
uv run winml analyze -m model.onnx --ep qnn --optim-config optim_config.json
```

This produces `optim_config.json` with the auto-discovered optimization flags. Apply them with `winml optimize`, then re-analyze:

```bash
# Apply recommended optimizations
uv run winml optimize -m model.onnx -o model_optimized.onnx -c optim_config.json

# Re-analyze to check if unsupported nodes are resolved
uv run winml analyze -m model_optimized.onnx --ep qnn
```

If unsupported nodes still remain after optimization, consider:

- **Manually modifying problematic nodes** — use tools like `onnx-graphsurgeon` to replace or remove operators the EP cannot handle
- **Using a different EP** (`--ep dml` or `--ep cpu`) that supports the operators in question
- **Checking if a newer opset version** resolves the compatibility gap (re-export with `--opset-version 18`)

---

### Many "unknown" results from constant nodes

When `winml analyze` reports a large number of nodes as "unknown", the model likely hasn't been normalized — it contains raw constant-folding subgraphs, missing shape annotations, or redundant initializer nodes that the analyzer cannot classify.

**Solution:** Run `winml optimize` with no optimization flags to normalize the model (constant folding, shape inference, dead-node elimination), then re-analyze:

```bash
# Normalize only (no fusion flags)
uv run winml optimize -m model.onnx -o model_normalized.onnx

# Re-analyze — constant nodes are now folded, shapes are inferred
uv run winml analyze -m model_normalized.onnx --ep qnn
```

This baseline pass collapses constant subgraphs into initializers and propagates tensor shapes throughout the graph, giving the analyzer enough information to classify nodes correctly.

---

## Build / Cache

### Disk full / out of space

Build artifacts (exported ONNX, optimized graphs, quantized models, compiled EPContext files) are cached under:

```
C:\Users\<user>\.cache\winml
```

This directory can grow significantly after multiple builds with large models. If you encounter disk-full errors or want to reclaim space, it is safe to delete the entire folder:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\winml"
```

The next `winml build` will re-create the cache as needed. Use `--rebuild` to force a full rebuild without relying on cached intermediates.

---

## General Tips

| Tip | Command |
|-----|---------|
| **Diagnose environment** | `uv run winml sys` |
| **Check EP compatibility** | `uv run winml analyze -m model.onnx --ep <ep>` |
| **Verbose output** | Add `-v` or `--verbose` to any command |
| **Skip a pipeline stage** | `--no-quant`, `--no-compile`, `--no-optimize` |
| **Force rebuild (ignore cache)** | `uv run winml build -c config.json -m <model> -o output/ --rebuild` |
| **Regenerate config** | `uv run winml config -m <model> -d <device> -o dir/` |
| **Free disk space** | Delete `C:\Users\<user>\.cache\winml` |

## See also

- [winml sys](commands/sys.md) — system diagnostics
- [winml analyze](commands/analyze.md) — EP compatibility analysis
- [EP and Device](concepts/eps-and-devices.md) — execution provider reference
