# Troubleshooting

Common issues and solutions when working with winml-cli.

---

## Build and Pipeline Errors

### Cannot enable compilation: no compile section

```text
UsageError: Cannot enable compilation: no compile section found in the config file
```

**Cause:** Compilation is **off by default** in `winml build`. You passed `--compile` to explicitly enable it, but the config JSON has no `"compile"` section (it's `null`). This happens when the config was generated without a device target that supports EPContext (e.g., `--device cpu` or `--device auto` on a machine without NPU).

**Solution:** Regenerate the config targeting a device that supports compilation (NPU or GPU with an EP that produces EPContext):

```bash
uv run winml config -m <model> -d npu -o output/
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

## Analysis and Compatibility

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

## Device and EP Issues

### Unknown EP or device mismatch

```text
UsageError: Unknown EP: invalid_ep
UsageError: --ep QNNExecutionProvider cannot run on --device gpu
```

**Cause:** The specified EP doesn't exist or doesn't support the requested device.

**Solution:** Check available EPs on your system:

```bash
uv run winml sys --list-ep
```

Valid EP aliases: `qnn`, `openvino`, `dml`, `cpu`, `tensorrt`, `migraphx`, `vitisai`.

---

### No NPU device detected

```text
Available Devices (priority order)
  #1  GPU   ...
  #2  CPU   ...
```

**Cause:** NPU driver not installed, or Windows version is too old.

**Solution:**

1. Verify Windows 11 24H2 or later
2. Check for NPU driver updates in Device Manager → Neural processors
3. Install the latest Qualcomm AI Engine Direct SDK (for Snapdragon NPUs)
4. Re-run `uv run winml sys` to confirm

!!! note
    All winml-cli commands work without NPU hardware. Use `--device auto` to fall back to GPU or CPU.

---

## Quantization and Compilation Failures

### Quantization failed

```text
RuntimeError: Quantization failed: [error details]
```

**Cause:** Quantization encountered an incompatible graph structure or calibration error.

**Solution:**

1. Add `--verbose` to see detailed error output
2. Ensure the model has been optimized first (run `winml optimize` before `winml quantize`)
3. Try a different calibration method: `--calibration-method entropy`
4. Exclude problematic nodes: use `nodes_to_exclude` in the quant config

---

### No output file produced after compile

```text
Warning: Compilation finished but no output file was written
ClickException: No output file produced
```

**Cause:** The compiler ran but didn't generate the expected `_ctx.onnx` file. Common with DML/CPU (which don't produce EPContext).

**Solution:** Verify you're targeting an EP that supports EPContext:

```bash
# Correct — QNN supports EPContext
uv run winml compile -m model.onnx -d npu --ep qnn

# Won't produce output — DML doesn't support EPContext
uv run winml compile -m model.onnx -d gpu --ep dml
```

---

## Output and File Issues

### Output path exists but is not a directory

```text
ValueError: Output path exists but is not a directory: output.onnx
```

**Cause:** The `-o` flag expects a directory path, but you passed a file path.

**Solution:** Use a directory:

```bash
uv run winml build -c config.json -m model -o output_dir/
```

---

## General Tips

| Tip | Command |
|-----|---------|
| **Diagnose environment** | `uv run winml sys` |
| **Check EP compatibility** | `uv run winml analyze -m model.onnx --ep <ep>` |
| **Verbose output** | Add `-v` or `--verbose` to any command |
| **Skip a pipeline stage** | `--no-quant`, `--no-compile`, `--no-optimize` |
| **Regenerate config** | `uv run winml config -m <model> -d <device> -o dir/` |

## See also

- [winml sys](commands/sys.md) — system diagnostics
- [winml analyze](commands/analyze.md) — EP compatibility analysis
- [EP and Device](concepts/eps-and-devices.md) — execution provider reference
