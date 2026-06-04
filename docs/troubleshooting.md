# Troubleshooting

Common issues and solutions when working with winml-cli.

---

## Build and Pipeline Errors

### Config file is empty or invalid JSON

```text
UsageError: Config file is empty: config.json
UsageError: Invalid JSON in config: Expecting value: line 1 column 1
```

**Cause:** The config file passed to `winml build -c` is empty, malformed, or not valid JSON.

**Solution:** Validate the file with `python -m json.tool config.json`, or regenerate it:

```bash
uv run winml config -m <model> -d <device> -o output/
```

---

### Cannot enable compilation: no compile section

```text
UsageError: Cannot enable compilation: no compile section found in the config file
```

**Cause:** You passed `--compile` but the config JSON has no `"compile"` section (it's `null`).

**Solution:** Regenerate the config with compilation enabled, or add a compile section manually:

```bash
uv run winml config -m <model> -d npu -o output/
```

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

**Solution:** Run `winml analyze` first to identify which operators are problematic:

```bash
uv run winml analyze -m model.onnx --ep qnn
```

Then consider:

- Using a different EP (`--ep dml` or `--ep cpu`)
- Running optimization to fuse unsupported patterns into supported ones
- Checking if a newer opset version resolves the compatibility gap

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
