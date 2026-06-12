# EP and Device

An **Execution Provider (EP)** is a pluggable backend in ONNX Runtime that claims and runs a subset of graph nodes on a specific hardware target. When ONNX Runtime loads a model it partitions the graph among the registered EPs: operators that an EP claims are dispatched to it, and the remainder fall back to the CPU EP. This design lets a single [ONNX](graphs-and-ir.md) model exploit an NPU, GPU, or CPU without any change to the graph itself.

A **device** is the hardware category that an EP targets — one of `npu`, `gpu`, or `cpu`. winml-cli exposes both levels of control: the high-level `--device` flag selects a hardware category, while the low-level `--ep` flag pins a specific ONNX Runtime provider name. In most workflows you set `--device` and let winml-cli resolve the best available EP; you reach for `--ep` when you need to compare or force a specific provider.

## EPs winml-cli supports

The table below lists every Execution Provider that winml-cli has explicit support for. EP names are the canonical ONNX Runtime strings accepted by `--ep`. You can also use the short **alias** (case-insensitive) anywhere the full name is accepted.

| EP | Alias | Device | Hardware | When to use |
|----|-------|--------|----------|-------------|
| `QNNExecutionProvider` | `qnn` | npu / gpu | Qualcomm NPU (Hexagon DSP) / Qualcomm GPU (Adreno) | Snapdragon-based Copilot+ PCs; best latency and power efficiency on Qualcomm silicon |
| `VitisAIExecutionProvider` | `vitisai` | npu | AMD NPU (XDNA) | AMD Ryzen AI platforms; targets the AMD AI Engine via the Vitis AI stack |
| `OpenVINOExecutionProvider` | `openvino` | npu / gpu / cpu | Intel CPU / GPU / NPU | Intel Core Ultra platforms; flexible device targeting across all three Intel compute types |
| `DmlExecutionProvider` | `dml` | gpu | GPU (DirectML) | Any DirectX 12 GPU on Windows; broad compatibility across AMD, Intel, and NVIDIA discrete/integrated graphics |
| `NvTensorRTRTXExecutionProvider` | `nv_tensorrt_rtx` | gpu | NVIDIA GPU (TensorRT RTX) | NVIDIA RTX GPUs; maximum throughput via TensorRT graph optimization |
| `MIGraphXExecutionProvider` | `migraphx` | gpu | AMD GPU (MIGraphX) | AMD discrete GPUs; hardware-accelerated inference via the MIGraphX graph engine |
| `CPUExecutionProvider` | `cpu` | cpu | CPU | Universal fallback; always available regardless of hardware |

To see which EPs are available on the current machine, run:

```bash
winml sys --list-ep
```

## Device vs. EP on the CLI

winml-cli exposes two overlapping flags for targeting hardware. Understanding their relationship prevents confusion when using `winml analyze`, `winml compile`, or `winml build`.

**`--device` (high-level)**

Accepts one of four values: `auto`, `cpu`, `gpu`, or `npu`. When set to `auto` (the default), winml-cli inspects the machine and selects the highest-priority device class that has a compatible EP available, in the order NPU > GPU > CPU. Setting an explicit value such as `--device npu` requests a device category without naming the EP.

For `winml analyze`, `--device` also accepts `all` — this evaluates the model against every device that has rule data, producing a side-by-side compatibility report.

```bash
# Let winml-cli pick the best available device
winml analyze --model model.onnx --device auto

# Target the NPU device class
winml analyze --model model.onnx --device npu

# Analyze against all devices at once (analyze only)
winml analyze --model model.onnx --device all
```

**`--ep` (low-level override)**

Accepts a valid EP name or alias (for example `qnn`, `vitisai`, `dml`, `openvino`), or `auto` to let winml-cli resolve the EP from the device. When `--ep` is provided with a specific value it takes precedence over `--device` and bypasses device-class resolution entirely. Use `--ep` when you need to pin a specific provider — for instance to compare `QNNExecutionProvider` against `DmlExecutionProvider` on the same machine.

For `winml analyze`, `--ep` also accepts `all` — this evaluates the model against every registered EP simultaneously.

```bash
# Force Qualcomm QNN regardless of device selection
winml analyze --model model.onnx --ep QNNExecutionProvider --device npu

# Use the short alias; winml-cli normalizes it to the full name
winml analyze --model model.onnx --ep qnn

# Analyze against all EPs at once (analyze only)
winml analyze --model model.onnx --ep all
```

The `--ep` flag accepts a free-form string and is not restricted to the choices listed above. This allows forward compatibility with EP names that winml-cli does not yet enumerate.

## See also

- [Graphs and IR](graphs-and-ir.md) — ONNX graph format, operator sets, and the IR that EPs consume
- [Weight and Activation](weight-and-activation.md) — tensor roles relevant to EP compatibility
- [winml sys](../commands/sys.md) — list available devices and EPs on the current machine
- [winml analyze](../commands/analyze.md) — check ONNX operator compatibility against a specific EP
