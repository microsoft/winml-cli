# Troubleshooting `winml optimize` Failure for Swin-Large on QNN (Snapdragon X Elite)

You're hitting a common rough edge with Swin Transformers on QNN. The fact that `inspect` and `export` worked but `optimize` fails on an "unsupported op pattern" almost always points at the same root cause: **Swin's window-attention block contains ops or graph patterns that QNN's optimizer can't fuse or lower as-is**. Below is a systematic recovery plan.

## 1. Read the actual error carefully

Before changing anything, capture the full optimize log. The phrase "unsupported op pattern" can mean several different things:

- An operator QNN's HTP backend doesn't implement (e.g. `Roll`, `Mod`, `ScatterND` with non-trivial reductions, certain `Einsum` equations).
- A *shape/rank* limitation (QNN HTP often dislikes >4D tensors, dynamic axes, or rank-5 reshapes that appear in window partitioning).
- A pattern the optimizer tries to fuse (e.g. LayerNorm, GELU, MatMul+Add) but fails because of an unexpected constant layout or extra `Cast`/`Reshape` in the middle.

Re-run with verbose logging if your CLI supports it (e.g. `--verbose`, `--log-level debug`) and grab:
- The op type that's flagged.
- The node name (so you can locate it in Netron).
- Whether it's during *graph optimization*, *quantization prep*, or *EP partitioning*.

## 2. Inspect the offending node in Netron

Open the exported `model.onnx` in Netron and search for the node name from the error. For Swin-Large, the usual suspects are:

| Pattern | Where it comes from | Why QNN chokes |
|---|---|---|
| `Roll` | Shifted window attention (SW-MSA) | Not in QNN HTP op set; needs decomposition into `Slice` + `Concat` |
| 5D `Reshape` + `Transpose` | `window_partition` / `window_reverse` | HTP prefers <=4D; rank-5 layouts often fail to lower |
| `Where` + `Add` with `relative_position_bias` | Attention bias | Large constant tensor, sometimes `int64` indices |
| `Einsum` | Some HF export paths emit it for QKV | QNN supports a limited set of equations |
| `Gather` with `int64` indices | Relative position index table | `int64` not supported on HTP — needs cast to `int32` |
| Dynamic `Reshape` (shape from `Shape` op) | Dynamic batch / sequence | HTP wants fully static shapes |

## 3. Fixes, in order of cheapest first

### (a) Re-export with a friendlier opset and static shapes
HF's default export often leaves dynamic batch + dynamic sequence. Re-run your export pinning:

- `opset=17` (or whatever your toolchain's QNN backend prefers — usually 17 or 18; 20+ sometimes introduces ops QNN hasn't caught up to).
- A **fixed batch size of 1** and **fixed input resolution 224x224**.
- Disable dynamic axes entirely.

For `optimum`/`transformers.onnx`, that means dropping `--dynamic-axes` or passing a static dummy input. A re-export with static shapes alone fixes a surprising fraction of "unsupported pattern" failures because it lets constant folding eliminate the problematic `Shape`/`Gather`/`Reshape` chains.

### (b) Run ONNX shape inference + constant folding before optimize
Use `onnxsim` (onnx-simplifier) or `onnxruntime.tools.symbolic_shape_infer`:

```bash
python -m onnxsim model.onnx model_sim.onnx --overwrite-input-shape 1,3,224,224
```

This collapses many of the `Roll`/`Reshape`/`Transpose` chains into simpler equivalents, and removes redundant `Cast` nodes that block QNN's fusion patterns. Point `winml optimize` at `model_sim.onnx`.

### (c) Decompose `Roll` manually
If the error specifically names `Roll`, replace each `Roll` with a `Slice` + `Concat` pair. You can do this with a small `onnx.helper` script that walks the graph, but the easier path is: re-export with a Swin variant that already disables the shift (some HF configs allow `shift_size=0`), or use a community-patched export that pre-decomposes `Roll`.

### (d) Cast `int64` indices/constants to `int32`
QNN HTP doesn't support `int64`. A pre-processing pass that rewrites `int64` initializers and the surrounding `Gather`/`Cast` nodes to `int32` resolves a lot of "pattern not supported" cases. ONNX Runtime ships `onnxruntime.quantization.preprocess` which does some of this; alternatively the QNN SDK provides `qnn-onnx-converter` with a `--use_int32_indices` style flag depending on version.

### (e) Try a different EP first to isolate the problem
Run `winml optimize` targeting **CPU** (or just plain ORT optimization level 99) and confirm the model optimizes cleanly. If CPU optimize also fails, the problem is in the ONNX graph itself, not QNN-specific — fix it there first. If CPU optimize succeeds but QNN fails, you've confirmed it's a QNN op-coverage issue and (a)-(d) above are the right tools.

### (f) Fall back: let QNN partition around it
If one specific subgraph genuinely isn't supported, you don't *have* to make the whole model run on NPU. Configure the QNN EP to fall back to CPU for unsupported nodes (this is the default behavior for ORT's QNN EP — `qnn_context_priority` / partitioning options). You lose some perf on the fallback subgraph but the model runs end-to-end. For Swin-Large this often still yields most of the NPU speedup because the bulk of FLOPs are in the MatMuls, which QNN handles well.

## 4. Reality check on Swin-Large + Snapdragon X Elite

Two honest caveats:

1. **Swin-Large is big** (~197M params). Even when it compiles, you may hit HTP memory pressure. If you have flexibility, validate the pipeline with **swin-tiny** or **swin-base** first — if those optimize cleanly, you've confirmed your toolchain works and the issue is large-model-specific (often constant tensor sizes for relative position bias).
2. **Quantization is usually required** for HTP. QNN HTP runs INT8/INT16; FP16 support is limited and FP32 isn't a target. If `winml optimize` is the step that prepares for quantization, the "unsupported pattern" may actually be a *quantization* pattern failure (e.g. can't find a Q/DQ-friendly subgraph around the attention softmax). In that case, switching from static to dynamic quantization for the attention block, or excluding specific nodes from quantization, often unblocks it.

## 5. Concrete next step I'd take

1. Paste the full error message (op name + node name).
2. Re-export with `opset=17`, batch=1, static 224x224, no dynamic axes.
3. Run `onnxsim` on it.
4. Re-run `winml optimize` against the simplified model.
5. If it still fails, share the specific op type from the new error — that determines whether it's (c) Roll decomposition, (d) int64 cast, or a genuine QNN op-gap that needs CPU fallback.

If you can share the exact error string, I can narrow this down to one fix instead of five.
