Swin Transformer is a known pain point for QNN on Snapdragon because of the window-shift / roll / cyclic-shift pattern and the masked-attention reshape pattern. The "unsupported op pattern" from `winml optimize` almost always traces back to one of those. Here's how I'd attack it:

## 1. Pin down the exact failing op/pattern

Re-run with verbose logging to get the actual node name and op type:

```
winml optimize <model.onnx> --provider qnn --verbose
```

The error usually points at one of:
- `Roll` / `ScatterND` (from the cyclic shift)
- `Where` + `Softmax` with a non-broadcastable mask (attention mask)
- `Reshape` -> `Transpose` -> `Reshape` chains that QNN's HTP backend can't fold
- `LayerNormalization` with axes QNN doesn't support
- `MatMul` with rank-5 inputs (window attention)

Knowing which one matters — the workaround differs.

## 2. Check opset and operator domains

QNN EP has fairly tight opset support. Make sure your inspect output shows opset 17 or 18 (not 20+). If you exported with a newer opset, re-export pinning:

```python
torch.onnx.export(..., opset_version=17)
```

## 3. Try simplification before optimize

Run `onnxsim` (or `onnxruntime`'s offline transforms) on the exported model first. A lot of the rank-5 reshape/transpose chains in Swin collapse once constant folding runs:

```
python -m onnxsim model.onnx model_simplified.onnx
```

Then point `winml optimize` at the simplified model.

## 4. If it's the cyclic shift / Roll

`Roll` isn't supported on QNN. The HuggingFace Swin export sometimes emits this. Two options:
- Patch the model code to replace `torch.roll` with a `torch.cat` / slice equivalent before export (there are gists for this — search "swin onnx roll workaround").
- Use `transformers`' `swin` with `attn_implementation="eager"` and re-export.

## 5. If it's the attention mask `Where`

The relative position bias / attention mask sometimes exports as a `Where` with dynamic shapes. Fixing input shapes at export time (`dynamic_axes=None`, fixed 224x224) usually makes it constant-foldable.

## 6. Fallback: hybrid execution

If you can't get a clean QNN graph, configure the EP with CPU fallback for the offending subgraph. QNN EP supports partitioning — the unsupported nodes will fall back to CPU. You won't get full NPU acceleration, but the rest of the model still runs on HTP. Check the EP options for `qnn.disable_cpu_ep_fallback=0`.

## 7. Quick sanity check

Before going deep, try the Microsoft-published Swin variant from the WinML model zoo or `onnx/models` — if their version optimizes cleanly, diff the graph against yours to spot what your export added.

Can you paste the exact error line from `winml optimize`? That'll narrow it down a lot — Swin has at least 3 different patterns that commonly trip QNN and the fix is different for each.
