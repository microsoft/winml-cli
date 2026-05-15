# eval-optimize-failure-recovery

## Prompt

> i'm building microsoft/swin-large-patch4-window7-224 for QNN on snapdragon x elite. inspect and export went through fine but winml optimize keeps erroring with something about an unsupported op pattern. what do i do?

## with_skill — 6/6

### Response

Good news first: Swin is in scope for the toolkit (it's one of the families explicitly designed to flow through the pipeline), and a Snapdragon X Elite is exactly the QNN target. An unsupported-op-pattern error at the optimize stage is the classic "the default optimizer applied a fusion/rewrite that QNN's HTP backend can't lower" situation — it's a configuration problem, not a "this model doesn't work" problem.

The right move is to stop guessing at flags and let the analyzer tell you which pattern is the culprit. Run analyze against your **exported** ONNX (not the HF id) and target QNN — it lints every operator and pattern per EP and will name the offending op or pattern explicitly:

```
uv run winml analyze -m <your_exported>.onnx --ep qnn --device npu --output analyze.json
```

A couple of things to do with that output:

1. Look for any operators or patterns marked unsupported or partial. The JSON dump (`analyze.json`) is the easiest place to read the per-node detail.
2. There's also an `--optim-config` flag on `winml analyze` that writes an auto-discovered optimization config to JSON — pass `--optim-config optim.json` on the analyze invocation and then feed that file to optimize via `-c optim.json`. That's the path of least resistance because analyze picks the rewrite/fusion set that QNN actually accepts.

So a clean recovery looks like:

```
uv run winml analyze -m <your_exported>.onnx --ep qnn --device npu --optim-config optim.json --output analyze.json
uv run winml optimize -m <your_exported>.onnx -o swin_opt.onnx -c optim.json
```

If you'd rather steer it manually, `winml optimize --help` (and `winml optimize --list-rewrites` / `--list-capabilities`) shows the full set of `--enable-...` / `--disable-...` toggles. The patterns that most often bite Swin on QNN are the LayerNorm rewrites, the attention rewrites, and the high-dim reshape-transpose-reshape rewrite, e.g. `--disable-layernormalization-singlelayernorm`, `--disable-attention-expandedattention`, `--disable-highdimRTR-lowdimRTR`. Don't shotgun all of them — let analyze name the one that's failing and disable that specific family. If the error mentions a specific op (Gelu, MatMul+Add, RMSNorm, etc.), there's a matching `--disable-...` flag for it.

One thing to avoid: don't hand-edit the ONNX graph, and don't try to layer a `--preset` or `--profile` style flag — those don't exist on `winml optimize`. The knobs are the per-capability `--enable-...` / `--disable-...` toggles plus `-c <config>`.

If after disabling the named pattern analyze still reports unsupported ops on QNN (not just unsupported patterns), the next step is to share the analyze JSON — at that point we're past optimize and into "does QNN HTP actually support this op at all," which is a different conversation.

Quick sanity checks before you start:
- `uv run winml sys --list-ep` to confirm QNNExecutionProvider is registered on the box.
- If you haven't already, `uv run winml inspect -m microsoft/swin-large-patch4-window7-224` should show a Supported verdict — worth confirming since you mentioned inspect "went through fine" but not what verdict it printed.

Paste the analyze output (or just the offending pattern name) back and I'll give you the exact disable flag.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends running `winml analyze` against the exported ONNX as the first diagnostic step. | PASS | Quote: 'Run analyze against your exported ONNX (not the HF id) and target QNN' with concrete command. |
| ✓ | Explains analyze's linter will name the offending op pattern per EP. | PASS | Quote: 'it lints every operator and pattern per EP and will name the offending op or pattern explicitly'. |
| ✓ | Does NOT recommend hand-editing the ONNX graph. | PASS | Quote: 'do not hand-edit the ONNX graph'. |
| ✓ | Suggests trying different optim/quant config OR escalating to 'not a fit'. | PASS | Both: --disable-... toggles + --optim-config flow + escalation path. |
| ✓ | Encourages reading actual error message before guessing. | PASS | Asks user to share analyze JSON / 'just the offending pattern name'; lists Swin patterns 'If the error mentions a specific op'. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 4/6 ⚠

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends `winml analyze` against the exported ONNX as the first diagnostic step. | **FAIL** | No mention of winml analyze. Goes straight to verbose logging + onnxsim + opset checks. |
| ✗ | Explains analyze will name the offending op pattern per EP. | **FAIL** | Same â€” no analyze tool in the response. |
| ✓ | Does NOT recommend hand-editing the ONNX graph. | PASS | Doesn't suggest direct ONNX graph editing. (Does suggest patching PyTorch model code with `torch.cat`/slice replacements for `torch.roll` â€” invasive source-model surgery, but not ONNX-graph editing… |
| ✓ | Suggests trying different optim/quant config OR escalating to 'not a fit'. | PASS | Offers multiple paths: onnxsim, pin opset, simplification, CPU fallback for offending subgraph (hybrid execution). |
| ✓ | Encourages reading actual error message before guessing. | PASS | Step 1 is 'Pin down the exact failing op/pattern' via verbose logs, and the response ends with 'Can you paste the exact error line from `winml optimize`?'. |
| ✓ | All `winml` commands quoted in the response are runnable — every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All 1 winml commands verified runnable against `--help`. |

---
