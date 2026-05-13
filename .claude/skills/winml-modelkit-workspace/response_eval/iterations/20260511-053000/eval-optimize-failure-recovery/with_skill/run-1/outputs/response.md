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
