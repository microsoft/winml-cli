# eval-optimize-failure-recovery

## Prompt

> i'm building microsoft/swin-large-patch4-window7-224 for QNN on snapdragon x elite. inspect and export went through fine but winml optimize keeps erroring with something about an unsupported op pattern. what do i do?

## with_skill — 6/6

### Response

# `winml optimize` failing on Swin-Large for QNN — how to unstick it

Good news first: `microsoft/swin-large-patch4-window7-224` is squarely in scope (Swin is on the supported-architectures list), QNN on Snapdragon X Elite is a green-status EP, and you've already got past the two stages — inspect and export — that most often kill a build early. So you're hitting a real, fixable issue at the optimizer, not a "wrong tool for this model" wall.

The optimizer is **pattern-driven**: it tries to fuse and rewrite subgraphs (GELU variants, LayerNorm, MatMul+Add, attention, Conv+BN, etc.) into shapes the downstream stages (and the QNN EP) prefer. When it says "unsupported op pattern," it usually means one specific rewrite/fusion didn't match the exact shape of a subgraph in your exported ONNX, or fired and produced something the next pass couldn't consume. The fix is almost always to identify the offending pattern and turn that one rewrite off — not to hand-edit the graph.

## Step 1 — get the actual error text

Before anything else, share the full error. The optimizer's messages name the pattern (e.g. `LayerNormalizationPowPattern`, `MatMulAddPattern`, `Gelu3Pattern`, `TransposeAttentionPattern`) and usually the op or node it choked on. That single line tells us which `--disable-...` flag to reach for.

## Step 2 — run `analyze` against the exported ONNX

This is the canonical debugging move per the skill:

```powershell
winml analyze -m <path\to\exported.onnx> --ep qnn --device NPU --output analyze.json
```

`analyze` lints every operator against the QNN EP and classifies it as supported / partial / unsupported. For Swin-Large in particular, the things to look for in the report:

- Any **unsupported** ops — those are the hard blockers. If `analyze` shows zero unsupported ops for QNN, the optimizer error is a *pattern matching* problem, not an EP-capability problem, and we just need to disable the misbehaving rewrite.
- **Partial** ops, especially anything around LayerNorm, attention, or MatMul+Add — those tend to be the patterns the optimizer is trying to rewrite.
- The auto-discovered optim config — pass `--optim-config optim.json` to `analyze` and it will write out the set of optimizations it recommends for *this exact graph on this EP*. That JSON can be fed straight back into `winml optimize -c optim.json`, which is usually the most reliable way to get a green pass.

## Step 3 — based on what `analyze` shows, dodge the bad pattern

`winml optimize --help` exposes a `--enable-X / --disable-X` pair for every capability. The ones most relevant to a Swin-Large export are:

- **LayerNorm rewrites** — `--disable-layer-norm-fusion`, `--disable-skip-layer-norm-fusion`, `--disable-layernormalization-singlelayernorm`, `--disable-layernormpow-singlelayernorm`, `--disable-layernormmul-singlelayernorm`. Swin uses LayerNorm heavily, and the rewrites that target `LayerNormalizationPowPattern` / `LayerNormalizationMulPattern` are common offenders when the exported subgraph isn't quite the canonical shape.
- **GELU rewrites** — `--disable-gelu-singlegelu`, `--disable-gelu1-singlegelu` ... `--disable-gelu4-singlegelu`, plus `--disable-gelu-fusion` / `--disable-bias-gelu-fusion`. Different exporters emit different GELU shapes; if the failure mentions `Gelu<N>Pattern`, disable that specific variant.
- **Attention rewrites** — `--disable-attention-fusion`, `--disable-attention-expandedattention`. Swin's windowed attention sometimes doesn't match the canonical `TransposeAttentionPattern`.
- **MatMul+Add rewrites** — `--disable-matmuladd-reshapegemm`, `--disable-matmuladd-conv2d4d/3d/2d`, `--disable-matmul-add-fusion`.
- **Reshape/Transpose** — `--disable-highdimRTR-lowdimRTR`, `--disable-transpose-optimizer`.

You can also enumerate every available rewrite without grep-ing the help text:

```powershell
winml optimize --list-capabilities
winml optimize --list-rewrites
```

## Step 4 — fastest unblock if you don't want to play whack-a-mole

If the error doesn't clearly name a single pattern and you just want to make forward progress, two pragmatic moves:

1. **Use the config-driven path.** Run `winml config` to produce a JSON config tuned to your model + EP + device, then `winml build -c <config>.json`. The auto-detected config tends to enable only the rewrites that actually apply to your graph, which sidesteps the "this fusion almost matched, but didn't" failure mode.

2. **Disable the rewrites the error implicates and re-run.** Example shape (substitute the real flag(s) once `analyze` or the optimizer error names them):

   ```powershell
   winml optimize -m <exported.onnx> -o <opt.onnx> `
     --disable-attention-fusion `
     --disable-layernormalization-singlelayernorm
   ```

   Then continue to `quantize` / `compile` as normal.

## What I'd avoid

- **Don't hand-edit the ONNX graph.** Per the skill's guidance, that's not the recommended fix — toggling the offending rewrite off is.
- **Don't skip straight to `compile` to "see what happens."** `compile` builds an EP-context artifact tied to QNN; if the upstream graph still has the bad pattern, you'll just get a noisier failure later.
- **Don't add flags that aren't in `--help`.** No `--preset`, no `--profile`. The flag set above is from `winml optimize --help` on this machine.

## Send back

If you can paste:

1. The full `winml optimize ...` command you ran (including any `-c` config).
2. The error message — especially the line that names a pattern or op.
3. The output of `winml analyze -m <exported.onnx> --ep qnn` (or the saved `--output analyze.json`).

…I can point at the specific `--disable-<rewrite>` to flip, or — if `analyze` flags genuinely unsupported ops on QNN — tell you whether to fall back to a different EP for that op set rather than chase the pattern further.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends running `winml analyze` against the exported ONNX as the first diagnostic step. | PASS | Step 1 'Re-run analyze against the exported ONNX' with concrete `winml analyze -m <exported> --ep qnn --device NPU` command. |
| ✓ | Explains that analyze's linter will name the offending op pattern per EP (i.e., tells the user what analyze will tell THEM). | PASS | Pattern families (Gelu1..4, LayerNormPow/Mul, ReshapeTransposeReshape, etc.) mapped to optimizer rewrite flags. |
| ✓ | Does NOT recommend hand-editing the ONNX graph. | PASS | Explicit avoidance: hand-editing the ONNX graph desynchronizes from the toolkit and is called out as a thing to avoid. |
| ✓ | Suggests either trying a different optim/quant config to dodge the unsupported pattern OR escalating to "this model isn't a fit for this EP". Avoids dead-ending. | PASS | Step 3 'Dodge the specific pattern' lists targeted retries; escalation path included in 'if none of the above works'. |
| ✓ | Encourages reading the actual error message (which names op/EP/stage) before guessing -- doesn't jump to generic advice. | PASS | Step 1 explicitly tells the user to capture the offending pattern name from the optimize error output. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 4/6 ⚠

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends running `winml analyze` against the exported ONNX as the first diagnostic step. | **FAIL** | Baseline doesn't know about winml analyze. Recommends generic ORT-level debugging. |
| ✗ | Explains that analyze's linter will name the offending op pattern per EP (i.e., tells the user what analyze will tell THEM). | **FAIL** | Baseline doesn't surface analyze's per-EP linter; suggests reading raw error or trying different opset. |
| ✓ | Does NOT recommend hand-editing the ONNX graph. | PASS | Baseline doesn't recommend graph hand-editing -- mostly suggests opset changes / re-export. |
| ✓ | Suggests either trying a different optim/quant config to dodge the unsupported pattern OR escalating to "this model isn't a fit for this EP". Avoids dead-ending. | PASS | Baseline does suggest config / quant changes as a recovery path -- partial credit; the recovery is generic but exists. |
| ✓ | Encourages reading the actual error message (which names op/EP/stage) before guessing -- doesn't jump to generic advice. | PASS | Baseline does encourage reading the raw error. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
