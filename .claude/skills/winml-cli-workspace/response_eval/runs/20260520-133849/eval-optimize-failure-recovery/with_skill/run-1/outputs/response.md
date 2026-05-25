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
