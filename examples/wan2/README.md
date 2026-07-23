# Wan-AI/Wan2.1-T2V-1.3B-Diffusers — ONNX denoiser export

End-to-end walkthrough for exporting the **denoising transformer** of
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (text-to-video) to **fp16 ONNX** and running
the full diffusers pipeline with that graph on **ONNX Runtime (CUDA)**.

Only the transformer is exported. In a diffusion pipeline the transformer is the
module that runs on *every* denoising step (100 calls for a 50-step,
classifier-free-guidance run), so it dominates runtime. The T5 text encoder and
the VAE run **once** per video and stay in PyTorch.

```
text prompt ──► T5 (umt5-xxl)  ──►┐
                                  │   ┌────────── denoise loop (N steps) ──────────┐
random latent noise ──────────────┼──►│  WanTransformer3DModel  ← THIS is exported │──► latent ──► VAE decode ──► frames ──► mp4
[1,16,21,60,104]                  │   │  (fp16 ONNX / ORT-CUDA)                     │      (PyTorch fp32, tiled)
                                  │   └─────────────────────────────────────────────┘
                                  └── stays in PyTorch (runs once)
```

---

## Files

| File | Purpose |
|---|---|
| `diffusers_t2v_sample.py`     | Baseline: generate a video with the stock PyTorch pipeline (bf16). |
| `export_transformer_onnx.py`  | Export the transformer to fp32 ONNX, then convert to fp16. |
| `verify_onnx.py`              | Numerical parity check: ORT fp16 graph vs PyTorch fp16 module. |
| `run_onnx_pipeline.py`        | Full pipeline with the ONNX transformer swapped in via ORT-CUDA. |
| `test_mha_symbolic.py`        | Minimal, self-contained demo of the fused-attention export trick. |
| `export_vae_decoder_onnx.py`  | Export the VAE **decoder** (single-tile graph, dynamic spatial) to fp32 ONNX. |
| `run_vae_onnx.py`             | `OrtVaeDecoder`: portable tiled VAE decode on any ORT execution provider. |
| `verify_vae_onnx.py`          | Parity check: ORT tiled decode vs PyTorch `AutoencoderKLWan.decode`. |
| `requirements.txt`            | Pinned dependency versions used for the numbers below. |

---

## 0. Setup

Use a CUDA-enabled `torch` (the numbers below are cu128 on an RTX 5090 D, 32 GB).

```powershell
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

All scripts download the model from the HuggingFace Hub on first run and resolve
their own paths, so they can be run from anywhere.

---

## 1. Baseline (PyTorch)

Confirms the model works and gives a reference video + timing.

```powershell
python diffusers_t2v_sample.py
```

Writes `output.mp4` (480×832, 81 frames, 16 fps). Settings mirror the model card:
`flow_shift=3.0`, `guidance_scale=6.0`, 50 steps (diffusers default).

---

## 2. Export the transformer to ONNX

```powershell
python export_transformer_onnx.py
```

Produces, under `model/`:

- `wan_transformer_fp32.onnx` (+ `.data`, ~5.4 GB)
- `wan_transformer_fp16.onnx` (+ `.data`, ~2.7 GB)  ← used for inference

The export is **static-shaped** for 480P / 81 frames, batch 1
(latent `[1,16,21,60,104]`, sequence length `21×30×52 = 32,760` tokens).

### How the export works (and why it's done this way)

The naive `softmax(QK^T)V` attention is the crux of the whole problem: at 32,760
tokens each layer materializes a **~26 GB** score tensor, which makes the ONNX
graph unusable in ORT (it hangs / near-OOMs). The fix is to emit a single
**fused `com.microsoft::MultiHeadAttention` contrib node** per attention, which
ORT executes with a flash-attention kernel.

The node is emitted directly with `torch.onnx.ops.symbolic` from a monkeypatched
`WanAttnProcessor.__call__` (see `wan_mha_call` in `export_transformer_onnx.py`):

```python
out = torch.onnx.ops.symbolic(
    "com.microsoft::MultiHeadAttention",
    (query, key, value),               # each packed [B, S, num_heads*head_dim]
    attrs={"num_heads": int(attn.heads)},
    dtype=query.dtype,
    shape=(B, Sq, D),
    version=1,
)
```

`test_mha_symbolic.py` is a 60-line standalone proof of this mechanism — it
exports one node and shows ORT matches PyTorch SDPA to ~1e-3. Run it first if you
want to understand the trick in isolation:

```powershell
python test_mha_symbolic.py
# nodes: ['MultiHeadAttention(com.microsoft)']
# max_abs_diff vs torch SDPA: ~9e-4
```

Other things the export script handles, each of which otherwise breaks:

- **Dynamo (`torch.export`) exporter, not the legacy TorchScript one.** Dynamo
  traces with fake tensors, so the multi-GB activations are never allocated. The
  legacy tracer allocates real activations across all 30 layers and OOMs (~75 GB).
- **ScatterND-free RoPE.** The diffusers reference uses `out[..., 0::2] = ...`
  slice-assignment, which lowers to *hundreds* of `ScatterND` ops. It is replaced
  by `torch.stack((o1, o2), -1).flatten(-2)` (0 ScatterND in the final graph).
- **RMSNorm swap.** `torch.nn.RMSNorm` dispatches to `aten._fused_rms_norm`,
  which the exporter can't lower; `OnnxRMSNorm` computes the same thing with
  primitive ops.
- **Uniform fp32 export, then convert to fp16.** Exporting directly in mixed
  precision crashes the type-promotion pass at `scale_shift_table + temb`.
  fp16 conversion uses `onnxconverter_common.float16` with
  `disable_shape_infer=True` (required for >2 GB models).
- **RoPE buffers cast to fp32** (they are float64 by default; no float64 in the graph).

The final fp16 graph has **60 MultiHeadAttention nodes, 0 Softmax, 0 ScatterND**.

#### Approaches that do NOT work (don't retry these)

| Attempt | Result |
|---|---|
| Naive `softmax(QK^T)V` attention | ORT hang / near-OOM (26 GB score tensor per layer) |
| Legacy TorchScript exporter | CUDA OOM (~75 GB real activations across 30 layers) |
| Dynamo + onnxscript `custom_translation_table` | version-converter inliner crash: "number of values and replacements must match" |
| ORT `optimize_model` transformer fusion | did not fuse the attention |

---

## 3. Verify parity

```powershell
python verify_onnx.py
```

Compares the ORT fp16 graph against the PyTorch fp16 module on the same random
input. Expected: `max_abs_diff ≈ 0.013`, `mean rel error ≈ 0.29%` — i.e. within
fp16 noise.

---

## 4. Run the full pipeline on ONNX Runtime

```powershell
# denoiser in ONNX; VAE decode in PyTorch (tiled, CUDA only)
python run_onnx_pipeline.py --steps 50 --frames 81 --vae torch --out output_onnx.mp4

# denoiser AND VAE decode in ONNX/ORT (portable path, default)
python run_onnx_pipeline.py --steps 50 --frames 81 --vae onnx --out output_onnx.mp4
```

`OrtTransformer` is a drop-in for `WanTransformer3DModel` exposing only what the
`WanPipeline` touches (`config`, `dtype`, a no-op `cache_context`, and a
`__call__` returning `(noise_pred,)`). T5 always stays PyTorch.

With `--vae onnx` (default) the runner drives the pipeline with
`output_type="latent"`, then reproduces WanPipeline's latent un-scaling
(`latents / std + mean`) and decodes with `OrtVaeDecoder` (see §5) — so **both**
the per-step denoiser and the VAE decode run on ONNX Runtime, and the torch VAE
weights are moved off the GPU. With `--vae torch` the VAE decode stays in PyTorch.

### VRAM management (needed to fit in 32 GB)

- **T5 stays bf16/fp32** — umt5-xxl overflows in fp16.
- **VAE decode is tiled** either way — a full fp32 decode of 81 frames OOMs 32 GB
  (`vae.enable_tiling()` for `--vae torch`; the single-tile ONNX graph for `--vae onnx`).
- **ORT arenas capped** — `gpu_mem_limit = 8 GB` + `arena_extend_strategy=kSameAsRequested`
  on both the denoiser and VAE sessions, so ORT doesn't grab all the VRAM.
- With `--vae onnx`, the denoiser ORT session is freed before the VAE session decodes.

Peak VRAM ≈ **23 GB**, occurring during VAE decode (T5 ~11 GB resident + ORT
arena + tiled VAE working set).

---

## 5. Export the VAE decoder (needed for `--vae onnx`, and for non-CUDA GPUs)

The T5 encoder stays in PyTorch, so with `--vae torch` the *decode* path only runs
on CUDA. To run the VAE decode on ONNX Runtime — required by the default
`--vae onnx` in §4, and the only way to run the decode on **other GPU backends**
(DirectML, OpenVINO, ROCm, …) — the decoder has to be ONNX too. For T2V only the
**decoder** matters (the encoder is unused).

```powershell
python export_vae_decoder_onnx.py   # -> model/wan_vae_decoder_fp32.onnx (+ .data, ~280 MB)
python verify_vae_onnx.py           # parity vs PyTorch tiled decode
```

### Why the VAE is trickier than the transformer

The Wan VAE decode is a **stateful, sequential** process, not a single forward:
`AutoencoderKLWan._decode` walks the latent frames one chunk at a time, threading
a **causal temporal feature cache** through 33 `WanCausalConv3d` layers, and
`tiled_decode` wraps that in a spatial tile loop with overlap blending. The ops
themselves (Conv3d, `F.normalize` RMSNorm, nearest `Upsample`, a small ~1024-token
mid-block attention) all export cleanly — the challenge is the state + the loops.

### How the export works

`export_vae_decoder_onnx.py` exports the decode of **one spatial tile** as a
single graph:

- The **21-frame loop is unrolled** inside the traced forward, so the temporal
  feature cache becomes ordinary intermediate tensors — there is no 33-tensor
  cache to plumb in/out, and the `first_chunk` / cache-padding branches collapse
  to compile-time constants. Numerically identical to PyTorch.
- The cache is threaded through **local Python lists** (not module attributes) so
  the dynamo exporter traces it without side-effect errors.
- Frames are fixed at 21 (→81 output); **spatial H/W are dynamic** (`torch.export.Dim`)
  so the smaller edge tiles decode at their true size — exact parity, no padding.
- `post_quant_conv` is folded in; the graph has 817 Convs, 21 small Softmaxes, and
  **0 ScatterND**.

`run_vae_onnx.py` provides `OrtVaeDecoder`, which reproduces `tiled_decode`'s outer
spatial tiling + `blend_v`/`blend_h` orchestration in ~40 lines of Python/NumPy
around the single-tile ORT session — so all heavy compute runs on the chosen EP:

```python
from run_vae_onnx import OrtVaeDecoder
dec = OrtVaeDecoder(providers=["DmlExecutionProvider"])   # or OpenVINO/ROCm/CPU
frames = dec.decode(latents)   # [1,16,21,60,104] -> [1,3,81,480,832]
```

### Notes

- **fp32, portable.** The Wan VAE prefers fp32 for stability, and fp32 is the
  safest common denominator across EPs. Parity vs PyTorch tiled decode:
  **max_abs_diff 0.0042, mean rel error 0.037%**.
- **Memory-bounded by tiling.** The single-tile graph keeps peak memory small,
  which matters on lower-VRAM non-NVIDIA GPUs.
- **Verify 3D-conv support on your target EP.** These are the standard ONNX ops,
  but 5D `Conv` coverage varies by backend (e.g. DirectML historically) — run
  `verify_vae_onnx.py` with your providers before relying on it.

---

## 6. Why T5 stays in PyTorch

The T5 (umt5-xxl) text encoder is deliberately **not** exported. It is the one
component where ONNX adds cost without benefit:

- **It runs once per video, not per step.** T5 encodes the prompt a single time;
  the ~180 s run is dominated by the 100 denoiser calls. Its latency is
  irrelevant — even 4.4 s on CPU is noise next to the denoise loop.
- **It's big and fp16-hostile.** umt5-xxl is ~5.6 B params and overflows in fp16,
  so an ONNX export would be fp32 (~22 GB) or bf16 (~11 GB) — a huge artifact for
  a once-per-video op.
- **PyTorch-CPU already covers portability.** The point of exporting was the
  *per-step GPU compute*. On a non-CUDA machine you simply run T5 in PyTorch on
  CPU (4.4 s); ORT wouldn't make that faster.

Only export T5 if you need a **100% pure-ORT deployment** with zero PyTorch
anywhere — and even then, INT8-dynamic-quantize it (halves the size) rather than
shipping fp32. If your prompts are fixed, a simpler win is to **precompute and
cache the T5 embeddings once** and skip the encoder entirely at inference.

---

## Benchmarks

RTX 5090 D (32 GB), 50 steps, 81 frames, 480×832.

| Phase | GPU | CPU (fp32) | Notes |
|---|---|---|---|
| T5 text encode (both prompts) | 0.64 s | 4.4 s (~14×) | once per video |
| Denoise (per transformer call) | 1.82 s | — | 100 calls / 50 steps → ~98% of runtime |
| VAE decode (81 frames, PyTorch, tiled) | 11 s | 397 s (~36×) | once per video |
| VAE decode (81 frames, ONNX/ORT-CUDA, tiled) | 11.3 s | — | portable graph, matches PyTorch |
| **Total pipeline** | **~186 s** | — | vs ~289 s for the bf16 PyTorch baseline |

The ONNX/ORT denoiser is faster than the bf16 PyTorch transformer here, but the
main reason to export is **portability**, not raw speed.

---

## Recommendation

- **Export the denoiser always.** It runs every step and dominates runtime.
- **Export the VAE decoder only if you need non-CUDA portability.** On NVIDIA it's
  a wash with PyTorch (11.3 s vs 11 s); its value is running on DirectML / OpenVINO
  / ROCm via ORT. For T2V you never need the encoder.
- **T5 (umt5-xxl) is the remaining PyTorch piece.** It runs once and must stay
  bf16/fp32 (fp16 overflows). Export it too only if the *whole* pipeline must
  leave CUDA; otherwise keep it in PyTorch. ONNX is for portability, not speed —
  CPU execution is 14–36× slower, and only INT8 quantization would meaningfully
  help there.
