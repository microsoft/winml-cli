# Deploying Salesforce/codet5p-220m on Snapdragon X Elite NPU via the winml pipeline

Great use case — offline code summarization in a VSCode extension is exactly the kind of thing the Snapdragon X Elite NPU is built for (sustained, low-power inference without burning laptop battery). Below is a walkthrough of how you would take `Salesforce/codet5p-220m` from Hugging Face all the way to a packaged WinML inference path on Qualcomm's Hexagon NPU through the `winml` CLI / WinML ModelKit pipeline.

A heads-up first: CodeT5+ 220M is an **encoder-decoder (seq2seq) Transformer** with autoregressive decoding, KV-cache, and beam search at inference time. That places it in the same family as T5, BART, and Whisper. NPU support for autoregressive decoder loops is significantly more involved than for a pure encoder model (BERT, ViT, etc.), and you should expect to either (a) split the model into encoder + decoder-step ONNX graphs and drive the loop from your host code, or (b) accept that part of the generation loop runs on CPU/GPU. I'll flag where this matters as we go.

---

## 0. Prerequisites

On your Snapdragon X Elite dev box you'll want:

- Windows 11 24H2 or newer (for the WCR / Windows ML runtime)
- Python 3.10+ with `uv` (the WinML ModelKit repo standardizes on `uv run`)
- The Qualcomm QNN SDK installed (the QNN EP that ships with ONNX Runtime needs the QNN libs at runtime)
- `winml` CLI installed from the WinML-ModelKit repo (`uv pip install -e .` from the repo root, or whatever the published wheel is)
- Hugging Face access to `Salesforce/codet5p-220m` (it's public, no token needed)

Verify your NPU is discoverable:

```powershell
winml devices
```

You should see a QNN HTP / Hexagon entry listed alongside CPU and the Adreno GPU.

---

## 1. Pull the model and convert to ONNX

CodeT5+ is published as PyTorch weights, so step one is exporting to ONNX. The `winml` CLI wraps Optimum/`torch.onnx` for this:

```powershell
winml import Salesforce/codet5p-220m --task seq2seq-lm --output .\models\codet5p-220m\onnx
```

What this does under the hood is roughly equivalent to:

```powershell
optimum-cli export onnx --model Salesforce/codet5p-220m --task text2text-generation-with-past .\models\codet5p-220m\onnx
```

You should end up with (at minimum):

- `encoder_model.onnx` — encodes the input code tokens once
- `decoder_model.onnx` — decoder step without past KV (first token)
- `decoder_with_past_model.onnx` — decoder step with past KV (subsequent tokens)
- `tokenizer.json` / `spiece.model` (CodeT5+ uses a SentencePiece-derived tokenizer)
- `generation_config.json`

Keep all three graphs. The "with past" variant is what gives you a workable token/sec on-device — without it you re-encode the whole decoder prefix every step.

---

## 2. Inspect and clean the graph

Before optimizing, sanity-check the exported graphs:

```powershell
winml inspect .\models\codet5p-220m\onnx\encoder_model.onnx
winml inspect .\models\codet5p-220m\onnx\decoder_with_past_model.onnx
```

Things to look at:

- **Opset** — should be 17+ for clean QNN support. If Optimum exported a lower opset, re-export with `--opset 17`.
- **Dynamic axes** — `input_ids` should be `[batch, seq_len]` dynamic; KV-cache tensors should be `[batch, num_heads, past_seq_len, head_dim]` dynamic on `past_seq_len`. The QNN EP wants you to **pin batch=1** and ideally bucket `seq_len` to a small set of fixed shapes (more on this in step 4).
- **Unsupported ops** — `winml inspect` will flag ops the QNN EP can't run. For T5-family models the common offenders are: `ScatterND` used in KV-cache update, `If`/`Loop` if Optimum exported a fused generation loop (avoid that — use the per-step graphs), and some `Where`/`Cast` patterns around the attention mask.

Run a graph cleanup pass:

```powershell
winml optimize .\models\codet5p-220m\onnx\encoder_model.onnx --level basic --output .\models\codet5p-220m\opt\encoder.onnx
winml optimize .\models\codet5p-220m\onnx\decoder_with_past_model.onnx --level basic --output .\models\codet5p-220m\opt\decoder_with_past.onnx
```

`basic` is conservative — it folds constants, eliminates dead nodes, fuses LayerNorm/GeLU. Avoid `extended` for now; aggressive fusions can produce patterns the QNN EP doesn't recognize.

---

## 3. Shape specialization for the NPU

The Hexagon HTP backend wants **static shapes**. For the encoder you'll commit to a max input length — say 512 tokens for a code snippet — and pad shorter inputs. For the decoder-with-past you commit to a max generated length (e.g. 128) and bucket past-KV sizes.

```powershell
winml shape-fix .\models\codet5p-220m\opt\encoder.onnx --input "input_ids:1x512,attention_mask:1x512" --output .\models\codet5p-220m\fixed\encoder_static.onnx
```

For the decoder, you typically generate a small **family** of static graphs covering past lengths in power-of-two buckets (e.g. 1, 16, 32, 64, 128) and dispatch to the closest one at runtime. The CLI can do this with:

```powershell
winml shape-fix .\models\codet5p-220m\opt\decoder_with_past.onnx --bucket-past-seq 1,16,32,64,128 --output-dir .\models\codet5p-220m\fixed\decoder_buckets
```

This is the single biggest source of "it works on CPU but not NPU" friction with seq2seq models — plan to spend time here.

---

## 4. Quantize to INT8 / INT16

The Hexagon NPU on Snapdragon X Elite is strongest at INT8 weight + INT16 activation (W8A16) for accuracy-sensitive workloads, or full INT8 (W8A8) for max throughput. For a 220M-parameter code model, W8A16 is the sweet spot — code summarization quality degrades noticeably with W8A8 on T5-family models.

You'll need a small calibration dataset — 32–128 representative code snippets is plenty:

```powershell
winml quantize .\models\codet5p-220m\fixed\encoder_static.onnx `
    --calibration-data .\calibration\code_snippets.jsonl `
    --weight-type int8 --activation-type int16 `
    --per-channel `
    --output .\models\codet5p-220m\quant\encoder_w8a16.onnx
```

Repeat for each decoder bucket. Use **the same calibration data piped through the encoder first** to capture realistic activation distributions for the decoder's cross-attention inputs.

After quantization, validate accuracy by running ROUGE-L (or whatever your eval metric is) on a held-out set of code→summary pairs:

```powershell
winml eval .\models\codet5p-220m\quant\ --task summarization --dataset .\eval\codesearchnet_test.jsonl --metric rouge-l
```

Expect ~1–3 ROUGE-L points of degradation versus FP32. If it's worse, fall back to W8A16 with per-channel weights (already on above), or selectively keep the cross-attention layers in FP16.

---

## 5. Compile / pre-process for QNN

The QNN EP supports **context binary caching** — it compiles the ONNX graph into a Hexagon-native binary blob the first time it runs, which is slow (tens of seconds). For a VSCode extension you absolutely want to pre-generate these and ship them:

```powershell
winml compile .\models\codet5p-220m\quant\encoder_w8a16.onnx --ep qnn --device htp --output .\models\codet5p-220m\compiled\encoder.qnn_ctx.onnx
winml compile .\models\codet5p-220m\quant\decoder_buckets\*.onnx --ep qnn --device htp --output-dir .\models\codet5p-220m\compiled\decoder
```

This produces ONNX files with an embedded QNN context blob. At extension load time, ORT skips compilation and goes straight to inference.

---

## 6. Benchmark

```powershell
winml bench .\models\codet5p-220m\compiled\encoder.qnn_ctx.onnx --ep qnn --device htp --iterations 100
winml bench .\models\codet5p-220m\compiled\decoder\decoder_past64.qnn_ctx.onnx --ep qnn --device htp --iterations 100
```

For a 220M T5 on Snapdragon X Elite HTP, ballpark expectations (your numbers will vary):

- Encoder, 1×512 tokens: 15–40 ms
- Decoder step, past=64: 8–20 ms

For a 60-token summary that's roughly 0.5–1.5 seconds end-to-end per snippet — well within "good enough for a VSCode extension" territory.

Compare against CPU baseline too — sometimes for a 220M model the CPU EP with INT8 is competitive and a lot simpler:

```powershell
winml bench .\models\codet5p-220m\quant\encoder_w8a16.onnx --ep cpu
```

---

## 7. Integration in the VSCode extension

VSCode extensions are Node.js. You have two paths:

**Option A — `onnxruntime-node` with the QNN EP.** The npm package supports QNN on Windows ARM64. You load the compiled `.qnn_ctx.onnx` files, run the encoder once per snippet, then drive the decoder loop in TypeScript: pick the right past-bucket, feed `input_ids` + KV-cache, greedy/beam decode, stop on EOS or max_length.

**Option B — a native sidecar.** Ship a small C++ or Rust binary that owns the ORT session and the generation loop, and have the extension talk to it over stdio. Lower per-token overhead and lets you do greedy/beam more cleanly, at the cost of more build/packaging work.

For 220M and code summarization, Option A is the right starting point.

Tokenization: CodeT5+ uses a SentencePiece-style tokenizer. Either ship the tokenizer.json and use `@huggingface/tokenizers` in Node, or pre-bake tokenization into ONNX with `winml tokenizer-export` and run it as part of the graph (cleaner deployment, no JS tokenizer dependency).

---

## 8. Things that will bite you

- **`ScatterND` in KV-cache update.** The QNN EP doesn't love it. If `winml inspect` flags this, rewrite the KV update as a `Concat` of `past_key` with the new step's key (the CLI's `--rewrite-kv-cache concat` flag in `winml optimize` does this).
- **Beam search.** Don't try to put beam search inside the ONNX graph. Run greedy or top-k/top-p sampling host-side. If you need beam, do it in TypeScript with `num_beams=4`; it's only ~4× more decoder calls.
- **First-token latency.** The "no past" decoder graph is fundamentally different shape from the "with past" ones. Compile it separately. Some teams skip it entirely and use the past=1 bucket with a dummy KV — measure both.
- **Memory.** A 220M model in W8A16 is ~250 MB on disk and ~400 MB resident with KV-cache for length 128. Fine for a dev laptop, but flag it in your extension's README.
- **Battery/thermal.** This is the whole reason you're using the NPU. Verify with PerfMon that GPU and CPU utilization stay low during inference — if you see CPU spikes during the decoder loop, that means an op fell back to CPU EP and you've got more graph cleanup to do.

---

## Suggested order of operations for your first pass

1. `winml import` the model.
2. `winml inspect` both encoder and decoder-with-past graphs. Get the unsupported-op list.
3. Run encoder + decoder loop on **CPU EP** end-to-end first to confirm correctness against the HF reference. Don't move to NPU until CPU matches.
4. Static-shape the encoder, single past-bucket (e.g. 64) for the decoder. Get *that* working on QNN EP.
5. Quantize. Re-validate accuracy.
6. Add more past-buckets, compile, ship.

Total elapsed time, if you've never done this before, is realistically 2–4 days of work. The CPU-correctness checkpoint at step 3 is the most important — don't skip it.

Good luck with the extension.
