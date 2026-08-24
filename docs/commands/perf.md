# winml perf

> Benchmark an ONNX model's latency and throughput on a target device.

## When to use this

Use `winml perf` when you want a quantitative latency and throughput baseline for a model on a specific device, or when you need to compare the performance impact of different precision settings, execution providers, or batch sizes.

## Synopsis

```bash
$ winml perf [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | `TEXT` | — | HuggingFace model ID or path to a local `.onnx` file. Required. With `--runtime winml-genai`, also accepts a prebuilt genai **bundle directory**, or a HuggingFace model ID that is auto-built into a bundle on demand. |
| `--runtime` | | `winml\|winml-genai` | `winml` | Inference runtime. `winml` benchmarks single-shot ONNX inference; `winml-genai` benchmarks an onnxruntime-genai bundle (LLM generation: time-to-first-token + decode tokens/sec). With `winml-genai`, a model ID that is not a bundle directory is auto-built into one (cached under `~/.cache/winml/`, targeting the NPU HTP via QNN) before benchmarking. GenAI cache controls are tracked in issue #1275. |
| `--task` | | `TEXT` | auto-detected | Explicit task override (e.g., `image-classification`). Inferred from the model if omitted. |
| `--iterations` | | `INTEGER` | `100` | Number of timed inference iterations used to compute statistics. |
| `--warmup` | | `INTEGER` | `10` | Number of warm-up iterations run before timing begins; excluded from statistics. |
| `--device` | `-d` | `auto\|cpu\|gpu\|npu` | `auto` | Device to run the benchmark on. `auto` selects the highest-priority available device. |
| `--precision` | | `TEXT` | `auto` | Precision mode applied during model build: `auto`, `fp32`, `fp16`, `int8`, `int16`, or compound forms such as `w8a16`. |
| `--ep` | | `TEXT` | — | Force a specific execution provider (e.g., `qnn`, `dml`, `vitisai`, `openvino`, `cpu`). Overrides the device-to-provider mapping. |
| `--ep-options` | | `KEY=VALUE` (multiple) | — | Runtime EP provider option forwarded to the inference session (e.g., `--ep-options htp_performance_mode=burst`). Repeatable. Applies to both HuggingFace model IDs and ONNX file inputs. When detail op-tracing automatically compiles a raw ONNX model, these options are also applied to that compilation. |
| `--output` | `-o` | `PATH` | `~/.cache/winml/perf/<slug>/<timestamp>.json` | Output JSON file path for the benchmark report. |
| `--batch-size` | | `INTEGER` | `1` | Batch size used when generating synthetic input tensors. Ignored when `--input-data` is set. |
| `--input-data` | | `PATH` | — | Path to a `.npz` file of real input tensors to benchmark with instead of randomly generated inputs. The archive's keys must match the model's inputs exactly; dtypes are cast to the model's expected dtype (with a warning) to mirror normal inference. Not supported with `--module`, `--runtime winml-genai`, or composite (dual-encoder) models. |
| `--shape-config` | | `PATH` | — | Path to a JSON file containing shape overrides (e.g., `{"height": 480, "width": 480}`). Used for Hugging Face export and random input generation; ignored in `--module` mode and when `--input-data` is set. |
| `--input-specs` | | `PATH` | — | JSON input tensor specs to merge into the Hugging Face export config before benchmarking. Symbolic string dimensions infer dynamic axes. Ignored for pre-exported ONNX files and in `--module` mode. |
| `--export-config` | | `PATH` | — | JSON ONNX export config overrides to apply when `perf` builds a Hugging Face model before benchmarking. Ignored for pre-exported ONNX files and in `--module` mode. |
| `--dynamic-axes` | | `PATH` | — | JSON dynamic axes mapping for Hugging Face ONNX export, for example `{"input_ids": {"0": "batch", "1": "sequence"}}`. Ignored for pre-exported ONNX files and in `--module` mode. |
| `--quantize/--no-quantize` | | flag | `true` | Run quantization during model build (use `--no-quantize` to skip it). Useful for measuring the fp32 baseline. |
| `--use-cache/--no-use-cache` | | flag | `true` | Reuse persistent model build artifacts. `--no-use-cache` performs a fresh build in a temporary folder and discards it after benchmarking. |
| `--rebuild/--no-rebuild` | | flag | `false` | Force model rebuild even if a cached artifact already exists. |
| `--module` | | `TEXT` | — | PyTorch module class name for per-module benchmarking (e.g., `BertAttention`). Builds and times each matching instance separately. See [Load and export](../concepts/load-and-export.md). |
| `--monitor/--no-monitor` | | flag | `false` | Show a live NPU/CPU utilization chart while the benchmark runs and include hardware metrics in the JSON report. With `--runtime winml-genai`, the monitor wraps the genai load + generation benchmark. |
| `--memory/--no-memory` | | flag | `true` | Record process RSS and adapter memory lifecycle snapshots, plus continuously sampled inference peaks. |
| `--op-tracing` | | `basic\|detail` | — | Enable operator-level profiling. QNN detail tracing requires an EPContext model; a raw ONNX input is detected and compiled automatically with the required profiling options. |
| `--compile` / `--no-compile` | | flag | `false` | Compile the model to EPContext binaries during build. QNN detail op-tracing enables this automatically for a raw ONNX input unless `--no-compile` or `--skip-build` was explicitly specified. For `--runtime winml-genai` on the NPU, `--compile` pre-compiles each QNN stage (in an isolated subprocess) before generation. |
| `--compile-timeout` | | `INTEGER` | `300` | *(winml-genai)* Max seconds to compile each EPContext stage before falling back to the original ONNX. Requires `--compile`. |
| `--prompt` | | `TEXT` | `Explain the theory of relativity in simple terms.` | *(winml-genai)* Prompt text to generate from. Wrapped in the bundle's chat template unless `--no-apply-template`. |
| `--apply-template/--no-apply-template` | | flag | `true` | *(winml-genai)* Wrap `--prompt` in the bundle's chat template before timing. |
| `--max-new-tokens` | | `INTEGER` | `128` | *(winml-genai)* Number of new tokens to generate per iteration. |

## How it works

`winml perf` resolves and registers the target EP as setup before the measured model/session lifecycle. It then loads the model through `WinMLAutoModel` — accepting both HuggingFace IDs and local ONNX files — and generates input tensors from the model's I/O configuration. It runs the specified number of warm-up iterations (excluded from statistics) followed by the timed iterations, collecting per-sample latency. The final report includes mean, min, max, P50, P90, P95, P99, standard deviation, and throughput in samples per second. When `--monitor` is active, a hardware polling loop runs in parallel and records NPU / GPU utilization, CPU usage, and device memory alongside the timing data.

Both runtime reports include `schema_version: 2` and a `benchmark_info.runtime` discriminator (`winml` or `winml-genai`). Shared metadata such as `model_id`, `running_model_path`, `device`, `ep`, `iterations`, `warmup`, and `timestamp` uses the same field names where the concepts overlap; GenAI also keeps `bundle_dir` because the runnable artifact is a bundle directory.

When `--memory` is enabled, both `winml` and `winml-genai` use the same lifecycle and field set. `rss_process_baseline_mb` is captured before EP setup and model/session creation. `rss_before_session_mb` is the post-setup baseline, `rss_after_session_mb` follows eager or lazy native session creation, and `rss_after_inference_mb` is the final diagnostic snapshot. A background sampler records `rss_peak_during_inference_mb`; `rss_peak_delta_mb` is `max(0, peak - process baseline)` and is the usage increase shown by the CLI. `rss_end_delta_mb` remains signed so working-set trimming is visible without being mislabeled as usage. `setup_duration_ms` reports the separated setup span.

Adapter memory uses matching `vram_local_*` and `vram_shared_*` stages. Because the adapter is identified during setup, VRAM peak and end deltas use `vram_*_before_session_mb` as their baseline. If no concrete adapter or PDH counter can be resolved, every VRAM value is `null` in JSON and `N/A` in the CLI; a successfully collected zero remains `0.0`.

The previous `rss_baseline_mb`, `rss_after_compile_mb`, `rss_checkpoint_peak_mb`, `rss_model_load_delta_mb`, `rss_inference_delta_mb`, `rss_total_delta_mb`, and corresponding `vram_*` fields remain for schema compatibility. They are listed in `memory.deprecated_fields`; new consumers should use the lifecycle fields above. In particular, the deprecated `*_checkpoint_peak_mb` fields remain discrete checkpoint maxima rather than true peaks.

With `--runtime winml-genai`, `winml perf` benchmarks the onnxruntime-genai decoder pipeline rather than a single `session.run()`. The JSON report uses a phase-based schema: `load` contains startup spans, `requests` contains one warmup or timed generation sample per request, `aggregate` summarizes timed requests only, `memory` contains optional RAM/VRAM deltas, and `hw_monitor` contains optional monitor output. The optional `memory` and `hw_monitor` top-level names match the classic `winml` perf report; GenAI keeps `load`/`requests`/`aggregate` instead of classic `latency_ms`/`throughput` because generation has distinct prompt, first-token, and decode phases.

### GenAI metric definitions

| Core metric | JSON field(s) | Definition |
|---|---|---|
| Model Load Time | `load.session_load_duration_ms`, `load.native_load_duration_ms` | `session_load_duration_ms` is the outer `GenaiSession.load()` wall-clock span. `native_load_duration_ms` is the onnxruntime-genai `og.Config` + `og.Model` + `og.Tokenizer` span. |
| Weight Upload Time | `load.weight_upload_duration_ms`, `load.weight_upload_estimate_duration_ms` | Exact upload telemetry is `null` today because onnxruntime-genai does not expose it. The estimate is `model_create_duration_ms` and is labeled by `weight_upload_estimate_source`. |
| Cold Start Time | `aggregate.cold_start_ttft_duration_ms`, `aggregate.cold_start_total_duration_ms` | Load plus the first request's TTFT, or load plus the first request's total request duration. |
| Warm Start Time / Latency | `aggregate.request_duration_ms` | Timed-request full duration after warmups: template + tokenization + generator creation + model compute + sequence fetch + detokenization. |
| TTFT | `requests[].model_ttft_duration_ms`, `requests[].request_ttft_duration_ms`, `aggregate.*ttft*` | Model TTFT is prefill + first-token compute. Request TTFT also includes template, tokenization, and generator creation. |
| Prefill TPS | `requests[].prefill_tokens_per_second`, `aggregate.prefill_tokens_per_second` | Prompt tokens divided by `prefill_duration_ms`. |
| Decode TPS | `requests[].steady_state_decode_tokens_per_second`, `aggregate.steady_state_decode_tokens_per_second` | Tokens after the first divided by the sum of per-token decode durations after the first. |
| RAM Usage | `memory.rss_*` | Shared lifecycle snapshots, continuously sampled inference peak, non-negative peak delta, and signed end delta. Requires `--memory`. |
| VRAM Usage | `memory.vram_*` | Local/shared adapter lifecycle snapshots and continuously sampled inference peaks. Unavailable adapter data is `null`, not zero. Requires `--memory`. |

## Examples

Basic benchmark on the best available device:

```bash
$ winml perf -m microsoft/resnet-50
```

```text
Device:      npu
Precision:   auto
Task:        image-classification
Iterations:  100 (+ 10 warmup)
Batch Size:  1

Latency (ms)
  Avg    P50    P90    P95    P99    Min    Max    Std
 2.14   2.11   2.38   2.51   2.79   1.97   3.04   0.12

Throughput: 467.29 samples/sec

Results saved to: ~/.cache/winml/perf/microsoft_resnet-50/2026-05-27T120000.json
```

Benchmark a pre-exported ONNX file on CPU with more iterations:

```bash
$ winml perf -m model.onnx --device cpu --iterations 500
```

Benchmark a text model with an explicit task, targeting the NPU:

```bash
$ winml perf -m bert-base-uncased --task text-classification --device npu --precision w8a16
```

Benchmark with live hardware monitoring enabled:

```bash
$ winml perf -m microsoft/resnet-50 --device npu --monitor
```

Pass runtime EP provider options to tune the session (repeatable):

```bash
$ winml perf -m model.onnx --device npu \
    --ep-options htp_performance_mode=burst \
    --ep-options htp_graph_finalization_optimization_mode=3
```

Per-module benchmarking to find latency hot-spots across all attention blocks:

```bash
$ winml perf -m bert-base-uncased --module BertAttention --iterations 200
```

Benchmark with real inputs from a `.npz` file instead of random data:

```bash
$ winml perf -m model.onnx --input-data inputs.npz
```

Benchmark a Hugging Face model with dynamic export axes before measuring:

```bash
$ winml perf -m microsoft/resnet-50 --dynamic-axes dynamic_axes.json
```

The archive must contain one array per model input, keyed by the input name —
for example:

```python
import numpy as np
np.savez("inputs.npz", pixel_values=np.zeros((4, 3, 224, 224), dtype=np.float32))
```

Array dtypes are cast to the model's expected dtype (with a warning) if they
differ, so `.npz` files saved with default integer/float widths still work.

When `--op-tracing` is combined with `--input-data`, the op trace runs on the
same real tensors as the latency benchmark (not on fresh random inputs). If the
traced graph's inputs don't match the provided data — for example a compiled
context model with different input names — the trace falls back to random inputs
and logs a warning.

## Common pitfalls

- **Warm-up too low on NPU.** The first several inferences on an NPU EP can be significantly slower due to kernel compilation and caching. The default of 10 warm-up iterations is usually enough for vision models, but transformer models with many operators may need `--warmup 30` or higher to reach steady-state latency.
- **Hidden third-party diagnostics.** Normal `winml perf` output suppresses noisy native warning-level diagnostics and Hugging Face download/progress chatter so benchmark results stay readable. Use `-v`/`-vv` or set `WINMLCLI_SHOW_ALL_WARNINGS=1` to show those warnings when debugging provider or Hub issues.
- **`--input-data` keys must match; dtypes are cast.** The `.npz` keys must equal the model's input names — a missing or unexpected key is a hard error (typo protection). Array dtypes are cast to the model's expected dtype with a warning (matching normal inference), so you don't have to hand-match widths. `.npy` files are not supported — save named arrays as `.npz`. When `--input-data` is set, `--batch-size` and `--shape-config` are ignored (the tensors define their own shapes). It is also rejected for `--module` mode, `--runtime winml-genai`, and composite (dual-encoder) models such as CLIP/SigLIP, where each sub-model has its own inputs that a single `.npz` cannot address.
- **Real data only binds if the export kept axes dynamic.** When `-m` is a HuggingFace model ID, `perf` exports it with default shapes (because `--shape-config`/`--batch-size` are ignored under `--input-data`). If that export baked in static shapes, ORT will reject differently-shaped `--input-data`. Use `--dynamic-axes`/symbolic `--input-specs` for the Hugging Face build, or point `-m` at an ONNX file that already has dynamic axes.
- **`--shape-config` is ignored when real or module inputs own the shape.** It is ignored in `--module` mode and when `--input-data` is set. The command prints a warning in both situations.
- **Random inputs do not represent real data distributions.** Latency numbers are accurate, but memory access patterns may differ from production because the generated tensors are uniform random values. For memory-bandwidth-sensitive models this can understate real-world latency.
- **Cross-device comparison.** To compare performance across devices, run `winml perf` separately with different `--device` values and compare the resulting JSON reports.

## See also

- [winml eval](eval.md) — measure accuracy after benchmarking
- [winml build](build.md) — build the quantized artifact that `perf` benchmarks
- [Load and export concept](../concepts/load-and-export.md) — how `--module` per-instance benchmarking works
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — understand `--device` vs `--ep`
