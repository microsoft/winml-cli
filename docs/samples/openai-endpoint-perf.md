# OpenAI-compatible server — text generation speed (Experimental)

`winml perf --endpoint` benchmarks text generation on a server that is already
running and speaks the OpenAI chat completions API, such as llama.cpp's
`llama-server`. The server loads the model, applies its chat template and
tokenizes the prompt; winml-cli sends text prompts, streams the responses, and
measures what arrives. Because winml-cli loads no model, the same command
compares different servers, builds, settings and machines.

!!! warning "Experimental"
    The `--endpoint` option and its report fields are experimental and
    can change between releases.

## What is measured

Requests are sent one at a time with `temperature` 0 (greedy decoding) and
`stream: true`. For each request winml-cli records:

| Measurement | Where it comes from |
|---|---|
| Time to first token (`request_ttft_duration_ms`) | Client: from sending the request to the first streamed event that carries generated text. |
| Decode throughput (`steady_state_decode_tokens_per_second`) | Client: `generated_tokens - 1` divided by the time from the first to the last event that carries text. `steady_state_tpot_ms` is the inverse. |
| Request duration (`request_duration_ms`) | Client: from sending the request to the end of the stream. |
| Prompt, cached prompt and generated tokens | Server: the `usage` object of the final event. |
| Stop reason, server build | Server: `finish_reason` and `system_fingerprint`. |
| Server prefill and decode spans, draft tokens and accepted draft tokens | Server: a llama.cpp-style `timings` object, when the server sends one. |

A server that uses speculative decoding can stream several tokens in one event.
Throughput is therefore computed from the token count, not from gaps between
events; `tokens_per_event` in the report shows how many tokens each event carried.
At one token per event, the values equal what GGUF and CGC runs report under the
same names.

## Step 1: Start a server

Any server that implements streaming `/v1/chat/completions` works. This example
uses `llama-server` from a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)
with a small public model. Review the model license before you download it:

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models'))"
llama-server -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf --alias qwen2.5-1.5b-instruct -ngl 99 -c 4096 --port 8080
```

`--alias` sets the model name the server reports; without it, `llama-server`
reports the model's file path, which winml-cli reduces to the file name in reports.

## Step 2: Benchmark the server

Pass the server's OpenAI base URL, the part before `/chat/completions`:

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --max-new-tokens 256
```

The run sends 2 warmup and 10 timed requests unless you set `--warmup` and
`--iterations`, prints a report, and saves the JSON report under
`~/.cache/winml/perf/endpoint-<model>/`. Pass `-o` to choose the path.

When the server lists one model at `/v1/models`, winml-cli uses it. Otherwise pass
the model name with `--endpoint-model`. Some servers, `llama-server` among them,
answer a request for any model name with the model they serve, so the report names
the model the server answered with (`benchmark_info.model_id`), records the
requested name in `benchmark_info.requested_model`, and warns when the two differ.
When the `OPENAI_API_KEY` environment variable is set, its value is sent as a
bearer token; it is never written to the report.

## Measure a full prefill

Servers can reuse a cached prompt prefix. Repeating one prompt then measures a
mostly cached prefill, and the report notes cached prompt tokens. Use
`--extra-body` to pass server-specific request fields, for example to turn off
the `llama-server` prompt cache:

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --extra-body '{"cache_prompt": false}'
```

This quoting needs PowerShell 7.3 or newer. Windows PowerShell 5.1 strips the
inner double quotes when it starts winml, so the JSON arrives invalid; there, write
the value as `'{\"cache_prompt\": false}'`.

`--extra-body` is merged into every request body last, so it can also override
`temperature` or add sampling fields such as `top_k`. The report records it.

## Compare runs

To compare runs later, label each one and record the files the server loaded.
For example, the same model without and with speculative decoding:

```powershell
llama-server -m models\target.gguf -ngl 99 -c 4096 --port 8080
winml perf --endpoint http://127.0.0.1:8080/v1 --run-label config=plain --run-index 0 `
    --hash-file model=models\target.gguf -o results\plain-0.json

llama-server -m models\target.gguf --spec-draft-model models\draft.gguf --spec-type draft-simple -ngl 99 -c 4096 --port 8080
winml perf --endpoint http://127.0.0.1:8080/v1 --run-label config=draft --run-index 0 `
    --hash-file model=models\target.gguf --hash-file draft=models\draft.gguf -o results\draft-0.json
```

With a draft model, the report adds the drafted and accepted draft tokens and the
acceptance rate. Repeat each configuration with increasing `--run-index` values,
and alternate the configurations, so that drift in clocks or temperature affects
every configuration alike.

## Report fields

Endpoint reports use the `schema_version: 3` layout of
[GGUF](gguf-llama-perf.md#report-fields) and
[CGC](cgc-text-generation-perf.md#report-fields) text generation reports, with
`benchmark_info.runtime: openai-endpoint` and `benchmark_info.timing_source:
http-stream`, so the same fields compare a server with a local run of the same
model. Fields the client cannot observe, such as the `task_*` spans, the `load`
block and cold-start latencies, are `null` or absent; endpoint-only data are extra
fields and blocks.

| Block | Contents |
|---|---|
| `benchmark_info` | The schema 3 fields (model the server answered with, iterations, warmup, `max_new_tokens`, prompt, first timed request's token counts and stop reason), then package versions, `requested_model`, `endpoint` (without credentials or query), `prompt_source`, `request_parameters` including `extra_body`, and the timestamp. |
| `provenance` | `run_index`, `run_labels`, `file_hashes` (file name, size and SHA-256; never the folder), `host` (OS, CPUs, GPUs with driver versions, memory; never the host name) and `server` (`system_fingerprint` and the `/v1/models` list). |
| `requests` | One entry per request, warmup included: the schema 3 request fields (`prompt_tokens`, `generated_tokens`, `request_ttft_duration_ms`, `request_duration_ms`, `steady_state_decode_tokens_per_second`, `steady_state_tpot_ms`, `decode_token_durations_ms` between text events), plus `cached_prompt_tokens`, `stop_reason`, `tokens_per_event`, a `server` block of normalized server timings (`prefill_duration_ms`, `decode_tokens_per_second`, `draft_tokens`, `draft_tokens_accepted`, `draft_acceptance_rate`, ...), the raw `server_timings`, and the response text. |
| `aggregate` | Schema 3 percentile summaries over the timed requests and `stop_reason_counts`, plus `tokens_per_event`, `server` summaries of the server rates, and `totals` that pool tokens and time across requests (for example the pooled draft acceptance rate). |

Server timings stay in the `server` blocks instead of the schema 3 `prefill_*` and
`response_eval_*` fields because llama.cpp splits prefill from decode at a
different point. Server-reported fields are `null` when the server does not send them.

## Limitations

- Only streaming `/v1/chat/completions` is supported, with one user message per
  request. The server's chat template and tokenizer decide the prompt tokens, so
  compare prompt token counts only between servers that run the same model.
- Requests are sequential; concurrent load is not measured.
- Client-side times include network and HTTP overhead. Run the server on the same
  machine for decode measurements, and compare with the server timings when the
  server reports them.
- `winml eval` does not support `--endpoint`.

## See also

- [`winml perf`](../commands/perf.md)
- [GGUF — llama.cpp on Windows ML Runtime](gguf-llama-perf.md)
