# OpenAI-compatible server — text generation speed (Experimental)

`winml perf --endpoint` benchmarks text generation on a server that is already
running and speaks the OpenAI chat completions API, such as llama.cpp's
`llama-server`. The server loads the model, applies its chat template and
tokenizes the prompt; winml-cli sends text prompts, streams the responses, and
measures what arrives. Because winml-cli loads no model, the same command
compares different servers, builds, settings and machines.

!!! warning "Experimental"
    The `--endpoint` option and its `schema_version: 4` report are experimental and
    can change between releases.

## What is measured

Requests are sent one at a time with `temperature` 0 (greedy decoding) and
`stream: true`. For each request winml-cli records:

| Measurement | Where it comes from |
|---|---|
| Time to first token (TTFT) | Client: from sending the request to the first streamed event that carries generated text. |
| Decode throughput | Client: `completion_tokens - 1` divided by the time from the first to the last event that carries text. |
| Request duration | Client: from sending the request to the end of the stream. |
| Prompt, cached prompt and completion tokens | Server: the `usage` object of the final event. |
| Finish reason, server build | Server: `finish_reason` and `system_fingerprint`. |
| Server prefill and decode spans, draft tokens and accepted draft tokens | Server: a llama.cpp-style `timings` object, when the server sends one. |

A server that uses speculative decoding can stream several tokens in one event.
Throughput is therefore computed from the token count, not from gaps between
events; `tokens_per_event` in the report shows how many tokens each event carried.

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
the model name with `--endpoint-model`. When the `OPENAI_API_KEY` environment
variable is set, its value is sent as a bearer token; it is never written to the
report.

## Measure a full prefill

Servers can reuse a cached prompt prefix. Repeating one prompt then measures a
mostly cached prefill, and the report notes cached prompt tokens. Use
`--extra-body` to pass server-specific request fields, for example to turn off
the `llama-server` prompt cache:

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --extra-body '{"cache_prompt": false}'
```

`--extra-body` is merged into every request body last, so it can also override
`temperature` or add sampling fields such as `top_k`. The report records it.

## Benchmark a prompt dataset

One prompt says little about a server across different kinds of requests.
`--dataset` sends the rows of a dataset instead, with the same dataset options as
`winml eval`. It accepts a Hugging Face dataset ID, a folder saved with
`Dataset.save_to_disk`, or a local folder or file of `.jsonl`, `.json`,
`.parquet` or `.csv` rows. For example, a folder `prompts` with a file
`prompts\data.jsonl`:

```json
{"question_id": "q1", "category": "math", "turns": ["What is 17 times 23?", "Now divide by 3."]}
{"question_id": "q2", "category": "coding", "turns": ["Write a Python function that reverses a string."]}
{"question_id": "q3", "category": "math", "turns": ["List the first five prime numbers."]}
```

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --dataset prompts `
    --column input_column=turns --column id_column=question_id --column cohort_column=category
```

`--column` maps the dataset's columns:

| Key | Default | Meaning |
|---|---|---|
| `input_column` | `prompt` | The prompt: a string (one user message), a list of turns (the first turn is sent as the user message), or a list of `{"role": ..., "content": ...}` messages (sent as they are). |
| `id_column` | `id` | The prompt ID recorded with each request. When the dataset has no such column, the row index is used. IDs must be unique. |
| `cohort_column` | — | Comma-separated columns whose values group prompts into cohorts, for example `category` or `category,difficulty`. Each column is a separate grouping. |

An unknown key, or a column the dataset does not have, fails with the list of
available columns. An empty prompt fails with its row index.

Each `--iterations` value is a pass over the selected rows, so the default of
`1` sends every row once. `--warmup` requests send the first selected rows and
are excluded from statistics. `--samples N` sends the first `N` rows; add
`--shuffle` to take them in a shuffled order with a fixed seed that the report
records. Use `--split` when the dataset has several splits, and `--dataset-name`
and `--dataset-revision` for a Hugging Face dataset.

The console report adds a table per cohort column, with request count, TTFT,
decode throughput and request duration for each cohort value.

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

Endpoint reports use `schema_version: 4`, with `benchmark_info.runtime:
openai-endpoint` and `benchmark_info.timing_source: http-stream`. The JSON Schema
ships with winml-cli as `winml/modelkit/commands/assets/perf-endpoint-report.schema.json`.

| Block | Contents |
|---|---|
| `benchmark_info` | Endpoint (without credentials or query), model, iterations, warmup, `max_new_tokens`, prompt (`null` with `--dataset`), `prompt_source`, request parameters including `extra_body`, package versions, timestamp. |
| `provenance` | `run_index`, `run_labels`, `file_hashes` (file name, size and SHA-256; never the folder), `host` (OS, CPUs, GPUs with driver versions, memory; never the host name) and `server` (`system_fingerprint` and the `/v1/models` list). |
| `dataset` | `null` without `--dataset`. Otherwise `source` (`hub` or `local`), `path` (the dataset ID, or only the last component of a local path), `name`, `revision`, `split`, `samples`, `shuffle`, `seed`, `column_mapping`, `row_count`, `selected_row_count`, `cohort_columns`, and `prompts_sha256`: a SHA-256 over the selected prompt IDs and messages in sending order, equal across runs that sent the same prompts. |
| `requests` | One entry per request, warmup included: `pass` (the timed pass, `null` for warmup), `prompt_id` and `row_index` (`null` without `--dataset`), `cohorts` (`{column: value}`), token counts, `ttft_ms`, `request_duration_ms`, `decode_tokens_per_second`, `token_event_intervals_ms`, normalized server timings (`server_prefill_ms`, `server_decode_tokens_per_second`, `draft_tokens`, `draft_tokens_accepted`, `draft_acceptance_rate`), the raw `server_timings`, the response text, and `response_sha256` (plus `reasoning_sha256` when the server streams reasoning text) to check that runs produced the same output. |
| `aggregate` | Percentile summaries over the timed requests, `totals` that pool tokens and time across requests (for example the pooled draft acceptance rate), `finish_reason_counts`, and `cohorts`: `{column: {value: summary}}`, where each summary has `request_count`, `ttft_ms`, `decode_tokens_per_second` and `request_duration_ms` percentile summaries, and pooled `totals`. |

Server-reported fields are `null` when the server does not send them.

## Limitations

- Only streaming `/v1/chat/completions` is supported. Each request sends one user
  message, or the messages of a `--dataset` row. The server's chat template and
  tokenizer decide the prompt tokens, so
  compare prompt token counts only between servers that run the same model.
- Requests are sequential; concurrent load is not measured.
- Client-side times include network and HTTP overhead. Run the server on the same
  machine for decode measurements, and compare with the server timings when the
  server reports them.
- `winml eval` does not support `--endpoint`.

## See also

- [`winml perf`](../commands/perf.md)
- [GGUF — llama.cpp on Windows ML Runtime](gguf-llama-perf.md)
