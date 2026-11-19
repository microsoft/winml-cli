---
name: llm-speed-bench
description: 'Use when measuring LLM text generation speed on an OpenAI-compatible server (for example llama.cpp llama-server) with SPEED-Bench prompts and `winml perf --endpoint --dataset`: acquiring and preparing SPEED-Bench subsets, starting a server with or without a speculative-decoding draft model, and running repeatable, labeled benchmark passes. Also use when a user asks how to get SPEED-Bench data, why its Hugging Face rows are masked, or how to compare decode throughput, TTFT, or draft acceptance across prompt categories and input lengths. Do not use for building, optimizing, or benchmarking classic ONNX models (use use-winml-cli), for accuracy evaluation, or for training.'
---

# SPEED-Bench with `winml perf`

[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) is NVIDIA's
prompt set for speculative decoding: a `qualitative` config with 11 semantic
categories, and `throughput_*` configs that fill prompts to fixed input lengths
(1k to 32k tokens, plus 100k) in low-, mixed- and high-entropy categories. This
skill prepares local subsets of it and runs them against a server with
`winml perf --endpoint`. `winml perf` only reads the prepared local files; the
preparation script below has its own dependencies and is not part of winml-cli.

Read the CLI help before running commands: `winml perf --help`. The workflow
needs a winml-cli version with `--endpoint` and `--dataset`.

## Rules

- Never commit prepared prompts or benchmark reports. Keep them in a folder
  outside the repository.
- Prepared prompts are rebuilt from third-party datasets. SPEED-Bench is under
  the NVIDIA Evaluation Dataset License, and each source dataset keeps its own
  license and terms. Tell the user to review them before use.
- Do not hand-pick prompts. Use the script's subsets, `--samples`, or a new
  deterministic rule.
- Compare only runs that sent the same prompts: equal `dataset.prompts_sha256`
  in the reports.

## Step 1: Acquire SPEED-Bench

The Hub copy masks many prompts with the text `FULL BENCHMARK DATA SHOULD BE
FETCHED FROM THE SOURCE USING SPECDEC_BENCH` and records where each one comes
from. NVIDIA's standalone `prepare.py` rebuilds them from about 14 source
datasets. [`scripts/prepare_speed_bench.py`](scripts/prepare_speed_bench.py)
wraps it, and `winml perf` then reads only the local files it writes:

1. Downloads `prepare.py` from NVIDIA NeMo Gym at a pinned commit and refuses to
   run it unless its SHA-256 matches.
2. Loads SPEED-Bench at a pinned revision (tag `v1.1`).
3. Selects each subset's rows deterministically, rebuilds only those rows with
   `prepare.py`, and drops any row that still holds the placeholder.
4. Writes `<output-dir>/<subset>/test.jsonl` and `manifest.json`.

Prerequisites:

- [uv](https://docs.astral.sh/uv/). The script declares its dependencies inline
  (PEP 723), so `uv run` builds an isolated environment for it.
- Network access to Hugging Face, GitHub and the source sites.
- Optional: a Hugging Face login that has accepted the terms of the gated
  dataset `cais/hle`. Without it, the script skips the prompts rebuilt from that
  dataset before selecting rows and reports how many it skipped. To include
  them, accept the terms on the dataset page, run `hf auth login`, and rerun.
- Windows on Arm64: `pyarrow` has no Arm64 wheel for the needed Python
  versions. Run the script with an x64 Python, which runs under emulation:
  `uv run --python cpython-3.11-windows-x86_64-none ...`.

```powershell
uv run skills\llm-speed-bench\scripts\prepare_speed_bench.py --output-dir D:\bench\speed-bench
uv run skills\llm-speed-bench\scripts\prepare_speed_bench.py --output-dir D:\bench\speed-bench --subset full --subset long
```

| Subset | Selection | Suggested `--max-new-tokens` |
|---|---|---|
| `mini` (default) | 2 qualitative prompts per category; 2 throughput prompts per entropy category at 1k, 8k and 16k | 256 |
| `throughput` | 2 prompts per sub-category family (for example `code`, `sort`, `open`, `creative`, `NIAH`) per input length, 1k to 32k | 256 |
| `long` | `mini`'s qualitative prompts; 2 throughput prompts per entropy category at every input length, 1k to 32k | 4096 |
| `full` | Every qualitative prompt and every throughput prompt from 1k to 32k | 1024 |

Selection orders each group's rows by the SHA-256 of `question_id`, alternates
between sub-categories inside a group, and interleaves the groups. Any prefix is
therefore balanced, so `--samples N` is a fair smaller run. The first run
downloads the source datasets into the Hugging Face cache; `full` downloads
and rebuilds the most.

Columns of `test.jsonl`, chosen to match the `winml perf --dataset` defaults:

| Column | Contents |
|---|---|
| `id` | SPEED-Bench `question_id`. |
| `prompt` | Text of the first user turn. Later turns of multi-turn prompts are not sent. |
| `category` | Qualitative category, or entropy category for throughput prompts. |
| `sub_category` | SPEED-Bench sub-category; can be `null`. |
| `bucket` | `qualitative`, or the input length: `1k`, `2k`, `8k`, `16k`, `32k`. |
| `source` | Source dataset URL recorded by SPEED-Bench. |

`manifest.json` records the pins (dataset revision, `prepare.py` commit and
SHA-256), counts per bucket and category, skipped and dropped rows, the
`data_sha256` of `test.jsonl`, package versions, and the suggested command. Two
machines that report the same `data_sha256` prepared the same prompts.

Throughput lengths are counted with the `o200k_base` tokenizer. A server's
tokenizer produces a different count, so size its context with headroom: at
least the bucket length plus `--max-new-tokens`, plus about 25 percent.

## Step 2: Start a server

Any server with streaming `/v1/chat/completions` works. Example with
`llama-server` from a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)
and public models (check each model's license first):

```powershell
# Target model only
llama-server -m models\target.gguf --alias target -ngl 99 -c 24576 --parallel 1 --port 8080

# Target model with a draft model for speculative decoding
llama-server -m models\target.gguf --spec-draft-model models\draft.gguf --spec-type draft-simple --alias target -ngl 99 -c 24576 --parallel 1 --port 8080
```

Use `--parallel 1` so one request owns the server, and size `-c` for the
largest bucket in the subset. Speculative decoding options change between
llama.cpp releases; check `llama-server --help`.

## Step 3: Run the benchmark

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --dataset D:\bench\speed-bench\mini `
    --max-new-tokens 256 --column cohort_column=bucket,category `
    --run-label config=draft --run-index 0 `
    --hash-file target=models\target.gguf --hash-file draft=models\draft.gguf `
    -o D:\bench\results\draft-0.json
```

- `--column cohort_column=bucket,category` adds per-bucket and per-category
  tables to the console and `aggregate.cohorts` to the report.
- `--samples N` sends a balanced prefix for a quick check.
- `--run-label KEY=VALUE` names the configuration; `--run-index` numbers the
  repeats; `--hash-file LABEL=PATH` records the size and SHA-256 of each file
  the server loaded.
- To measure a full prefill on `llama-server`, turn off its prompt cache:
  `--extra-body '{"cache_prompt": false}'`. Warmup requests reuse the first
  rows, so without this they prime the cache for the timed pass.

## Measurement hygiene

- Alternate configurations across run indexes (plain 0, draft 0, plain 1,
  draft 1, ...) instead of running all repeats of one configuration in a row,
  so that thermal and clock drift affects every configuration alike.
- Run one large model at a time and stop other servers first. Before each run,
  check that the GPU is idle (for example `nvidia-smi` or Task Manager).
- Expect run-to-run noise of a few percent. Use at least 3 run indexes per
  configuration before drawing conclusions, and compare medians.
- Greedy decoding is used, so `response_sha256` should match between runs of
  the same model and settings; a mismatch shows that outputs changed.
- Store reports outside the repository, one file per run.

## Next step

To compare runs, load the JSON reports that share `dataset.prompts_sha256`,
group them by `provenance.run_labels`, and compare per-cohort medians across
run indexes. A report-building workflow can consume these files directly.

See also the [SPEED-Bench sample](../../docs/samples/speed-bench.md) and the
[OpenAI-compatible server sample](../../docs/samples/openai-endpoint-perf.md).
