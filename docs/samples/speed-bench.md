# SPEED-Bench — text generation speed by category and input length (Experimental)

[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) is a prompt set
from NVIDIA for measuring speculative decoding. Its `qualitative` config has 80
prompts in each of 11 categories (coding, humanities, math, multilingual, qa,
rag, reasoning, roleplay, stem, summarization, writing). Its `throughput_*`
configs fill prompts to fixed input lengths from 1k to 32k tokens (and 100k),
grouped into low-entropy, mixed and high-entropy categories.

This sample prepares a local subset of SPEED-Bench and runs it with
`winml perf --endpoint --dataset` against an OpenAI-compatible server, so the
report breaks down time to first token and decode throughput by category and
input length. It builds on
[OpenAI-compatible server — text generation speed](openai-endpoint-perf.md).

!!! warning "Experimental"
    `--endpoint`, `--dataset` and the preparation script can change between releases.

## Licensing

SPEED-Bench is governed by the NVIDIA Evaluation Dataset License Agreement, and
the prompts it points to come from third-party datasets with their own licenses
and terms. Review them before you prepare the data. Keep prepared prompts and
reports out of source control.

## Step 1: Prepare the prompts

On the Hugging Face Hub, many SPEED-Bench rows hold the placeholder text
`FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`
instead of a prompt, plus a reference to the source dataset. NVIDIA publishes a
standalone `prepare.py` that rebuilds the prompts from those sources. A
separate script runs NVIDIA's `prepare.py` to write local prompt files, and
`winml perf` reads only those files.

The script
[`skills/llm-speed-bench/scripts/prepare_speed_bench.py`](https://github.com/microsoft/winml-cli/blob/main/skills/llm-speed-bench/scripts/prepare_speed_bench.py)
downloads `prepare.py` from NVIDIA NeMo Gym at a pinned commit, checks its
SHA-256, and loads SPEED-Bench at a pinned revision (tag `v1.1`). It then selects
a subset deterministically, rebuilds only the selected prompts, and writes them
as a dataset folder. It declares its own dependencies, so run it with
[uv](https://docs.astral.sh/uv/) from a winml-cli checkout:

```powershell
uv run skills\llm-speed-bench\scripts\prepare_speed_bench.py --output-dir D:\bench\speed-bench
```

On Windows on Arm64, run it with an x64 Python, because `pyarrow` has no Arm64
wheel for the Python versions it needs:

```powershell
uv run --python cpython-3.11-windows-x86_64-none skills\llm-speed-bench\scripts\prepare_speed_bench.py --output-dir D:\bench\speed-bench
```

Some prompts are rebuilt from the gated dataset `cais/hle`. Without a Hugging
Face login that accepted its terms, the script skips those rows before selecting
prompts and reports the count. To include them, accept the terms on the dataset
page, run `hf auth login`, and rerun the script.

Access changes which prompts the fixed-size subsets select. To match a machine
without gated access, pass `--skip-gated` on every machine. Compare prompt
hashes even when subset names and pins match. A sub-category's family is its
prefix before the first underscore; HLE subject labels without underscores
each form their own family and can increase the `throughput` subset's size.

Preparation can download whole source corpora for a few selected prompts.
Allow gigabytes of cache space (`full` needs substantially more than `mini`).
Set `HF_HOME` before `hf auth login` and preparation to choose the cache drive;
it also moves the saved token. To relocate only caches, set `HF_HUB_CACHE` and
`HF_DATASETS_CACHE` instead. `--output-dir` only
places the prepared files. The cache is not required to benchmark written JSONL.
Preparation still checks Hub access with warm caches or `--skip-gated`, so
`HF_HUB_OFFLINE=1` is not supported.
The pins cover SPEED-Bench metadata and `prepare.py`, not every upstream source
file or dependency. Upstream AdaLEval files use plain HTTP at
`opencompass.openxlab.space`; review the pinned script's source URLs and terms.

Pass `--subset` (repeatable) to choose what to write:

| Subset | Selection | Suggested `--max-new-tokens` |
|---|---|---|
| `mini` (default) | 2 qualitative prompts per category, and 2 throughput prompts per entropy category at 1k, 8k and 16k | 256 |
| `throughput` | 2 throughput prompts per sub-category family at each input length from 1k to 32k | 256 |
| `long` | `mini`'s qualitative prompts, and 2 throughput prompts per entropy category at each input length from 1k to 32k | 4096 |
| `full` | Every qualitative prompt and every throughput prompt from 1k to 32k | 1024 |

Each subset is a folder that `--dataset` loads directly:

```text
D:\bench\speed-bench\
    mini\
        test.jsonl      id, prompt, category, sub_category, bucket, source
        manifest.json   pins, counts, data_sha256 and the suggested command
```

`prompt` is the first user turn of the SPEED-Bench row. `bucket` is
`qualitative` or the input length (`1k`, `2k`, `8k`, `16k`, `32k`). Rows are
interleaved across input lengths, then category groups within each length,
rotating the starting group per bucket.
The first rows include one prompt per nonempty bucket before revisiting a
bucket, so `--samples 4` covers all four `mini` buckets when available; small
prefixes do not cover every category. `manifest.json` records the `data_sha256` of `test.jsonl`;
equal values on two machines mean the same prompts.

## Step 2: Start a server

Start a server as in the
[OpenAI-compatible server sample](openai-endpoint-perf.md#step-1-start-a-server),
with a context large enough for the longest prompts in the server's tokenizer
plus the generated tokens.
Input lengths are counted with the `o200k_base` tokenizer, so add headroom for
your model's tokenizer. For the `mini` subset (up to 16k):

```powershell
llama-server -m models\target.gguf --alias target -ngl 99 -c 24576 --parallel 1 --port 8080
```

The other subsets reach 32k in `o200k_base` and need a larger context. The
example's context size is not a guarantee for every tokenizer; measure the
longest prepared prompts with the server's tokenizer rather than relying on
a fixed percentage of headroom.
With `llama-server`, POST each prepared prompt in the largest bucket to
`/tokenize` with `{"content": "<prompt>"}` and count the returned tokens.
Add room for the chat template (inspect it with `/apply-template`) and the
requested generated tokens; the preflight alone does not establish a worst case.

## Step 3: Run the benchmark

```powershell
winml perf --endpoint http://127.0.0.1:8080/v1 --dataset D:\bench\speed-bench\mini `
    --max-new-tokens 256 --column cohort_column=bucket,category -o D:\bench\results\mini.json
```

The defaults of `--column` already match the folder (`input_column=prompt`,
`id_column=id`); `cohort_column=bucket,category` adds a table per input length
and per category to the console report, and `aggregate.cohorts` to the JSON
report. Before a long run, add `--samples B` for one prompt per bucket:
4 for `mini`, 5 for `throughput`, 6 for `long` and `full` when all buckets are
available. This reaches the longest bucket but does not test every prompt.
One failed request stops the run without writing a report.

To measure the full prefill on `llama-server`, add
`--extra-body '{"cache_prompt": false}'`. Otherwise the warmup requests, which
reuse the first rows, leave those prompts in the server's cache.
This quoting requires PowerShell 7.3 or newer. In Windows PowerShell 5.1,
write `--extra-body '{\"cache_prompt\": false}'` instead.

## Compare configurations

To compare, for example, a target model without and with a draft model, label
each run, record the model files, and alternate the configurations across run
indexes:

```powershell
# Server: llama-server -m models\target.gguf ... --port 8080
winml perf --endpoint http://127.0.0.1:8080/v1 --dataset D:\bench\speed-bench\mini `
    --max-new-tokens 256 --column cohort_column=bucket,category `
    --run-label config=plain --run-index 0 --hash-file target=models\target.gguf `
    -o D:\bench\results\plain-0.json

# Server: llama-server -m models\target.gguf --spec-draft-model models\draft.gguf --spec-type draft-simple ... --port 8080
winml perf --endpoint http://127.0.0.1:8080/v1 --dataset D:\bench\speed-bench\mini `
    --max-new-tokens 256 --column cohort_column=bucket,category `
    --run-label config=draft --run-index 0 `
    --hash-file target=models\target.gguf --hash-file draft=models\draft.gguf `
    -o D:\bench\results\draft-0.json
```

Then repeat with `--run-index 1`, `2`, and so on, alternating plain and draft.
For stable results:

- Run one large model at a time, and check that the GPU is otherwise idle before
  each run.
- Expect a few percent of run-to-run noise; use at least 3 run indexes per
  configuration and compare medians.
- Compare only reports with the same `dataset.prompts_sha256`.
- With greedy decoding, `response_sha256` per prompt should match between runs
  of the same model and settings.

## See also

- [OpenAI-compatible server — text generation speed](openai-endpoint-perf.md)
- [`winml perf`](../commands/perf.md)
- The [`llm-speed-bench` agent skill](https://github.com/microsoft/winml-cli/blob/main/skills/llm-speed-bench/SKILL.md)
