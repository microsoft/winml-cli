# Deterministic multi-prompt GenAI benchmarks

`run_genai_multi_prompt_perf.py` benchmarks one or more existing GenAI bundles
with deterministic prompts that round-trip through each bundle tokenizer to an
exact token count. The saved prompts and SHA-256 hashes let another runtime use
identical input content instead of only matching context lengths.

```bash
uv run python scripts/e2e_eval/run_genai_multi_prompt_perf.py \
  --model-bundle qwen3=out/qwen3-bundle \
  --contexts 256 512 1024 \
  --prompt-count 5 \
  --repeats-per-prompt 5 \
  --output-dir eval_results/qwen3-multi-prompt
```

For each context, the runner automatically writes these reusable artifacts:

```text
<output-dir>/
└── <model-name>/
    └── context_<bucket>/
        ├── prompt_00.txt
        ├── ...
        ├── prompt_<count - 1>.txt
        ├── warmup_prompt.txt
        └── perf.json
```

No input `prompts.txt` is required. A downstream runtime can use the generated
model directory as its prompt root. When running that downstream benchmark by
itself, run this script first or otherwise provide the same numbered prompt
files and `warmup_prompt.txt`.

`perf.json` contains aggregate latency and throughput statistics, raw samples,
per-prompt records, prompt seeds, and prompt SHA-256 hashes.
