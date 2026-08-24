# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Benchmark WinML GenAI bundles with deterministic shared prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from winml.modelkit.session import GenaiSession, GenerationConfig


CONTEXT_DIRS = {256: "256", 512: "512", 1024: "1k"}
PROMPT_WORDS = (
    "adapter",
    "array",
    "async",
    "benchmark",
    "branch",
    "buffer",
    "cache",
    "class",
    "compile",
    "context",
    "data",
    "decode",
    "device",
    "function",
    "graph",
    "input",
    "kernel",
    "latency",
    "memory",
    "model",
    "output",
    "pipeline",
    "prompt",
    "provider",
    "runtime",
    "session",
    "tensor",
    "thread",
    "token",
    "validate",
)


class Tokenizer(Protocol):
    """Minimal tokenizer interface needed to construct exact-length prompts."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs into text."""
        ...


def parse_named_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for {option}, got: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or not path.is_dir():
            raise FileNotFoundError(f"Invalid {option} value: {value}")
        result[name] = path
    return result


def build_seeded_prompt(tokenizer: Tokenizer, target_tokens: int, seed: int) -> str:
    """Return deterministic text that round-trips to exactly target_tokens."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")

    rng = random.Random(seed)
    words = [rng.choice(PROMPT_WORDS) for _ in range(target_tokens * 2)]
    tokens = tokenizer.encode(" ".join(words))
    while len(tokens) < target_tokens:
        words.extend(rng.choice(PROMPT_WORDS) for _ in range(target_tokens))
        tokens = tokenizer.encode(" ".join(words))

    prompt = tokenizer.decode(tokens[:target_tokens])
    round_trip = tokenizer.encode(prompt)
    if len(round_trip) == target_tokens:
        return prompt

    while len(round_trip) > target_tokens and prompt:
        prompt = prompt[:-1]
        round_trip = tokenizer.encode(prompt)

    fillers = (" a", " x", " 0", ".", "\n")
    for _ in range(target_tokens * 8):
        if len(round_trip) == target_tokens:
            return prompt
        for filler in fillers:
            candidate = prompt + filler
            candidate_tokens = tokenizer.encode(candidate)
            if len(round_trip) < len(candidate_tokens) <= target_tokens:
                prompt = candidate
                round_trip = candidate_tokens
                break
        else:
            raise RuntimeError(
                f"Could not construct a {target_tokens}-token prompt for seed {seed}"
            )

    raise RuntimeError(f"Could not construct a {target_tokens}-token prompt for seed {seed}")


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        index = min(int(len(ordered) * percent / 100), len(ordered) - 1)
        return ordered[index]

    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def run_context(
    model_name: str,
    bundle: Path,
    output_root: Path,
    context: int,
    prompt_count: int,
    repeats_per_prompt: int,
    prompt_seed: int,
    max_new_tokens: int,
) -> None:
    if context not in CONTEXT_DIRS:
        raise ValueError(f"Unsupported context bucket: {context}")
    context_dir = output_root / model_name / f"context_{CONTEXT_DIRS[context]}"
    context_dir.mkdir(parents=True, exist_ok=True)

    session = GenaiSession(bundle)
    session.load()
    try:
        prompts = []
        for index in range(prompt_count):
            seed = prompt_seed + index
            text = build_seeded_prompt(session, context, seed)
            path = context_dir / f"prompt_{index:02d}.txt"
            path.write_text(text, encoding="utf-8")
            prompts.append(
                {
                    "index": index,
                    "seed": seed,
                    "text": text,
                    "file": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

        warmup_text = build_seeded_prompt(session, context, prompt_seed - 1)
        (context_dir / "warmup_prompt.txt").write_text(warmup_text, encoding="utf-8")
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        session.generate_timed(warmup_text, generation_config)

        records = []
        for prompt in prompts:
            prompt_decode = []
            for repeat in range(repeats_per_prompt):
                timing = session.generate_timed(prompt["text"], generation_config)
                if timing.input_tokens != context:
                    raise RuntimeError(
                        f"Prompt token mismatch: expected={context} actual={timing.input_tokens}"
                    )
                if timing.decode_tokens_per_sec <= 0:
                    raise RuntimeError(
                        "WinML generation produced fewer than two measured output tokens"
                    )
                prompt_decode.append(timing.decode_tokens_per_sec)
                records.append(
                    {
                        "prompt_index": prompt["index"],
                        "prompt_seed": prompt["seed"],
                        "repeat": repeat,
                        "prompt_file": prompt["file"],
                        "prompt_sha256": prompt["sha256"],
                        "prompt_tokens": timing.input_tokens,
                        "generated_tokens": timing.generated_tokens,
                        "prefill_ms": timing.prefill_s * 1000,
                        "ttft_ms": timing.ttft_s * 1000,
                        "decode_tokens_per_sec": timing.decode_tokens_per_sec,
                        "tpot_ms": timing.tpot_s * 1000,
                        "total_generation_ms": timing.total_s * 1000,
                    }
                )
            print(
                f"[PROMPT] context={context} "
                f"prompt={prompt['index'] + 1}/{prompt_count} "
                f"repeat={repeats_per_prompt} "
                f"decode_mean={statistics.fmean(prompt_decode):.2f}",
                flush=True,
            )

        decode_samples = [float(record["decode_tokens_per_sec"]) for record in records]
        prefill_samples = [float(record["prefill_ms"]) for record in records]
        ttft_samples = [float(record["ttft_ms"]) for record in records]
        tpot_samples = [float(record["tpot_ms"]) for record in records]
        total_samples = [float(record["total_generation_ms"]) for record in records]
        generated_samples = [int(record["generated_tokens"]) for record in records]
        report = {
            "schema_version": "multi-prompt-1.0",
            "benchmark_info": {
                "runtime": "winml-genai",
                "model_id": model_name,
                "bundle_dir": str(bundle),
                "prompt_tokens": context,
                "generated_tokens": min(generated_samples),
                "context_length": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_count": prompt_count,
                "repeats_per_prompt": repeats_per_prompt,
                "prompt_seed": prompt_seed,
                "max_new_tokens": max_new_tokens,
                "warmup": 1,
                "apply_template": False,
            },
            "prefill_ms": stats(prefill_samples),
            "ttft_ms": stats(ttft_samples),
            "total_generation_ms": stats(total_samples),
            "decode": {
                "tokens_per_sec": statistics.fmean(decode_samples),
                "tpot_ms": statistics.fmean(tpot_samples),
            },
            "raw": {
                "decode_tokens_per_sec": decode_samples,
                "prefill_ms": prefill_samples,
                "ttft_ms": ttft_samples,
                "tpot_ms": tpot_samples,
                "total_generation_ms": total_samples,
            },
            "per_prompt": records,
        }
        (context_dir / "perf.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[DONE] context={context} "
            f"decode={statistics.fmean(decode_samples):.2f} "
            f"tok/s samples={len(decode_samples)}",
            flush=True,
        )
    finally:
        session.unload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-bundle",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXT_DIRS))
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--repeats-per-prompt", type=int, default=5)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_count < 1 or args.repeats_per_prompt < 1:
        raise ValueError("prompt-count and repeats-per-prompt must be positive")
    bundles = parse_named_paths(args.model_bundle, "--model-bundle")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, bundle in bundles.items():
        for context in args.contexts:
            print(f"[RUN] WinML {model_name} context={context}", flush=True)
            run_context(
                model_name,
                bundle,
                args.output_dir,
                context,
                args.prompt_count,
                args.repeats_per_prompt,
                args.prompt_seed,
                args.max_new_tokens,
            )


if __name__ == "__main__":
    main()
