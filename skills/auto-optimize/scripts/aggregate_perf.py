# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Aggregate explicitly selected compatible performance sessions from raw samples."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def aggregate(sessions):
    """Pool timed samples; report linear percentiles and serial inference rate."""
    samples = []
    for session in sessions:
        values = session.get("raw_samples_ms")
        if not isinstance(values, list) or not values:
            raise ValueError("raw_samples_ms must be a nonempty list; summaries cannot be pooled")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("timed samples must be finite positive numbers")
        samples.extend(values)
    if not samples:
        raise ValueError("at least one session required")
    samples.sort()

    def percentile(q):
        index = (len(samples) - 1) * q
        lo = math.floor(index)
        hi = math.ceil(index)
        return samples[lo] + (samples[hi] - samples[lo]) * (index - lo)

    return {
        "p50_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "p99_ms": percentile(0.99),
        "throughput_ips": 1000 * len(samples) / sum(samples),
        "sample_count": len(samples),
        "method": "Linear pooled percentiles; serial batch-1 timed rate; excludes setup/warmup.",
    }


def main():
    """Write derived metrics and hash-bound source receipts without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len({p.resolve() for p in args.inputs}) != len(args.inputs):
        parser.error("duplicate input session")
    payloads = [p.read_bytes() for p in args.inputs]
    sessions = [json.loads(b) for b in payloads]
    identities = []
    for session in sessions:
        info = session.get("benchmark_info", {})
        identities.append(
            {
                k: info.get(k)
                for k in (
                    "running_model_path",
                    "ep",
                    "device",
                    "ep_options",
                    "batch_size",
                    "effective_batch_size",
                    "precision",
                )
            }
        )
    if any(i != identities[0] for i in identities):
        parser.error("incompatible session identity; select one artifact/provider configuration")
    for i in identities:
        if i["batch_size"] != 1 or i["effective_batch_size"] != 1:
            parser.error("only explicit batch-1 sessions supported")
    result = aggregate(sessions)
    result["identity"] = identities[0]
    result["sources"] = [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(b).hexdigest()}
        for p, b in zip(args.inputs, payloads, strict=True)
    ]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
