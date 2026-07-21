# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Scan ONNX models and summarize Attention structure signatures.

This script is a lightweight structural pre-check to identify models that could
match the QNN TransposeAttentionPattern rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan Attention pattern structure.")
    parser.add_argument("--models-root", type=Path, default=Path(r"D:\Models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "scan_attention_structure_result.json",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=30,
        help="Max number of candidate sample paths to keep.",
    )
    parser.add_argument(
        "--signature-sample-limit",
        type=int,
        default=50,
        help="Max number of attention signatures to include.",
    )
    args = parser.parse_args()

    files = sorted(args.models_root.rglob("*.onnx"))
    total = len(files)

    models_with_attention = 0
    attention_nodes = 0
    attention_input_ge4 = 0
    k_from_transpose = 0
    k_from_transpose_perm_0213 = 0

    candidate_paths: set[str] = set()
    failed_paths: list[str] = []
    attention_model_paths: list[str] = []
    attention_signatures: list[dict[str, object]] = []

    for path in files:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except Exception:
            failed_paths.append(str(path))
            continue

        nodes = list(model.graph.node)
        output_to_node = {output: node for node in nodes for output in node.output}

        has_attention = False
        for node in nodes:
            if node.op_type != "Attention":
                continue

            has_attention = True
            attention_nodes += 1

            if len(node.input) >= 4:
                attention_input_ge4 += 1

            k_source_op = None
            k_source_perm: list[int] | None = None

            if len(node.input) >= 2:
                k_input = node.input[1]
                producer = output_to_node.get(k_input)
                if producer is not None:
                    k_source_op = producer.op_type

                if producer is not None and producer.op_type == "Transpose":
                    k_from_transpose += 1

                    perm = None
                    for attr in producer.attribute:
                        if attr.name == "perm":
                            perm = list(attr.ints)
                            break

                    if perm == [0, 2, 1, 3]:
                        k_from_transpose_perm_0213 += 1
                        candidate_paths.add(str(path))

                    k_source_perm = perm

            if len(attention_signatures) < args.signature_sample_limit:
                attention_signatures.append(
                    {
                        "model_path": str(path),
                        "attention_node_name": node.name,
                        "attention_input_count": len(node.input),
                        "k_source_op": k_source_op,
                        "k_source_perm": k_source_perm,
                    }
                )

        if has_attention:
            models_with_attention += 1
            attention_model_paths.append(str(path))

    result = {
        "models_root": str(args.models_root),
        "total_models": total,
        "failed_models": len(failed_paths),
        "models_with_attention": models_with_attention,
        "attention_nodes": attention_nodes,
        "attention_input_ge4": attention_input_ge4,
        "k_from_transpose": k_from_transpose,
        "k_from_transpose_perm_0213": k_from_transpose_perm_0213,
        "attention_model_count": len(attention_model_paths),
        "attention_model_samples": attention_model_paths[: args.sample_limit],
        "attention_signatures": attention_signatures,
        "candidate_model_count": len(candidate_paths),
        "candidate_samples": sorted(candidate_paths)[: args.sample_limit],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"[SUMMARY] result saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
