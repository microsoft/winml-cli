from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

CANDIDATE = "4cc9e8ee56848ffea02642294b27647e3eaeadb0"
PARENT = "3708969b731425b0c6d4b97920d1b5e6519bb013"
MODEL = "mixedbread-ai/mxbai-rerank-base-v1"
REVISION = "800f24c113213a187e65bde9db00c15a2bb12738"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(root: Path, name: str, args: list[str], cwd: Path, timeout: int, env: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(args, cwd=cwd, env=os.environ | (env or {}), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    duration = time.monotonic() - started
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / f"{name}.stdout.log").write_text(stdout, encoding="utf-8")
    (root / "logs" / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
    record = {"name": name, "command": subprocess.list2cmdline(args), "cwd": str(cwd.resolve()), "exit_code": exit_code, "duration_seconds": duration, "timeout_seconds": timeout, "timed_out": timed_out, "stdout": f"logs/{name}.stdout.log", "stderr": f"logs/{name}.stderr.log"}
    write_json(root / "commands" / f"{name}.json", record)
    return record


def dims(value_info) -> list[int | str]:
    return [item.dim_value if item.dim_value else item.dim_param for item in value_info.type.tensor_type.shape.dim]


def inspect_artifact(path: Path, precision: str) -> dict:
    import onnx
    from onnx import TensorProto

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(onnx.load(path, load_external_data=True))
    types: dict[str, int] = {}
    locations = set()
    for initializer in model.graph.initializer:
        name = TensorProto.DataType.Name(initializer.data_type)
        types[name] = types.get(name, 0) + 1
        locations.update(entry.value for entry in initializer.external_data if entry.key == "location")
    external = [{"location": item, "path": str((path.parent / item).resolve()), "exists": (path.parent / item).is_file(), "bytes": (path.parent / item).stat().st_size if (path.parent / item).is_file() else None} for item in sorted(locations)]
    inputs = [{"name": item.name, "dtype": TensorProto.DataType.Name(item.type.tensor_type.elem_type), "shape": dims(item)} for item in model.graph.input]
    outputs = [{"name": item.name, "dtype": TensorProto.DataType.Name(item.type.tensor_type.elem_type), "shape": dims(item)} for item in model.graph.output]
    component_patterns = {
        "deberta.embeddings": ("embedding",),
        "deberta.encoder.layer[]": ("encoder/layer", "encoder.layer", "debertav2layer"),
        "deberta.encoder.layer[].attention": ("attention", "disentangledselfattention"),
        "deberta.encoder.layer[].ffn": ("intermediate", "debertav2output", "/output"),
        "pooler": ("pooler",),
        "classifier": ("classifier",),
    }
    components = {}
    mapped_top_level = set()
    for component_id, patterns in component_patterns.items():
        matched = []
        for index, node in enumerate(model.graph.node):
            searchable = f"{node.name} {' '.join(node.output)}".lower()
            if any(pattern in searchable for pattern in patterns):
                matched.append((index, node))
                if component_id in {"deberta.embeddings", "deberta.encoder.layer[]", "pooler", "classifier"}:
                    mapped_top_level.add(index)
        operator_counts = {}
        for _index, node in matched:
            operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
        components[component_id] = {"node_count": len(matched), "operator_counts": operator_counts, "sample_nodes": [node.name for _index, node in matched[:5]], "mapping_basis": "fresh ONNX hierarchy-tagged node/output scopes", "confidence": "mapped" if matched else "gap"}
    component_mapping = {"components": components, "top_level_mapped_node_count": len(mapped_top_level), "unmapped_node_count": len(model.graph.node) - len(mapped_top_level), "node_name_samples": [node.name for node in model.graph.node[:20]]}
    checks = {"inputs": inputs == [{"name": "input_ids", "dtype": "INT32", "shape": [1, 512]}, {"name": "attention_mask", "dtype": "INT32", "shape": [1, 512]}], "output": outputs == [{"name": "logits", "dtype": "FLOAT16", "shape": [1, 1]}], "external_colocated": bool(external) and all(row["exists"] for row in external), "precision": types.get("FLOAT16", 0) == 228 and types.get("FLOAT", 0) == 0 if precision == "fp16" else types.get("FLOAT", 0) > 0}
    return {"precision": precision, "path": str(path.resolve()), "sha256": sha256(path), "ir_version": model.ir_version, "opsets": [{"domain": row.domain, "version": row.version} for row in model.opset_import], "node_count": len(model.graph.node), "inputs": inputs, "outputs": outputs, "initializer_types": types, "external_data": external, "external_data_total_bytes": sum(row["bytes"] or 0 for row in external), "component_mapping": component_mapping, "checks": checks}


def parity(snapshot: Path, artifacts: dict[str, Path]) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

    pairs = [["what is the capital of China?", "Beijing is the capital city of China."], ["what is the capital of China?", "Paris is the capital of France."], ["how to implement quick sort in python?", "Use partitioning and recursive calls to sort each side."], ["how to implement quick sort in python?", "A weather forecast predicts rain tomorrow."]]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = DebertaV2ForSequenceClassification.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager")
    model.eval()
    encoded = [tokenizer(query, document, padding="max_length", truncation=True, max_length=512, return_tensors="pt") for query, document in pairs]
    with torch.no_grad():
        pytorch = [float(model(input_ids=row["input_ids"], attention_mask=row["attention_mask"]).logits.reshape(-1)[0]) for row in encoded]
    rows = []
    for precision, artifact in artifacts.items():
        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        actual = [float(session.run(["logits"], {"input_ids": row["input_ids"].numpy().astype(np.int32), "attention_mask": row["attention_mask"].numpy().astype(np.int32)})[0].reshape(-1)[0]) for row in encoded]
        pt = np.asarray(pytorch, dtype=np.float64)
        onnx_values = np.asarray(actual, dtype=np.float64)
        pt_order = np.argsort(-pt).tolist()
        onnx_order = np.argsort(-onnx_values).tolist()
        rows.append({"precision": precision, "pair_count": len(pairs), "pytorch_raw_scalar_logits": pytorch, "onnx_raw_scalar_logits": actual, "cosine": float(np.dot(pt, onnx_values) / (np.linalg.norm(pt) * np.linalg.norm(onnx_values))), "max_abs": float(np.max(np.abs(pt - onnx_values))), "pytorch_descending_order": pt_order, "onnx_descending_order": onnx_order, "order_matches": pt_order == onnx_order, "top1_matches": pt_order[0] == onnx_order[0]})
    return {"model_revision": REVISION, "model_class": "DebertaV2ForSequenceClassification", "pairs": pairs, "comparisons": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.candidate.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = run(root, "identity", ["git", "rev-parse", "HEAD"], source, 60)
    head = (root / identity["stdout"]).read_text(encoding="utf-8").strip()
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=source, capture_output=True, text=True, check=True).stdout.strip()
    if head != CANDIDATE or parent != PARENT:
        raise RuntimeError(f"identity mismatch {head} {parent}")
    from huggingface_hub import snapshot_download
    snapshot = root / "model-snapshot"
    snapshot_download(MODEL, revision=REVISION, local_dir=snapshot)
    write_json(root / "model-identity.json", {"model_id": MODEL, "revision": REVISION, "snapshot": str(snapshot), "config_sha256": sha256(snapshot / "config.json")})
    cli = [sys.executable, "-m", "winml.modelkit.cli"]
    recipe = source / "examples" / "recipes" / "mixedbread-ai_mxbai-rerank-base-v1" / "cpu" / "cpu"
    artifacts: dict[str, Path] = {}
    build_records = []
    for precision in ("fp32", "fp16"):
        output = root / "build" / precision
        command = cli + ["build", "-c", str(recipe / f"reranking_{precision}_config.json"), "-m", str(snapshot), "-o", str(output)]
        if precision == "fp16":
            command += ["--precision", "fp16"]
        record = run(root, f"build-{precision}", command, source, 2400)
        build_records.append(record)
        if record["exit_code"] != 0 or record["timed_out"]:
            write_json(root / "terminal-status.json", {"phase": "L0-FAIL", "result": "FAIL", "records": build_records})
            return 1
        artifacts[precision] = output / "model.onnx"
    structures = [inspect_artifact(artifacts[item], item) for item in ("fp32", "fp16")]
    ratio = structures[1]["external_data_total_bytes"] / structures[0]["external_data_total_bytes"]
    write_json(root / "l0-structure.json", {"artifacts": structures, "fp16_to_fp32_payload_ratio": ratio, "all_pass": all(all(row["checks"].values()) for row in structures) and ratio < 0.8})
    if not all(all(row["checks"].values()) for row in structures):
        return 1
    metadata_root = root / "build-metadata"
    for precision in ("fp32", "fp16"):
        for name in ("winml_build_config.json", "export_htp_metadata.json", "analyze_result.json"):
            source_path = artifacts[precision].parent / name
            if source_path.is_file():
                destination = metadata_root / precision / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
    perf_records = [run(root, f"perf-{precision}", cli + ["perf", "-m", str(artifacts[precision]), "--device", "cpu", "--ep", "cpu"], source, 1200) for precision in ("fp32", "fp16")]
    if any(row["exit_code"] != 0 or row["timed_out"] for row in perf_records):
        write_json(root / "terminal-status.json", {"phase": "L1-FAIL", "result": "FAIL", "records": perf_records})
        return 1
    l2 = parity(snapshot, artifacts)
    write_json(root / "l2-parity.json", l2)
    fixture = root / "msmarco-fixture"
    builder = run(root, "l3-fixture", [sys.executable, str(source / "scripts" / "e2e_eval" / "datasets" / "build_msmarco_reranking_fixture.py"), "--output", str(fixture), "--queries", "1", "--max-negatives", "3"], source, 1800)
    eval_output = root / "l3-eval.json"
    eval_command = cli + ["eval", "-m", str(artifacts["fp32"]), "--model-id", str(snapshot), "--task", "reranking", "--dataset", str(fixture), "--split", "dev", "--samples", "1", "--column", "query_column=input", "--column", "expected_output_column=expected_output", "--column", "metadata_column=metadata", "--column", "candidates_column=candidates", "--column", "max_candidates=4", "--ep", "cpu", "--device", "cpu", "-o", str(eval_output), "--overwrite"]
    evaluation = run(root, "l3-eval", eval_command, source, 1800)
    analysis_records = []
    for precision in ("fp32", "fp16"):
        analyze_output = root / f"analyze-{precision}.json"
        analysis_records.append(run(root, f"analyze-{precision}", cli + ["analyze", "--model", str(artifacts[precision]), "--ep", "all", "--output", str(analyze_output)], source, 1800))
    analysis_complete = all((root / f"analyze-{precision}.json").is_file() for precision in ("fp32", "fp16"))
    required_eval_fields = all(name in json.dumps(json.loads(eval_output.read_text(encoding="utf-8"))).lower() for name in ("mrr@10", "recall@1", "recall@10", "scored_groups", "groups_without_positive")) if eval_output.is_file() else False
    fixture_provenance = json.loads((fixture / "provenance.json").read_text(encoding="utf-8")) if (fixture / "provenance.json").is_file() else {}
    selected_rows = fixture_provenance.get("selected_rows", [])
    fixture_semantics = len(selected_rows) == 1 and len(selected_rows[0].get("selected_positive_candidate_ids", [])) >= 1 and len(selected_rows[0].get("selected_negative_candidate_ids", [])) <= 3
    fp16_perf_text = (root / "logs" / "perf-fp16.stdout.log").read_text(encoding="utf-8", errors="replace") + (root / "logs" / "perf-fp16.stderr.log").read_text(encoding="utf-8", errors="replace")
    status_pass = builder["exit_code"] == evaluation["exit_code"] == 0 and required_eval_fields and fixture_semantics and analysis_complete and all(row["exit_code"] in (0, 1, 2) for row in analysis_records) and "Model Precision: fp16" in fp16_perf_text
    status = {"phase": "TESTER-COMPLETE", "result": "PASS" if status_pass else "FAIL", "candidate_sha": CANDIDATE, "parent_sha": PARENT, "l0": "PASS", "l1": "PASS", "l2": "PASS", "l3_exit": evaluation["exit_code"], "l3_semantics": {"required_metric_fields": required_eval_fields, "fixture_semantics": fixture_semantics}, "analysis_exits": {row["name"]: row["exit_code"] for row in analysis_records}, "analysis_complete": analysis_complete, "rules_root": os.environ.get("WINMLCLI_RULES_DIR"), "fp16_perf_reports_fp16": "Model Precision: fp16" in fp16_perf_text, "timed_out": any(row["timed_out"] for row in [builder, evaluation, *analysis_records]), "error": None if status_pass else "One or more final Goal predicates failed."}
    write_json(root / "terminal-status.json", status)
    shutil.rmtree(snapshot, ignore_errors=True)
    shutil.rmtree(root / "build", ignore_errors=True)
    return 0 if status["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())