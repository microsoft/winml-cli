# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evidence-bound report preparation, resume and single delivery entry point."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path


GATES = ("correctness", "performance", "review", "replay")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2) + chr(10), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def status(path):
    """Find the first unrecorded or changed gate, never trust a stale status."""
    state = _read(path) if Path(path).exists() else {"schema_version": 1, "gates": {}}
    if state.get("schema_version") != 1:
        raise ValueError("unsupported workflow schema")
    for gate in GATES:
        records = state.get("gates", {}).get(gate, [])
        if not records:
            return {"next_gate": gate, "reason": "missing evidence"}
        for record in records:
            try:
                valid = _record(record["path"]) == record
            except OSError:
                valid = False
            if not valid:
                return {
                    "next_gate": gate,
                    "reason": "evidence changed or missing",
                    "subject": record["path"],
                }
    return {"next_gate": "deliver", "reason": "all recorded evidence hashes match"}


def record(path, gate, evidence):
    """Record an explicit gate attestation and invalidate downstream records."""
    if gate not in GATES:
        raise ValueError("unknown gate")
    next_gate = status(path)["next_gate"]
    if next_gate != "deliver" and GATES.index(gate) > GATES.index(next_gate):
        raise ValueError("complete " + next_gate + " before " + gate)
    if not evidence:
        raise ValueError("evidence files required")
    records = [_record(p) for p in evidence]
    state = _read(path) if Path(path).exists() else {"schema_version": 1, "gates": {}}
    for later in GATES[GATES.index(gate) :]:
        state["gates"].pop(later, None)
    state["gates"][gate] = records
    _write(path, state)
    return status(path)


def _metrics(sessions):
    if not sessions:
        raise ValueError("raw performance sessions required")
    keys = (
        "running_model_path",
        "ep",
        "device",
        "ep_options",
        "precision",
        "batch_size",
        "effective_batch_size",
    )
    identities = [{k: s.get("benchmark_info", {}).get(k) for k in keys} for s in sessions]
    if any(any(v is None for v in i.values()) for i in identities):
        raise ValueError("session identity missing")
    if any(i != identities[0] for i in identities):
        raise ValueError("incompatible performance sessions")
    if identities[0]["batch_size"] != 1 or identities[0]["effective_batch_size"] != 1:
        raise ValueError("only batch-1 supported")
    return _load("aggregate_perf").aggregate(sessions)


def read_sessions(paths):
    """Reject duplicate paths before pooling session data."""
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("duplicate performance session path")
    return [_read(p) for p in paths]


def prepare(report, baseline, candidate, analyzer):
    """Populate deterministic fields; never manufacture narrative or review verdicts."""
    result = copy.deepcopy(report)
    for section, sessions in (("baseline", baseline), ("leader", candidate)):
        metrics = _metrics(sessions)
        result[section].update(metrics)
        for key in metrics:
            result[section].get("missing_reasons", {}).pop(key, None)
    comparison_keys = (
        "ep",
        "device",
        "ep_options",
        "precision",
        "batch_size",
        "effective_batch_size",
    )
    if any(
        baseline[0]["benchmark_info"].get(k) != candidate[0]["benchmark_info"].get(k)
        for k in comparison_keys
    ):
        raise ValueError("incompatible baseline/candidate comparison")
    target = baseline[0]["benchmark_info"]["ep"]
    matches = [r for r in analyzer["results"] if r.get("ep_type") == target]
    if len(matches) != 1:
        raise ValueError("analyzer must contain exactly one matching provider")
    result["evidence"]["analyzer"] = {
        "coverage": [
            {"classification": k, "count": len(v)} for k, v in matches[0]["classification"].items()
        ],
        "optimizations": [
            {
                "name": x["name"],
                "status": x.get("worst_support", "unknown"),
                "instances": sum(x.get("support_counts", {}).values()),
            }
            for x in analyzer["optimization_output_support"]["optimizations"]
        ],
        "scope": "Static analyzer categories, not runtime fallback or measured gains",
    }
    return result


def deliver(state, config, output):
    """Reuse finalizer and promotion; emit success receipt only after validation."""
    next_gate = status(state)["next_gate"]
    if next_gate != "deliver":
        raise ValueError("complete " + next_gate + " before delivery")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("delivery directory exists; choose a new version")
    finalizer = _load("finalize_output")
    required = (
        "report",
        "champion",
        "winml_config",
        "rebuild_config",
        "repro_script",
        "repro_lock",
        "promotion_context",
    )
    for key in required:
        if not config.get(key):
            raise ValueError("missing delivery input: " + key)
    # Delivery inputs must be included in recorded evidence, not silently swapped afterward.
    bound = {r["path"]: r for records in _read(state)["gates"].values() for r in records}
    paths = (
        [config[k] for k in required]
        + config.get("companions", [])
        + config.get("repro_assets", [])
    )
    for path in paths:
        if bound.get(str(Path(path).resolve())) != _record(path):
            raise ValueError("unbound delivery input: " + str(path))
    finalizer._load_renderer().validate_report(_read(config["report"]), final=True)
    output.mkdir(parents=True)
    bundle = finalizer.finalize_output(
        Path(config["report"]),
        Path(config["champion"]),
        Path(config["winml_config"]),
        [Path(p) for p in config.get("companions", [])],
        output / "bundle",
        rebuild_config=Path(config["rebuild_config"]),
        repro_script=Path(config["repro_script"]),
        repro_lock=Path(config["repro_lock"]),
        repro_assets=[Path(p) for p in config.get("repro_assets", [])],
    )
    finalizer.validate_output_bundle(bundle)
    promotion = _load("promotion")
    handoff = promotion.create_handoff(bundle, Path(config["promotion_context"]))
    promotion.validate_handoff(handoff)
    if status(state)["next_gate"] != "deliver":
        raise ValueError("evidence changed during delivery")
    receipt = {
        "schema_version": 1,
        "status": "validated",
        "manifest": _record(bundle / "manifest.json"),
        "handoff": _record(handoff),
        "state": _read(state),
        "report_integrity": "validated",
        "model_evidence": "recorded attestation; hashes verified, not independently re-executed",
        "independent_review": "recorded attestation; reviewer identity not authenticated",
        "replay": "recorded evidence; not rerun by deliver",
        "visual_review": "not performed",
    }
    _write(output / "receipt.json", receipt)
    return receipt


def main():
    """Expose bounded actions with structured failure diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status")
    p.add_argument("--state", required=True, type=Path)
    p = sub.add_parser("record")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--gate", choices=GATES, required=True)
    p.add_argument("--evidence", action="append", required=True, type=Path)
    p = sub.add_parser("prepare")
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--baseline", required=True, action="append", type=Path)
    p.add_argument("--candidate", required=True, action="append", type=Path)
    p.add_argument("--analyzer", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("deliver")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "status":
            result = status(args.state)
        elif args.command == "record":
            result = record(args.state, args.gate, args.evidence)
        elif args.command == "deliver":
            result = deliver(args.state, _read(args.config), args.output)
        else:
            result = prepare(
                _read(args.report),
                read_sessions(args.baseline),
                read_sessions(args.candidate),
                _read(args.analyzer),
            )
            result["preparation_sources"] = [
                _record(p) for p in [args.report, *args.baseline, *args.candidate, args.analyzer]
            ]
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "action": args.command,
                    "diagnostic": str(error),
                    "next_action": "Repair named input; preserve frozen bundles.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
