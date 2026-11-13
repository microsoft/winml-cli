#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rebuild and execute the final composed WinML config in a clean directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]


class ConfigValidationError(ValueError):
    """Raised when the final composed config cannot reproduce the champion."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigValidationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ConfigValidationError(f"missing {label}: {resolved}")
    return resolved


def _run(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    runner: Runner,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode:
        raise ConfigValidationError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )
    return result


def _find_rebuilt_model(build_dir: Path, relative: str | None) -> Path:
    if relative:
        candidate = (build_dir / relative).resolve()
        if build_dir.resolve() not in candidate.parents or not candidate.is_file():
            raise ConfigValidationError(f"rebuilt model not found: {relative}")
        return candidate
    direct = build_dir / "model.onnx"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(build_dir.rglob("model.onnx"))
    if len(candidates) != 1:
        raise ConfigValidationError(
            "build must produce one discoverable model.onnx; pass --rebuilt-model-relative"
        )
    return candidates[0].resolve()


def _value_info(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return {"name": value.name, "dtype": tensor.elem_type, "shape": shape}


def _public_io(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as error:
        raise ConfigValidationError(f"rebuilt ONNX validation failed: {path}") from error
    initializers = {item.name for item in model.graph.initializer}
    return {
        "inputs": [_value_info(item) for item in model.graph.input if item.name not in initializers],
        "outputs": [_value_info(item) for item in model.graph.output],
    }


def _source_identity(source: str) -> dict[str, Any]:
    path = Path(source)
    if path.is_file():
        resolved = path.resolve()
        return {"kind": "file", "value": str(resolved), "sha256": _sha256(resolved)}
    return {"kind": "model-id", "value": source}


def _cosine_means(evaluation: dict[str, Any]) -> dict[str, float]:
    values = evaluation.get("metrics", {}).get("cosine_similarity_mean")
    if not isinstance(values, dict) or not values:
        raise ConfigValidationError("comparison output lacks cosine_similarity_mean")
    result: dict[str, float] = {}
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ConfigValidationError(f"invalid cosine similarity for output {name}")
        result[str(name)] = float(value)
    return result


def validate_config(
    *,
    config_path: Path,
    source: str,
    champion_path: Path,
    input_data: Path,
    work_dir: Path,
    receipt_path: Path,
    device: str,
    ep: str,
    min_cosine: float,
    winml: str = "winml",
    rebuilt_model_relative: str | None = None,
    keep_build: bool = False,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run a clean public-CLI rebuild and compare it with the confirmed champion."""
    config = _require_file(config_path, "final rebuild config")
    champion = _require_file(champion_path, "confirmed champion")
    inputs = _require_file(input_data, "fixed validation inputs")
    _load_json(config, "final rebuild config")
    if type(min_cosine) not in (int, float) or not 0 < min_cosine <= 1:
        raise ConfigValidationError("min_cosine must be in (0, 1]")
    work = work_dir.resolve()
    receipt = receipt_path.resolve()
    if work.exists():
        raise ConfigValidationError(f"validation work directory must not exist: {work}")
    if receipt.exists():
        raise ConfigValidationError(f"validation receipt already exists: {receipt}")
    work.mkdir(parents=True)
    build_dir = work / "build"
    evaluation_path = work / "comparison.json"
    build_stdout = work / "build.stdout.log"
    build_stderr = work / "build.stderr.log"
    eval_stdout = work / "eval.stdout.log"
    eval_stderr = work / "eval.stderr.log"
    build_command = [
        winml,
        "build",
        "-c",
        str(config),
        "-m",
        source,
        "-o",
        str(build_dir),
        "--rebuild",
    ]
    _run(build_command, work, build_stdout, build_stderr, runner)
    rebuilt = _find_rebuilt_model(build_dir, rebuilt_model_relative)
    rebuilt_io = _public_io(rebuilt)
    champion_io = _public_io(champion)
    if rebuilt_io != champion_io:
        raise ConfigValidationError("rebuilt model public I/O does not match the champion")

    eval_command = [
        winml,
        "eval",
        "--mode",
        "compare",
        "-m",
        str(rebuilt),
        "--device",
        device,
        "--ep",
        ep,
        "--skip-build",
        "--reference",
        str(champion),
        "--reference-device",
        device,
        "--reference-ep",
        ep,
        "--input-data",
        str(inputs),
        "--output",
        str(evaluation_path),
        "--overwrite",
        "--format",
        "json",
    ]
    _run(eval_command, work, eval_stdout, eval_stderr, runner)
    evaluation = _load_json(evaluation_path, "comparison output")
    cosine = _cosine_means(evaluation)
    observed = min(cosine.values())
    if observed < min_cosine:
        raise ConfigValidationError(
            f"rebuilt model cosine {observed:.8f} is below required {min_cosine:.8f}"
        )

    rebuilt_sha256 = _sha256(rebuilt)
    result = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clean_directory": True,
        "identity": {
            "config_sha256": _sha256(config),
            "source": _source_identity(source),
            "champion_sha256": _sha256(champion),
            "input_data_sha256": _sha256(inputs),
            "device": device,
            "ep": ep,
        },
        "build": {
            "status": "pass",
            "command": build_command,
            "rebuilt_model_sha256": rebuilt_sha256,
            "public_io": "pass",
        },
        "runtime_load": {"status": "pass", "command": eval_command},
        "correctness": {
            "status": "pass",
            "metric": "minimum output cosine_similarity_mean",
            "threshold": min_cosine,
            "observed": observed,
            "outputs": cosine,
        },
        "evidence": [
            {"path": path.name, "sha256": _sha256(path)}
            for path in (build_stdout, build_stderr, eval_stdout, eval_stderr, evaluation_path)
        ],
        "rebuilt_model_retained": keep_build,
    }
    _write_json(receipt, result)
    if not keep_build:
        shutil.rmtree(build_dir)
    return result


def verify_receipt(
    receipt_path: Path,
    config_path: Path,
    champion_path: Path,
    input_paths: list[Path] | tuple[Path, ...] = (),
    repro_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Verify that a passing receipt still binds the files being delivered."""
    receipt = _load_json(_require_file(receipt_path, "config validation receipt"), "receipt")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "pass":
        raise ConfigValidationError("config validation receipt is not passing schema v1")
    if receipt.get("clean_directory") is not True:
        raise ConfigValidationError("config validation did not run in a clean directory")
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        raise ConfigValidationError("config validation receipt lacks identity")
    if identity.get("config_sha256") != _sha256(_require_file(config_path, "config")):
        raise ConfigValidationError("config changed after composed-config validation")
    if identity.get("champion_sha256") != _sha256(_require_file(champion_path, "champion")):
        raise ConfigValidationError("champion changed after composed-config validation")
    if repro_lock_path is not None:
        lock = _load_json(_require_file(repro_lock_path, "reproduction lock"), "lock")
        expected_source_sha256 = lock.get("source", {}).get("prepared_model_sha256")
        source = identity.get("source")
        if (
            not isinstance(source, dict)
            or source.get("kind") != "file"
            or source.get("sha256") != expected_source_sha256
        ):
            raise ConfigValidationError(
                "validated prepared source does not match the reproduction lock"
            )
    if input_paths:
        hashes = {_sha256(_require_file(path, "reproduction input")) for path in input_paths}
        if identity.get("input_data_sha256") not in hashes:
            raise ConfigValidationError("validated input is not present in reproduction assets")
    for section in ("build", "runtime_load", "correctness"):
        if receipt.get(section, {}).get("status") != "pass":
            raise ConfigValidationError(f"config validation {section} did not pass")
    correctness = receipt["correctness"]
    if correctness.get("observed", -math.inf) < correctness.get("threshold", math.inf):
        raise ConfigValidationError("config validation correctness threshold was not met")
    return receipt


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--input-data", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ep", required=True)
    parser.add_argument("--min-cosine", required=True, type=float)
    parser.add_argument("--winml", default="winml")
    parser.add_argument("--rebuilt-model-relative")
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_config(
            config_path=args.config,
            source=args.source,
            champion_path=args.champion,
            input_data=args.input_data,
            work_dir=args.work_dir,
            receipt_path=args.receipt,
            device=args.device,
            ep=args.ep,
            min_cosine=args.min_cosine,
            winml=args.winml,
            rebuilt_model_relative=args.rebuilt_model_relative,
            keep_build=args.keep_build,
        )
        print(json.dumps(result))
        return 0
    except (ConfigValidationError, OSError) as error:
        print(json.dumps({"status": "blocked", "diagnostic": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))