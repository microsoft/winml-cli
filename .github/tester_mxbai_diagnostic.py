from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

LARGE_FILE_BYTES = 1024 * 1024
PAYLOAD_SUFFIXES = {".bin", ".data", ".onnx", ".safetensors"}
REPORT_SUFFIXES = {".json", ".log", ".md", ".ps1", ".py", ".txt", ".yml", ".yaml"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_onnx(path: Path) -> dict[str, object]:
    import onnx
    from onnx import TensorProto

    model = onnx.load(path, load_external_data=False)
    initializer_types: dict[str, int] = {}
    external_locations: set[str] = set()
    for initializer in model.graph.initializer:
        data_type = TensorProto.DataType.Name(initializer.data_type)
        initializer_types[data_type] = initializer_types.get(data_type, 0) + 1
        external_locations.update(
            entry.value for entry in initializer.external_data if entry.key == "location"
        )
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_types": initializer_types,
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
        "external_locations": sorted(external_locations),
    }


def build_manifest(work: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in work.rglob("*") if item.is_file()):
        size = path.stat().st_size
        if size < LARGE_FILE_BYTES and path.suffix.lower() not in PAYLOAD_SUFFIXES:
            continue
        row: dict[str, object] = {
            "path": path.relative_to(work).as_posix(),
            "bytes": size,
            "sha256": sha256(path),
            "hash_provenance": "runner-computed-from-retained-payload",
        }
        if path.suffix.lower() == ".onnx":
            try:
                row["onnx"] = inspect_onnx(path)
            except Exception:
                row["onnx_inspection_traceback"] = traceback.format_exc()
        rows.append(row)
    return rows


def copy_reports(work: Path, slim: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(item for item in work.rglob("*") if item.is_file()):
        relative = source.relative_to(work)
        if source.suffix.lower() not in REPORT_SUFFIXES:
            continue
        destination = slim / "reports" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination.relative_to(slim).as_posix())
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--original-harness", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--slim", type=Path, required=True)
    args = parser.parse_args()

    work = args.work.resolve()
    slim = args.slim.resolve()
    output = work / "evidence"
    work.mkdir(parents=True, exist_ok=False)
    slim.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    child_exit = 255
    wrapper_error: str | None = None
    command = [
        sys.executable,
        str(args.original_harness.resolve()),
        "--candidate",
        str(args.candidate.resolve()),
        "--output",
        str(output),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        child_exit = completed.returncode
        (slim / "harness.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (slim / "harness.stderr.log").write_text(completed.stderr, encoding="utf-8")
    except BaseException:
        wrapper_error = traceback.format_exc()
        (slim / "wrapper.traceback.log").write_text(wrapper_error, encoding="utf-8")
    finally:
        manifest_error: str | None = None
        copy_error: str | None = None
        manifest: list[dict[str, object]] = []
        copied: list[str] = []
        try:
            manifest = build_manifest(work)
            write_json(slim / "large-model-files.manifest.json", manifest)
        except BaseException:
            manifest_error = traceback.format_exc()
            (slim / "manifest.traceback.log").write_text(manifest_error, encoding="utf-8")
        try:
            copied = copy_reports(work, slim)
        except BaseException:
            copy_error = traceback.format_exc()
            (slim / "report-copy.traceback.log").write_text(copy_error, encoding="utf-8")
        status = {
            "schema_version": "1.0",
            "phase": "DIAGNOSTIC-COMPLETE",
            "result": "PASS" if child_exit == 0 and not wrapper_error else "FAIL",
            "child_exit_code": child_exit,
            "duration_seconds": time.monotonic() - started,
            "command": subprocess.list2cmdline(command),
            "wrapper_traceback": wrapper_error,
            "manifest_traceback": manifest_error,
            "report_copy_traceback": copy_error,
            "large_file_count": len(manifest),
            "copied_report_count": len(copied),
        }
        write_json(slim / "diagnostic-status.json", status)
        write_json(slim / "copied-reports.json", copied)
        shutil.copy2(Path(__file__), slim / Path(__file__).name)

    return child_exit if child_exit != 0 else (1 if wrapper_error else 0)


if __name__ == "__main__":
    raise SystemExit(main())