#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Stage, validate, and atomically publish a champion output bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
else:
    Sequence = importlib.import_module("collections.abc").Sequence
    ModuleType = type(sys)


RESERVED_NAMES = {
    "champion.onnx",
    "winml_config.json",
    "report.json",
    "report.html",
    "manifest.json",
    "rebuild_config.json",
    "repro.ps1",
    "repro-run.ps1",
    "repro.lock.json",
}
REQUIRED_ROLES = {
    "champion.onnx": "champion_onnx",
    "winml_config.json": "winml_config",
    "report.json": "report_json",
    "report.html": "report_html",
}
REQUIRED_POINTERS = {
    "champion": "champion.onnx",
    "winml_config": "winml_config.json",
    "report_json": "report.json",
    "report_html": "report.html",
}
REPRODUCTION_POINTERS = {
    "script": "repro.ps1",
    "run_script": "repro-run.ps1",
    "rebuild_config": "rebuild_config.json",
    "lock": "repro.lock.json",
}
REPRODUCTION_ROLES = {
    "repro.ps1": "reproduction_wrapper",
    "repro-run.ps1": "reproduction_script",
    "rebuild_config.json": "rebuild_config",
    "repro.lock.json": "reproduction_lock",
}
DELIVERABLE_STATUSES = {
    "confirmed",
    "confirmed-performance-provisional-quality",
}
PASSING_CORRECTNESS = {"pass", "passed"}
HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
ALLOWED_SLASH_SWITCHES = {"/c", "/noprofile"}
SLASH_SWITCH_COMMANDS = {
    "/c": {"cmd", "cmd.exe"},
    "/noprofile": {"pwsh", "pwsh.exe", "powershell", "powershell.exe"},
}

REPRO_WRAPPER_TEXT = (
    """param([switch]$ValidateOnly)
#requires -Version 7.3
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Read-BundleJson($RelativePath) {
    $Path = Join-Path $PSScriptRoot $RelativePath
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-BundleSha256($Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$Manifest = Read-BundleJson 'manifest.json'
$Lock = Read-BundleJson 'repro.lock.json'
if ($null -eq $Lock) {
    throw 'Failed to parse repro.lock.json'
}

foreach ($File in $Manifest.files) {
    $RelativePath = [string]$File.path
"""
    "    if ([string]::IsNullOrWhiteSpace($RelativePath) "
    "-or $RelativePath.Contains('\\') -or $RelativePath.Contains(':') "
    "-or $RelativePath.Contains('..') "
    "-or [System.IO.Path]::IsPathRooted($RelativePath)) {\n"
    """        throw "Unsafe manifest path: $RelativePath"
    }
    $Path = Join-Path $PSScriptRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing manifest file: $RelativePath"
    }
    $Item = Get-Item -LiteralPath $Path
    if ([int64]$File.size_bytes -ne [int64]$Item.Length) {
        throw "Size mismatch: $RelativePath"
    }
    if ([string]$File.sha256 -ne (Get-BundleSha256 $Path)) {
        throw "SHA-256 mismatch: $RelativePath"
    }
}

Get-Command winml -ErrorAction Stop | Out-Null

if ($ValidateOnly) {
    exit 0
}

& (Join-Path $PSScriptRoot 'repro-run.ps1')
exit $LASTEXITCODE
"""
)


class OutputBundleError(ValueError):
    """Raised when a final output bundle is incomplete or inconsistent."""


def _load_renderer() -> ModuleType:
    path = Path(__file__).with_name("render_report.py")
    spec = importlib.util.spec_from_file_location("auto_optimize_bundle_renderer", path)
    if spec is None or spec.loader is None:
        raise OutputBundleError(f"cannot load report renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OutputBundleError(f"invalid {label}: {path}") from error


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise OutputBundleError(f"missing {label}: {path}")
    return path


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and ":" not in value


def _safe_basename(value: Any) -> bool:
    if not _safe_relative(value):
        return False
    path = PurePosixPath(value)
    return path.name == value and len(path.parts) == 1


def _entry(path: Path, role: str, source: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "role": role,
        "source": source,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_rebuild_config(path: Path) -> dict[str, Any]:
    config = _load_json(path, "rebuild config")
    if not isinstance(config, dict) or not config:
        raise OutputBundleError("rebuild config must be a non-empty JSON object")
    if config.get("skip_optimize") is True:
        raise OutputBundleError("rebuild config must not set skip_optimize: true")
    return config


def _prepare_inputs(
    report_path: Path,
    champion_onnx: Path,
    winml_config: Path,
    companions: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[Path]]:
    report_path = _require_file(report_path, "report JSON")
    champion_onnx = _require_file(champion_onnx, "champion ONNX")
    if champion_onnx.suffix.lower() != ".onnx":
        raise OutputBundleError("champion artifact must be an ONNX file")
    winml_config = _require_file(winml_config, "WinML config")
    config = _load_json(winml_config, "WinML config")
    if not isinstance(config, dict) or not config:
        raise OutputBundleError("WinML config must be a non-empty JSON object")
    report = _load_json(report_path, "report JSON")
    if not isinstance(report, dict):
        raise OutputBundleError("report JSON must be an object")
    leader = report.get("leader")
    if (
        not isinstance(leader, dict)
        or str(leader.get("status", "")).lower() not in DELIVERABLE_STATUSES
    ):
        raise OutputBundleError("champion output requires a confirmed leader")
    if str(leader.get("correctness", "")).lower() not in PASSING_CORRECTNESS:
        raise OutputBundleError("champion output requires passing correctness")
    renderer = _load_renderer()
    try:
        renderer.validate_quality_gate(leader)
    except renderer.ReportError as error:
        raise OutputBundleError(str(error)) from error

    resolved_companions = [_require_file(Path(path), "champion companion") for path in companions]
    names: set[str] = set()
    for companion in resolved_companions:
        folded = companion.name.casefold()
        if folded in {name.casefold() for name in RESERVED_NAMES}:
            raise OutputBundleError(f"companion uses a reserved output name: {companion.name}")
        if folded in names:
            raise OutputBundleError(f"duplicate companion name: {companion.name}")
        names.add(folded)
    return report, config, champion_onnx, winml_config, resolved_companions


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OutputBundleError(f"reproduction lock {label} must be an object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OutputBundleError(f"reproduction lock {label} must be non-empty")
    return value


def _require_hex(value: Any, label: str, pattern: re.Pattern[str]) -> None:
    text = _require_nonempty_string(value, label)
    if pattern.fullmatch(text) is None:
        raise OutputBundleError(f"reproduction lock {label} must be lowercase hex")


def _validate_winml_identity(winml: dict[str, Any]) -> None:
    kind = _require_nonempty_string(winml.get("kind"), "toolchain.winml.kind")
    if kind == "git":
        if "version" in winml:
            raise OutputBundleError(
                "reproduction lock toolchain.winml git identity must not include version"
            )
        _require_hex(winml.get("revision"), "toolchain.winml.revision", HEX40)
        return
    if kind == "release":
        if "revision" in winml:
            raise OutputBundleError(
                "reproduction lock toolchain.winml release identity must not include revision"
            )
        _require_nonempty_string(winml.get("version"), "toolchain.winml.version")
        return
    raise OutputBundleError("reproduction lock toolchain.winml.kind is invalid")


def _validate_repro_lock(lock: Any, assets: Sequence[Path]) -> list[str]:
    lock = _require_object(lock, "root")
    if lock.get("schema_version") != 1:
        raise OutputBundleError("reproduction lock schema_version must be 1")
    status = lock.get("status")
    if status not in {"available", "requires-unmerged-pr"}:
        raise OutputBundleError("reproduction lock status is invalid")
    if status == "requires-unmerged-pr":
        dependencies = lock.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise OutputBundleError("requires-unmerged-pr reproduction lock needs dependencies")
        for dependency in dependencies:
            dependency = _require_object(dependency, "dependency")
            _require_nonempty_string(dependency.get("url"), "dependency.url")
            _require_hex(dependency.get("revision"), "dependency.revision", HEX40)

    source = _require_object(lock.get("source"), "source")
    for field in ("kind", "id"):
        _require_nonempty_string(source.get(field), f"source.{field}")
    _require_hex(source.get("revision"), "source.revision", HEX40)
    _require_hex(source.get("prepared_model_sha256"), "source.prepared_model_sha256", HEX64)

    toolchain = _require_object(lock.get("toolchain"), "toolchain")
    winml = _require_object(toolchain.get("winml"), "toolchain.winml")
    _validate_winml_identity(winml)
    for field in ("python", "runtime", "provider", "sdk", "device", "driver"):
        _require_nonempty_string(toolchain.get(field), f"toolchain.{field}")

    if not isinstance(lock.get("provider_options"), dict):
        raise OutputBundleError("reproduction lock provider_options must be an object")
    expected = _require_object(lock.get("expected"), "expected")
    for field in ("public_io", "correctness", "topology", "performance"):
        if not isinstance(expected.get(field), dict) or not expected[field]:
            raise OutputBundleError(f"reproduction lock expected.{field} must be non-empty")
    replay = _require_object(lock.get("replay_validation"), "replay_validation")
    if replay != {"status": "pass", "clean_directory": True}:
        raise OutputBundleError(
            "reproduction lock replay_validation must be pass in a clean directory"
        )

    inputs = lock.get("inputs")
    if not isinstance(inputs, list):
        raise OutputBundleError("reproduction lock inputs must be a list")
    asset_by_name = {asset.name: asset for asset in assets}
    folded_asset_names = {asset.name.casefold() for asset in assets}
    input_names: list[str] = []
    folded_input_names: set[str] = set()
    for item in inputs:
        item = _require_object(item, "input")
        name = item.get("path")
        if not _safe_basename(name):
            raise OutputBundleError("reproduction lock input path must be a safe POSIX basename")
        folded = str(name).casefold()
        if folded in folded_input_names:
            raise OutputBundleError(f"duplicate reproduction input basename: {name}")
        folded_input_names.add(folded)
        input_names.append(str(name))
        asset = asset_by_name.get(str(name))
        if asset is None:
            raise OutputBundleError(f"missing reproduction input asset: {name}")
        _require_hex(item.get("sha256"), f"input {name} sha256", HEX64)
        if item.get("sha256") != _sha256(asset):
            raise OutputBundleError(f"reproduction input hash mismatch: {name}")
        _require_nonempty_string(item.get("purpose"), f"input {name} purpose")
    undeclared = sorted(folded_asset_names - folded_input_names)
    if undeclared:
        raise OutputBundleError("undeclared reproduction asset is not allowed")
    return input_names


def _script_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    token: list[str] = []
    active_quote: str | None = None
    index = 0
    while index < len(text):
        character = text[index]
        if character == "`" and active_quote != "'":
            token.append(character)
            index += 1
            if index < len(text):
                token.append(text[index])
        elif active_quote is not None:
            token.append(character)
            if character == active_quote:
                if index + 1 < len(text) and text[index + 1] == active_quote:
                    index += 1
                    token.append(text[index])
                else:
                    active_quote = None
        elif character in {'"', "'"}:
            active_quote = character
            token.append(character)
        elif character.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(character)
        index += 1
    if token:
        tokens.append("".join(token))
    return tokens


def _strip_token_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _decode_token_value(value: str) -> str:
    decoded: list[str] = []
    active_quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if character == "`" and active_quote != "'":
            index += 1
            if index < len(value):
                decoded.append(value[index])
        elif active_quote is not None:
            if character == active_quote:
                if index + 1 < len(value) and value[index + 1] == active_quote:
                    decoded.append(character)
                    index += 1
                else:
                    active_quote = None
            else:
                decoded.append(character)
        elif character in {'"', "'"}:
            active_quote = character
        else:
            decoded.append(character)
        index += 1
    return "".join(decoded)


def _assignment_values(token: str) -> list[str]:
    values: list[str] = []
    for index, character in enumerate(token):
        is_drive_colon = (
            index == 1 and token[0].isalpha() and len(token) > 2 and token[2] in {"\\", "/"}
        )
        if character == "=" or (character == ":" and not is_drive_colon):
            values.append(_strip_token_quotes(token[index + 1 :]))
    return values


def _script_command_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _is_allowed_shell_switch(value: str, previous_token: str | None) -> bool:
    allowed_commands = SLASH_SWITCH_COMMANDS.get(value.casefold())
    if allowed_commands is None or previous_token is None:
        return False
    return _script_command_basename(previous_token) in allowed_commands


def _is_absolute_script_path(
    value: str,
    *,
    after_assignment: bool = False,
    previous_token: str | None = None,
) -> bool:
    if re.match(r"(?i)\A[a-z]:[\\/]", value) is not None:
        return True
    if value.startswith("\\"):
        return True
    if value.startswith("/"):
        if not after_assignment and _is_allowed_shell_switch(value, previous_token):
            return False
        return len(value) > 1
    return False


def _contains_absolute_script_path(text: str) -> bool:
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        previous_token: str | None = None
        for raw_token in _script_tokens(line):
            token = _decode_token_value(raw_token)
            if _is_absolute_script_path(token, previous_token=previous_token):
                return True
            if any(
                _is_absolute_script_path(value, after_assignment=True)
                for value in _assignment_values(token)
            ):
                return True
            previous_token = token
    return False


def _validate_repro_script(path: Path) -> None:
    data = path.read_bytes()
    if not data:
        raise OutputBundleError("reproduction script must be non-empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OutputBundleError("reproduction script must be UTF-8 decodable") from error
    if "$PSScriptRoot" not in text:
        raise OutputBundleError("reproduction script must use $PSScriptRoot")
    if _contains_absolute_script_path(text):
        raise OutputBundleError("reproduction script must not contain absolute filesystem paths")


def _repro_wrapper_bytes() -> bytes:
    return REPRO_WRAPPER_TEXT.encode("utf-8")


def _prepare_reproduction(
    rebuild_config: Path | None,
    repro_script: Path | None,
    repro_lock: Path | None,
    repro_assets: Sequence[Path],
    companions: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Path], dict[str, bytes]] | None:
    primary = [rebuild_config, repro_script, repro_lock]
    if not any(primary):
        if repro_assets:
            raise OutputBundleError("reproduction assets require the primary reproduction trio")
        return None
    if not all(primary):
        raise OutputBundleError("reproduction requires rebuild config, script, and lock together")

    assert rebuild_config is not None
    assert repro_script is not None
    assert repro_lock is not None
    rebuild_config = _require_file(Path(rebuild_config), "rebuild config")
    repro_script = _require_file(Path(repro_script), "reproduction script")
    repro_lock = _require_file(Path(repro_lock), "reproduction lock")
    resolved_assets = [_require_file(Path(path), "reproduction asset") for path in repro_assets]

    _validate_rebuild_config(rebuild_config)
    _validate_repro_script(repro_script)

    names: dict[str, str] = {}
    for name in (*REPRODUCTION_POINTERS.values(), *(path.name for path in companions)):
        folded = name.casefold()
        if folded in names:
            raise OutputBundleError(f"reproduction name collision: {name}")
        names[folded] = name
    reserved = {name.casefold() for name in RESERVED_NAMES}
    for asset in resolved_assets:
        if not _safe_basename(asset.name):
            raise OutputBundleError(
                f"reproduction asset must publish as a safe basename: {asset.name}"
            )
        folded = asset.name.casefold()
        if folded in names or folded in reserved:
            raise OutputBundleError(f"reproduction name collision: {asset.name}")
        names[folded] = asset.name

    asset_names = _validate_repro_lock(_load_json(repro_lock, "reproduction lock"), resolved_assets)
    reproduction = {
        "script": "repro.ps1",
        "run_script": "repro-run.ps1",
        "rebuild_config": "rebuild_config.json",
        "lock": "repro.lock.json",
        "assets": sorted(asset_names, key=str.casefold),
    }
    sources = {
        "rebuild_config.json": rebuild_config,
        "repro-run.ps1": repro_script,
        "repro.lock.json": repro_lock,
    }
    sources.update({path.name: path for path in resolved_assets})
    generated = {"repro.ps1": _repro_wrapper_bytes()}
    return reproduction, sources, generated


def _publish_directory(stage: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise OutputBundleError(f"output directory already exists: {output}")
    backup = output.parent / f".{output.name}.{uuid.uuid4().hex}.backup"
    moved_old = False
    try:
        if output.exists():
            output.replace(backup)
            moved_old = True
        stage.replace(output)
        if moved_old:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
    except Exception:
        if not output.exists() and moved_old and backup.exists():
            backup.replace(output)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if backup.exists() and output.exists():
            if backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)


def finalize_output(
    report_path: Path,
    champion_onnx: Path,
    winml_config: Path,
    companions: Sequence[Path],
    output_dir: Path,
    *,
    overwrite: bool = False,
    rebuild_config: Path | None = None,
    repro_script: Path | None = None,
    repro_lock: Path | None = None,
    repro_assets: Sequence[Path] = (),
) -> Path:
    """Publish a validated champion artifact set and deterministic reports."""
    report, config, champion, config_source, companion_sources = _prepare_inputs(
        Path(report_path),
        Path(champion_onnx),
        Path(winml_config),
        companions,
    )
    reproduction_bundle = _prepare_reproduction(
        rebuild_config,
        repro_script,
        repro_lock,
        repro_assets,
        companion_sources,
    )
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise OutputBundleError(f"output directory already exists: {output}")
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        shutil.copy2(champion, stage / "champion.onnx")
        for companion in companion_sources:
            shutil.copy2(companion, stage / companion.name)
        (stage / "winml_config.json").write_bytes(_json_bytes(config))
        reproduction: dict[str, Any] | None = None
        reproduction_sources: dict[str, Path] = {}
        generated_sources: dict[str, bytes] = {}
        if reproduction_bundle is not None:
            reproduction, reproduction_sources, generated_sources = reproduction_bundle
            for output_name, content in generated_sources.items():
                (stage / output_name).write_bytes(content)
            for output_name, source_path in reproduction_sources.items():
                shutil.copy2(source_path, stage / output_name)

        facts = json.loads(json.dumps(report))
        facts["leader"]["model_path"] = "champion.onnx"
        facts["artifacts"] = {
            "champion_onnx": "champion.onnx",
            "companions": [{"path": path.name, "role": "companion"} for path in companion_sources],
            "winml_config": "winml_config.json",
            "manifest": "manifest.json",
        }
        if reproduction is not None:
            facts["artifacts"]["reproduction"] = reproduction
            facts["conclusion"]["reproduce"] = ["pwsh -File ./repro.ps1"]
        renderer = _load_renderer()
        renderer.validate_report(facts, final=True)
        (stage / "report.json").write_bytes(_json_bytes(facts))
        renderer.render_report(facts, stage / "report.html")

        sources = {
            "champion.onnx": str(champion),
            "winml_config.json": str(config_source),
            "report.json": str(Path(report_path).resolve()),
            "report.html": "generated:report.json",
        }
        sources.update({path.name: str(path) for path in companion_sources})
        sources.update(dict.fromkeys(generated_sources, "generated:reproduction_wrapper"))
        sources.update({name: str(path) for name, path in reproduction_sources.items()})
        roles = dict(REQUIRED_ROLES)
        roles.update({path.name: "companion" for path in companion_sources})
        roles.update(REPRODUCTION_ROLES)
        if reproduction is not None:
            roles.update(dict.fromkeys(reproduction["assets"], "reproduction_asset"))
        paths = sorted(
            (path for path in stage.iterdir() if path.name != "manifest.json"),
            key=lambda path: path.name.casefold(),
        )
        manifest = {
            "schema_version": 1,
            "champion": "champion.onnx",
            "champion_dependencies": [path.name for path in companion_sources],
            "winml_config": "winml_config.json",
            "report_json": "report.json",
            "report_html": "report.html",
            "files": [_entry(path, roles[path.name], sources[path.name]) for path in paths],
        }
        if reproduction is not None:
            manifest["reproduction"] = reproduction
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        validate_output_bundle(stage)
        _publish_directory(stage, output, overwrite=overwrite)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def validate_output_bundle(output_dir: Path) -> dict[str, Any]:
    """Validate file membership, hashes, dependencies, config, and report facts."""
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise OutputBundleError(f"output bundle is not a directory: {output}")
    manifest = _load_json(output / "manifest.json", "bundle manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise OutputBundleError("bundle manifest schema_version must be 1")
    for field, expected in REQUIRED_POINTERS.items():
        if manifest.get(field) != expected:
            raise OutputBundleError(f"bundle manifest {field} must point to {expected}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise OutputBundleError("bundle manifest files must be non-empty")

    declared = {
        str(entry.get("path"))
        for entry in files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    untracked = sorted(actual - declared - {"manifest.json"})
    if untracked:
        raise OutputBundleError(f"bundle contains untracked files: {', '.join(untracked)}")
    missing_declared = sorted(declared - actual)
    if missing_declared:
        raise OutputBundleError(f"bundle is missing declared files: {', '.join(missing_declared)}")

    entries: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or not _safe_relative(entry.get("path")):
            raise OutputBundleError("bundle manifest contains an invalid file path")
        relative = entry["path"]
        folded = relative.casefold()
        if folded in entries:
            raise OutputBundleError(f"duplicate manifest path: {relative}")
        entries[folded] = entry
        path = (output / relative).resolve()
        if output not in path.parents or not path.is_file():
            raise OutputBundleError(f"missing bundled file: {relative}")
        if entry.get("sha256") != _sha256(path):
            raise OutputBundleError(f"hash mismatch: {relative}")
        if entry.get("size_bytes") != path.stat().st_size:
            raise OutputBundleError(f"size mismatch: {relative}")
        if not entry.get("role") or not entry.get("source"):
            raise OutputBundleError(f"missing role/source metadata: {relative}")

    for name, role in REQUIRED_ROLES.items():
        entry = entries.get(name.casefold())
        if entry is None or entry.get("role") != role:
            raise OutputBundleError(f"missing required role {role}: {name}")
    dependencies = manifest.get("champion_dependencies")
    if not isinstance(dependencies, list):
        raise OutputBundleError("champion_dependencies must be a list")
    for dependency in dependencies:
        entry = entries.get(str(dependency).casefold())
        if entry is None or entry.get("role") != "companion":
            raise OutputBundleError(f"missing champion companion: {dependency}")

    reproduction = manifest.get("reproduction")
    if reproduction is not None:
        if not isinstance(reproduction, dict):
            raise OutputBundleError("bundle manifest reproduction must be an object")
        for field, expected in REPRODUCTION_POINTERS.items():
            if reproduction.get(field) != expected:
                raise OutputBundleError(
                    f"bundle manifest reproduction {field} must point to {expected}"
                )
            entry = entries.get(expected.casefold())
            if entry is None or entry.get("role") != REPRODUCTION_ROLES[expected]:
                raise OutputBundleError(
                    f"missing required role {REPRODUCTION_ROLES[expected]}: {expected}"
                )
        assets = reproduction.get("assets")
        if not isinstance(assets, list):
            raise OutputBundleError("bundle manifest reproduction assets must be a list")
        if assets != sorted(assets, key=str.casefold):
            raise OutputBundleError("bundle manifest reproduction assets must be sorted")
        asset_paths: list[Path] = []
        folded_assets: set[str] = set()
        for asset in assets:
            if not _safe_basename(asset):
                raise OutputBundleError(
                    "bundle manifest reproduction asset path must be a safe basename"
                )
            folded = str(asset).casefold()
            if folded in folded_assets:
                raise OutputBundleError(f"duplicate reproduction asset: {asset}")
            folded_assets.add(folded)
            entry = entries.get(folded)
            if entry is None or entry.get("role") != "reproduction_asset":
                raise OutputBundleError(f"missing reproduction_asset role: {asset}")
            asset_paths.append(output / str(asset))
        if (output / "repro.ps1").read_bytes() != _repro_wrapper_bytes():
            raise OutputBundleError("reproduction wrapper does not match generated template")
        _validate_rebuild_config(output / "rebuild_config.json")
        _validate_repro_lock(
            _load_json(output / "repro.lock.json", "reproduction lock"),
            asset_paths,
        )
        _validate_repro_script(output / "repro-run.ps1")

    config = _load_json(output / "winml_config.json", "WinML config")
    if not isinstance(config, dict) or not config:
        raise OutputBundleError("WinML config must be a non-empty JSON object")
    report = _load_json(output / "report.json", "report JSON")
    renderer = _load_renderer()
    renderer.validate_report(report, final=True)
    artifacts = report["artifacts"]
    if artifacts.get("champion_onnx") != "champion.onnx":
        raise OutputBundleError("report champion_onnx does not match the bundle")
    if artifacts.get("winml_config") != "winml_config.json":
        raise OutputBundleError("report winml_config does not match the bundle")
    if artifacts.get("manifest") != "manifest.json":
        raise OutputBundleError("report manifest does not match the bundle")
    report_dependencies = [item.get("path") for item in artifacts.get("companions", [])]
    if report_dependencies != dependencies:
        raise OutputBundleError("report companion dependencies do not match manifest")
    report_reproduction = artifacts.get("reproduction")
    if reproduction is None:
        if report_reproduction is not None:
            raise OutputBundleError("report reproduction does not match manifest")
    elif report_reproduction != reproduction:
        raise OutputBundleError("report reproduction does not match manifest")
    if reproduction is not None and report["conclusion"].get("reproduce") != [
        "pwsh -File ./repro.ps1"
    ]:
        raise OutputBundleError("report reproduction command does not match the bundle")
    return manifest


def main(argv: list[str]) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--winml-config", required=True, type=Path)
    parser.add_argument("--companion", action="append", default=[], type=Path)
    parser.add_argument("--rebuild-config", type=Path)
    parser.add_argument("--repro-script", type=Path)
    parser.add_argument("--repro-lock", type=Path)
    parser.add_argument("--repro-asset", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        validate_output_bundle(args.output)
        print(args.output.resolve())
        return 0
    output = finalize_output(
        args.report,
        args.champion,
        args.winml_config,
        args.companion,
        args.output,
        overwrite=args.overwrite,
        rebuild_config=args.rebuild_config,
        repro_script=args.repro_script,
        repro_lock=args.repro_lock,
        repro_assets=args.repro_asset,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
