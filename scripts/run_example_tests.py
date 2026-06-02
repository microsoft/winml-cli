#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Run build+eval tests for example configs under a given EP bucket.

Flow per config:
1) build:  winml build -m <hf_id> -c <config> -o <build_dir>
2) eval:   winml eval  -m <built_onnx> --model-id <hf_id> ...

Notes:
- Perf is intentionally not run in this workflow.
- For VitisAI EP, build is forced with --no-compile.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from winml.modelkit.utils.constants import ALL_EP_NAMES, EP_NAME_TO_ALIAS, normalize_ep_name


REPO_ROOT = Path(__file__).resolve().parent.parent

EP_TO_EXAMPLES_FOLDER = {
    "cpu": "mlas",
}

EP_CHOICES = [name for name in ALL_EP_NAMES if name not in ("cuda", "CUDAExecutionProvider")]
EP_CHOICES_MAP = {name.lower(): name for name in EP_CHOICES}
DEVICE_CHOICES = ["cpu", "gpu", "npu"]

ROLE_MAP_BY_TASK: dict[str, tuple[str, ...]] = {
    "zero-shot-image-classification": ("image-encoder", "text-encoder"),
    "image-to-text": ("encoder", "decoder"),
}


def parse_ep(ep_arg: str) -> str:
    """Parse EP argument case-insensitively using winml choices."""
    ep = EP_CHOICES_MAP.get(ep_arg.lower())
    if ep is None:
        choices_text = ", ".join(EP_CHOICES)
        raise argparse.ArgumentTypeError(f"Invalid --ep '{ep_arg}'. Choices: {choices_text}")
    return ep


def parse_device(device_arg: str) -> str:
    """Parse device argument case-insensitively using winml choices."""
    device = device_arg.lower()
    if device not in DEVICE_CHOICES:
        choices_text = ", ".join(DEVICE_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Invalid --device '{device_arg}'. Choices: {choices_text}"
        )
    return device


def resolve_ep_and_examples_folder(ep_arg: str) -> tuple[str, str]:
    """Normalize EP value for winml and map to examples folder name."""
    canonical_ep = normalize_ep_name(ep_arg)
    if canonical_ep is None:
        return ep_arg, ep_arg

    ep_for_winml = EP_NAME_TO_ALIAS.get(canonical_ep, ep_arg.lower())
    examples_folder = EP_TO_EXAMPLES_FOLDER.get(ep_for_winml, ep_for_winml)
    return ep_for_winml, examples_folder


def infer_hf_id(config_path: Path) -> str | None:
    """Extract HF model ID from quant.model_name or fallback to directory slug."""
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = (cfg.get("quant") or {}).get("model_name")
        if model_name:
            return model_name
        slug = config_path.parent.name
        return slug.replace("_", "/", 1)
    except Exception:
        return None


def infer_task(config_path: Path) -> str | None:
    """Extract eval task from config."""
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        eval_cfg = cfg.get("eval") if isinstance(cfg.get("eval"), dict) else {}
        task = eval_cfg.get("task")
        if isinstance(task, str) and task:
            return task
        loader = cfg.get("loader") if isinstance(cfg.get("loader"), dict) else {}
        task = loader.get("task")
        if isinstance(task, str) and task:
            return task
        return None
    except Exception:
        return None


def needs_trust_remote_code(config_path: Path) -> bool:
    """Check if config has dataset build_script requiring --trust-remote-code."""
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        dataset = (cfg.get("eval") or {}).get("dataset") or {}
        return bool(dataset.get("build_script"))
    except Exception:
        return False


def clean_caches() -> None:
    """Clean HF and winml caches to free disk space between models."""
    for cache_dir in [
        Path.home() / ".cache" / "winml",
        Path.home() / ".cache" / "huggingface",
    ]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def _build_error_path(config_path: Path, stem: str) -> Path:
    return config_path.parent / f"{stem}_build_result.error.txt"


def _build_timeout_path(config_path: Path, stem: str) -> Path:
    return config_path.parent / f"{stem}_build_result.timeout"


def run_build(
    hf_id: str,
    config_path: Path,
    build_dir: Path,
    ep_for_winml: str,
    timeout: int,
    rebuild: bool,
) -> str:
    """Run winml build and return PASS/FAIL/TIMEOUT."""
    cmd = [
        sys.executable,
        "-m",
        "winml.modelkit",
        "build",
        "-m",
        hf_id,
        "-c",
        str(config_path),
        "-o",
        str(build_dir),
    ]
    if ep_for_winml == "vitisai":
        cmd.append("--no-compile")
    if rebuild:
        cmd.append("--rebuild")

    stem = config_path.stem.replace("_config", "")
    err_path = _build_error_path(config_path, stem)
    timeout_path = _build_timeout_path(config_path, stem)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            if err_path.exists():
                err_path.unlink()
            if timeout_path.exists():
                timeout_path.unlink()
            return "PASS"
        err_path.write_text(result.stderr[-4000:] if result.stderr else "Unknown build error", encoding="utf-8")
        if timeout_path.exists():
            timeout_path.unlink()
        return "FAIL"
    except subprocess.TimeoutExpired:
        timeout_path.write_text("timeout", encoding="utf-8")
        return "TIMEOUT"


def _find_single_onnx(build_dir: Path) -> Path | None:
    direct = build_dir / "model.onnx"
    if direct.exists():
        return direct
    candidates = sorted(p for p in build_dir.rglob("model.onnx") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    return None


def _role_keyword(role: str) -> tuple[str, ...]:
    if role == "image-encoder":
        return ("vision", "image")
    if role == "text-encoder":
        return ("text",)
    if role == "encoder":
        return ("encoder",)
    if role == "decoder":
        return ("decoder",)
    return (role,)


def _resolve_composite_model_args(build_dir: Path, task: str) -> list[str] | None:
    """Resolve role=path args for composite evaluators from module_summary outputs."""
    roles = ROLE_MAP_BY_TASK.get(task)
    if not roles:
        return None

    summary = build_dir / "module_summary.json"
    if not summary.exists():
        return None

    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        instances = payload.get("instances") or []
    except Exception:
        return None

    if not isinstance(instances, list) or not instances:
        return None

    # Build candidate list: (module_path, onnx_path)
    cands: list[tuple[str, Path]] = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        module_path = str(inst.get("module_path") or "")
        out_dir = inst.get("output_dir")
        if not out_dir:
            continue
        onnx_path = Path(out_dir) / "model.onnx"
        if onnx_path.exists():
            cands.append((module_path.lower(), onnx_path))

    if len(cands) < len(roles):
        return None

    assigned: dict[str, Path] = {}
    used_idx: set[int] = set()

    for role in roles:
        keywords = _role_keyword(role)
        picked = None
        for idx, (module_path, path) in enumerate(cands):
            if idx in used_idx:
                continue
            if any(k in module_path for k in keywords):
                picked = (idx, path)
                break
        if picked is None:
            for idx, (_module_path, path) in enumerate(cands):
                if idx in used_idx:
                    continue
                picked = (idx, path)
                break
        if picked is None:
            return None
        used_idx.add(picked[0])
        assigned[role] = picked[1]

    args: list[str] = []
    for role in roles:
        args.extend(["-m", f"{role}={assigned[role]}"])
    return args


def resolve_built_model_args(build_dir: Path, task: str | None) -> list[str] | None:
    """Return eval -m args pointing to built ONNX artifacts."""
    onnx_single = _find_single_onnx(build_dir)
    if onnx_single is not None:
        return ["-m", str(onnx_single)]

    if task:
        comp = _resolve_composite_model_args(build_dir, task)
        if comp:
            return comp

    return None


def run_eval(
    model_args: list[str],
    hf_id: str,
    config_path: Path,
    output_path: Path,
    ep: str,
    device: str,
    timeout: int,
    trust_remote_code: bool = False,
) -> str:
    """Run winml eval and return PASS, FAIL, or TIMEOUT."""
    cmd = [
        sys.executable,
        "-m",
        "winml.modelkit",
        "eval",
        *model_args,
        "--model-id",
        hf_id,
        "--ep",
        ep,
        "--device",
        device,
        "-c",
        str(config_path),
        "-o",
        str(output_path),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0 and output_path.exists():
            return "PASS"
        err_path = output_path.with_suffix(".error.txt")
        err_path.write_text(result.stderr[-4000:] if result.stderr else "Unknown error", encoding="utf-8")
        return "FAIL"
    except subprocess.TimeoutExpired:
        timeout_path = output_path.with_suffix(".timeout")
        timeout_path.write_text("timeout", encoding="utf-8")
        return "TIMEOUT"


def main() -> None:
    """Entrypoint for running build+eval on example configs."""
    parser = argparse.ArgumentParser(description="Run build+eval tests for example configs")
    parser.add_argument(
        "--ep",
        required=True,
        type=parse_ep,
        help="Execution provider (same accepted values as winml --ep)",
    )
    parser.add_argument(
        "--device",
        required=True,
        type=parse_device,
        help="Device (same accepted values as winml --device)",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per build/eval (default: 3600s)")
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        default=False,
        help="Clean ~/.cache/winml and ~/.cache/huggingface between different models (default: disabled)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Pass --rebuild to winml build (force rebuild instead of reusing existing build artifacts)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="If set, delete existing *_eval_result.error.txt / *.timeout and retry those configs.",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model slugs")
    args = parser.parse_args()

    ep_for_winml, examples_ep = resolve_ep_and_examples_folder(args.ep)
    ep_dir = REPO_ROOT / "examples" / examples_ep / args.device
    if not ep_dir.exists():
        print(f"EP directory not found: {ep_dir}")
        sys.exit(1)

    model_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())
    if args.models:
        allowed = set(args.models.split(","))
        model_dirs = [d for d in model_dirs if d.name in allowed]

    configs = sorted(cfg_file for model_dir in model_dirs for cfg_file in model_dir.glob("*_config.json"))

    print(f"EP: {ep_for_winml}, Device: {args.device} (examples/{examples_ep}/{args.device})")
    print(f"Models: {len(model_dirs)}, Configs: {len(configs)}")
    print()

    results = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "SKIP": 0}
    prev_model = None

    for i, cfg_path in enumerate(configs, 1):
        stem = cfg_path.stem.replace("_config", "")
        model_slug = cfg_path.parent.name
        hf_id = infer_hf_id(cfg_path)
        task = infer_task(cfg_path)
        if not hf_id:
            print(f"[{i}/{len(configs)}] {model_slug}/{stem} ... SKIP (no model ID)")
            results["SKIP"] += 1
            continue

        eval_output = cfg_path.parent / f"{stem}_eval_result.json"
        eval_err = eval_output.with_suffix(".error.txt")
        eval_tmo = eval_output.with_suffix(".timeout")

        had_failed_marker = eval_err.exists() or eval_tmo.exists()
        if args.retry_failed:
            if eval_err.exists():
                eval_err.unlink()
            if eval_tmo.exists():
                eval_tmo.unlink()

        if eval_output.exists():
            results["SKIP"] += 1
            continue
        if (eval_err.exists() or eval_tmo.exists()) and not args.retry_failed:
            results["SKIP"] += 1
            continue

        # Clean caches between different models when enabled
        if args.clean_cache and model_slug != prev_model and prev_model is not None:
            clean_caches()
        prev_model = model_slug

        trust = needs_trust_remote_code(cfg_path)
        build_dir = cfg_path.parent / f"{stem}_build_artifacts"

        print(f"[{i}/{len(configs)}] {hf_id} / {stem} build ...", end=" ", flush=True)
        build_status = run_build(
            hf_id,
            cfg_path,
            build_dir,
            ep_for_winml,
            args.timeout,
            rebuild=(args.rebuild or (args.retry_failed and had_failed_marker)),
        )
        print(build_status)
        if build_status != "PASS":
            results[build_status] += 1
            continue

        model_args = resolve_built_model_args(build_dir, task)
        if not model_args:
            eval_err.write_text(
                f"Could not resolve built ONNX artifacts for task={task!r} in {build_dir}",
                encoding="utf-8",
            )
            results["FAIL"] += 1
            print(f"[{i}/{len(configs)}] {hf_id} / {stem} eval ... FAIL")
            continue

        print(f"[{i}/{len(configs)}] {hf_id} / {stem} eval ...", end=" ", flush=True)
        status = run_eval(
            model_args,
            hf_id,
            cfg_path,
            eval_output,
            ep_for_winml,
            args.device,
            args.timeout,
            trust,
        )
        results[status] += 1
        print(status)

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
