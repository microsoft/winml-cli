#!/usr/bin/env python3
"""Generate example configs for the canonical 57 models across EPs and precisions.

Usage:
    python scripts/generate_example_configs.py

Generates configs in examples/<ep>/<hardware>/<model_slug>/<task>_<precision>_config.json
with the current eval schema.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_57 = REPO_ROOT / "scripts" / "e2e_eval" / "testsets" / "models_57.txt"

# EP configurations: (ep_flag, ep_folder, hardware)
# EP names/aliases are validated by `winml config --help`.
EPS = [
    ("qnn", "qnn", "npu"),
    ("qnn", "qnn", "gpu"),
    ("openvino", "openvino", "npu"),
    ("openvino", "openvino", "cpu"),
    ("openvino", "openvino", "gpu"),
    ("vitisai", "vitisai", "npu"),
    ("nv_tensorrt_rtx", "nv_tensorrt_rtx", "gpu"),
    ("cpu", "mlas", "cpu"),
    ("dml", "dml", "gpu"),
]

# NPU targets are tested across multiple quantization precisions.
# CPU/GPU targets generate a single config without an explicit --precision
# (uses the EP's default), so we don't sweep precisions there.
NPU_PRECISIONS = ["w8a8", "w8a16", "fp16"]
CPU_GPU_PRECISIONS: list[str | None] = [None]

def load_model_pairs() -> list[tuple[str, str]]:
    """Load the canonical 57 (hf_id, task) tuples from the shared list."""
    out: list[tuple[str, str]] = []
    for line in MODELS_57.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hf_id, task = s.split("|", 1)
        out.append((hf_id, task))
    return out


def load_eval_lookup() -> dict:
    """Load eval dataset data from models_with_acc.json."""
    acc_path = REPO_ROOT / "scripts" / "e2e_eval" / "testsets" / "models_with_acc.json"
    if not acc_path.exists():
        return {}
    acc = json.loads(acc_path.read_text(encoding="utf-8"))
    lookup = {}
    for m in acc:
        key = (m["hf_id"], m["task"])
        if "dataset_config" in m:
            dc = m["dataset_config"]
            dataset: dict = {}
            if dc.get("path"):
                dataset["path"] = dc["path"]
            if dc.get("name"):
                dataset["name"] = dc["name"]
            if dc.get("split"):
                dataset["split"] = dc["split"]
            if dc.get("samples"):
                dataset["samples"] = dc["samples"]
            if dc.get("shuffle") is not None:
                dataset["shuffle"] = dc["shuffle"]
            if dc.get("columns_mapping"):
                dataset["columns_mapping"] = dc["columns_mapping"]
            if dc.get("label_mapping_file"):
                dataset["label_mapping_file"] = dc["label_mapping_file"]
            if dc.get("build_script"):
                dataset["build_script"] = dc["build_script"]
            lookup[key] = dataset
    return lookup


def merge_eval_section(
    config: dict,
    *,
    task: str,
    dataset: dict | None,
) -> None:
    """Write/merge the new eval section into a build config."""
    eval_cfg = config.get("eval") if isinstance(config.get("eval"), dict) else {}
    eval_cfg["task"] = eval_cfg.get("task") or task
    # Keep eval config device-agnostic; runtime device is chosen at test-time.
    eval_cfg.pop("device", None)
    if dataset:
        ds = eval_cfg.get("dataset") if isinstance(eval_cfg.get("dataset"), dict) else {}
        for key, value in dataset.items():
            if key not in ds:
                ds[key] = value
        eval_cfg["dataset"] = ds
    config["eval"] = eval_cfg


def generate_config(
    hf_id: str, task: str, device: str, ep: str, precision: str | None,
    out_file: Path,
) -> list[Path] | None:
    """Run `winml config -o <out_file>` and return the list of files it wrote.

    - Single-model: returns ``[out_file]``.
    - Composite model (e.g., CLIP zero-shot): `winml config` writes one file per
      sub-component using its own naming (``<stem>_<role>.json``); we return
      those paths. We do NOT merge them into a wrapper, because
      ``winml build -c`` does not understand a ``{"components": [...]}`` shape.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot pre-existing sibling split files so we can identify newly-written ones.
    stem = out_file.stem
    pre_existing = set(out_file.parent.glob(f"{stem}_*.json"))

    cmd = [
        sys.executable, "-m", "winml.modelkit", "config",
        "-m", hf_id,
        "--task", task,
        "--device", device,
        "--ep", ep,
        "-o", str(out_file),
    ]
    if precision is not None:
        cmd.extend(["--precision", precision])
    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            print(f"  FAIL: {result.stderr.strip()[-200:]}")
            return None
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None

    # Single-model: the exact file we asked for exists.
    if out_file.exists():
        return [out_file]

    # Composite: collect newly-written split files matching `<stem>_*.json`.
    split_files = sorted(
        p for p in out_file.parent.glob(f"{stem}_*.json") if p not in pre_existing
    )
    if not split_files:
        print("  FAIL: no config output")
        return None
    return split_files


def model_slug(hf_id: str) -> str:
    """Convert HF model ID to folder-safe slug."""
    return hf_id.replace("/", "_")


def main() -> None:
    """Entrypoint for config generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate example configs")
    parser.add_argument("--ep", help="Filter by EP folder name (e.g. qnn, openvino)")
    parser.add_argument("--device", help="Filter by device (e.g. npu, gpu, cpu)")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated HF model IDs (e.g. 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K') "
             "to restrict generation to those models",
    )
    args = parser.parse_args()

    eps = EPS
    if args.ep or args.device:
        eps = [
            (ef, folder, hw)
            for ef, folder, hw in EPS
            if (not args.ep or folder == args.ep) and (not args.device or hw == args.device)
        ]
        if not eps:
            print(f"No matching EP config for --ep={args.ep} --device={args.device}")
            sys.exit(1)

    eval_lookup = load_eval_lookup()
    models = load_model_pairs()
    if args.models:
        allowed = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [(hf, task) for hf, task in models if hf in allowed]
        if not models:
            print(f"No models match --models={args.models}")
            sys.exit(1)
    examples_dir = REPO_ROOT / "examples"

    # Pre-compute precision list per EP target (NPU sweeps, CPU/GPU single config).
    ep_precisions = [
        (ep_flag, ep_folder, hw, NPU_PRECISIONS if hw == "npu" else CPU_GPU_PRECISIONS)
        for ep_flag, ep_folder, hw in eps
    ]
    total = sum(len(models) * len(precs) for *_, precs in ep_precisions)
    done = 0
    created = 0
    skipped = 0
    failed = 0

    for hf_id, task in models:
        slug = model_slug(hf_id)
        eval_dataset = eval_lookup.get((hf_id, task))

        for ep_flag, ep_folder, hardware, precisions in ep_precisions:
            for precision in precisions:
                done += 1
                out_dir = examples_dir / ep_folder / hardware / slug
                if precision is None:
                    out_file = out_dir / f"{task}_config.json"
                    label = "default"
                else:
                    out_file = out_dir / f"{task}_{precision}_config.json"
                    label = precision

                # Skip if either the single-model file or any composite split
                # file (``<stem>_*.json``) is already present.
                if out_file.exists() or any(out_dir.glob(f"{out_file.stem}_*.json")):
                    skipped += 1
                    continue

                print(f"[{done}/{total}] {hf_id} / {task} / {ep_folder} / {label} ...", end=" ")
                written = generate_config(
                    hf_id, task, hardware, ep_flag, precision, out_file
                )
                if written is None:
                    failed += 1
                    continue

                # Merge eval section into each emitted file (single or split).
                for path in written:
                    cfg = json.loads(path.read_text(encoding="utf-8"))
                    merge_eval_section(cfg, task=task, dataset=eval_dataset)
                    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                created += 1
                suffix = f" ({len(written)} files)" if len(written) > 1 else ""
                print(f"OK{suffix}")

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
