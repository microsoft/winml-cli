#!/usr/bin/env python3
"""Generate example configs for builtin models across EPs and precisions.

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

PRECISIONS = ["w8a8", "w8a16", "fp16"]

# Builtin models: (hf_id, task)
MODELS = [
    ("facebook/dino-vits16", "image-feature-extraction"),
    ("Salesforce/blip-image-captioning-base", "image-to-text"),
    ("StanfordAIMI/dinov2-base-xray-224", "image-feature-extraction"),
    ("BAAI/bge-base-en-v1.5", "feature-extraction"),
    ("BAAI/bge-base-en-v1.5", "sentence-similarity"),
    ("BAAI/bge-small-en-v1.5", "feature-extraction"),
    ("BAAI/bge-small-en-v1.5", "sentence-similarity"),
    ("Babelscape/wikineural-multilingual-ner", "token-classification"),
    ("deepset/bert-large-uncased-whole-word-masking-squad2", "question-answering"),
    ("dslim/bert-base-NER", "token-classification"),
    ("facebook/convnext-tiny-224", "image-classification"),
    ("facebook/dinov2-large", "image-feature-extraction"),
    ("FacebookAI/xlm-roberta-large", "fill-mask"),
    ("google-bert/bert-base-multilingual-cased", "feature-extraction"),
    ("Intel/bert-base-uncased-mrpc", "feature-extraction"),
    ("Intel/bert-base-uncased-mrpc", "text-classification"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "zero-shot-image-classification"),
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "zero-shot-image-classification"),
    ("microsoft/table-transformer-detection", "object-detection"),
    ("openai/clip-vit-base-patch16", "zero-shot-image-classification"),
    ("openai/clip-vit-base-patch32", "zero-shot-image-classification"),
    ("patrickjohncyh/fashion-clip", "zero-shot-image-classification"),
    ("ProsusAI/finbert", "text-classification"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "feature-extraction"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sentence-similarity"),
    ("BAAI/bge-large-en-v1.5", "sentence-similarity"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "text-classification"),
    ("dbmdz/bert-large-cased-finetuned-conll03-english", "token-classification"),
    ("deepset/roberta-base-squad2", "question-answering"),
    ("deepset/tinyroberta-squad2", "question-answering"),
    ("facebook/dino-vitb16", "image-feature-extraction"),
    ("facebook/dinov2-base", "image-feature-extraction"),
    ("facebook/dinov2-small", "image-feature-extraction"),
    ("FacebookAI/roberta-base", "fill-mask"),
    ("FacebookAI/roberta-large", "fill-mask"),
    ("FacebookAI/xlm-roberta-base", "fill-mask"),
    ("google-bert/bert-base-multilingual-uncased", "fill-mask"),
    ("google-bert/bert-base-uncased", "fill-mask"),
    ("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", "question-answering"),
    ("google/vit-base-patch16-224", "image-classification"),
    ("google/vit-base-patch16-224-in21k", "image-feature-extraction"),
    ("joeddav/xlm-roberta-large-xnli", "zero-shot-classification"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "feature-extraction"),
    ("mattmdjaga/segformer_b2_clothes", "image-segmentation"),
    ("microsoft/rad-dino", "image-feature-extraction"),
    ("microsoft/resnet-50", "image-classification"),
    ("microsoft/swin-large-patch4-window7-224", "image-classification"),
    ("microsoft/trocr-base-handwritten", "image-to-text"),
    ("microsoft/trocr-base-printed", "image-to-text"),
    ("microsoft/trocr-large-handwritten", "image-to-text"),
    ("microsoft/trocr-large-printed", "image-to-text"),
    ("nvidia/segformer-b1-finetuned-ade-512-512", "image-segmentation"),
    ("nvidia/segformer-b2-finetuned-ade-512-512", "image-segmentation"),
    ("nvidia/segformer-b5-finetuned-ade-640-640", "image-segmentation"),
    ("openai/clip-vit-base-patch16", "feature-extraction"),
    ("openai/clip-vit-base-patch32", "feature-extraction"),
    ("openai/clip-vit-large-patch14", "zero-shot-image-classification"),
    ("openai/clip-vit-large-patch14-336", "zero-shot-image-classification"),
    ("rizvandwiki/gender-classification", "image-classification"),
    ("sentence-transformers/all-MiniLM-L6-v2", "feature-extraction"),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-similarity"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-similarity"),
    ("w11wo/indonesian-roberta-base-posp-tagger", "token-classification"),
]


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
    device: str,
    dataset: dict | None,
) -> None:
    """Write/merge the new eval section into a build config."""
    eval_cfg = config.get("eval") if isinstance(config.get("eval"), dict) else {}
    eval_cfg["task"] = eval_cfg.get("task") or task
    eval_cfg["device"] = eval_cfg.get("device") or device
    if dataset:
        ds = eval_cfg.get("dataset") if isinstance(eval_cfg.get("dataset"), dict) else {}
        for key, value in dataset.items():
            if key not in ds:
                ds[key] = value
        eval_cfg["dataset"] = ds
    config["eval"] = eval_cfg


def generate_config(hf_id: str, task: str, device: str, ep: str, precision: str) -> dict | None:
    """Run winml config and return the parsed JSON config.

    For composite models (e.g., CLIP with image-encoder + text-encoder),
    returns a dict with 'components' list containing all sub-configs.
    """
    cmd = [
        sys.executable, "-m", "winml.modelkit", "config",
        "-m", hf_id,
        "--task", task,
        "--device", device,
        "--ep", ep,
        "--precision", precision,
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            print(f"  FAIL: {result.stderr.strip()[-200:]}")
            return None
        # Extract JSON from stdout (may have non-JSON lines before it)
        stdout = result.stdout.strip()

        # Parse multiple JSON objects (for composite models like CLIP)
        configs = []
        pos = 0
        while True:
            json_start = stdout.find("{", pos)
            if json_start < 0:
                break
            # Find matching closing brace
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(stdout)):
                if stdout[i] == "{":
                    brace_count += 1
                elif stdout[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            try:
                config = json.loads(stdout[json_start:json_end])
                configs.append(config)
                pos = json_end
            except json.JSONDecodeError:
                break

        if not configs:
            print("  FAIL: no JSON in output")
            return None

        # If single config, return as-is; if multiple, wrap in components
        if len(configs) == 1:
            return configs[0]
        return {"components": configs}
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    except json.JSONDecodeError as e:
        print(f"  FAIL: JSON parse error: {e}")
        return None


def model_slug(hf_id: str) -> str:
    """Convert HF model ID to folder-safe slug."""
    return hf_id.replace("/", "_")


def main() -> None:
    """Entrypoint for config generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate example configs")
    parser.add_argument("--ep", help="Filter by EP folder name (e.g. qnn, openvino)")
    parser.add_argument("--hardware", help="Filter by hardware (e.g. npu, gpu, cpu)")
    args = parser.parse_args()

    eps = EPS
    if args.ep or args.hardware:
        eps = [
            (ef, folder, hw)
            for ef, folder, hw in EPS
            if (not args.ep or folder == args.ep) and (not args.hardware or hw == args.hardware)
        ]
        if not eps:
            print(f"No matching EP config for --ep={args.ep} --hardware={args.hardware}")
            sys.exit(1)

    eval_lookup = load_eval_lookup()
    examples_dir = REPO_ROOT / "examples"

    total = len(MODELS) * len(eps) * len(PRECISIONS)
    done = 0
    created = 0
    skipped = 0
    failed = 0

    for hf_id, task in MODELS:
        slug = model_slug(hf_id)
        eval_dataset = eval_lookup.get((hf_id, task))

        for ep_flag, ep_folder, hardware in eps:
            for precision in PRECISIONS:
                done += 1
                out_dir = examples_dir / ep_folder / hardware / slug
                out_file = out_dir / f"{task}_{precision}_config.json"

                if out_file.exists():
                    skipped += 1
                    continue

                print(f"[{done}/{total}] {hf_id} / {task} / {ep_folder} / {precision} ...", end=" ")
                config = generate_config(hf_id, task, hardware, ep_flag, precision)
                if config is None:
                    failed += 1
                    continue

                merge_eval_section(config, task=task, device=hardware, dataset=eval_dataset)

                out_dir.mkdir(parents=True, exist_ok=True)
                out_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                created += 1
                print("OK")

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
