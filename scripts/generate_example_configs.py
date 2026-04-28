#!/usr/bin/env python3
"""Generate example configs for builtin models across EPs and precisions.

Usage:
    python scripts/generate_example_configs.py

Generates configs in examples/<ep>/<model_slug>/<task>_<precision>_config.json
for all builtin models × 3 EPs (amd, qnn, ov) × 3 precisions (w8a8, w8a16, fp16).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# EP configurations: (ep_name, folder_name, device, ep_flag)
EPS = [
    ("vitisai", "amd", "npu", "vitisai"),
    ("qnn", "qnn", "npu", "qnn"),
    ("openvino", "ov", "npu", "openvino"),
]

PRECISIONS = ["w8a8", "w8a16", "fp16"]

# 48 builtin models: (hf_id, task)
MODELS = [
    ("microsoft/swin-large-patch4-window7-224", "image-classification"),
    ("BAAI/bge-base-en-v1.5", "feature-extraction"),
    ("BAAI/bge-base-en-v1.5", "sentence-similarity"),
    ("BAAI/bge-small-en-v1.5", "feature-extraction"),
    ("BAAI/bge-small-en-v1.5", "sentence-similarity"),
    ("Babelscape/wikineural-multilingual-ner", "token-classification"),
    ("dslim/bert-base-NER", "token-classification"),
    ("facebook/dino-vits16", "image-feature-extraction"),
    ("FacebookAI/xlm-roberta-large", "fill-mask"),
    ("google-bert/bert-base-multilingual-cased", "feature-extraction"),
    ("Intel/bert-base-uncased-mrpc", "feature-extraction"),
    ("Intel/bert-base-uncased-mrpc", "text-classification"),
    ("microsoft/table-transformer-detection", "object-detection"),
    ("ProsusAI/finbert", "text-classification"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "feature-extraction"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sentence-similarity"),
    ("BAAI/bge-large-en-v1.5", "sentence-similarity"),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "text-classification"),
    ("dbmdz/bert-large-cased-finetuned-conll03-english", "token-classification"),
    ("deepset/bert-large-uncased-whole-word-masking-squad2", "question-answering"),
    ("deepset/roberta-base-squad2", "question-answering"),
    ("deepset/tinyroberta-squad2", "question-answering"),
    ("facebook/dino-vitb16", "image-feature-extraction"),
    ("facebook/dinov2-base", "image-feature-extraction"),
    ("facebook/dinov2-large", "image-feature-extraction"),
    ("facebook/dinov2-small", "image-feature-extraction"),
    ("FacebookAI/roberta-base", "fill-mask"),
    ("FacebookAI/roberta-large", "fill-mask"),
    ("FacebookAI/xlm-roberta-base", "fill-mask"),
    ("google-bert/bert-base-multilingual-uncased", "fill-mask"),
    ("google-bert/bert-base-uncased", "fill-mask"),
    ("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", "question-answering"),
    ("google/vit-base-patch16-224-in21k", "image-feature-extraction"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "feature-extraction"),
    ("mattmdjaga/segformer_b2_clothes", "image-segmentation"),
    ("microsoft/rad-dino", "image-feature-extraction"),
    ("microsoft/resnet-50", "image-classification"),
    ("nvidia/segformer-b1-finetuned-ade-512-512", "image-segmentation"),
    ("nvidia/segformer-b2-finetuned-ade-512-512", "image-segmentation"),
    ("nvidia/segformer-b5-finetuned-ade-640-640", "image-segmentation"),
    ("openai/clip-vit-base-patch16", "feature-extraction"),
    ("openai/clip-vit-base-patch32", "feature-extraction"),
    ("rizvandwiki/gender-classification", "image-classification"),
    ("sentence-transformers/all-MiniLM-L6-v2", "feature-extraction"),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-similarity"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-similarity"),
    ("StanfordAIMI/dinov2-base-xray-224", "image-feature-extraction"),
    ("w11wo/indonesian-roberta-base-posp-tagger", "token-classification"),
]


def load_eval_option_lookup() -> dict:
    """Load eval_option data from models_with_acc.json."""
    acc_path = REPO_ROOT / "scripts" / "e2e_eval" / "testsets" / "models_with_acc.json"
    if not acc_path.exists():
        return {}
    acc = json.loads(acc_path.read_text())
    lookup = {}
    for m in acc:
        key = (m["hf_id"], m["task"])
        if "dataset_config" in m:
            dc = m["dataset_config"]
            # Convert to eval_option schema
            eval_opt: dict = {"dataset": {}}
            ds = eval_opt["dataset"]
            if dc.get("path"):
                ds["path"] = dc["path"]
            if dc.get("name"):
                ds["name"] = dc["name"]
            if dc.get("split"):
                ds["split"] = dc["split"]
            if dc.get("samples"):
                ds["samples"] = dc["samples"]
            if dc.get("columns_mapping"):
                ds["columns_mapping"] = dc["columns_mapping"]
            if dc.get("label_mapping_file"):
                eval_opt["label_mapping_file"] = dc["label_mapping_file"]
            if dc.get("build_script"):
                eval_opt["dataset_script"] = dc["build_script"]
            lookup[key] = eval_opt
    return lookup


def generate_config(hf_id: str, task: str, device: str, ep: str, precision: str) -> dict | None:
    """Run winml config and return the parsed JSON config."""
    cmd = [
        sys.executable, "-m", "winml.modelkit", "config",
        "-m", hf_id,
        "--task", task,
        "--device", device,
        "--ep", ep,
        "--precision", precision,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            print(f"  FAIL: {result.stderr.strip()[-200:]}")
            return None
        # Extract JSON from stdout (may have non-JSON lines before it)
        stdout = result.stdout.strip()
        # Find the first '{' to start of JSON
        json_start = stdout.find("{")
        if json_start < 0:
            print(f"  FAIL: no JSON in output")
            return None
        return json.loads(stdout[json_start:])
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        return None
    except json.JSONDecodeError as e:
        print(f"  FAIL: JSON parse error: {e}")
        return None


def model_slug(hf_id: str) -> str:
    """Convert HF model ID to folder-safe slug."""
    return hf_id.replace("/", "_")


def main() -> None:
    eval_lookup = load_eval_option_lookup()
    examples_dir = REPO_ROOT / "examples"

    total = len(MODELS) * len(EPS) * len(PRECISIONS)
    done = 0
    created = 0
    skipped = 0
    failed = 0

    for hf_id, task in MODELS:
        slug = model_slug(hf_id)
        eval_opt = eval_lookup.get((hf_id, task))

        for ep_name, ep_folder, device, ep_flag in EPS:
            for precision in PRECISIONS:
                done += 1
                out_dir = examples_dir / ep_folder / slug
                out_file = out_dir / f"{task}_{precision}_config.json"

                if out_file.exists():
                    skipped += 1
                    continue

                print(f"[{done}/{total}] {hf_id} / {task} / {ep_folder} / {precision} ...", end=" ")
                config = generate_config(hf_id, task, device, ep_flag, precision)
                if config is None:
                    failed += 1
                    continue

                # Add eval_option if available
                if eval_opt:
                    config["eval_option"] = eval_opt

                out_dir.mkdir(parents=True, exist_ok=True)
                out_file.write_text(json.dumps(config, indent=2) + "\n")
                created += 1
                print("OK")

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
