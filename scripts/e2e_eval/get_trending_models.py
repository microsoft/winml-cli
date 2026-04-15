# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Fetch trending HuggingFace models (library=transformers).

Excludes models already in testsets/models_all.json and those with excluded pipeline tags.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


EXCLUDED_PIPELINE_TAGS = {
    "text-generation",
    "image-text-to-text",
    "any-to-any",
    "video-text-to-text",
}

MODELS_ALL_PATH = Path(__file__).parent / "testsets" / "models_all.json"
FETCH_LIMIT = 100


def load_existing_model_ids() -> set[str]:
    with MODELS_ALL_PATH.open() as f:
        models = json.load(f)
    return {m["hf_id"] for m in models}


def fetch_trending_models(limit: int) -> list:
    api = HfApi()
    return list(
        api.list_models(
            filter="transformers",
            sort="trending_score",
            direction=-1,
            limit=limit,
            expand=["pipeline_tag", "downloads", "likes", "lastModified", "trendingScore"],
        )
    )


def main() -> None:
    existing_ids = load_existing_model_ids()
    print(f"Loaded {len(existing_ids)} existing models from models_all.json")

    print(f"Fetching top {FETCH_LIMIT} trending transformers models from HuggingFace...")
    models = fetch_trending_models(FETCH_LIMIT)
    print(f"Fetched {len(models)} models")

    results = []
    for model in models:
        model_id = model.id
        pipeline_tag = model.pipeline_tag or ""

        if pipeline_tag in EXCLUDED_PIPELINE_TAGS:
            continue
        if model_id in existing_ids:
            continue

        results.append(
            {
                "hf_id": model_id,
                "pipeline_tag": pipeline_tag,
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "trending_score": getattr(model, "trending_score", None),
                "last_modified": str(model.last_modified) if model.last_modified else "",
            }
        )

    print(
        f"\nFound {len(results)} new trending models "
        f"(excluded tags + already in models_all.json filtered out)\n"
    )
    print(f"{'#':<4} {'hf_id':<60} {'pipeline_tag':<35} {'downloads':>10} {'likes':>6}")
    print("-" * 120)
    for i, m in enumerate(results, 1):
        print(
            f"{i:<4} {m['hf_id']:<60} {m['pipeline_tag']:<35} {m['downloads']:>10,} {m['likes']:>6,}"
        )

    output_path = Path(__file__).parent / "trending_models.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
