# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image classification dataset for testing quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import load_dataset
from torchvision import transforms

from .registry import DataRegistry


if TYPE_CHECKING:
    from .data_config import DataConfig


@DataRegistry.register_dataset()
class ImageClassificationDataset:
    """Image classification dataset using Hugging Face datasets.

    Loads mini-imagenet dataset and preprocesses images for model input.
    """

    def __init__(self, config: DataConfig | None = None):
        """Initialize dataset.

        Args:
            config: Optional DataConfig with dataset_name, split, stream, and size settings.
                    Defaults to "timm/mini-imagenet", split="train", stream=True, size=256
        """
        load_dataset_config = config.load_dataset_config if config else {}

        dataset_name = load_dataset_config.get("dataset_name", "timm/mini-imagenet")
        split = load_dataset_config.get("split", "train")
        stream = load_dataset_config.get("stream", True)
        size = load_dataset_config.get("size", 256)

        self.dataset = load_dataset(dataset_name, split=split, streaming=stream)

        # TODO: Image preprocessing is temporarily hardcoded;
        # will integrate with Hugging Face data processor
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Cache preprocessed samples (limited by size)
        self.images = []
        self.labels = []
        for i, sample in enumerate(self.dataset):
            if i >= size:
                break
            img = sample["image"]
            # Convert grayscale to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0)
            self.images.append(tensor.numpy())
            self.labels.append(0)  # Placeholder label

    def __len__(self):
        """Return dataset length."""
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx):
        """Get item by index.

        Args:
            idx: Index of the sample

        Returns:
            Dict with pixel_values key containing numpy array
        """
        return {"pixel_values": self.images[idx]}
