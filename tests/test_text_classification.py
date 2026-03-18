"""
Comprehensive tests for TextDataset (renamed from TextClassificationDataset).

Tests cover:
1. Basic initialization and defaults
2. io_config priority (io_config > explicit > default)
3. io_mapping with rename_column()
4. Feature-based column detection
5. Universal tokenization pattern
6. Properties and readonly access
7. Sentence pair classification
8. Edge cases and error handling

Design principles validated:
- io_config as source of truth for ONNX shapes
- io_mapping for custom ONNX input names
- Explicit attributes over dict access
- Configuration priority: io_config > explicit > DEFAULT_SEQ_LEN
"""

from __future__ import annotations

import torch


class TestTextDatasetBasic:
    """Basic initialization and defaults."""

    def test_class_exists(self):
        """Test that the class exists and is importable."""
        from winml.modelkit.datasets.text import TextDataset

        assert TextDataset is not None

    def test_default_seq_len_constant(self):
        """Test DEFAULT_SEQ_LEN class constant exists."""
        from winml.modelkit.datasets.text import TextDataset

        assert hasattr(TextDataset, "DEFAULT_SEQ_LEN")
        assert TextDataset.DEFAULT_SEQ_LEN == 128

    def test_default_dataset_glue_mrpc(self):
        """Test default dataset is glue/mrpc when none specified."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert dataset.dataset_name == "glue"
        assert dataset.data_split == "train"

    def test_explicit_dataset_name(self):
        """Test explicit dataset name is used."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            dataset_name="glue",
            data_split="validation",
            max_samples=5,
            subset="sst2",
        )

        assert dataset.dataset_name == "glue"
        assert dataset.data_split == "validation"

    def test_max_samples_limits_dataset_size(self):
        """Test max_samples parameter limits dataset size."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=10,
        )

        assert len(dataset) == 10


class TestMaxLengthPriority:
    """Test max_length priority: io_config > explicit > DEFAULT_SEQ_LEN."""

    def test_default_max_length(self):
        """Test max_length defaults to DEFAULT_SEQ_LEN (128)."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert dataset.max_length == 128

    def test_explicit_max_length_overrides_default(self):
        """Test explicit max_length overrides default."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=256,
        )

        assert dataset.max_length == 256

    def test_io_config_overrides_explicit(self):
        """Test io_config max_length overrides explicit param."""
        from winml.modelkit.datasets.text import TextDataset

        io_config = {
            "input_ids": {"shape": [1, 64], "dtype": "int64"},
            "attention_mask": {"shape": [1, 64], "dtype": "int64"},
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=256,  # Should be overridden
            io_config=io_config,
        )

        # io_config shape[1]=64 should override explicit max_length=256
        assert dataset.max_length == 64

    def test_io_config_overrides_default(self):
        """Test io_config max_length overrides default."""
        from winml.modelkit.datasets.text import TextDataset

        io_config = {
            "input_ids": {"shape": [1, 512], "dtype": "int64"},
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_config=io_config,
        )

        assert dataset.max_length == 512

    def test_io_config_with_remapped_input_name(self):
        """Test io_config with remapped input name via io_mapping."""
        from winml.modelkit.datasets.text import TextDataset

        # Custom ONNX with different input name
        io_config = {
            "custom_input_ids": {"shape": [1, 96], "dtype": "int64"},
        }
        io_mapping = {
            "input_ids": "custom_input_ids",  # Map standard name to custom
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_config=io_config,
            io_mapping=io_mapping,
        )

        # Should extract max_length from custom_input_ids shape
        assert dataset.max_length == 96


class TestIoMapping:
    """Test io_mapping for renaming columns to match ONNX input names."""

    def test_no_io_mapping_keeps_standard_names(self):
        """Test standard HF names are preserved without io_mapping."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        sample = dataset[0]
        # Standard tokenizer output names
        assert "input_ids" in sample
        assert "attention_mask" in sample

    def test_io_mapping_renames_columns(self):
        """Test io_mapping renames columns to ONNX input names."""
        from winml.modelkit.datasets.text import TextDataset

        io_mapping = {
            "input_ids": "custom_ids",
            "attention_mask": "custom_mask",
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_mapping=io_mapping,
        )

        sample = dataset[0]
        # Columns should be renamed
        assert "custom_ids" in sample
        assert "custom_mask" in sample
        # Original names should not exist
        assert "input_ids" not in sample
        assert "attention_mask" not in sample

    def test_partial_io_mapping(self):
        """Test partial io_mapping only renames specified columns."""
        from winml.modelkit.datasets.text import TextDataset

        io_mapping = {
            "input_ids": "custom_ids",
            # attention_mask not mapped
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_mapping=io_mapping,
        )

        sample = dataset[0]
        assert "custom_ids" in sample
        assert "attention_mask" in sample  # Original name preserved


class TestColumnDetection:
    """Test feature-based column detection."""

    def test_detects_text_columns(self):
        """Test detection of text columns by Value dtype=string."""
        from winml.modelkit.datasets.text import TextDataset

        # GLUE/MRPC has sentence1 and sentence2 text columns
        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert len(dataset.text_columns) > 0
        assert all(isinstance(col, str) for col in dataset.text_columns)

    def test_detects_label_column(self):
        """Test detection of label column by ClassLabel type."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert dataset.label_col is not None
        assert isinstance(dataset.label_col, str)

    def test_sentence_pair_detection_mrpc(self):
        """Test sentence pair detection for MRPC (2 text columns)."""
        from winml.modelkit.datasets.text import TextDataset

        # GLUE/MRPC is a sentence pair task
        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert dataset.is_sentence_pair is True
        assert len(dataset.text_columns) == 2

    def test_single_sentence_detection_sst2(self):
        """Test single sentence detection for SST2 (1 text column)."""
        from winml.modelkit.datasets.text import TextDataset

        # GLUE/SST2 is a single sentence task
        dataset = TextDataset(
            model_name="bert-base-uncased",
            dataset_name="glue",
            data_split="train",
            max_samples=5,
            subset="sst2",
        )

        assert dataset.is_sentence_pair is False
        assert len(dataset.text_columns) == 1


class TestTokenization:
    """Test universal tokenization pattern."""

    def test_tokenized_output_has_correct_keys(self):
        """Test tokenized output contains expected keys."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        sample = dataset[0]

        # Standard BERT tokenizer outputs
        assert "input_ids" in sample
        assert "attention_mask" in sample

    def test_tokenized_tensors_have_correct_shape(self):
        """Test tokenized tensors have correct shape [batch, seq_len]."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=64,
        )

        sample = dataset[0]

        # Shape should be [1, max_length] for return_tensors="pt"
        assert sample["input_ids"].shape == torch.Size([1, 64])
        assert sample["attention_mask"].shape == torch.Size([1, 64])

    def test_tokenized_tensors_are_torch_tensors(self):
        """Test tokenized output is torch tensors."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        sample = dataset[0]

        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)

    def test_max_length_applied_to_tokenization(self):
        """Test max_length is applied during tokenization."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=32,
        )

        sample = dataset[0]

        # All sequences should be padded/truncated to max_length
        assert sample["input_ids"].shape[1] == 32


class TestReadonlyProperties:
    """Test readonly property access."""

    def test_max_length_property(self):
        """Test max_length property is accessible."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=64,
        )

        assert dataset.max_length == 64

    def test_label_col_property(self):
        """Test label_col property is accessible."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert dataset.label_col == "label"

    def test_label_names_property(self):
        """Test label_names property returns class names."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert isinstance(dataset.label_names, list)
        assert len(dataset.label_names) > 0  # MRPC has 2 classes

    def test_is_sentence_pair_property(self):
        """Test is_sentence_pair property."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert isinstance(dataset.is_sentence_pair, bool)

    def test_text_columns_property(self):
        """Test text_columns property returns column names."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert isinstance(dataset.text_columns, list)
        assert len(dataset.text_columns) > 0


class TestSamplingBehavior:
    """Test sampling and shuffling behavior."""

    def test_sequential_sampling_without_shuffle(self):
        """Test sequential sampling when shuffle=False."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=10,
            shuffle=False,
        )

        assert len(dataset) == 10

    def test_random_sampling_with_shuffle(self):
        """Test random sampling when shuffle=True."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=10,
            shuffle=True,
            seed=42,
        )

        assert len(dataset) == 10

    def test_seed_reproducibility(self):
        """Test same seed produces same samples."""
        from winml.modelkit.datasets.text import TextDataset

        dataset1 = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            shuffle=True,
            seed=42,
        )

        dataset2 = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            shuffle=True,
            seed=42,
        )

        # Same seed should produce same first sample
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        assert torch.equal(sample1["input_ids"], sample2["input_ids"])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_getitem_returns_dict(self):
        """Test __getitem__ returns dict."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_len_returns_int(self):
        """Test __len__ returns int."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
        )

        assert isinstance(len(dataset), int)
        assert len(dataset) == 5

    def test_io_config_missing_input_ids_uses_default(self):
        """Test io_config without input_ids falls back to explicit/default."""
        from winml.modelkit.datasets.text import TextDataset

        # io_config without input_ids
        io_config = {
            "some_other_input": {"shape": [1, 256], "dtype": "int64"},
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            max_length=64,  # Should be used since io_config has no input_ids
            io_config=io_config,
        )

        # Should use explicit max_length since io_config has no input_ids
        assert dataset.max_length == 64

    def test_empty_io_mapping(self):
        """Test empty io_mapping dict."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_mapping={},
        )

        sample = dataset[0]
        # Standard names should be preserved
        assert "input_ids" in sample


class TestIntegration:
    """Integration tests with real data."""

    def test_full_pipeline_glue_mrpc(self):
        """Test full pipeline with GLUE/MRPC."""
        from winml.modelkit.datasets.text import TextDataset

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=10,
        )

        # Check dataset properties
        assert len(dataset) == 10
        assert dataset.max_length == 128
        assert dataset.is_sentence_pair is True

        # Check sample structure
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "label" in sample

        # Check tensor shapes
        assert sample["input_ids"].shape == torch.Size([1, 128])

    def test_full_pipeline_with_io_config(self):
        """Test full pipeline with io_config override."""
        from winml.modelkit.datasets.text import TextDataset

        io_config = {
            "input_ids": {"shape": [1, 64], "dtype": "int64"},
            "attention_mask": {"shape": [1, 64], "dtype": "int64"},
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_config=io_config,
        )

        assert dataset.max_length == 64

        sample = dataset[0]
        assert sample["input_ids"].shape == torch.Size([1, 64])

    def test_full_pipeline_with_io_mapping(self):
        """Test full pipeline with io_mapping renaming."""
        from winml.modelkit.datasets.text import TextDataset

        io_mapping = {
            "input_ids": "ids",
            "attention_mask": "mask",
        }

        dataset = TextDataset(
            model_name="bert-base-uncased",
            max_samples=5,
            io_mapping=io_mapping,
        )

        sample = dataset[0]
        assert "ids" in sample
        assert "mask" in sample
        assert "input_ids" not in sample

    def test_calibration_reader_compatibility(self):
        """Test dataset works with DatasetCalibrationReader."""
        from winml.modelkit.datasets import DatasetCalibrationReader

        reader = DatasetCalibrationReader(
            model_name="bert-base-uncased",
            task="text-classification",
            max_samples=5,
        )

        # Get first sample
        sample = reader.get_next()
        assert sample is not None
        assert "input_ids" in sample
        assert "attention_mask" in sample

        # Should be numpy arrays (not torch tensors)
        import numpy as np
        assert isinstance(sample["input_ids"], np.ndarray)

        # Label should be excluded
        assert "label" not in sample
