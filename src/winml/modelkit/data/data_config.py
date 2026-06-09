# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configuration class for dataset loading and preprocessing."""


class DataConfig:
    """Simple configuration container for dataset loading and preprocessing."""

    def __init__(
        self,
        load_dataset_config=None,
        pre_process_data_config=None,
        model_input=None,
    ):
        """Initialize DataConfig.

        Args:
            load_dataset_config: Parameters for dataset loading
            pre_process_data_config: Parameters for preprocessing
            model_input: Path to model input file
        """
        self.load_dataset_config = load_dataset_config or {}
        self.pre_process_data_config = pre_process_data_config or {}
        self.model_input = model_input
