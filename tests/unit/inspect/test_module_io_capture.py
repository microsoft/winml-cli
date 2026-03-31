# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for multi-input module I/O capture."""

from __future__ import annotations

import torch
import torch.nn as nn

from winml.modelkit.inspect.module_io_capture import _extract_tensors, capture_module_io


class MultiInputModule(nn.Module):
    """Test module that takes multiple inputs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(hidden_states) * attention_mask.unsqueeze(-1)


class WrapperModel(nn.Module):
    """Test model containing a multi-input submodule."""

    def __init__(self):
        super().__init__()
        self.sub = MultiInputModule()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.sub(hidden_states, attention_mask=attention_mask)


class TestCaptureModuleIO:
    def test_captures_multiple_inputs(self) -> None:
        model = WrapperModel()
        inputs = {
            "hidden_states": torch.randn(1, 10, 64),
            "attention_mask": torch.ones(1, 10),
        }
        result = capture_module_io(model, inputs, target_class="MultiInputModule")
        assert "sub" in result
        info = result["sub"]
        assert len(info.input_shapes) == 2
        assert info.input_shapes[0] == [1, 10, 64]  # hidden_states
        assert info.input_shapes[1] == [1, 10]  # attention_mask
        assert "hidden_states" in info.input_names
        assert "attention_mask" in info.input_names

    def test_captures_output(self) -> None:
        model = WrapperModel()
        inputs = {
            "hidden_states": torch.randn(1, 10, 64),
            "attention_mask": torch.ones(1, 10),
        }
        result = capture_module_io(model, inputs, target_class="MultiInputModule")
        info = result["sub"]
        assert len(info.output_shapes) >= 1
        assert info.output_shapes[0] == [1, 10, 32]

    def test_captures_dtypes(self) -> None:
        model = WrapperModel()
        inputs = {
            "hidden_states": torch.randn(1, 10, 64),
            "attention_mask": torch.ones(1, 10, dtype=torch.int64),
        }
        result = capture_module_io(model, inputs, target_class="MultiInputModule")
        info = result["sub"]
        assert "float32" in info.input_dtypes
        assert "int64" in info.input_dtypes

    def test_no_target_captures_all(self) -> None:
        model = WrapperModel()
        inputs = {
            "hidden_states": torch.randn(1, 10, 64),
            "attention_mask": torch.ones(1, 10),
        }
        result = capture_module_io(model, inputs)
        # Should capture both 'sub' and 'sub.linear'
        assert len(result) >= 2


class TestExtractTensors:
    def test_single_tensor(self) -> None:
        t = torch.randn(2, 3)
        result = _extract_tensors(t)
        assert len(result) == 1
        assert result[0][0] == "tensor"

    def test_dict_output(self) -> None:
        output = {"last_hidden_state": torch.randn(1, 10, 64), "pooler_output": torch.randn(1, 64)}
        result = _extract_tensors(output)
        assert len(result) == 2
        names = [r[0] for r in result]
        assert "last_hidden_state" in names
        assert "pooler_output" in names

    def test_tuple_output(self) -> None:
        output = (torch.randn(1, 10, 64), torch.randn(1, 64))
        result = _extract_tensors(output)
        assert len(result) == 2

    def test_nested_tuple(self) -> None:
        output = (torch.randn(1, 10, 64), (torch.randn(1, 5), torch.randn(1, 3)))
        result = _extract_tensors(output)
        assert len(result) == 3

    def test_none_in_dict(self) -> None:
        output = {"hidden": torch.randn(1, 64), "cache": None}
        result = _extract_tensors(output)
        assert len(result) == 1
        assert result[0][0] == "hidden"

    def test_empty_container(self) -> None:
        result = _extract_tensors(())
        assert len(result) == 0
        result = _extract_tensors({})
        assert len(result) == 0


class TestCaptureEdgeCases:
    def test_hooks_cleaned_on_forward_failure(self) -> None:
        """Hooks are removed even if forward pass raises."""

        class FailModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = nn.Linear(10, 5)

            def forward(self, x):
                raise RuntimeError("intentional failure")

        model = FailModel()
        inputs = (torch.randn(1, 10),)
        # Should not raise (captured dict may be empty, but hooks cleaned up)
        try:
            capture_module_io(model, inputs)
        except RuntimeError:
            pass  # Expected
        # Verify no hooks remain
        assert len(model.sub._forward_hooks) == 0

    def test_target_class_no_match(self) -> None:
        """Returns empty dict when target_class matches nothing."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        inputs = (torch.randn(1, 10),)
        result = capture_module_io(model, inputs, target_class="NonExistent")
        assert result == {}
