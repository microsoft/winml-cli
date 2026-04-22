# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pseudo-perplexity metric for MLMs (Salazar et al. 2020)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


class PseudoPerplexityMetric:
    r"""Accumulates per-token log P(w_i | w_\i); reports PPPL = exp(mean NLL)."""

    def __init__(self) -> None:
        self._sum_log_p = 0.0
        self._n = 0

    def update(self, token_log_probs: torch.Tensor) -> None:
        """Add a 1-D tensor of per-token log-probabilities to the aggregate."""
        if token_log_probs.numel() == 0:
            return
        self._sum_log_p += float(token_log_probs.sum().item())
        self._n += int(token_log_probs.numel())

    def compute(self) -> dict[str, Any]:
        """Return pseudo-perplexity and mean NLL over all accumulated tokens."""
        if self._n == 0:
            raise ValueError("No tokens accumulated; call update() first.")
        mean_nll = -self._sum_log_p / self._n
        return {
            "pseudo_perplexity": round(math.exp(mean_nll), 4),
            "nll": round(mean_nll, 4),
        }

    def reset(self) -> None:
        """Clear accumulated state."""
        self._sum_log_p = 0.0
        self._n = 0
