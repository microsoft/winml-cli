# inference/lora_loader.py
# Verbatim Appendix A loader from docs/lora_qnn_investigation.md.
# Phase Q3 validates this code unmodified.
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

# PEFT files store:
#   base_model.model.<orig_layer_path>.lora_A.weight  : [r, in]
#   base_model.model.<orig_layer_path>.lora_B.weight  : [out, r]
_PEFT_KEY_RE = re.compile(
    r"^base_model\.model\.(?P<layer>.+)\.lora_(?P<which>[AB])\.weight$"
)


def load_adapter(
    adapter_path: str | Path,
    onnx_input_names: list[str],
    *,
    alpha: float | None = None,        # if None, read adapter_config.json
    rank: int | None = None,
    fold_scaling_into: str = "B",      # "A" or "B"
) -> dict[str, np.ndarray]:
    """
    Build a feeds dict mapping ONNX LoRA input names -> numpy arrays.

    The ONNX surgery in this repo names inputs as:
        "<base_weight_name>_lora_A"
        "<base_weight_name>_lora_B"
    where <base_weight_name> typically ends in something like
        "model.layers.0.self_attn.q_proj.weight"
    which matches the PEFT layer path verbatim (modulo the
    "base_model.model." prefix and the ".lora_X.weight" suffix).
    """
    adapter_path = Path(adapter_path)
    raw = load_file(str(adapter_path))

    # 1. Resolve scaling
    if alpha is None or rank is None:
        cfg = json.loads((adapter_path.parent / "adapter_config.json").read_text())
        alpha = alpha if alpha is not None else cfg["lora_alpha"]
        rank = rank if rank is not None else cfg["r"]
    scaling = float(alpha) / float(rank)

    # 2. Group PEFT entries by layer
    peft_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for k, v in raw.items():
        m = _PEFT_KEY_RE.match(k)
        if not m:
            continue   # ignore optimizer state, embeddings tweaks, etc.
        peft_by_layer.setdefault(m["layer"], {})[m["which"]] = v

    # 3. Build the feeds dict by matching ONNX input names
    feeds: dict[str, np.ndarray] = {}
    for inp in onnx_input_names:
        if not (inp.endswith("_lora_A") or inp.endswith("_lora_B")):
            continue
        which = inp[-1]                          # 'A' or 'B'
        base = inp[: -len("_lora_X")]            # strip "_lora_A"/"_lora_B"
        layer = base[: -len(".weight")] if base.endswith(".weight") else base

        peft_entry = peft_by_layer.get(layer)
        if peft_entry is None or which not in peft_entry:
            # Layer not adapted — feed zeros so this layer behaves as base.
            # (Shape must be inferred from the ONNX input metadata; left as
            #  an exercise — typically use sess.get_inputs() shape.)
            continue

        w = peft_entry[which].astype(np.float32, copy=False)
        # PEFT:  A:[r,in], B:[out,r].
        # ONNX surgery expects A:[in,r], B:[r,out]  (plain MatMul order).
        w = w.T
        if which == fold_scaling_into:
            w = w * scaling
        feeds[inp] = np.ascontiguousarray(w)

    return feeds
