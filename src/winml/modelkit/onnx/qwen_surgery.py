# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Ad-hoc ONNX surgery to turn a Qwen3 decoder ONNX into a transformer-only graph.

Applied as a post-export surgery on the fused decoder ONNX produced by
``WinMLQwen3Model`` (``decoder_prefill.onnx`` / ``decoder_gen.onnx``).

The resulting transformer-only ONNX has:
  - ``input_ids`` graph input replaced by ``inputs_embeds`` (FLOAT,
    ``[batch, seq, hidden_size]``)  — the upstream embedding Gather is
    removed.
  - ``logits`` graph output replaced by ``output_hidden_states``
    (FLOAT, ``[batch, seq, hidden_size]``) — the final ``lm_head`` MatMul
    is removed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from .persistence import load_onnx, save_onnx


logger = logging.getLogger(__name__)


def _dim(d: onnx.TensorShapeProto.Dimension) -> int | str:
    if d.HasField("dim_value"):
        return d.dim_value
    return d.dim_param or "?"


def make_transformer_only(
    model_path: str | Path,
    output_path: str | Path,
    *,
    input_ids_name: str = "input_ids",
    logits_name: str = "logits",
    inputs_embeds_name: str = "inputs_embeds",
    output_hidden_states_name: str = "output_hidden_states",
) -> Path:
    """Strip the embedding Gather and the lm_head MatMul from a Qwen3 ONNX.

    Args:
        model_path: Path to the fused decoder ONNX (logits output, input_ids input).
        output_path: Destination for the transformer-only ONNX.
        input_ids_name: Name of the input_ids graph input to drop.
        logits_name: Name of the logits graph output to drop.
        inputs_embeds_name: Display name for the new embeddings input
            (used only for logging; the actual tensor keeps its existing
            internal name so downstream nodes need no rewiring).
        output_hidden_states_name: Display name for the new hidden-state output.

    Returns:
        The output path.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_onnx(model_path, load_weights=True, validate=False)
    graph = model.graph
    init_by_name = {init.name: init for init in graph.initializer}

    # -------------------- Embedding removal --------------------
    embed_idx = next(
        (i for i, n in enumerate(graph.node) if input_ids_name in n.input),
        None,
    )
    if embed_idx is None:
        msg = f"No node consumes graph input {input_ids_name!r}"
        raise RuntimeError(msg)

    embed_node = graph.node[embed_idx]
    embed_out_name = embed_node.output[0]

    embed_weight = None
    for ipt in embed_node.input:
        init = init_by_name.get(ipt)
        if init is not None and len(init.dims) == 2:
            embed_weight = init
            break
    if embed_weight is None:
        msg = f"Could not find 2-D embedding weight initializer on node {embed_node.name!r}"
        raise RuntimeError(msg)
    hidden_size = int(embed_weight.dims[1])

    ids_input = next(i for i in graph.input if i.name == input_ids_name)
    batch_dim = _dim(ids_input.type.tensor_type.shape.dim[0])
    seq_dim = _dim(ids_input.type.tensor_type.shape.dim[1])

    logger.info(
        "Removing embedding node %r (%s) — exposing %r as new input %r [%s, %s, %d]",
        embed_node.name,
        embed_node.op_type,
        embed_out_name,
        inputs_embeds_name,
        batch_dim,
        seq_dim,
        hidden_size,
    )

    new_embed_input = helper.make_tensor_value_info(
        inputs_embeds_name,
        TensorProto.FLOAT,
        [batch_dim, seq_dim, hidden_size],
    )

    del graph.node[embed_idx]
    graph.input.remove(ids_input)
    graph.input.append(new_embed_input)
    graph.initializer.remove(embed_weight)

    # Rewire any consumer of the removed embedding output to the new input.
    for n in graph.node:
        for i, name in enumerate(n.input):
            if name == embed_out_name:
                n.input[i] = inputs_embeds_name

    # -------------------- lm_head removal --------------------
    lmh_idx = next(
        (i for i, n in enumerate(graph.node) if logits_name in n.output),
        None,
    )
    if lmh_idx is None:
        msg = f"No node produces graph output {logits_name!r}"
        raise RuntimeError(msg)

    lmh_node = graph.node[lmh_idx]
    init_names = {init.name for init in graph.initializer}
    hidden_in: str | None = None
    weight_in: str | None = None
    for ipt in lmh_node.input:
        if ipt in init_names:
            weight_in = ipt
        else:
            hidden_in = ipt
    if hidden_in is None:
        msg = f"lm_head node {lmh_node.name!r} has no non-initializer input ({list(lmh_node.input)})"
        raise RuntimeError(msg)

    logger.info(
        "Removing lm_head node %r (%s) — exposing %r as new output %r",
        lmh_node.name,
        lmh_node.op_type,
        hidden_in,
        output_hidden_states_name,
    )

    logits_output = next(o for o in graph.output if o.name == logits_name)
    new_hidden_output = helper.make_tensor_value_info(
        output_hidden_states_name,
        TensorProto.FLOAT,
        [batch_dim, seq_dim, hidden_size],
    )

    del graph.node[lmh_idx]
    graph.output.remove(logits_output)
    # Put hidden states first so it mirrors the original logits position.
    graph.output.insert(0, new_hidden_output)

    # Rename the producer of ``hidden_in`` to emit the new graph output name.
    for n in graph.node:
        for i, name in enumerate(n.output):
            if name == hidden_in:
                n.output[i] = output_hidden_states_name
        for i, name in enumerate(n.input):
            if name == hidden_in:
                n.input[i] = output_hidden_states_name

    if weight_in is not None and not any(weight_in in n.input for n in graph.node):
        wi = next(init for init in graph.initializer if init.name == weight_in)
        graph.initializer.remove(wi)

    save_onnx(model, output_path)
    logger.info("Wrote transformer-only ONNX → %s", output_path)
    return output_path


__all__ = ["make_transformer_only"]
