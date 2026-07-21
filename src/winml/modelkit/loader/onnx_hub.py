# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download pre-exported ONNX files hosted on the HuggingFace Hub.

This module is the **download** half of Hub-hosted ONNX support.
Classification (deciding whether a ``-m/--model`` value is a Hub ONNX
ref, a local ``.onnx`` file, an HF model ID, or a build directory) lives
in :mod:`winml.modelkit.utils.model_input` and is the single entry point
that all CLI commands and library APIs should go through.

The function exposed here, :func:`resolve_hf_onnx_path`, is called by
``resolve_model_input`` for the ``hub_onnx`` case and downloads the
``.onnx`` file (plus any ``.onnx_data`` sidecar) via
``huggingface_hub.hf_hub_download``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GraphContract:
    path: Path
    inputs: tuple[tuple[str, str, int], ...]
    outputs: tuple[tuple[str, str, int], ...]
    precision: str
    has_quantized_weights: bool = False
    input_shapes: tuple[tuple[Any, ...], ...] = ()
    output_shapes: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class _CompositeGraphPair:
    encoder: _GraphContract
    decoder: _GraphContract
    shared_outputs: tuple[str, ...]
    prompt_component: str


def _inspect_runnable_graph(path: Path) -> _GraphContract | None:
    """Return a runnable graph's I/O contract, or ``None`` when ORT rejects it."""
    import onnx
    import onnxruntime as ort

    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model = onnx.load(path, load_external_data=False)
    except Exception as error:
        logger.warning("Ignoring unusable published ONNX graph %s: %s", path.name, error)
        return None

    type_counts = Counter(initializer.data_type for initializer in model.graph.initializer)
    fp16 = type_counts[onnx.TensorProto.FLOAT16]
    fp32 = type_counts[onnx.TensorProto.FLOAT]
    precision = "fp16" if fp16 > fp32 else "fp32"
    has_quantized_weights = bool(
        type_counts[onnx.TensorProto.INT8] or type_counts[onnx.TensorProto.UINT8]
    )

    def contract(nodes: list[Any]) -> tuple[tuple[str, str, int], ...]:
        return tuple((node.name, node.type, len(node.shape)) for node in nodes)

    return _GraphContract(
        path=path,
        inputs=contract(session.get_inputs()),
        outputs=contract(session.get_outputs()),
        precision=precision,
        has_quantized_weights=has_quantized_weights,
        input_shapes=tuple(tuple(node.shape) for node in session.get_inputs()),
        output_shapes=tuple(tuple(node.shape) for node in session.get_outputs()),
    )


def _has_point_prompt(contract: _GraphContract) -> bool:
    """Whether a graph contract exposes one coordinate/label prompt pair."""
    ports = contract.inputs
    has_integer_labels = any(
        rank in {2, 3} and "int" in dtype.lower() for _name, dtype, rank in ports
    )
    has_point_tensor = any(
        rank in {3, 4} and "float" in dtype.lower() for _name, dtype, rank in ports
    )
    # Some deployment exports cast labels to float. Their unambiguous prompt
    # shape is coordinates [B,N,2] plus labels [B,N]. Embedding-only decoders
    # have no rank-2 input, so this does not mistake sparse embeddings for prompts.
    has_float_pair = any(rank == 2 for _name, _dtype, rank in ports) and any(
        rank == 3 for _name, _dtype, rank in ports
    )
    return has_point_tensor and (has_integer_labels or has_float_pair)


def _has_mask_outputs(contract: _GraphContract) -> bool:
    ranks = [rank for _name, _dtype, rank in contract.outputs]
    return sum(rank >= 4 for rank in ranks) == 1 and sum(1 <= rank <= 3 for rank in ranks) == 1


def _select_encoder_decoder_pair(
    graphs: list[_GraphContract],
    *,
    precision: str,
) -> _CompositeGraphPair:
    """Select one graph pair by connectivity, image, prompt, and mask contracts."""
    preferred_precisions = ("fp16", "fp32") if precision == "fp16" else ("fp32",)
    candidates: list[tuple[int, int, _CompositeGraphPair]] = []
    for encoder in graphs:
        image_inputs = [port for port in encoder.inputs if port[2] == 4]
        if len(image_inputs) != 1:
            continue
        encoder_outputs = {name for name, _dtype, _rank in encoder.outputs}
        for decoder in graphs:
            if (
                decoder.path == encoder.path
                or decoder.precision != encoder.precision
                or decoder.has_quantized_weights != encoder.has_quantized_weights
                or not _has_mask_outputs(decoder)
            ):
                continue
            decoder_inputs = {name for name, _dtype, _rank in decoder.inputs}
            shared = tuple(sorted(encoder_outputs & decoder_inputs))
            if not shared:
                continue
            encoder_prompt = _has_point_prompt(encoder)
            decoder_prompt = _has_point_prompt(decoder)
            if encoder_prompt == decoder_prompt:
                continue
            try:
                precision_rank = preferred_precisions.index(encoder.precision)
            except ValueError:
                continue
            pair = _CompositeGraphPair(
                encoder=encoder,
                decoder=decoder,
                shared_outputs=shared,
                prompt_component="image-encoder" if encoder_prompt else "prompt-decoder",
            )
            candidates.append((precision_rank, int(encoder.has_quantized_weights), pair))

    if not candidates:
        contracts = [
            {
                "file": graph.path.name,
                "precision": graph.precision,
                "inputs": [name for name, _dtype, _rank in graph.inputs],
                "outputs": [name for name, _dtype, _rank in graph.outputs],
            }
            for graph in graphs
        ]
        raise ValueError(
            "Published ONNX graphs do not contain a runnable promptable "
            f"encoder/decoder pair for precision {precision!r}: {contracts}"
        )

    best_source_rank = min(candidate[:2] for candidate in candidates)
    preferred = sorted(
        (candidate[2] for candidate in candidates if candidate[:2] == best_source_rank),
        key=lambda pair: (str(pair.encoder.path), str(pair.decoder.path)),
    )
    if len(preferred) != 1:
        pairs = [
            {
                "image-encoder": pair.encoder.path.name,
                "prompt-decoder": pair.decoder.path.name,
                "precision": pair.encoder.precision,
                "shared_outputs": list(pair.shared_outputs),
                "prompt_component": pair.prompt_component,
            }
            for pair in preferred
        ]
        raise ValueError(
            "Published ONNX graphs contain multiple valid encoder/decoder pairs "
            f"for precision {precision!r}; unable to select one unambiguously. "
            f"Candidate pairs: {pairs}"
        )
    return preferred[0]


def resolve_hf_onnx_encoder_decoder(
    model_id: str,
    *,
    revision: str | None = None,
    precision: str = "fp32",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> dict[str, Path]:
    """Discover a runnable image-encoder/prompt-decoder pair in a Hub repo.

    Selection is graph-contract driven rather than filename driven. The encoder
    has one rank-4 tensor input; the decoder consumes one or more encoder outputs
    plus a rank-3 integer prompt tensor. Invalid published variants are ignored
    after an ORT CPU session probe. An fp16 build may safely fall back to fp32
    source graphs because the normal build pipeline performs fp16 conversion.
    """
    from huggingface_hub import list_repo_files

    files = sorted(
        name
        for name in list_repo_files(model_id, revision=revision, token=token)
        if name.lower().endswith(".onnx")
    )
    if not files:
        raise FileNotFoundError(f"No published ONNX graphs found in Hub repo {model_id!r}.")

    graphs: list[_GraphContract] = []
    for filename in files:
        path = resolve_hf_onnx_path(
            f"{model_id}/{filename}",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
        graph = _inspect_runnable_graph(path)
        if graph is not None:
            graphs.append(graph)

    pair = _select_encoder_decoder_pair(graphs, precision=precision)
    return {"image-encoder": pair.encoder.path, "prompt-decoder": pair.decoder.path}


def resolve_hf_release_onnx_encoder_decoder(
    model_id: str,
    *,
    revision: str | None = None,
    precision: str = "fp32",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> dict[str, Path] | None:
    """Resolve a config-less release archive into one promptable graph pair."""
    from .release_assets import acquire_hf_release_asset

    release = acquire_hf_release_asset(
        model_id,
        revision=revision,
        precision=precision,
        format_name="onnx",
        cache_dir=cache_dir,
        token=token,
    )
    if release is None:
        return None
    pipeline_tag = release.provenance.get("pipeline_tag")
    if pipeline_tag not in {"image-segmentation", "mask-generation"}:
        raise ValueError(
            f"Release archive graph pair requires an image-segmentation or "
            f"mask-generation pipeline tag; got {pipeline_tag!r}."
        )
    graphs = [
        graph
        for path in sorted(release.root.rglob("*.onnx"))
        if (graph := _inspect_runnable_graph(path)) is not None
    ]
    pair = _select_encoder_decoder_pair(graphs, precision=precision)
    return {"image-encoder": pair.encoder.path, "prompt-decoder": pair.decoder.path}


def resolve_hf_onnx_path(
    model_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """Download a Hub-hosted ONNX file and return the local path.

    Splits ``model_id`` into ``(repo_id, filename)``, downloads the
    ``.onnx`` file, and best-effort fetches an optional
    ``<filename>.onnx_data`` sidecar so the ONNX loader can find external
    initializers.

    Args:
        model_id: A Hub ONNX reference such as
            ``"onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"``.
        revision: Optional Hub revision (branch, tag, or commit SHA).
        cache_dir: Optional override for the ``huggingface_hub`` cache directory.
        token: Optional auth token forwarded to ``hf_hub_download``.

    Returns:
        The local path to the downloaded ``.onnx`` file.

    Raises:
        ValueError: If ``model_id`` does not have at least three ``/``-separated
            components.
        FileNotFoundError: If the referenced ``.onnx`` file does not exist in
            the repo. The error message lists the ``.onnx`` files that *are*
            present so the user can correct the path.
        huggingface_hub.utils.RepositoryNotFoundError: If the repo itself does
            not exist (re-raised unchanged).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo_id, filename = _split_hf_onnx_path(model_id)
    logger.info("Downloading ONNX from Hub: repo=%s file=%s", repo_id, filename)

    try:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
    except EntryNotFoundError as e:
        # The repo exists but ``filename`` does not. Surface the available
        # ``.onnx`` files so the user can pick the right one without leaving
        # the terminal. Re-raise as ``FileNotFoundError`` so callers that
        # already handle local-file-missing errors get a consistent type.
        hint = _format_available_onnx_files(repo_id, revision=revision, token=token)
        raise FileNotFoundError(
            f"ONNX file '{filename}' not found in Hub repo '{repo_id}'.\n{hint}"
        ) from e

    # External-data sidecars (used for >2 GiB models) live next to the .onnx
    # file with a ``.onnx_data`` suffix. The main download above just
    # succeeded for the same repo, so the only expected reason the sidecar
    # is missing is that the model inlined its weights -- catch
    # ``EntryNotFoundError`` quietly. Any other failure (disk full,
    # permissions, network blip, etc.) is real and surfaced as a warning
    # so the user is not left with a half-downloaded model that fails
    # later at load time with a confusing error.
    sidecar_filename = f"{filename}_data"
    try:
        sidecar_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=sidecar_filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
        logger.info("Downloaded external-data sidecar: %s", sidecar_path.name)
    except EntryNotFoundError:
        # Expected: model has no separate weights file (weights are inlined).
        logger.debug("No external-data sidecar at %s (weights inlined)", sidecar_filename)
    except OSError as e:
        # Unexpected: disk/permission/network problem. Warn loudly --
        # silent failure here would make the model unloadable later.
        logger.warning(
            "Failed to download external-data sidecar %s for %s: %s. "
            "If the model uses external weights, loading will fail.",
            sidecar_filename,
            repo_id,
            e,
        )

    return local_path


def _split_hf_onnx_path(model_id: str) -> tuple[str, str]:
    """Split a Hub ONNX reference into ``(repo_id, filename)``."""
    parts = [p for p in model_id.split("/") if p]
    if len(parts) < 3:
        raise ValueError(
            f"Hub ONNX reference must have form 'org/repo/path/to/file.onnx', got: {model_id!r}"
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def _format_available_onnx_files(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Build a human-readable hint listing ``.onnx`` files in a Hub repo.

    Used to enrich ``EntryNotFoundError`` messages so users who guessed the
    wrong filename can see the available options without leaving the
    terminal. Best-effort: if listing fails for any reason (network,
    auth, gated repo) we return a generic fallback hint instead of
    masking the original error.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
    except Exception as list_err:
        logger.debug("Could not list files for %s: %s", repo_id, list_err)
        return (
            f"Could not list available .onnx files in '{repo_id}' "
            f"(see https://huggingface.co/{repo_id}/tree/{revision or 'main'})."
        )

    onnx_files = sorted(f for f in files if f.lower().endswith(".onnx"))
    if not onnx_files:
        return (
            f"No .onnx files were found in '{repo_id}'. "
            f"This repo may not host pre-exported ONNX weights; "
            f"see https://huggingface.co/{repo_id}/tree/{revision or 'main'}."
        )

    listing = "\n".join(f"  - {repo_id}/{f}" for f in onnx_files)
    return f"Available .onnx files in '{repo_id}':\n{listing}"


__all__ = [
    "resolve_hf_onnx_encoder_decoder",
    "resolve_hf_onnx_path",
    "resolve_hf_release_onnx_encoder_decoder",
]
