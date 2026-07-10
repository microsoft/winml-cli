# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Registry-driven genai-bundle build support (architecture-agnostic).

A *genai bundle* is the onnxruntime-genai directory layout produced for a
decoder-LLM: a transformer split into a prefill (``ctx.onnx``) and a decode
(``iter.onnx``) graph, plus CPU-side ``embeddings.onnx`` / ``lm_head.onnx``
companions and a ``genai_config.json`` + tokenizer.

This module owns the **generic mechanism** — a small family -> recipe table and
an orchestrator that drives the existing composite/auto model builders — while
every model-specific value (component model types, precisions, sub-model
names, the assembler and any ONNX passes) lives in the model package that
registers a :class:`GenaiBundleRecipe`.  Nothing here hardcodes an architecture:
the orchestrator only reads recipe data and delegates to
:class:`~winml.modelkit.models.auto.WinMLAutoModel`.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from ...utils.constants import normalize_ep_name


if TYPE_CHECKING:
    from collections.abc import Mapping


# =========================================================================
# Recipe specs (pure data)
# =========================================================================


@dataclass(frozen=True)
class GenaiTransformerSpec:
    """How to build the transformer half of a genai bundle.

    The transformer is a single registered composite model (see
    :data:`~winml.modelkit.models.winml.composite_model.COMPOSITE_MODEL_REGISTRY`)
    whose two sub-models become the bundle's context (prefill) and iterator
    (decode) graphs.

    Attributes:
        model_type: Composite ``model_type`` override selecting the
            transformer-only build (e.g. ``"qwen3_transformer_only"``).
        task: Composite task key (e.g. ``"text-generation"``).
        precision: Default transformer precision when the caller passes none.
        context_sub_model: Sub-model name feeding ``ctx.onnx`` (prefill).
        iterator_sub_model: Sub-model name feeding ``iter.onnx`` (decode).
    """

    model_type: str
    task: str
    precision: str
    context_sub_model: str
    iterator_sub_model: str


@dataclass(frozen=True)
class GenaiCompanionSpec:
    """How to build one CPU-side companion (embeddings / lm_head).

    Attributes:
        role: Logical role, e.g. ``"embeddings"`` or ``"lm_head"``.  The
            orchestrator forwards the built ONNX to the assembler as
            ``f"{role}_src"``, so the role must match the assembler's keyword.
        model_type: ``model_type`` override selecting the companion build.
        task: Task key used to build the companion.
        precision: Companion precision.
    """

    role: str
    model_type: str
    task: str
    precision: str


@dataclass(frozen=True)
class GenaiBundleRecipe:
    """Declarative description of a decoder-LLM genai bundle.

    Attributes:
        family: HF ``config.model_type`` that selects this recipe (e.g.
            ``"qwen3"``).  Used as the registry key.
        transformer: Transformer (ctx/iter) build spec.
        companions: CPU-side companion build specs.
        assemble: Callable assembling the copied/converted components into a
            bundle directory and writing ``genai_config.json``; must accept the
            keywords the orchestrator passes (see :func:`build_genai_bundle`).
        transformer_onnx_passes: ONNX graph transforms applied to the
            transformer graphs during assembly.
        max_cache_len: Default static KV cache length (context_length).
        prefill_seq_len: Default prefill/context sequence length.
        soc_model: Default SoC model number forwarded to the assembler.
    """

    family: str
    transformer: GenaiTransformerSpec
    companions: tuple[GenaiCompanionSpec, ...]
    assemble: Callable[..., Path]
    transformer_onnx_passes: tuple[Callable[[onnx.ModelProto], onnx.ModelProto], ...] = ()
    max_cache_len: int = 2048
    prefill_seq_len: int = 64
    soc_model: str = "60"


# =========================================================================
# Registry
# =========================================================================

# family (model_type) -> recipe.  Populated by model packages at import time.
GENAI_BUNDLE_REGISTRY: dict[str, GenaiBundleRecipe] = {}


def register_genai_bundle(recipe: GenaiBundleRecipe) -> GenaiBundleRecipe:
    """Register *recipe* under ``recipe.family`` and return it.

    Raises:
        ValueError: If a recipe is already registered for the family.
    """
    key = recipe.family
    if key in GENAI_BUNDLE_REGISTRY:
        raise ValueError(
            f"genai bundle recipe already registered for {key!r}: "
            f"{GENAI_BUNDLE_REGISTRY[key]!r}. Cannot register {recipe!r}."
        )
    GENAI_BUNDLE_REGISTRY[key] = recipe
    return recipe


def _genai_bundle_registry() -> dict[str, GenaiBundleRecipe]:
    """Return the populated registry, importing the model packages that fill it.

    ``winml.modelkit.models.hf`` is where recipes register (as a side effect of
    import); importing it here is the REQUIRED trigger and is kept lazy so
    lightweight callers stay import-cheap.  The non-empty check turns a
    "registrations moved/renamed" refactor mistake into a loud failure instead
    of silently disabling the feature.
    """
    import winml.modelkit.models.hf  # noqa: F401  # REQUIRED: populates the registry

    if not GENAI_BUNDLE_REGISTRY:
        raise RuntimeError(
            "GENAI_BUNDLE_REGISTRY is empty after importing winml.modelkit.models.hf "
            "— genai bundle registrations are missing or have moved; update the import "
            "trigger in _genai_bundle_registry()."
        )
    return GENAI_BUNDLE_REGISTRY


def resolve_genai_bundle(model_type: str | None) -> GenaiBundleRecipe | None:
    """Return the genai-bundle recipe for *model_type*, or ``None`` if unregistered."""
    if model_type is None:
        return None
    return _genai_bundle_registry().get(model_type)


# =========================================================================
# Orchestrator
# =========================================================================


def _node_summary(path: str | Path, *, top: int = 6) -> str:
    """One-line op-type histogram of an ONNX graph (loads shape metadata only).

    Architecture-agnostic: reports the total node count, the number of distinct
    op types and the *top* most common op types by count.  Hardcodes no
    operator names, so it stays valid for any model the recipe registry drives.
    """
    model = onnx.load(str(path), load_external_data=False)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    total = sum(counts.values())
    top_ops = " ".join(f"{op}={n}" for op, n in counts.most_common(top))
    return f"{total} nodes, {len(counts)} op types: {top_ops}"


def build_genai_bundle(
    model_id: str,
    output_dir: str | Path,
    recipe: GenaiBundleRecipe,
    *,
    ep: str = "qnn",
    device: str = "npu",
    precision: str | None = None,
    max_cache_len: int | None = None,
    prefill_seq_len: int | None = None,
    soc_model: str | None = None,
    companion_overrides: Mapping[str, str | Path] | None = None,
    force_rebuild: bool = False,
    cache_dir: str | Path | None = None,
    emit: Callable[[str], None] | None = None,
) -> Path:
    """Build (or reuse) every bundle component and assemble the genai bundle.

    Drives the existing composite/auto builders per *recipe*, then hands the
    resulting ONNX paths to ``recipe.assemble``.  This function is
    architecture-agnostic: all model-specific values come from *recipe*.

    Args:
        model_id: HF model id or local path.
        output_dir: Destination bundle directory.
        recipe: Bundle recipe (component specs + assembler).
        ep: Short bundle execution-provider token routing the transformer
            stages (e.g. ``"qnn"`` for the NPU HTP, ``"cpu"`` for CPU).  Also
            normalized to the full ORT name for the transformer build.
        device: Device for the transformer build (companions always use CPU).
        precision: Transformer precision override; falls back to the recipe.
        max_cache_len: Static KV cache length override; falls back to the recipe.
        prefill_seq_len: Prefill sequence length override; falls back to recipe.
        soc_model: SoC model override forwarded to the assembler.
        companion_overrides: ``role -> prebuilt ONNX path`` map skipping a
            companion build (e.g. ``{"embeddings": Path(...)}``).
        force_rebuild: Rebuild components even if cached.
        cache_dir: Build cache directory override.
        emit: Optional progress sink invoked with human-readable status lines.

    Returns:
        Path to the written ``genai_config.json``.
    """
    _emit = emit if emit is not None else (lambda _msg: None)

    max_cache_len = recipe.max_cache_len if max_cache_len is None else max_cache_len
    prefill_seq_len = recipe.prefill_seq_len if prefill_seq_len is None else prefill_seq_len
    soc_model = recipe.soc_model if soc_model is None else soc_model
    transformer = recipe.transformer
    transformer_precision = precision or transformer.precision
    transformer_ep = normalize_ep_name(ep) or ep
    companion_ep = normalize_ep_name("cpu") or "cpu"

    from ..auto import WinMLAutoModel

    # --- Transformer (context + iterator) via the composite builder ---
    _emit(f"building transformer stages (device={device}, precision={transformer_precision})")
    built = WinMLAutoModel.from_pretrained(
        model_id,
        task=transformer.task,
        model_type=transformer.model_type,
        device=device,
        precision=transformer_precision,
        ep=transformer_ep,
        no_compile=True,
        use_cache=True,
        force_rebuild=force_rebuild,
        cache_dir=cache_dir,
        sub_model_kwargs={
            transformer.context_sub_model: {
                "shape_config": {"max_cache_len": max_cache_len, "seq_len": prefill_seq_len}
            },
            transformer.iterator_sub_model: {
                "shape_config": {"max_cache_len": max_cache_len, "seq_len": 1}
            },
        },
    )
    context_onnx = Path(built.sub_models[transformer.context_sub_model].onnx_path)
    iterator_onnx = Path(built.sub_models[transformer.iterator_sub_model].onnx_path)
    for label, model_path in (("ctx", context_onnx), ("iter", iterator_onnx)):
        _emit(f"  [{label}] {model_path}")
        _emit(f"        {_node_summary(model_path)}")

    # --- CPU-side companions (embeddings, lm_head, ...) ---
    overrides = companion_overrides or {}
    companion_srcs: dict[str, Path] = {
        role: Path(path) for role, path in overrides.items() if path is not None
    }
    for spec in recipe.companions:
        if spec.role in companion_srcs:
            _emit(f"using provided {spec.role}: {companion_srcs[spec.role]}")
            continue
        _emit(f"building {spec.role} (model_type={spec.model_type}, precision={spec.precision})")
        companion = WinMLAutoModel.from_pretrained(
            model_id,
            task=spec.task,
            model_type=spec.model_type,
            device="cpu",
            precision=spec.precision,
            ep=companion_ep,
            no_compile=True,
            use_cache=True,
            force_rebuild=force_rebuild,
            cache_dir=cache_dir,
        )
        companion_path = Path(companion.onnx_path)
        _emit(f"  [{spec.role}] {companion_path}")
        _emit(f"        {_node_summary(companion_path)}")
        companion_srcs[spec.role] = companion_path

    # --- Assemble ---
    _emit(f"assembling bundle -> {output_dir}")
    config_path = recipe.assemble(
        output_dir,
        context_onnx=context_onnx,
        iterator_onnx=iterator_onnx,
        model_id=model_id,
        max_cache_len=max_cache_len,
        prefill_seq_len=prefill_seq_len,
        ep=ep,
        soc_model=soc_model,
        transformer_onnx_passes=list(recipe.transformer_onnx_passes),
        **{f"{role}_src": path for role, path in companion_srcs.items()},
    )
    _emit(f"  genai_config.json -> {config_path}")
    return Path(config_path)


__all__ = [
    "GENAI_BUNDLE_REGISTRY",
    "GenaiBundleRecipe",
    "GenaiCompanionSpec",
    "GenaiTransformerSpec",
    "build_genai_bundle",
    "register_genai_bundle",
    "resolve_genai_bundle",
]
