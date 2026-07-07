# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Recipe discovery for the example configs under ``examples/recipes/``.

A *recipe* is an authored ``winml build`` config (``export``/``optim``/
``quant``/``compile``/``loader``/``eval`` sections) checked into
``examples/recipes/<slug>/``.  Files are named::

    <task>_<precision>_config.json            # single model
    <task>_<precision>_config_<role>.json     # composite component

where ``<slug>`` is the HuggingFace id with ``/`` replaced by ``_``
(e.g. ``microsoft/resnet-50`` -> ``microsoft_resnet-50``), ``<precision>``
is one of :data:`KNOWN_PRECISIONS`, and ``<role>`` is a component name for
composite models (e.g. ``encoder``/``decoder`` or ``image-encoder``/
``text-encoder``).

The runner builds each authored config directly (``winml build -c``) and runs
perf + accuracy on every precision variant that exists on disk, so adding a new
precision (e.g. ``w8a8``) recipe is automatically picked up with no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Precisions recognised in recipe filenames.  Order is the canonical display
# order (fp16 first).  Discovery is data-driven: any precision listed here that
# has files on disk is run, so dropping a new ``*_w8a8_config.json`` into a
# recipe dir is picked up automatically.
KNOWN_PRECISIONS: tuple[str, ...] = ("fp16", "w8a16", "w8a8")


# Composite component roles per task.  The recipe filename's ``_config_<role>``
# suffix carries the role verbatim; this map only documents the expected set
# and drives a stable ordering for ``winml eval -m <role>=<path>`` args.
ROLE_ORDER_BY_TASK: dict[str, tuple[str, ...]] = {
    "zero-shot-image-classification": ("image-encoder", "text-encoder"),
    "image-to-text": ("encoder", "decoder"),
}


def model_slug(hf_id: str) -> str:
    """Return the recipe directory slug for a HuggingFace id.

    ``microsoft/resnet-50`` -> ``microsoft_resnet-50``.  Only the first ``/``
    is replaced so org/name ids map 1:1 to the on-disk directory names.
    """
    return hf_id.replace("/", "_", 1)


def split_config_stem(path: Path) -> tuple[str, str | None]:
    """Split a recipe filename into ``(group_stem, role)``.

    ``image-to-text_fp16_config_encoder.json`` -> ``("image-to-text_fp16", "encoder")``
    ``image-classification_fp16_config.json``  -> ``("image-classification_fp16", None)``
    """
    stem = path.stem
    if "_config_" in stem:
        group_stem, role = stem.split("_config_", 1)
        return group_stem, role or None
    if stem.endswith("_config"):
        return stem[: -len("_config")], None
    return stem, None


def split_task_precision(group_stem: str) -> tuple[str, str | None]:
    """Split ``<task>_<precision>`` into ``(task, precision)``.

    Only the known precisions are stripped; an unrecognised trailing token is
    left as part of the task so a malformed name never silently changes scope.
    """
    for precision in KNOWN_PRECISIONS:
        suffix = f"_{precision}"
        if group_stem.endswith(suffix):
            return group_stem[: -len(suffix)], precision
    return group_stem, None


@dataclass(frozen=True)
class RecipeComponent:
    """One authored config file, optionally a composite component."""

    path: Path
    role: str | None = None  # None for single-model recipes


@dataclass
class RecipeVariant:
    """All config files for one ``(model, task, precision)`` recipe."""

    precision: str
    components: list[RecipeComponent] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        """True when this variant has split component configs (roles)."""
        return any(c.role is not None for c in self.components)

    @property
    def roles(self) -> list[str]:
        """Component roles in canonical task order, falling back to file order."""
        return [c.role for c in self.components if c.role is not None]


def discover_recipe_variants(
    recipes_dir: Path,
    hf_id: str,
    task: str,
) -> list[RecipeVariant]:
    """Find recipe variants for ``(hf_id, task)`` under ``recipes_dir``.

    Returns one :class:`RecipeVariant` per precision that has config files on
    disk, ordered by :data:`KNOWN_PRECISIONS` (fp16 first).  Returns an empty
    list when the model has no recipe directory or no configs for ``task`` —
    the caller then falls back to ``winml config`` generation.
    """
    model_dir = recipes_dir / model_slug(hf_id)
    if not model_dir.is_dir():
        return []

    # Group component configs by precision.  Only files whose task prefix
    # matches the requested task are considered, so a model dir that hosts
    # several tasks contributes only the relevant ones.
    by_precision: dict[str, list[RecipeComponent]] = {}
    for cfg in sorted(model_dir.glob(f"{task}_*_config*.json")):
        group_stem, role = split_config_stem(cfg)
        cfg_task, precision = split_task_precision(group_stem)
        if cfg_task != task or precision is None:
            continue
        by_precision.setdefault(precision, []).append(RecipeComponent(cfg, role))

    variants: list[RecipeVariant] = []
    for precision in KNOWN_PRECISIONS:
        components = by_precision.get(precision)
        if not components:
            continue
        ordered = _order_components(task, components)
        variants.append(RecipeVariant(precision=precision, components=ordered))
    return variants


def _order_components(task: str, components: list[RecipeComponent]) -> list[RecipeComponent]:
    """Order composite components by the task's canonical role order.

    Single-model variants (one component, role=None) are returned as-is.
    Components whose role is not in the task map keep their sorted file order
    after the known ones, so an unexpected role never drops a component.
    """
    if len(components) <= 1:
        return components
    order = ROLE_ORDER_BY_TASK.get(task, ())

    def sort_key(component: RecipeComponent) -> tuple[int, str]:
        role = component.role or ""
        rank = order.index(role) if role in order else len(order)
        return rank, role

    return sorted(components, key=sort_key)
