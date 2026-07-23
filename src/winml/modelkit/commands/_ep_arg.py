# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Module-level helper for the ``--ep <name>[@<source-tag>]`` CLI syntax.

Splits the raw click argument into ``(ep, source)`` and validates the
source-tag at parse time so malformed input surfaces at the CLI layer
instead of further down the chain inside ``EPDeviceTarget.__post_init__``.

Per design doc ``docs/design/session/2_coreloop.md`` §6.2 Scenarios A.5/A.6.
"""

from __future__ import annotations

from typing import Any

import click

from ..session import VALID_SOURCE_TAGS


def split_ep_at_source(value: str) -> tuple[str, str | None]:
    """Split ``"openvino@pypi"`` into ``("openvino", "pypi")``.

    Without ``@`` returns ``(value, None)`` (unqualified ``--ep openvino``
    form). The source tag is normalized to lowercase before being matched
    against :data:`VALID_SOURCE_TAGS`. The EP name is returned verbatim
    (case preserved) so full names like ``"OpenVINOExecutionProvider"``
    survive intact for :func:`expand_ep_name` (which lowercases
    short-name lookups itself) and :class:`EPDeviceTarget`'s
    case-sensitive full-name match.

    Raises:
        ValueError: On whitespace, multiple ``@``, empty ep or source
            substring, or unknown source tag.
    """
    if any(c.isspace() for c in value):
        raise ValueError(f"Invalid --ep value {value!r}: whitespace is not allowed")

    if value.count("@") > 1:
        raise ValueError(f"Invalid --ep value {value!r}: expected at most one '@'")

    if "@" not in value:
        return value, None

    ep, source = value.split("@", 1)
    if not ep or not source:
        raise ValueError(
            f"Invalid --ep value {value!r}: expected '<ep>@<source-tag>' "
            f"with non-empty ep and source"
        )

    source = source.lower()
    if source not in VALID_SOURCE_TAGS:
        raise ValueError(
            f"Unknown source tag {source!r}; expected one of {sorted(VALID_SOURCE_TAGS)}"
        )

    return ep, source


class EpAtSourceParamType(click.ParamType):
    """Click ParamType wrapping :func:`split_ep_at_source` for ``--ep``.

    Converts ``"openvino@pypi"`` into ``("openvino", "pypi")`` at click
    parse time. Empty / unset values pass through as ``None`` (click's
    standard "option not provided" shape).

    Failures from :func:`split_ep_at_source` surface as
    :class:`click.UsageError` via :meth:`click.ParamType.fail`, so each
    CLI command stops re-wrapping ``ValueError`` in its own try/except
    block.

    Commands wire this up as
    ``@click.option("--ep", type=EpAtSourceParamType())`` and receive
    the option value pre-split as a ``tuple[str, str | None] | None``.
    """

    name = "ep_at_source"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[str, str | None] | None:
        if value is None or value == "":
            return None
        # Idempotency: click may invoke convert() twice (e.g. when the
        # value came from a callback that already returned the parsed
        # tuple). Return pre-split tuples unchanged.
        if isinstance(value, tuple):
            return value
        try:
            return split_ep_at_source(value)
        except ValueError as e:
            self.fail(str(e), param, ctx)
            return None


def _reject_ep_source(
    ep: tuple[str, str | None] | None,
    command_name: str,
) -> str | None:
    """Reject the ``--ep <name>@<source>`` form at the CLI boundary.

    Used by commands that route ``--ep`` through
    :class:`EpAtSourceParamType` but whose downstream pipeline does not
    yet honor the source-tag (build, config). Collapses the verbatim
    try/except block that those commands previously duplicated.

    Args:
        ep: The pre-split value from :class:`EpAtSourceParamType` —
            ``None`` when ``--ep`` was not given, otherwise
            ``(ep_short_name, source_tag_or_None)``.
        command_name: User-visible command string for the error message
            (e.g. ``"winml build"``, ``"winml config"``).

    Returns:
        The bare ``ep`` short name (``str``) when given, or ``None``
        when ``--ep`` was not supplied.

    Raises:
        click.UsageError: when ``ep`` carries a non-``None`` source tag
            (e.g. ``--ep openvino@pypi``).
    """
    if ep is None:
        return None
    ep_part, ep_source = ep
    if ep_source is not None:
        raise click.UsageError(
            f"`{command_name}` does not yet support source pinning "
            f"(got --ep {ep_part}@{ep_source!r}); "
            f"use --ep {ep_part!r} without '@'."
        )
    return ep_part
