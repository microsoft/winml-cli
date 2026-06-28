# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the ``--ep <name>[@<source-tag>]`` CLI argument parser.

These tests cover the string-to-tuple split only. Downstream behavior
(EPDeviceTarget validation, resolve_device passthrough, auto_device
source-tag filtering) is pinned by existing tests:

- ``tests/unit/session/test_ep_device.py`` —
  ``test_resolve_device_passes_source_through_unchanged``
- ``tests/unit/session/test_auto_device.py`` —
  ``test_b_pinned_source_matches``,
  ``test_g_unmatched_source_tag_raises_unknown_listing_pick``

The parser pins the new feature: ``winml <cmd> --ep openvino@pypi`` should
split the raw value into ``(ep="openvino", source="pypi")``. Today the
parser does not exist — these tests are RED on import.

Per design doc ``docs/design/session/2_coreloop.md`` §6.2 Scenarios A.5/A.6.
"""

from __future__ import annotations

import pytest

from winml.modelkit.session.ep_device import VALID_SOURCE_TAGS


# ---------------------------------------------------------------------------
# Parser contract (RED until the helper is implemented).
# ---------------------------------------------------------------------------


def test_split_ep_no_at_returns_none_source() -> None:
    """Backward-compat — bare EP name yields ``source=None``.

    Existing CLI usage like ``--ep openvino`` must continue to parse to a
    target with ``source=None`` (Scenarios A.1-A.4). Adding the ``@<tag>``
    syntax must not break the unqualified form.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    assert split_ep_at_source("openvino") == ("openvino", None)


def test_split_ep_with_valid_source_returns_pair() -> None:
    """The main feature — ``--ep openvino@pypi`` -> ``("openvino", "pypi")``.

    This is the user-visible split that the design doc references but the
    CLI layer currently lacks (Scenarios A.5/A.6).
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    assert split_ep_at_source("openvino@pypi") == ("openvino", "pypi")


@pytest.mark.parametrize("source_tag", sorted(VALID_SOURCE_TAGS))
def test_split_ep_with_each_valid_source_tag(source_tag: str) -> None:
    """Every tag in ``VALID_SOURCE_TAGS`` must round-trip through the parser.

    Parametrized so the test set grows automatically when a new source tag
    is added to ``EPDeviceTarget``. Pins the contract: parser accepts the
    same vocabulary EPDeviceTarget's ``__post_init__`` validates.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    raw = f"openvino@{source_tag}"
    assert split_ep_at_source(raw) == ("openvino", source_tag)


def test_split_ep_with_invalid_source_raises() -> None:
    """Unknown source tag must be rejected at parse time with a clear message.

    Catches misspellings (e.g. ``openvino@msix`` — the real tags are
    ``msix-microsoft``/``msix-workload``) early, instead of propagating an
    invalid string into ``EPDeviceTarget(source=...)`` where the error
    message would be further from the user's input.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    with pytest.raises(ValueError, match=r"(?i)source"):
        split_ep_at_source("openvino@bogus")


def test_split_ep_with_empty_source_raises() -> None:
    """Bare trailing ``@`` (e.g. ``--ep openvino@``) is malformed and must raise.

    A trailing ``@`` reads as "I intended a source tag but forgot to type
    one" — treating it as ``source=None`` would silently swallow the user's
    typo and run as the unqualified-EP path.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    with pytest.raises(ValueError):
        split_ep_at_source("openvino@")


def test_split_ep_with_multiple_at_raises() -> None:
    """``openvino@msix-microsoft@pypi`` is ambiguous — must reject, not pick one.

    Splitting on the first ``@`` and ignoring later ones would let
    ``openvino@msix-microsoft@pypi`` parse as ``("openvino", "msix-microsoft@pypi")``
    which then fails downstream with a misleading "unknown source tag" error.
    Reject the malformed input at the parser layer so the user sees the
    actual problem.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    with pytest.raises(ValueError):
        split_ep_at_source("openvino@msix-microsoft@pypi")


def test_split_ep_with_empty_ep_before_at_raises() -> None:
    """``@pypi`` (bare leading ``@``) is malformed and must raise.

    A naive ``str.split("@", 1)`` yields ``("", "pypi")`` — an empty EP name
    that would then fail at ``EPDeviceTarget(__post_init__)`` with the
    confusing message "Unknown EP ''". Reject the malformed input at the
    parser layer so the user sees what they actually got wrong.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    with pytest.raises(ValueError):
        split_ep_at_source("@pypi")


@pytest.mark.parametrize(
    "raw",
    [
        "openvino @pypi",   # space before @
        " openvino@pypi",   # leading space
        "openvino@pypi ",   # trailing space
        "openvino@ pypi",   # space after @
    ],
)
def test_split_ep_rejects_whitespace(raw: str) -> None:
    """Whitespace anywhere in the argument is rejected, not silently stripped.

    Silent stripping would let ``--ep "openvino @pypi"`` parse identically to
    ``--ep "openvino@pypi"``, hiding the typo from the user. A CLI argument
    is a single token and any internal whitespace signals a quoting / shell
    mistake worth surfacing.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    with pytest.raises(ValueError):
        split_ep_at_source(raw)


def test_split_ep_normalizes_source_to_lowercase() -> None:
    """Source tag uppercase / mixed-case is normalized to lowercase.

    ``VALID_SOURCE_TAGS`` is lowercase-only, but users may type or paste
    ``--ep openvino@PYPI`` (e.g. from a copied config). Reject-as-invalid
    here would be hostile UX for what is genuinely the same source. Mirror
    the ``device`` field's existing lowercase normalization in
    ``EPDeviceTarget.__post_init__``.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    assert split_ep_at_source("openvino@PYPI") == ("openvino", "pypi")
    assert split_ep_at_source("openvino@MSIX-Microsoft") == ("openvino", "msix-microsoft")


def test_split_ep_preserves_ep_name_case() -> None:
    """EP name case is preserved; only the source tag is lowercased.

    Rationale: full EP names like ``"OpenVINOExecutionProvider"`` must
    survive the parse intact because :class:`EPDeviceTarget`'s
    ``_FULL_TO_SHORT`` lookup is case-sensitive. Short-name lookup in
    :func:`expand_ep_name` is already case-insensitive (it calls
    ``.lower()`` itself), so preserving case here is safe for both
    forms and required for full-name input.
    """
    from winml.modelkit.commands._ep_arg import split_ep_at_source

    # Short names — case preserved (downstream lowercases for lookup).
    assert split_ep_at_source("OPENVINO") == ("OPENVINO", None)
    assert split_ep_at_source("OPENVINO@PYPI") == ("OPENVINO", "pypi")
    assert split_ep_at_source("QNN@msix-microsoft") == ("QNN", "msix-microsoft")

    # Full names — case MUST be preserved for _FULL_TO_SHORT match.
    assert split_ep_at_source("OpenVINOExecutionProvider@pypi") == (
        "OpenVINOExecutionProvider",
        "pypi",
    )
    assert split_ep_at_source("QNNExecutionProvider") == (
        "QNNExecutionProvider",
        None,
    )


def test_ep_at_source_param_type_returns_tuple_for_valid_input() -> None:
    """The click ParamType converts valid input into a (ep, source) tuple."""
    from winml.modelkit.commands._ep_arg import EpAtSourceParamType

    pt = EpAtSourceParamType()
    assert pt.convert("openvino@pypi", None, None) == ("openvino", "pypi")
    assert pt.convert("qnn", None, None) == ("qnn", None)


def test_ep_at_source_param_type_passes_none_through() -> None:
    """Empty / None input passes through as None (click's unset-option shape)."""
    from winml.modelkit.commands._ep_arg import EpAtSourceParamType

    pt = EpAtSourceParamType()
    assert pt.convert(None, None, None) is None
    assert pt.convert("", None, None) is None


def test_ep_at_source_param_type_is_idempotent_on_pre_split_tuples() -> None:
    """Pre-split tuples pass through unchanged.

    Click can invoke ``convert`` twice (e.g. when the value was already
    transformed by an upstream callback). The ParamType must not
    re-split a tuple — calling ``split_ep_at_source(("qnn", None))``
    would raise a TypeError on the ``isspace`` check.
    """
    from winml.modelkit.commands._ep_arg import EpAtSourceParamType

    pt = EpAtSourceParamType()
    assert pt.convert(("qnn", "pypi"), None, None) == ("qnn", "pypi")
    assert pt.convert(("openvino", None), None, None) == ("openvino", None)


def test_ep_at_source_param_type_raises_usage_error_on_bad_tag() -> None:
    """Invalid source tag triggers click.BadParameter via self.fail()."""
    import click

    from winml.modelkit.commands._ep_arg import EpAtSourceParamType

    pt = EpAtSourceParamType()
    with pytest.raises(click.exceptions.BadParameter, match=r"(?i)source"):
        pt.convert("openvino@bogus", None, None)
