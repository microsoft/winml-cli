# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the listing-pick exception taxonomy added in Batch B.

Covers the three exception classes exported from
:mod:`winml.modelkit.session.ep_device`:

- :class:`UnknownListingPick`: raised when the user pins a source tag that
  does not match any discovered EPEntry for the requested EP.
- :class:`IncompatibleListingPick`: raised when the matched EPEntry is
  fundamentally incompatible with the host (e.g. wrong arch).
- :class:`AmbiguousListingPick`: defensive signal — multiple EPEntries
  matched the same (ep, source) pair (expected to be unreachable in normal
  operation, included to surface bugs early).
"""

from __future__ import annotations

import pytest

from winml.modelkit.session import (
    AmbiguousListingPick,
    IncompatibleListingPick,
    UnknownListingPick,
)


class TestUnknownListingPick:
    """Construction-time invariants of UnknownListingPick."""

    def test_attributes_match_constructor(self) -> None:
        exc = UnknownListingPick("openvino", "msix-does-not-exist")
        assert exc.ep_name == "openvino"
        assert exc.source_tag == "msix-does-not-exist"

    def test_message_contains_ep_and_tag(self) -> None:
        exc = UnknownListingPick("openvino", "pypi")
        msg = str(exc)
        assert "openvino" in msg
        assert "pypi" in msg

    def test_hint_in_message(self) -> None:
        """The 'winml sys --list-ep' hint helps users discover valid sources."""
        exc = UnknownListingPick("qnn", "fake-source")
        assert "winml sys --list-ep" in str(exc)

    def test_subclass_of_exception(self) -> None:
        assert issubclass(UnknownListingPick, Exception)

    def test_propagates_when_raised(self) -> None:
        """The exception is not silent — raising it propagates."""
        with pytest.raises(UnknownListingPick) as ei:
            raise UnknownListingPick("openvino", "msix-does-not-exist")
        assert ei.value.ep_name == "openvino"
        assert ei.value.source_tag == "msix-does-not-exist"


class TestIncompatibleListingPick:
    """Construction-time invariants of IncompatibleListingPick."""

    def test_attributes_match_constructor(self) -> None:
        exc = IncompatibleListingPick(
            "openvino", "pypi", "arch mismatch: x86_64 vs arm64"
        )
        assert exc.ep_name == "openvino"
        assert exc.source_tag == "pypi"
        assert exc.reason == "arch mismatch: x86_64 vs arm64"

    def test_message_contains_all_three_fields(self) -> None:
        exc = IncompatibleListingPick(
            "qnn", "msix-microsoft", "missing HTP driver"
        )
        msg = str(exc)
        assert "qnn" in msg
        assert "msix-microsoft" in msg
        assert "missing HTP driver" in msg

    def test_subclass_of_exception(self) -> None:
        assert issubclass(IncompatibleListingPick, Exception)

    def test_propagates_when_raised(self) -> None:
        with pytest.raises(IncompatibleListingPick) as ei:
            raise IncompatibleListingPick("qnn", "pypi", "wrong arch")
        assert ei.value.reason == "wrong arch"


class TestAmbiguousListingPick:
    """Construction-time invariants of AmbiguousListingPick (defensive bug signal)."""

    def test_attributes_match_constructor(self) -> None:
        exc = AmbiguousListingPick("openvino", "pypi", 3)
        assert exc.ep_name == "openvino"
        assert exc.source_tag == "pypi"
        assert exc.candidate_count == 3

    def test_message_contains_count(self) -> None:
        exc = AmbiguousListingPick("openvino", "pypi", 5)
        msg = str(exc)
        assert "openvino" in msg
        assert "pypi" in msg
        assert "5" in msg

    def test_subclass_of_exception(self) -> None:
        assert issubclass(AmbiguousListingPick, Exception)

    def test_propagates_when_raised(self) -> None:
        with pytest.raises(AmbiguousListingPick) as ei:
            raise AmbiguousListingPick("qnn", "msix-workload", 2)
        assert ei.value.candidate_count == 2
