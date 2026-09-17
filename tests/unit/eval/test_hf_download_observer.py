# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Regression tests for isolated download observation and timeout accounting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def observer_modules(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "scripts" / "e2e_eval"
    monkeypatch.syspath_prepend(str(directory))
    modules = []
    for name, path in (
        ("_hf_worker_test", directory / "hf_download_observer.py"),
        ("utils._hf_client_test", directory / "utils" / "download_observer.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_no_partial_files_never_enumerates_handles(observer_modules, monkeypatch, tmp_path):
    worker, _ = observer_modules
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda *_: pytest.fail("handle scan"))
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    assert tracker.poll(0)["state"] == "IDLE"
    assert tracker.poll(1)["state"] == "IDLE"


def test_unknown_does_not_reset_or_indefinitely_pause_budget(observer_modules):
    _, client = observer_modules
    budget = client.ExecutionBudget(timeout=10, stall_timeout=600, now=0)
    budget.update(4, client.Observation("ACTIVE", 4, 1, 0, 4))
    budget.update(6, client.Observation("UNKNOWN", 6))
    assert budget.execution_elapsed == pytest.approx(4)
    budget.update(11, client.Observation("UNKNOWN", 11))
    assert budget.execution_elapsed == pytest.approx(9)
    budget.update(12, client.Observation("IDLE", 12, 1, 1, 4))
    assert budget.timed_out


def test_confirmed_completion_resets_once(observer_modules):
    _, client = observer_modules
    budget = client.ExecutionBudget(timeout=10, stall_timeout=600, now=0)
    budget.update(4, client.Observation("ACTIVE", 4, 1, 0, 4))
    budget.update(6, client.Observation("IDLE", 6, 1, 1, 5))
    assert budget.execution_elapsed == 0
    budget.update(9, client.Observation("IDLE", 9, 1, 1, 5))
    assert budget.execution_elapsed == 3


def test_unrelated_download_never_becomes_active(observer_modules, monkeypatch, tmp_path):
    worker, _ = observer_modules
    partial = tmp_path / "hub" / "models--generated" / "blobs" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bytes(range(16)))
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda *_: set())
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    assert tracker.poll(0)["state"] == "IDLE"


def test_owned_download_ends_when_partial_disappears(observer_modules, monkeypatch, tmp_path):
    worker, _ = observer_modules
    partial = tmp_path / "hub" / "models--generated" / "blobs" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bytes(range(32)))
    scans = []
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda pid: (
        scans.append(pid) or {worker._normalized_path(partial)}
    ))
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    first = tracker.poll(0)
    assert first["state"] == "ACTIVE"
    assert tracker.poll(1)["last_progress"] == 0
    assert len(scans) == 1
    partial.rename(partial.with_suffix(""))
    finished = tracker.poll(2)
    assert finished["state"] == "IDLE"
    assert finished["completed_epoch"] == first["epoch"]
    assert tracker.poll(3)["completed_epoch"] == finished["completed_epoch"]


@pytest.mark.parametrize("existing_final", [False, True])
def test_download_end_does_not_depend_on_final_file_layout(
    observer_modules, monkeypatch, tmp_path, existing_final
):
    worker, _ = observer_modules
    partial = tmp_path / "hub" / "models--generated" / "blobs" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bytes(range(8)))
    if existing_final:
        partial.with_suffix("").write_bytes(bytes(range(32)))
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda *_: {
        worker._normalized_path(partial)
    })
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    assert tracker.poll(0)["state"] == "ACTIVE"
    partial.unlink()
    state = tracker.poll(1)
    assert state["state"] == "IDLE"
    assert state["completed_epoch"] == 1


def test_one_ended_file_does_not_discard_another_active_download(
    observer_modules, monkeypatch, tmp_path
):
    worker, _ = observer_modules
    directory = tmp_path / "hub" / "models--generated" / "blobs"
    directory.mkdir(parents=True)
    partials = [directory / f"weights-{index}.incomplete" for index in range(2)]
    for partial in partials:
        partial.write_bytes(bytes(range(8)))
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda *_: {
        worker._normalized_path(partial) for partial in partials
    })
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    assert tracker.poll(0)["state"] == "ACTIVE"
    partials[0].unlink()
    assert tracker.poll(1)["state"] == "ACTIVE"
    assert tracker.poll(2)["completed_epoch"] == 0
    partials[1].unlink()
    assert tracker.poll(3)["completed_epoch"] == 1


def test_failed_scan_does_not_report_download_end(observer_modules, monkeypatch, tmp_path):
    worker, _ = observer_modules
    partial = tmp_path / "hub" / "models--generated" / "blobs" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bytes(range(8)))
    monkeypatch.setattr(worker, "_process_tree_open_paths", lambda *_: {
        worker._normalized_path(partial)
    })
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    assert tracker.poll(0)["state"] == "ACTIVE"
    monkeypatch.setattr(worker, "_snapshot_hf_downloads", MagicMock(side_effect=OSError))
    with pytest.raises(OSError):
        tracker.poll(1)
    assert tracker.completed_epoch == 0


def test_recovered_active_episode_cannot_reset_after_unknown(observer_modules):
    _, client = observer_modules
    budget = client.ExecutionBudget(timeout=10, stall_timeout=600, now=0)
    budget.update(4, client.Observation("ACTIVE", 4, 1, 0, 4))
    budget.update(6, client.Observation("UNKNOWN", 6))
    budget.update(9, client.Observation("ACTIVE", 9, 1, 0, 9))
    budget.update(11, client.Observation("IDLE", 11, 1, 1, 9))
    assert budget.execution_elapsed == 7
    budget.update(14, client.Observation("IDLE", 14, 1, 1, 9))
    assert budget.timed_out


def test_stale_active_lease_expires_even_if_no_new_observation(observer_modules):
    _, client = observer_modules
    budget = client.ExecutionBudget(timeout=10, stall_timeout=600, now=0)
    active = client.Observation("ACTIVE", 4, 1, 0, 4)
    budget.update(4, active)
    budget.update(16, active)
    assert budget.execution_elapsed == 11
    assert budget.timed_out
    assert not budget.hf_download_stalled


def test_heartbeat_does_not_count_as_download_progress(observer_modules):
    _, client = observer_modules
    budget = client.ExecutionBudget(timeout=10, stall_timeout=5, now=0)
    for now in range(1, 7):
        budget.update(now, client.Observation("ACTIVE", now, 1, 0, 1))
    assert budget.hf_download_stalled
    assert not budget.timed_out


@pytest.fixture
def protocol_client(observer_modules):
    _, client = observer_modules
    observer = client.DownloadObserver.__new__(client.DownloadObserver)
    observer._socket = MagicMock()
    observer._owner = MagicMock()
    observer._owner.process.poll.return_value = None
    observer._token = "test-session"  # noqa: S105 - Fixed local protocol fixture, not a credential.
    observer._seq = 0
    observer._started = 10.0
    observer._last = client.Observation("UNKNOWN", 10.0)
    observer.failure = None
    return observer


def _message(**overrides):
    message = {"token": "test-session", "seq": 1, "state": "ACTIVE",
               "observed_at": 11.0, "epoch": 1, "completed_epoch": 0, "last_progress": 11.0}
    return json.dumps(message | overrides).encode(), ("127.0.0.1", 9999)


@pytest.mark.parametrize("overrides", [
    {"token": "other-session"}, {"seq": True}, {"seq": 0}, {"state": "invalid"},
    {"observed_at": float("nan")}, {"last_progress": float("inf")},
    {"observed_at": 20}, {"observed_at": 9}, {"last_progress": 12},
    {"epoch": -1}, {"completed_epoch": 2}, {"epoch": True},
])
def test_invalid_protocol_messages_are_ignored(protocol_client, overrides):
    protocol_client._socket.recvfrom.side_effect = [_message(**overrides), BlockingIOError]
    assert protocol_client.poll(11).state == "UNKNOWN"
    assert protocol_client._seq == 0


def test_message_processing_is_bounded_even_with_flood(protocol_client):
    protocol_client._socket.recvfrom.return_value = _message()
    assert protocol_client.poll(11).state == "ACTIVE"
    assert protocol_client._socket.recvfrom.call_count == 8


def test_replayed_or_backward_messages_cannot_extend_active_lease(protocol_client):
    protocol_client._socket.recvfrom.side_effect = [_message(), BlockingIOError]
    protocol_client.poll(11)
    protocol_client._socket.recvfrom.side_effect = [
        _message(seq=1, observed_at=12), _message(seq=2, observed_at=10.5),
        _message(seq=3, epoch=0), BlockingIOError,
    ]
    assert protocol_client.poll(12).observed_at == 11
    assert protocol_client._seq == 1


def test_helper_exit_discards_queued_active_state(protocol_client):
    owner = protocol_client._owner
    channel = protocol_client._socket
    owner.process.poll.return_value = 17
    assert protocol_client.poll(11).state == "UNKNOWN"
    channel.recvfrom.assert_not_called()
    owner.close.assert_called_once()
    assert protocol_client.failure == "observer process exited"


def test_silent_helper_is_reaped_and_client_fails_closed(protocol_client):
    owner = protocol_client._owner
    protocol_client._socket.recvfrom.side_effect = BlockingIOError
    assert protocol_client.poll(16).state == "UNKNOWN"
    owner.close.assert_called_once()
    assert "no completed observation" in protocol_client.failure


@pytest.mark.parametrize("cache_var", [
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE"
])
def test_cache_root_overrides(observer_modules, tmp_path, cache_var):
    worker, _ = observer_modules
    roots = worker._hf_cache_roots({"HF_HOME": str(tmp_path / "home"), cache_var: str(tmp_path)})
    index = 1 if cache_var == "HF_DATASETS_CACHE" else 2 if cache_var == "HF_XET_CACHE" else 0
    assert roots[index] == tmp_path


def test_slow_ownership_probe_timestamps_progress_after_scan(
    observer_modules, monkeypatch, tmp_path
):
    worker, _ = observer_modules
    partial = tmp_path / "hub" / "models--generated" / "blobs" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(bytes(range(8)))
    clock = [10.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])

    def open_paths(_pid):
        clock[0] += 3.0
        return {worker._normalized_path(partial)}

    monkeypatch.setattr(worker, "_process_tree_open_paths", open_paths)
    tracker = worker.DownloadTracker({"HF_HOME": str(tmp_path)}, 123)
    state = tracker.poll()
    assert state["state"] == "ACTIVE"
    assert state["last_progress"] == clock[0]
