# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNNMonitor — Qualcomm NPU per-op profiler via ORT's QNN EP.

Produces an :class:`OpTraceResult` with per-operator cycle counts
(``level="basic"``) or full QHAS roofline / DMA traffic
(``level="detail"``).

Contributes session options and provider options to a ``WinMLSession`` via
the two :class:`WinMLEPMonitor` hooks; owns the ``profiling_level`` and
``profiling_file_path`` provider-option keys (C-3 in PRD — never
user-overridable). Requires ``ort.InferenceSession`` teardown before
``__exit__`` because QNN EP flushes the profiling CSV only on session
destruction.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ..._env import env_flag_enabled
from ...onnx.epcontext import epcontext_partitions, select_main_epcontext_partition_name
from ._onnx_metadata import _load_onnx_operator_data
from .ep_monitor import WinMLEPMonitor
from .op_metrics import (
    OperatorMetrics,
    OpTraceResult,
    TraceFallbackReason,
    TraceStatus,
)
from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv
from .qnn.viewer import find_qnn_sdk, run_qhas_viewer_result


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


logger = logging.getLogger(__name__)
_OP_ADD_DATA_ENV = "WINMLCLI_OP_ADD_DATA"


# Maps user-facing level to QNN EP's `profiling_level` provider option.
_LEVEL_TO_PROFILING: dict[str, str] = {
    "basic": "detailed",
    "detail": "optrace",
}


def _metadata_int(value: object, field: str) -> int:
    """Normalize a QNN numeric metadata value, including float strings."""
    try:
        return round(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning(
            "QNNMonitor: could not parse %r as a number for metadata field %r; "
            "defaulting to 0. This may corrupt cycle_to_us and duration_us values.",
            value,
            field,
        )
        return 0


def _coalesce_partition_samples(
    samples: list[dict[str, Any]],
    partition_count: int,
) -> list[dict[str, Any]]:
    """Combine the per-partition profiling blocks emitted for each inference."""
    if partition_count == 1:
        return samples

    combined_samples: list[dict[str, Any]] = []
    for offset in range(0, len(samples), partition_count):
        partition_samples = samples[offset : offset + partition_count]
        combined: dict[str, Any] = {
            "metadata": {
                "hvx_threads": max(
                    _metadata_int(sample["metadata"]["hvx_threads"], "hvx_threads")
                    for sample in partition_samples
                ),
                "accel_execute_cycles": sum(
                    _metadata_int(
                        sample["metadata"]["accel_execute_cycles"],
                        "accel_execute_cycles",
                    )
                    for sample in partition_samples
                ),
                "accel_execute_us": sum(
                    _metadata_int(
                        sample["metadata"]["accel_execute_us"],
                        "accel_execute_us",
                    )
                    for sample in partition_samples
                ),
            },
            "samples": [],
        }
        for partition_index, sample in enumerate(partition_samples):
            metadata = sample["metadata"]
            for op in sample["samples"]:
                combined["samples"].append(
                    {
                        **op,
                        "_partition_index": partition_index,
                        "_accel_execute_cycles": metadata["accel_execute_cycles"],
                        "_accel_execute_us": metadata["accel_execute_us"],
                    }
                )
        combined_samples.append(combined)
    return combined_samples


class QNNMonitor(WinMLEPMonitor):
    """Qualcomm NPU per-op profiler via ORT's QNN EP.

    Produces an :class:`OpTraceResult` with per-operator cycle counts
    (``level="basic"``) or full QHAS roofline / DMA traffic
    (``level="detail"``).

    .. note::

       When ``output_dir`` is ``None``, a per-monitor temp directory
       (``qnn_profile_*``) is created under the OS tempdir and is **never
       auto-cleaned** so that profiling artifacts (CSV, QHAS JSON,
       schematic, QNN log) remain available for post-run inspection.
       Callers that care about disk hygiene should pass an explicit
       ``output_dir`` they manage. The chosen directory is exposed via
       :py:attr:`output_dir`.

    .. note::

       Contributes no ``get_session_options()`` override (uses the
       :class:`WinMLEPMonitor` default: empty dict). ``ep.context_enable=1``
       used to be set here to opt into EPContext caching, but the profiling
       session is always built from the model handed to benchmarking —
       which ``winml build`` has *already* compiled to an EPContext model
       (a placeholder node referencing an external ``.bin``, original ops
       stripped) before benchmarking ever starts. Asking QNN to *generate* a new EPContext
       output from a graph that's already just an EPContext node gives it
       nothing to compile, and ``OrtEp::Compile()`` returns a NULL
       EPContext node — a hard session-creation failure. Do not re-add this
       option: EPContext caching would only help a session's *next* run
       reuse the compiled artifact, but op-tracing sessions are single-use
       and torn down immediately after the perf window closes (see
       :attr:`requires_session_teardown`), so there is no future run to
       benefit from re-caching anyway.

       ``session.disable_cpu_ep_fallback`` is also intentionally not set:
       under ``onnxruntime-windowsml`` the WinML-registered QNN partitions
       a QDQ-wrapped EPContext model into Q/DQ-on-CPU + EPContext-on-QNN,
       which is correct behaviour (the boundary Q/DQ ops genuinely run on
       CPU). Disabling CPU fallback would reject that valid partition and
       cause NotImplemented errors even when QNN successfully claimed the
       EPContext node. The "no silent CPU fallback" guarantee is provided
       by ``add_provider_for_devices`` upstream — if the QNN device is
       absent, session creation fails loudly there.
    """

    #: QNN EP flushes the profiling CSV only on ``ort.InferenceSession``
    #: destruction; ``WinMLSession.perf().__exit__`` must drop the session
    #: before calling ``monitor.__exit__``.
    requires_session_teardown: ClassVar[bool] = True

    #: Pins ``WinMLSession`` to the QNN EP path so provider options
    #: (``profiling_level``, ``profiling_file_path``) flow through
    #: ``add_provider_for_devices``. Without this, the session would use
    #: ORT's policy-based selection which silently drops provider options.
    ep_name: ClassVar[str | None] = "qnn"

    def __init__(
        self,
        level: Literal["basic", "detail"] = "basic",
        output_dir: Path | None = None,
        extra_provider_options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            level: ``"basic"`` (cycles only) or ``"detail"`` (QHAS roofline +
                DMA traffic).
            output_dir: Directory for profiling artifacts. When ``None``, a
                per-monitor temp directory ``qnn_profile_*`` is created under
                the OS tempdir; that directory is **never auto-cleaned** so
                artifacts can be inspected post-run. Pass an explicit path if
                you want to manage cleanup yourself.
            extra_provider_options: Additional QNN EP provider options. The
                two profiling-control keys (``profiling_level``,
                ``profiling_file_path``) are owner-enforced per PRD C-3 and
                cannot be overridden via this argument.
        """
        if level not in _LEVEL_TO_PROFILING:
            raise ValueError(f"level must be 'basic' or 'detail', got {level!r}")
        self._level: str = level
        # Idempotency: paths produced at __init__, not per-call.
        # When output_dir is None we mint a fresh tempdir; we deliberately
        # do NOT register a finalizer to clean it up — see class docstring.
        self._output_dir: Path = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="qnn_profile_"))
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path: Path = (
            self._output_dir / f"profiling_output_{uuid.uuid4().hex}.csv"
        ).resolve()
        self._extra: dict[str, str] = dict(extra_provider_options or {})
        self._entered: bool = False
        self._result: OpTraceResult | None = None
        self._csv_signature_at_enter: tuple[int, int, int, int, int] | None = None
        self._qnn_log_signatures_at_enter: dict[Path, tuple[int, int, int, int, int]] = {}
        self._monitor_enter_time_ns: int | None = None
        self._warmup_samples: int = 0
        self._expected_measured_samples: int | None = None
        # v2.4: ONNX node.name -> node.op_type map injected by WinMLSession.perf
        # before __enter__. Populated only when an ONNX graph is available;
        # remains empty for the standalone parsing case (parse_existing_artifacts
        # without an onnx_op_types argument). Drives L1 of the fallback chain
        # in :py:meth:`_resolve_op_type`.
        self._onnx_op_types: dict[str, str] = {}
        self._onnx_model_path: Path | None = None
        self._running_model_path: Path | None = None

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Directory where profiling artifacts (CSV, QHAS JSON, schematic) are written.

        When ``output_dir=None`` was passed at construction, this is a
        per-monitor temp directory (``qnn_profile_*``) under the OS tempdir.
        The directory is **NOT auto-cleaned** — artifacts persist for
        post-hoc inspection. Callers that care about disk hygiene should
        pass an explicit ``output_dir`` they manage.
        """
        return self._output_dir

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether the QNN EP is usable on this system.

        Checks two paths in order:

        1. ``onnxruntime-qnn`` bundled wheel: ``QNNExecutionProvider`` is
           already in :func:`onnxruntime.get_available_providers`.
        2. ``onnxruntime-windowsml``: instantiate :class:`WinMLEPRegistry`
           to trigger plugin discovery (filesystem scan only — no eager
           DLL loading), then look for a QNN device in
           :func:`onnxruntime.get_ep_devices`.

        .. note::
           Behavior change (Batch D): the WinML path no longer eagerly
           preloads EP DLLs via the legacy ``ensure_initialized`` helper.
           Discovery is filesystem-only; DLL loading is deferred to
           :meth:`WinMLEPRegistry.register_ep` per the lazy-load contract.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        if "QNNExecutionProvider" in ort.get_available_providers():
            return True

        # WinML-registered path.
        try:
            from ..ep_registry import WinMLEPRegistry
        except ImportError:
            return False

        try:
            # TODO(qnn_monitor): WinMLEPRegistry.instance() runs the
            # filesystem-only discovery in __init__; it does NOT eagerly
            # load EP DLLs. If this probe later needs a specific DLL to be
            # loaded for ort.get_ep_devices() to see it, switch to calling
            # register_ep on the matching EPEntry directly.
            WinMLEPRegistry.instance()
            return any(
                getattr(d, "ep_name", None) == "QNNExecutionProvider" for d in ort.get_ep_devices()
            )
        except Exception as exc:
            # Real environmental failure (e.g., broken Windows App SDK,
            # denied registration, missing DLL) — surface at WARNING so
            # users can diagnose. NFR-2: this MUST NOT be silent.
            logger.warning(
                "QNNMonitor.is_available: WinML EP probe failed (%s: %s); reporting unavailable",
                type(exc).__name__,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Hook contributions
    # ------------------------------------------------------------------

    def get_provider_options(self) -> dict[str, str]:
        """Provider options for QNN EP with owner-enforced profiling keys.

        Only the two profiling keys (``profiling_level``, ``profiling_file_path``)
        are owner-set; everything else is pass-through from ``extra_provider_options``.
        This is deliberate: ORT's ``add_provider_for_devices`` merges these
        options on top of whatever the device source pre-configured. Under
        ``onnxruntime-windowsml`` the WinML-registered QNN device already has
        an absolute ``backend_path`` and tuned HTP defaults; supplying our own
        defaults here would *overwrite* WinML's and break DLL loading.

        Callers who need to tune HTP behaviour (e.g. ``backend_path`` for
        the bundled ``onnxruntime-qnn`` path, or ``htp_performance_mode``)
        pass them via ``extra_provider_options`` at construction time.

        Build order (last writer wins):

        1. ``self._extra`` — caller-supplied options (may include backend
           settings the bundled-ORT path needs).
        2. ``profiling_level`` and ``profiling_file_path`` — applied LAST;
           owner-enforced per C-3 (PRD). Assigned explicitly after
           :py:meth:`dict.update` to avoid Ruff ``F601`` on duplicate keys
           and to guarantee they cannot be shadowed by ``extra``.
        """
        opts: dict[str, str] = dict(self._extra)
        # C-3: these two keys are NEVER user-overridable.
        opts["profiling_level"] = _LEVEL_TO_PROFILING[self._level]
        opts["profiling_file_path"] = str(self._csv_path)
        return opts

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("QNNMonitor already entered")
        self._csv_signature_at_enter = self._artifact_signature(self._csv_path)
        self._qnn_log_signatures_at_enter = self._snapshot_qnn_log_signatures()
        self._monitor_enter_time_ns = time.time_ns()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Parse whatever artifacts are on disk. Never suppresses caller exceptions."""
        self._result = self._parse_artifacts_safe(require_fresh=True)
        # Implicit None return → does not suppress caller exception.

    def _parse_artifacts_safe(
        self,
        qhas_override: Path | None = None,
        *,
        require_fresh: bool = False,
    ) -> OpTraceResult:
        """Wrap :py:meth:`_parse_artifacts` with the parse-failure contract.

        Single source of truth for parse-failure handling: both ``__exit__``
        (live path) and :py:meth:`parse_existing_artifacts` (offline path)
        route through this helper so they cannot diverge — both produce
        ``OpTraceResult(status="parse_failed", error=str(exc))`` on
        exception, never propagate.

        Args:
            qhas_override: Optional pre-supplied QHAS JSON path; forwarded
                verbatim to :py:meth:`_parse_artifacts`.
        """
        try:
            if require_fresh:
                self._validate_live_csv_freshness()
            return self._parse_artifacts(qhas_override=qhas_override)
        except Exception as exc:
            logger.warning("QNNMonitor: artifact parse failed: %s", exc)
            return self._make_failure_result(status="parse_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def result(self) -> OpTraceResult | None:
        """Structured result object. Preferred by report writers."""
        return self._result

    # ------------------------------------------------------------------
    # v2.4 op-type resolution (FR-14, FR-15, FR-16)
    # ------------------------------------------------------------------

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Override the WinMLEPMonitor no-op default — QNN uses the map.

        Stores the ONNX ``node.name -> node.op_type`` map for use during
        ``__exit__`` parsing.  ``WinMLSession.perf`` calls this once,
        immediately before ``__enter__``, with the map built from the
        ONNX graph passed to the session.

        Defensively copies the input so later caller mutation cannot
        corrupt the resolver's L1 lookup table.  Empty / no-graph input
        is a valid no-op: the resolver simply falls through to L2/L3/L4.
        """
        self._onnx_op_types = dict(onnx_op_types)

    def set_onnx_model_path(self, onnx_model_path: Path) -> None:
        """Store a defensive path copy for opt-in basic-trace enrichment."""
        self._onnx_model_path = Path(onnx_model_path)

    def set_running_model_path(self, running_model_path: Path) -> None:
        """Store the ONNX model path ORT actually runs for sidecar binding."""
        self._running_model_path = Path(running_model_path)

    def set_perf_window(self, warmup: int, measured_iterations: int) -> None:
        """Record completed run counts for warmup filtering and validation."""
        self._warmup_samples = warmup
        self._expected_measured_samples = measured_iterations

    @staticmethod
    def _artifact_signature(path: Path) -> tuple[int, int, int, int, int] | None:
        """Return a metadata-only signature, or ``None`` when absent.

        The QNN session may already hold the profiling file open when the
        monitor enters, so freshness detection must not read its contents.
        """
        if not path.is_file():
            return None
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _validate_live_csv_freshness(self) -> None:
        """Reject a profiling CSV that predates and did not change in this window."""
        if self._csv_signature_at_enter is None:
            return
        current_signature = self._artifact_signature(self._csv_path)
        if current_signature == self._csv_signature_at_enter:
            raise ValueError(
                f"profiling CSV was unchanged during the monitor window: {self._csv_path}"
            )

    def _resolve_op_type(self, op_path: str, ep_authoritative: str | None = None) -> str:
        """Walk the v2.4 fallback chain: ONNX -> EP-authoritative -> heuristic -> raw.

        Implements FR-14 (ONNX primary + fallback chain) and §3.5 of the
        op-trace parser interface spec.

        - **L1**: ``self._onnx_op_types[op_path]`` lookup (primary, when
          graph is available and the path matches a node name verbatim).
          The lookup uses a truthy check on the value: an empty-string
          op_type (defensive guard against malformed ONNX input) falls
          through to L2/L3/L4 instead of short-circuiting with ``""``.
        - **L2**: ``ep_authoritative`` (e.g. QHAS ``qnn_op_type``) — only
          set when caller has it; basic-CSV path passes ``None``.
        - **L3**: :py:meth:`_heuristic_op_type` — leaf-split with strip
          safety, best-effort fallback for the CSV path.
        - **L4**: ``op_path`` verbatim — last resort, never empty.
        """
        mapped = self._onnx_op_types.get(op_path)
        if mapped:  # truthy: not None, not empty string
            return mapped
        if ep_authoritative:
            return ep_authoritative
        return self._heuristic_op_type(op_path) or op_path

    def _heuristic_op_type(self, op_path: str) -> str:
        r"""Heuristic-only fallback: leaf-split with strip safety.

        Preserves the strip semantics from the legacy ``_split_op_event_id``
        helper (spec §3.2 / coreloop §4.3 — Phase 0 fix):

        - Strips the ``_token_\d+(?:_\d+)?`` suffix injected by the QNN
          compiler (the CSV path's events carry this; the QHAS path's
          ``qnn_op`` does not, but stripping is idempotent).
        - Strips outer whitespace.
        - Splits at the trailing ``/`` and strips inner whitespace around
          the leaf.
        - For trailing-slash inputs the leaf is empty after split — fall
          back to the cleaned input so callers never receive an empty
          string they didn't supply.
        """
        cleaned = _TOKEN_SUFFIX.sub("", op_path).strip()
        if "/" not in cleaned:
            return cleaned
        leaf = cleaned.rsplit("/", 1)[-1].strip()
        return leaf if leaf else cleaned  # trailing-slash → fall back to full

    # ------------------------------------------------------------------
    # Standalone parsing (offline / post-hoc artifact analysis)
    # ------------------------------------------------------------------

    @classmethod
    def parse_existing_artifacts(
        cls,
        level: Literal["basic", "detail"],
        artifacts: dict[str, Path],
        onnx_op_types: dict[str, str] | None = None,
    ) -> OpTraceResult:
        """Parse pre-existing QNN profiling artifacts without running a benchmark.

        Use this for offline analysis of trace files from a previous run.
        Pass an ``onnx_op_types`` map to enable the ONNX op-type lookup
        (L1 of the fallback chain); pass ``None`` or ``{}`` to fall
        through to QHAS-authoritative or heuristic.

        Args:
            level: ``"basic"`` (CSV only) or ``"detail"`` (CSV + QHAS JSON).
            artifacts: Mapping of artifact kind to absolute path.  Must
                contain ``"csv"``; may contain ``"qhas"`` for the detail
                path.  When ``"qhas"`` is provided, the QHAS viewer
                shell-out is skipped and the JSON is parsed directly.
            onnx_op_types: Optional ONNX node.name -> op_type map for
                L1 resolution.  Defaults to empty (L2/L3/L4 only).

        Returns:
            :class:`OpTraceResult` with the parsed operators and summary.

        Raises:
            ValueError: if ``artifacts`` lacks the required ``"csv"`` key.
        """
        csv_path = artifacts.get("csv")
        if csv_path is None:
            raise ValueError("artifacts dict must contain a 'csv' key")
        csv_path = Path(csv_path)
        output_dir = csv_path.parent
        instance = cls(level=level, output_dir=output_dir)
        # Honour the caller's explicit path so this works for offline fixtures
        # with arbitrary names.
        instance._csv_path = csv_path.resolve()
        instance.set_onnx_op_types(onnx_op_types or {})

        qhas_path = artifacts.get("qhas")
        # Route through _parse_artifacts_safe so the offline path honours the
        # SAME parse-failure contract as __exit__: corrupt artifacts surface
        # as OpTraceResult(status="parse_failed", error=...) instead of
        # propagating an exception out of the classmethod.
        result = instance._parse_artifacts_safe(
            qhas_override=Path(qhas_path) if qhas_path else None
        )
        # M-2 carry-forward: leave the constructed instance internally
        # consistent so callers that hold onto it (e.g. via a wrapper) see
        # the parsed result via the typed accessor instead of None.
        instance._result = result
        return result

    # ------------------------------------------------------------------
    # Artifact parsing
    # ------------------------------------------------------------------

    def _parse_artifacts(self, qhas_override: Path | None = None) -> OpTraceResult:
        """Parse CSV (always) and optionally QHAS (detail mode).

        Windows file-handle lag mitigation (R-2): if the CSV is absent on
        the first check, sleep 50ms and retry once before giving up.

        Args:
            qhas_override: When provided, skip the QHAS viewer shell-out
                and parse this JSON directly.  Used by
                :py:meth:`parse_existing_artifacts` for offline analysis.
        """
        csv_path = self._csv_path
        if not csv_path.is_file():
            time.sleep(0.05)  # R-2: Windows file-handle flush lag
            if not csv_path.is_file():
                logger.warning("QNNMonitor: profiling CSV not produced at %s", csv_path)
                return self._make_failure_result(status="no_data", error=None)

        parsed = parse_qnn_profiling_csv(csv_path)
        samples = parsed.get("samples", [])
        if self._expected_measured_samples is not None:
            expected_total = self._warmup_samples + self._expected_measured_samples
            partition_count = self._epcontext_partition_count()
            expected_profile_blocks = expected_total * partition_count
            if (
                len(samples) != expected_profile_blocks
                and partition_count == 1
                and expected_total > 0
                and len(samples) % expected_total == 0
            ):
                partition_count = len(samples) // expected_total
                expected_profile_blocks = len(samples)
            if len(samples) != expected_profile_blocks:
                raise ValueError(
                    "profiling CSV sample count mismatch: "
                    f"expected {expected_profile_blocks} profiling blocks "
                    f"({self._warmup_samples} warmup + "
                    f"{self._expected_measured_samples} measured) across "
                    f"{partition_count} EPContext partition(s), got {len(samples)}"
                )
            samples = _coalesce_partition_samples(samples, partition_count)
            samples = samples[self._warmup_samples :]

        artifacts: dict[str, str] = {"csv": str(csv_path)}

        sample_metadata: list[dict[str, int]] = []
        operator_samples: dict[tuple[int, int], dict[str, Any]] = {}
        for sample in samples:
            sample_meta = sample.get("metadata", {})
            total_cycles = _metadata_int(
                sample_meta.get("accel_execute_cycles", 0) or 0,
                "accel_execute_cycles",
            )
            accel_us = _metadata_int(
                sample_meta.get("accel_execute_us", 0) or 0,
                "accel_execute_us",
            )
            hvx_threads = _metadata_int(
                sample_meta.get("hvx_threads", 0) or 0,
                "hvx_threads",
            )
            sample_metadata.append(
                {
                    "hvx_threads": hvx_threads,
                    "accel_execute_cycles": total_cycles,
                    "accel_execute_us": accel_us,
                }
            )
            for op in sample.get("samples", []):
                op_id = op["op_id"]
                partition_index = op.get("_partition_index", 0)
                key = (partition_index, op_id)
                entry = operator_samples.setdefault(
                    key,
                    {
                        "op_path": op["op_path"],
                        "op_id": op_id,
                        "durations_us": [],
                        "percentages": [],
                    },
                )
                op_total_cycles = _metadata_int(
                    op.get("_accel_execute_cycles", total_cycles),
                    "accel_execute_cycles",
                )
                op_accel_us = _metadata_int(
                    op.get("_accel_execute_us", accel_us),
                    "accel_execute_us",
                )
                cycle_to_us = (
                    op_accel_us / op_total_cycles if op_total_cycles > 0 else 0.0
                )
                entry["durations_us"].append(op["cycles"] * cycle_to_us)
                entry["percentages"].append(
                    op["cycles"] / total_cycles * 100 if total_cycles > 0 else 0.0
                )

        onnx_operator_data: dict[str, dict[str, Any]] = {}
        if (
            self._level == "basic"
            and self._onnx_model_path is not None
            and env_flag_enabled(_OP_ADD_DATA_ENV)
        ):
            try:
                onnx_operator_data = _load_onnx_operator_data(self._onnx_model_path)
            except Exception as error:
                logger.warning(
                    "Could not enrich QNN profiler metrics with ONNX metadata: %s",
                    error,
                    exc_info=True,
                )

        operators = []
        for entry in operator_samples.values():
            durations_us = entry["durations_us"]
            percentages = entry["percentages"]
            op_data = onnx_operator_data.get(entry["op_path"], {})
            operators.append(
                OperatorMetrics(
                    name=self._resolve_op_type(entry["op_path"], ep_authoritative=None),
                    op_path=entry["op_path"],
                    op_id=entry["op_id"],
                    duration_us=sum(durations_us) / len(durations_us),
                    percent_of_total=sum(percentages) / len(percentages),
                    samples_us=durations_us,
                    onnx_op_type=op_data.get("onnx_op_type"),
                    onnx_attributes=op_data.get("onnx_attributes"),
                    onnx_inputs=op_data.get("onnx_inputs"),
                    onnx_outputs=op_data.get("onnx_outputs"),
                )
            )
        operators.sort(key=lambda op: op.duration_us, reverse=True)

        def _metadata_mean(field: str) -> float:
            if not sample_metadata:
                return 0.0
            return sum(meta[field] for meta in sample_metadata) / len(sample_metadata)

        summary: dict[str, Any] = {
            "hvx_threads": _metadata_mean("hvx_threads"),
            "accel_execute_cycles": _metadata_mean("accel_execute_cycles"),
            "accel_execute_us": _metadata_mean("accel_execute_us"),
        }

        status: TraceStatus = "ok"
        fallback_reason: TraceFallbackReason | None = None
        # Detail mode: attempt QHAS post-processing.
        if self._level == "detail":
            qhas_summary, qhas_operators, qhas_path, fallback_reason = self._try_qhas(
                artifacts, qhas_override=qhas_override
            )
            if qhas_path is not None and qhas_operators is not None:
                operators = qhas_operators
                summary = qhas_summary or summary
                artifacts["qhas"] = str(qhas_path)
            else:
                # Fell back to CSV-only data in detail mode.
                status = "basic_fallback"
                logger.warning("QNNMonitor: QHAS unavailable; detail mode degraded to basic")

        return OpTraceResult(
            model=None,
            device="npu",
            tracing_level=self._level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=operators,
            summary=summary,
            num_samples=len(samples),
            artifacts=artifacts,
            status=status,
            fallback_reason=fallback_reason,
        )

    def _epcontext_partition_count(self) -> int:
        """Return the number of EPContext partitions represented in each inference."""
        if self._running_model_path is None:
            return 1
        try:
            return max(1, len(epcontext_partitions(self._running_model_path)))
        except Exception as exc:
            logger.info(
                "QNNMonitor: unable to inspect EPContext partitions from %s: %s",
                self._running_model_path,
                exc,
            )
            return 1

    def _try_qhas(
        self,
        artifacts: dict[str, str],
        qhas_override: Path | None = None,
    ) -> tuple[
        dict[str, Any] | None,
        list[OperatorMetrics] | None,
        Path | None,
        TraceFallbackReason | None,
    ]:
        """Attempt QHAS post-processing.

        Returns ``(summary, operators, qhas_path, fallback_reason)``. The
        reason is ``None`` on success and a stable code on failure. Never raises.

        Per C-5 / FR-12 this method does NOT call :func:`os.chdir`.
        Live-path QNN logs are bound by the profiling CSV stem: ORT writes
        ``<csv_stem>_qnn.log`` next to the CSV. QHAS is attempted only when
        an exact ``<partition_name>_schematic.bin`` can be selected from the
        running EPContext model metadata.

        Args:
            artifacts: Mutable artifact map; receives the schematic path
                on success.
            qhas_override: When provided, skip the viewer shell-out and
                parse this JSON directly.  Used by
                :py:meth:`parse_existing_artifacts`.
        """
        if qhas_override is not None:
            # Offline path: caller supplied the QHAS JSON; parse directly.
            try:
                qhas_available = qhas_override.is_file()
            except OSError as exc:
                logger.info("QNNMonitor: qhas_override %s is unavailable: %s", qhas_override, exc)
                return None, None, None, TraceFallbackReason.QHAS_OUTPUT_MISSING
            if not qhas_available:
                logger.info("QNNMonitor: qhas_override %s is not a file", qhas_override)
                return None, None, None, TraceFallbackReason.QHAS_OUTPUT_MISSING
            result_path = qhas_override
        else:
            # Live path: locate inputs and shell out to the QHAS viewer.
            try:
                qnn_log = self._select_fresh_qnn_log()
            except OSError as exc:
                logger.info("QNNMonitor: QNN log metadata unavailable: %s", exc)
                return None, None, None, TraceFallbackReason.QNN_LOG_MISSING
            if qnn_log is None:
                logger.info("QNNMonitor: no *_qnn.log found for QHAS")
                return None, None, None, TraceFallbackReason.QNN_LOG_MISSING

            # Find the schematic by EPContext partition metadata (never chdir).
            schematic = self._find_schematic()
            if schematic is None:
                logger.info("QNNMonitor: no *_schematic.bin found for QHAS")
                return None, None, None, TraceFallbackReason.SCHEMATIC_MISSING

            try:
                sdk_root = find_qnn_sdk()
            except OSError as exc:
                logger.info("QNNMonitor: QNN SDK discovery failed: %s", exc)
                return None, None, None, TraceFallbackReason.SDK_MISSING
            if sdk_root is None:
                logger.info("QNNMonitor: QNN SDK not located; skipping QHAS")
                return None, None, None, TraceFallbackReason.SDK_MISSING
            schematic = self._publish_schematic(schematic)
            if schematic is None:
                return None, None, None, TraceFallbackReason.SCHEMATIC_PUBLISH_FAILED

            qhas_output = self._qhas_output_path()
            viewer_result = run_qhas_viewer_result(
                qnn_log,
                schematic,
                qhas_output,
                sdk_root=sdk_root,
            )
            if viewer_result.path is None:
                logger.info(
                    "QNNMonitor: QHAS viewer unavailable (%s)",
                    viewer_result.failure_reason,
                )
                return None, None, None, viewer_result.failure_reason
            result_path = viewer_result.path

            artifacts["schematic"] = str(schematic)

        try:
            qhas_data = json.loads(result_path.read_text(encoding="utf-8"))
            parsed = parse_qhas(qhas_data)
        except Exception as exc:
            logger.warning("QNNMonitor: QHAS JSON parse failed: %s", exc)
            return None, None, None, TraceFallbackReason.QHAS_PARSE_FAILED

        # QHAS is inherently a single-snapshot summary (no per-sample
        # breakdown), so ``samples_us`` carries one entry equal to the
        # aggregated ``duration_us``.  This keeps downstream p90 / total
        # / count rendering consistent with the basic-CSV path.
        #
        # The QHAS dict's ``"name"`` field carries the QHAS-authoritative
        # ``qnn_op_type`` (e.g. ``"Conv2d"``).  Pass it as the L2 input
        # to the resolver so:
        # - L1 wins when the ONNX map is populated and contains op_path.
        # - L2 (qnn_op_type) wins when the ONNX map is empty/missing the path.
        # - L3/L4 are unreachable here because op["name"] is always truthy
        #   from the QHAS JSON.
        operators = [
            OperatorMetrics(
                name=self._resolve_op_type(op["op_path"], ep_authoritative=op["name"]),
                op_path=op["op_path"],
                duration_us=op["duration_us"],
                percent_of_total=op["percent_of_total"],
                dominant_path_us=op.get("dominant_path_us"),
                num_htp_ops=op.get("num_htp_ops"),
                dram_read_bytes=op.get("dram_read_bytes"),
                dram_write_bytes=op.get("dram_write_bytes"),
                vtcm_read_bytes=op.get("vtcm_read_bytes"),
                vtcm_write_bytes=op.get("vtcm_write_bytes"),
                vtcm_hit_ratio=op.get("vtcm_hit_ratio"),
                samples_us=[op["duration_us"]],
            )
            for op in parsed.get("operators", [])
        ]
        return parsed.get("summary"), operators, result_path, None

    def _snapshot_qnn_log_signatures(self) -> dict[Path, tuple[int, int, int, int, int]]:
        """Capture this run's QNN log metadata at monitor entry."""
        candidate = self._qnn_log_path()
        try:
            signature = self._artifact_signature(candidate)
        except OSError as exc:
            logger.info("QNNMonitor: unable to snapshot QNN log metadata: %s", exc)
            signature = None
        signatures: dict[Path, tuple[int, int, int, int, int]] = {}
        if signature is not None:
            signatures[candidate.resolve()] = signature
        return signatures

    def _select_fresh_qnn_log(self) -> Path | None:
        """Return this run's QNN log, derived from the profiling CSV path."""
        candidate = self._qnn_log_path()
        signature = self._artifact_signature(candidate)
        if signature is None:
            return None

        resolved = candidate.resolve()
        previous_signature = self._qnn_log_signatures_at_enter.get(resolved)
        if previous_signature == signature:
            return None

        if self._monitor_enter_time_ns is not None:
            # Reject files copied into the directory after entry but carrying
            # an old filesystem mtime from a previous monitor run.
            mtime_ns = signature[3]
            if mtime_ns < self._monitor_enter_time_ns - 5_000_000_000:
                return None

        return candidate

    def _qnn_log_path(self) -> Path:
        """QNN EP names the log by appending ``_qnn.log`` to the CSV stem."""
        return self._csv_path.with_name(f"{self._csv_path.stem}_qnn.log")

    def _qhas_output_path(self) -> Path:
        """Per-run QHAS JSON path paired with the profiling CSV stem."""
        return self._csv_path.with_name(f"{self._csv_path.stem}_qhas_output.json")

    def _find_schematic(self) -> Path | None:
        """Locate the schematic bound to this run's EPContext partition."""
        try:
            if not self._csv_path.is_file():
                return None
        except OSError as exc:
            logger.info(
                (
                    "QNNMonitor: unable to read profiling CSV metadata "
                    "for schematic discovery (%s): %s"
                ),
                self._csv_path,
                exc,
            )
            return None

        partition_name = self._schematic_partition_name()
        if partition_name is None:
            return None

        search_dirs: list[Path] = []
        if self._running_model_path is not None:
            search_dirs.append(self._running_model_path.parent)
        search_dirs.append(Path.cwd())

        seen: set[Path] = set()
        for search_dir in search_dirs:
            resolved_dir = search_dir.resolve()
            if resolved_dir in seen:
                continue
            seen.add(resolved_dir)
            candidate = search_dir / f"{partition_name}_schematic.bin"
            schematic = self._schematic_candidate(candidate)
            if schematic is not None:
                return schematic
        return None

    def _publish_schematic(self, schematic: Path) -> Path | None:
        """Copy a run-bound schematic beside this monitor's profiling artifacts."""
        destination = self._csv_path.with_name(f"{self._csv_path.stem}_schematic.bin")
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            if schematic.resolve() != destination.resolve():
                shutil.copy2(schematic, staging)
                os.link(staging, destination)
        except OSError as exc:
            logger.warning(
                "QNNMonitor: could not publish schematic %s to %s: %s",
                schematic,
                destination,
                exc,
            )
            return None
        finally:
            try:
                staging.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(
                    "QNNMonitor: could not remove schematic staging file %s: %s",
                    staging,
                    exc,
                )
        return destination

    def _schematic_partition_name(self) -> str | None:
        """Resolve the exact EPContext partition name for schematic lookup."""
        if self._running_model_path is None:
            return None
        try:
            return select_main_epcontext_partition_name(self._running_model_path)
        except Exception as exc:
            logger.info(
                "QNNMonitor: unable to inspect EPContext partition metadata from %s: %s",
                self._running_model_path,
                exc,
            )
            return None

    @staticmethod
    def _schematic_candidate(candidate: Path) -> Path | None:
        """Return candidate when the exact run-bound schematic exists."""
        try:
            if not candidate.is_file():
                return None
        except OSError as exc:
            logger.debug(
                "QNNMonitor: schematic candidate %s unavailable: %s",
                candidate,
                exc,
            )
            return None
        return candidate

    def _make_failure_result(self, status: TraceStatus, error: str | None) -> OpTraceResult:
        """Build a minimal ``OpTraceResult`` for parse-time failures."""
        return OpTraceResult(
            model=None,
            device="npu",
            tracing_level=self._level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=[],
            summary={},
            artifacts={"csv": str(self._csv_path)},
            status=status,
            error=error,
        )
