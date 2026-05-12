# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNNMonitor — Qualcomm NPU per-op profiler via ORT's QNN EP.

Produces an :class:`OpTraceResult` with per-operator cycle counts
(``level="basic"``) or full QHAS roofline / DMA traffic
(``level="detail"``).

Contributes session options and provider options to a ``WinMLSession`` via
the two :class:`EPMonitor` hooks; owns the ``profiling_level`` and
``profiling_file_path`` provider-option keys (C-3 in PRD — never
user-overridable). Requires ``ort.InferenceSession`` teardown before
``__exit__`` because QNN EP flushes the profiling CSV only on session
destruction.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .ep_monitor import EPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult, TraceStatus
from .qnn._internal import _TOKEN_SUFFIX, parse_qhas, parse_qnn_profiling_csv
from .qnn.viewer import find_qnn_sdk, run_qhas_viewer


if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self


logger = logging.getLogger(__name__)


# Maps user-facing level to QNN EP's `profiling_level` provider option.
_LEVEL_TO_PROFILING: dict[str, str] = {
    "basic": "detailed",
    "detail": "optrace",
}


class QNNMonitor(EPMonitor):
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
        self._csv_path: Path = (self._output_dir / "profiling_output.csv").resolve()
        self._extra: dict[str, str] = dict(extra_provider_options or {})
        self._entered: bool = False
        self._result: OpTraceResult | None = None
        # v2.4: ONNX node.name -> node.op_type map injected by WinMLSession.perf
        # before __enter__. Populated only when an ONNX graph is available;
        # remains empty for the standalone parsing case (parse_existing_artifacts
        # without an onnx_op_types argument). Drives L1 of the fallback chain
        # in :py:meth:`_resolve_op_type`.
        self._onnx_op_types: dict[str, str] = {}

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
        2. ``onnxruntime-windowsml``: call
           :func:`ep_registry.ensure_initialized` to trigger WinML EP
           registration, then look for a QNN device in
           :func:`onnxruntime.get_ep_devices`.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        if "QNNExecutionProvider" in ort.get_available_providers():
            return True

        # WinML-registered path.
        try:
            from ..ep_registry import ensure_initialized
        except ImportError:
            return False

        try:
            ensure_initialized()
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

    def get_session_options(self) -> dict[str, str]:
        """Session config entries required for QNN op-tracing.

        Only EPContext caching is opted into here — embed_mode=0 keeps the
        compiled binary external so the cached ONNX stays small.

        ``session.disable_cpu_ep_fallback`` is intentionally NOT set: under
        ``onnxruntime-windowsml`` the WinML-registered QNN partitions a
        QDQ-wrapped EPContext model into Q/DQ-on-CPU + EPContext-on-QNN,
        which is correct behaviour (the boundary Q/DQ ops genuinely run on
        CPU). Disabling CPU fallback would reject that valid partition and
        cause NotImplemented errors even when QNN successfully claimed the
        EPContext node. The "no silent CPU fallback" guarantee is provided
        by ``add_provider_for_devices`` upstream — if the QNN device is
        absent, session creation fails loudly there.
        """
        return {
            "ep.context_enable": "1",
            "ep.context_embed_mode": "0",
        }

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
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Parse whatever artifacts are on disk. Never suppresses caller exceptions."""
        self._result = self._parse_artifacts_safe()
        # Implicit None return → does not suppress caller exception.

    def _parse_artifacts_safe(self, qhas_override: Path | None = None) -> OpTraceResult:
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
        """Override the EPMonitor no-op default — QNN uses the map.

        Stores the ONNX ``node.name -> node.op_type`` map for use during
        ``__exit__`` parsing.  ``WinMLSession.perf`` calls this once,
        immediately before ``__enter__``, with the map built from the
        ONNX graph passed to the session.

        Defensively copies the input so later caller mutation cannot
        corrupt the resolver's L1 lookup table.  Empty / no-graph input
        is a valid no-op: the resolver simply falls through to L2/L3/L4.
        """
        self._onnx_op_types = dict(onnx_op_types)

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
        # The constructor pinned _csv_path to "<output_dir>/profiling_output.csv";
        # honour the caller's explicit path instead so this works for fixtures
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
        meta = parsed.get("metadata", {})
        artifacts: dict[str, str] = {"csv": str(csv_path)}

        # Convert cycles to microseconds via the CSV-reported ratio.
        total_cycles = int(meta.get("accel_execute_cycles", 0) or 0)
        accel_us = int(meta.get("accel_execute_us", 0) or 0)
        cycle_to_us = accel_us / total_cycles if total_cycles > 0 else 0.0

        # CSV path: no EP-authoritative op type column, so resolve via
        # L1 (ONNX) then fall through to L3 (heuristic) then L4 (raw).
        operators: list[OperatorMetrics] = [
            OperatorMetrics(
                name=self._resolve_op_type(op["op_path"], ep_authoritative=None),
                op_path=op["op_path"],
                op_id=op.get("op_id"),
                duration_us=op["cycles"] * cycle_to_us,
                percent_of_total=((op["cycles"] / total_cycles * 100) if total_cycles > 0 else 0.0),
                samples_us=[c * cycle_to_us for c in op.get("samples_cycles", [])],
            )
            for op in parsed.get("operators", [])
        ]

        summary: dict[str, Any] = {
            "hvx_threads": meta.get("hvx_threads", 0),
            "accel_execute_cycles": total_cycles,
            "accel_execute_us": accel_us,
        }

        status: TraceStatus = "ok"
        # Detail mode: attempt QHAS post-processing.
        if self._level == "detail":
            qhas_summary, qhas_operators, qhas_path = self._try_qhas(
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
            num_samples=int(meta.get("num_samples", 0) or 0),
            artifacts=artifacts,
            status=status,
        )

    def _try_qhas(
        self,
        artifacts: dict[str, str],
        qhas_override: Path | None = None,
    ) -> tuple[dict[str, Any] | None, list[OperatorMetrics] | None, Path | None]:
        """Attempt QHAS post-processing.

        Returns ``(summary, operators, qhas_path)`` on success, or
        ``(None, None, None)`` on any failure. Never raises.

        Per C-5 / FR-12 this method does NOT call :func:`os.chdir`. The
        ``*_schematic.bin`` is located via :py:meth:`Path.glob` in the
        output directory first, then the process CWD as a read-only
        fallback.

        Args:
            artifacts: Mutable artifact map; receives the schematic path
                on success.
            qhas_override: When provided, skip the viewer shell-out and
                parse this JSON directly.  Used by
                :py:meth:`parse_existing_artifacts`.
        """
        if qhas_override is not None:
            # Offline path: caller supplied the QHAS JSON; parse directly.
            if not qhas_override.is_file():
                logger.debug("QNNMonitor: qhas_override %s is not a file", qhas_override)
                return None, None, None
            result_path = qhas_override
        else:
            # Live path: locate inputs and shell out to the QHAS viewer.
            qnn_logs = list(self._output_dir.glob("*_qnn.log"))
            if not qnn_logs:
                logger.debug("QNNMonitor: no *_qnn.log found for QHAS")
                return None, None, None
            qnn_log = qnn_logs[0]

            # Find the schematic (glob, never chdir).
            schematic = self._find_schematic()
            if schematic is None:
                logger.debug("QNNMonitor: no *_schematic.bin found for QHAS")
                return None, None, None

            sdk_root = find_qnn_sdk()
            if sdk_root is None:
                logger.debug("QNNMonitor: QNN SDK not located; skipping QHAS")
                return None, None, None

            qhas_output = self._output_dir / "qhas_output.json"
            result_path = run_qhas_viewer(qnn_log, schematic, qhas_output, sdk_root=sdk_root)
            if result_path is None or not result_path.is_file():
                logger.debug("QNNMonitor: QHAS viewer produced no output")
                return None, None, None

            artifacts["schematic"] = str(schematic)

        try:
            qhas_data = json.loads(result_path.read_text(encoding="utf-8"))
            parsed = parse_qhas(qhas_data)
        except Exception as exc:
            logger.warning("QNNMonitor: QHAS JSON parse failed: %s", exc)
            return None, None, None

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
        return parsed.get("summary"), operators, result_path

    def _find_schematic(self) -> Path | None:
        """Locate ``*_schematic.bin`` without mutating CWD.

        Search order:

        1. :attr:`_output_dir` (where ``profiling_file_path`` points).
        2. Process CWD (glob-only; no :func:`os.chdir`) — the QNN SDK
           occasionally drops the schematic next to the process's current
           directory rather than next to the profiling CSV.

        The CWD fallback is **mtime-gated** against the profiling CSV: a
        schematic from a prior CI run sitting in CWD would otherwise be
        silently consumed and produce QHAS metrics for the wrong graph
        with ``status="ok"`` — silent data corruption. The schematic must
        be at least as new as the CSV (with a 5s tolerance for filesystem
        clock skew) to be accepted.
        """
        candidates = list(self._output_dir.glob("*_schematic.bin"))
        if candidates:
            return candidates[0]
        # Fallback: read-only glob of process CWD. No chdir.
        # Reject stale schematics older than the profiling CSV.
        csv_mtime = self._csv_path.stat().st_mtime if self._csv_path.is_file() else 0.0
        fresh = [
            p for p in Path.cwd().glob("*_schematic.bin") if p.stat().st_mtime >= csv_mtime - 5.0
        ]
        if fresh:
            logger.warning(
                "QNNMonitor: located *_schematic.bin in CWD (%s) rather than output dir (%s)",
                fresh[0].parent,
                self._output_dir,
            )
            return fresh[0]
        return None

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
