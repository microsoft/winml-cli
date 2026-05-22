# QNN Op-Type Resolution — Design Spec

**Initial Date:** 2026-05-03
**Last Revised:** 2026-05-08
**Version:** 2.0.1
**Status:** Spec — implementation pending
**Branch:** `feat/op-tracing-refactor`
**Anchor commit:** `bb3e2a91` (v1.0 of this spec)
**Companion docs:**
- `docs/design/perf/2026-04-28-console-mockup-design.md` — render-layer spec (v2.1)
- `docs/design/perf/2026-04-29-op-tracing-production-lift-plan.md` — T1-T8 lift plan (executed)
- `docs/design/perf/2026-05-01-op-tracing-production-lift-summary.md` — outcome summary
- `docs/design/session/monitor/1_prd.md` v2.4 — PRD that consumes this spec
- `docs/design/session/monitor/2_coreloop.md` v2.4 — companion core-loop design

### Revision history

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-05-03 | Initial draft (commit `bb3e2a91`) — separate `QNNOpTraceParser` companion class. |
| v1.1 | 2026-05-03 | Revised per user feedback: monitor IS the parser (multiple inheritance); delete `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules; concrete docstring examples; `_token_N` strip is a QNN-internal concern with worked examples. |
| v1.2 | 2026-05-06 | Added §3.4 Identifier mapping reference: 3-identifier table, bridge explanation, class responsibilities table, 4 worked walkthroughs, design notes. Consolidates info that was previously spread across §3, §5, §6. |
| v2.0 | 2026-05-08 | **Major simplification.** Drop the `OpTraceParser` ABC entirely — premature abstraction for a single concrete implementer (QNNMonitor); multiple inheritance + MRO ordering + a separate ABC file is too much complexity for one EP. Wait for the second op-tracing EP before extracting an abstraction. Also drop `to_dict()` from the `WinMLEPMonitor` ABC contract — it conflated op-tracing telemetry (QNN) with proof-of-execution signals (VitisAI/OpenVINO) under one polymorphic interface, which is dishonest. Replacement: extend the existing `WinMLEPMonitor` ABC with two concrete-default members (`set_onnx_op_types` no-op default, `result` property returning `None`); QNNMonitor stays single-inheritance and owns ONNX-graph lookup + fallback chain as private internals; `commands/perf.py` switches to isinstance-based typed accessor dispatch. Spec is now scoped to "QNN op-type resolution via ONNX graph lookup, implemented privately on QNNMonitor". |
| v2.0.1 | 2026-05-08 | Doc-review fixes: restored .strip() + trailing-slash safety in _heuristic_op_type pseudocode (§3.2, §4.1) per design-review C-6. |

## 1. Goal

Document the QNN op-type resolution mechanism — ONNX-graph `node.name → node.op_type` lookup with a four-layer fallback chain — as a **QNNMonitor-private** detail. The op-type field on each `OperatorMetrics` is resolved primarily from an injected ONNX-graph map; EP-authoritative fields (`qhas.qnn_op_type`) and a QNN-specific heuristic (`_token_N` strip + leaf-split) serve as fallbacks. The map is built once at session setup by `WinMLSession` and injected via a new concrete-default `WinMLEPMonitor.set_onnx_op_types(map)` hook. Non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) inherit the no-op default and are unaffected.

The v2.0 simplification rationale: the v1.x design introduced a separate `OpTraceParser` ABC implemented by `QNNMonitor` via multiple inheritance. After review, that abstraction was deemed premature for a single concrete implementer — multiple inheritance, MRO ordering, and a separate ABC file are too much complexity for one EP. When a second op-tracing EP (TensorRT, OpenVINO) lands, the abstraction can be extracted from the two concrete implementers; until then, the resolution logic lives on `QNNMonitor` as private methods.

The architectural principle is unchanged: **strict information hiding around the QNN module**. Nothing about CSV/QHAS parsing, `_token_N` suffix stripping, `qnn_op_type` field names, or QNN-compiler glue conventions leaks outside the QNN module. The only shapes visible to callers are the `WinMLEPMonitor` ABC, the concrete `QNNMonitor` class with its typed `result` accessor, and the canonical `OperatorMetrics` / `OpTraceResult` dataclasses.

## 2. Current state (post-`b77043a1`)

The op-tracing production lift (T1-T8 + four fix-ups, captured in `2026-05-01-op-tracing-production-lift-summary.md`) wired up `OperatorMetrics.samples_us` plus derived `p90_us / total_us / sample_count / avg_us`, rebuilt the basic and detail render layers per the mockup, and threaded QHAS-authoritative `qnn_op_type` through to the `name` field in detail mode (`b77043a1`). What works correctly today:

- **Canonical shape**: `OperatorMetrics` (in `src/winml/modelkit/session/monitor/op_metrics.py`) is the single output type of any parsing path. Its `name` field is documented as "QNN op type" but is in practice the user-facing **Type** column value, regardless of EP.
- **Per-sample retention**: basic-mode (CSV) parser preserves `samples_cycles` per op (`csv_parser.py:264-265`), and `qnn_monitor._parse_artifacts` converts those to `samples_us` for downstream stats (`qnn_monitor.py:316`).
- **Authoritative op-type in detail mode**: `_transform_op` reads `qhas.qnn_op_type` directly (`qhas_parser.py:97`) — no leaf-splitting in detail mode. `c3ac3d45` introduced the leaf-split helper for the basic-mode CSV path; `b77043a1` ensured detail mode bypasses it.

What is scattered and needs consolidation:

| Concern | Current location |
|---|---|
| CSV → list of partial dicts | `qnn/csv_parser.py::parse_qnn_profiling_csv` (clean) |
| Heuristic op-type recovery from event ID | `qnn/csv_parser.py::_split_op_event_id` (`csv_parser.py:39-77`) |
| `_token_N` suffix stripping | `qnn/csv_parser.py:220` (regex applied inside `_parse_node_event`) |
| QHAS JSON → list of partial dicts | `qnn/qhas_parser.py::parse_qhas` (clean) |
| Authoritative `qnn_op_type` extraction | inline in `_transform_op` (`qhas_parser.py:96-99`) |
| CSV partial dict → `OperatorMetrics` | inline list-comp in `qnn_monitor.py:309-319` |
| QHAS partial dict → `OperatorMetrics` | inline list-comp in `qnn_monitor.py:403-419` |
| Cycle-to-microsecond conversion (basic) | inline in `qnn_monitor.py:305-307` |
| Cycle-to-microsecond conversion (detail) | inline in `qhas_parser.py:33-35` |
| Mode dispatch (basic vs. detail) | inline `if self._level == "detail":` in `qnn_monitor.py:328-338` |

There is no abstraction for "given EP artifacts, produce `list[OperatorMetrics]`". Every concern that should belong to that abstraction (mode dispatch, op-type resolution, dict→dataclass plumbing) is fused into `_parse_artifacts` and `_try_qhas` on `QNNMonitor`. The scattering is manageable for one EP; it becomes a scaling problem the moment a second EP needs the same pipeline.

There is also no single place where op-type resolution is governed. Today the resolution rule is implicit: basic mode uses the leaf-split heuristic, detail mode uses `qhas.qnn_op_type`, neither path looks at the ONNX graph at all. That is the gap this spec closes.

A second gap, surfaced in v1.1: the QNN parsing helpers are reachable as public-ish module-level imports (`from winml.modelkit.session.monitor.qnn.csv_parser import _aggregate_operators` is a real test import today). That broadcasts QNN internals far past their containing module, and unit tests against private helpers couple test suites to QNN's CSV vocabulary — `_token_N`, `op_id`, `qnn_op_type` — that no other EP shares. The revised plan shuts those imports off.

## 3. Design

Three pieces, all narrower than v1.x:

1. **`WinMLEPMonitor` ABC extension** — two new concrete-default members. `set_onnx_op_types(map)` is a no-op by default; `result` is a property returning `None` by default. Op-tracing monitors override the former and populate `self._result` to make the latter meaningful. Non-op-tracing monitors inherit the defaults and ignore both. The ABC contract also drops `to_dict()` (see §3.4).
2. **`QNNMonitor` (single inheritance)** — `class QNNMonitor(WinMLEPMonitor)`. ALL QNN-specific concerns (CSV reading, QHAS reading, sample aggregation, `_token_N` stripping, leaf-split heuristic, op-type resolution chain) live as private methods on the monitor (or in a private sibling submodule `qnn/_internal.py`, see §7). Nothing leaks outside the QNN module. The monitor exposes a typed `result` property (`OpTraceResult | None`) for downstream consumers.
3. **`WinMLSession` ONNX-map injection** — at `perf().__enter__`, the session unconditionally calls `monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))` on every monitor. The no-op default makes the call safe for non-op-tracing monitors; QNNMonitor overrides to store the map.

There is no `OpTraceParser` ABC. There is no multiple inheritance. There is no `to_dict()` on the ABC.

### 3.1 WinMLEPMonitor extension (additive, concrete-default)

```python
# session/monitor/ep_monitor.py — additions to existing ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .op_metrics import OpTraceResult


class WinMLEPMonitor(ABC):
    """Per-EP observer attached to a WinMLSession for an inference window."""

    requires_session_teardown: ClassVar[bool] = False

    # ---- Existing optional config hooks (defaulted to {}) ----
    def get_session_options(self) -> dict[str, str]:
        return {}

    def get_provider_options(self) -> dict[str, str]:
        return {}

    # ---- NEW (v2.0): ONNX op-type map injection ----
    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Inject the ONNX node.name -> node.op_type map.

        Default: no-op. Op-tracing monitors override this to store the
        map for use during their __exit__ parsing pass. Non-op-tracing
        monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor)
        inherit this default and ignore the call.

        WinMLSession calls this unconditionally on every monitor before
        __enter__. Idempotent; the last value wins.
        """
        pass

    # ---- NEW (v2.0): typed op-trace result accessor ----
    @property
    def result(self) -> "OpTraceResult | None":
        """Wrapped op-trace result. None for monitors that don't produce it.

        Op-tracing monitors set ``self._result`` during ``__exit__`` after
        parsing artifacts; the default getattr returns None for monitors
        that never set it. The default implementation works for any
        subclass — no override needed unless the subclass wants to
        compute the result lazily.
        """
        return getattr(self, "_result", None)

    # ---- Mandatory contract (existing, unchanged) ----
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

Two concrete-default members are introduced; nothing on the existing contract is removed except `to_dict()` (see §3.4 for that removal). The defaults make the call sites in `WinMLSession.perf()` and `commands/perf.py` polymorphism-safe without forcing every monitor to opt in.

### 3.2 QNNMonitor (single inheritance)

QNNMonitor stays a plain `WinMLEPMonitor` subclass. The op-type resolution logic is its own private business.

```python
import re
from pathlib import Path
from typing import ClassVar, Self

from .ep_monitor import WinMLEPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult


class QNNMonitor(WinMLEPMonitor):
    """QNN per-op profiler. Owns CSV/QHAS parsing + ONNX op-type lookup
    as private internals.

    Produces an OpTraceResult exposed via the typed ``result`` property
    (inherited from WinMLEPMonitor; populated during ``__exit__``).
    """

    requires_session_teardown: ClassVar[bool] = True

    # _token_N suffix is a QNN-compiler artefact (token-position-tagged
    # repeats of the same op). Stripped before the ONNX-graph lookup so
    # event IDs like "/encoder/conv1/Conv_token_1_2" match ONNX node
    # names like "/encoder/conv1/Conv". This regex stays inside the QNN
    # module — it is not a general-purpose op-name normaliser.
    _TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")

    def __init__(
        self,
        level: Literal["basic", "detail"] = "basic",
        output_dir: Path | None = None,
        extra_provider_options: Mapping[str, str] | None = None,
        onnx_op_types: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._level = level
        # ... output_dir, csv_path, qhas_path setup ...
        self._onnx_op_types: dict[str, str] = dict(onnx_op_types or {})
        self._result: OpTraceResult | None = None

    # -- Override no-op default: QNN does use the map --------------------
    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Override the WinMLEPMonitor no-op default — QNN consumes the map."""
        self._onnx_op_types = dict(onnx_op_types)

    # -- Lifecycle (existing, unchanged) ---------------------------------
    @classmethod
    def is_available(cls) -> bool: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Parse artifacts based on ``self._level``; populate ``self._result``.

        Does NOT return True — does not suppress caller exceptions.
        """
        artifacts = (
            {"qhas": self._qhas_path, "csv": self._csv_path}
            if self._level == "detail"
            else {"csv": self._csv_path}
        )
        ops = (
            self._parse_detail(artifacts)
            if self._level == "detail"
            else self._parse_basic(artifacts)
        )
        self._result = OpTraceResult(
            operators=ops,
            # ... summary, status, artifacts ...
        )
        # No `return True` — exceptions from the with body propagate.

    # -- Private parsing internals (NOT part of any ABC) -----------------

    def _parse_basic(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]:
        csv_path = artifacts.get("csv")
        if csv_path is None or not csv_path.is_file():
            return []
        rows = self._read_qnn_csv(csv_path)             # private
        samples = self._extract_samples(rows)            # private
        ops, meta = self._aggregate_operators(samples)   # private
        cycle_to_us = self._compute_cycle_ratio(meta)    # private
        return [
            self._to_operator_metrics(
                op, cycle_to_us,
                name=self._resolve_op_type(op["op_path"], ep_authoritative=None),
            )
            for op in ops
        ]

    def _parse_detail(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]:
        qhas_path = artifacts.get("qhas")
        if qhas_path is None or not qhas_path.is_file():
            return []
        ops = self._read_qhas(qhas_path)                 # private
        return [
            self._to_operator_metrics_detail(
                op,
                name=self._resolve_op_type(
                    op["op_path"], ep_authoritative=op["qnn_op_type"]
                ),
            )
            for op in ops
        ]

    def _resolve_op_type(
        self, op_path: str, ep_authoritative: str | None = None
    ) -> str:
        """Walk the four-layer fallback chain.

        L1: ONNX graph lookup by ``op_path`` (== node.name post-strip).
        L2: ``ep_authoritative`` (e.g. ``qhas.qnn_op_type``).
        L3: ``_heuristic_op_type(op_path)`` (QNN leaf-split).
        L4: Raw ``op_path``.

        Each layer is monotonic in quality: a higher layer's hit always
        wins. Empty/None at any layer falls through.
        """
        onnx_hit = self._onnx_op_types.get(op_path)
        if onnx_hit:
            return onnx_hit
        if ep_authoritative:
            return ep_authoritative
        heuristic = self._heuristic_op_type(op_path)
        if heuristic:
            return heuristic
        return op_path

    def _heuristic_op_type(self, op_path: str) -> str:
        """Heuristic-only fallback: leaf-split with strip safety.

        Preserves the strip semantics from the legacy _split_op_event_id helper:
        outer whitespace is stripped, inner whitespace around the leaf is stripped,
        and trailing-slash inputs fall back to the original (never empty).
        """
        cleaned = self._TOKEN_SUFFIX.sub("", op_path).strip()
        if "/" not in cleaned:
            return cleaned
        leaf = cleaned.rsplit("/", 1)[-1].strip()
        return leaf if leaf else cleaned  # trailing-slash → fall back to full
```

This preserves the strip-safety semantics of the legacy `_split_op_event_id`
helper at `src/winml/modelkit/session/monitor/qnn/csv_parser.py:39-77`,
covered by `tests/unit/session/monitor/qnn/test_event_id_split.py`. Do not
simplify these guards in the production implementation — the strip-safety
tests are part of the migration's preserve-behavior contract.

```python

    @classmethod
    def parse_existing_artifacts(
        cls,
        level: Literal["basic", "detail"],
        artifacts: dict[str, Path],
        onnx_op_types: dict[str, str] | None = None,
    ) -> OpTraceResult:
        """Standalone parsing entry point — no live benchmark required.

        Useful for offline analysis of pre-existing CSV/QHAS files. Builds
        a transient QNNMonitor instance, runs the appropriate private
        parse method, and returns the wrapped result.
        """
        instance = cls(
            level=level,
            output_dir=artifacts["csv"].parent if "csv" in artifacts else None,
            onnx_op_types=onnx_op_types,
        )
        ops = (
            instance._parse_detail(artifacts)
            if level == "detail"
            else instance._parse_basic(artifacts)
        )
        return OpTraceResult(operators=ops, ...)
```

The two private parse methods preserve the exact arithmetic and field-mapping that exists today — only the **call site of op-type resolution** is centralised on the private `_resolve_op_type` template. The CSV/QHAS reading primitives, today public-ish under `qnn/csv_parser.py` and `qnn/qhas_parser.py`, are folded into private methods (or a private submodule, see §7).

In `_parse_detail`, the QHAS-side `qnn_op_type` (today written into `op["name"]` per `b77043a1`) is passed as `ep_authoritative` rather than directly into `OperatorMetrics.name`. This is the key behavioural change: when the ONNX graph has an entry for the node, ONNX wins; when it does not (QNN compiler-inserted ops, fused glue, etc.), QHAS wins. Today QHAS always wins because the ONNX layer does not exist. See §5 for the failure-mode walkthrough.

### 3.3 Worked example: `_token_N` stripping bridges QNN event IDs to ONNX names

The `_token_N` suffix is something the QNN compiler appends to operator names when an op is replicated across token positions in the graph (sequence-model artefact). It is not part of the original ONNX node name. The `_heuristic_op_type` method — and, by implication, the lookup key passed into `_resolve_op_type` — uses the cleaned form so the L1 ONNX lookup can succeed.

**Match path** (ONNX graph contains the node):

```
QNN profiling event ID:  "/encoder/conv1/Conv_token_1_2"
After _token_N strip:    "/encoder/conv1/Conv"     <- used as ONNX map key
ONNX node.name:          "/encoder/conv1/Conv"     <- matches!
ONNX node.op_type:       "Conv"                    <- L1 winner
```

The parser passes `"/encoder/conv1/Conv"` (post-strip) to `_resolve_op_type`. L1 hits, returns `"Conv"`.

**Glue-op path** (QNN-compiler-inserted op, no ONNX equivalent):

```
QNN profiling event ID:  "_qnn_compiler_glue_Add_3"   (no slash, no ONNX equivalent)
After _token_N strip:    "_qnn_compiler_glue_Add_3"   (regex did not match)
ONNX map lookup:         miss
QHAS qnn_op_type:        "ElementWiseAdd"             <- L2 fallback wins
```

When detail mode is active, L2 carries the QHAS-authoritative QNN op type and wins. When only basic mode is active (no QHAS), L3 calls `_heuristic_op_type` which returns the cleaned leaf (here just the bare string).

The strip-then-lookup contract is invisible to all callers — it lives entirely inside `_heuristic_op_type` and the private CSV-reading helpers that build the lookup key. No other EP needs to know `_token_N` exists.

### 3.4 WinMLEPMonitor.to_dict() removal — typed accessors instead

The `to_dict()` method is removed from the `WinMLEPMonitor` ABC contract in v2.0. v1.x carried it as a polymorphic method that every concrete monitor implemented, but it was a god-method conflating two unrelated concerns:

- **Op-tracing telemetry** (QNN): per-operator cycle counts, samples, summary statistics. The honest payload for QNNMonitor.
- **Proof-of-execution signals** (VitisAI/OpenVINO): xrt-smi snapshots, device-utilisation deltas, "did the NPU actually run" boolean flags. The honest payload for those monitors.

Stuffing both under one `to_dict()` and writing both to the perf JSON output under a single `ep_proof` key is dishonest — only one of the two is genuinely "proof"; the QNN side is op-tracing data. The CLI consumer pipeline cannot distinguish the two without inspecting the dict contents.

v2.0 replaces the polymorphic god-method with **two typed accessor properties**, only one of which is implemented per concrete monitor:

| Concrete monitor | Typed accessor | Returns |
|---|---|---|
| `QNNMonitor` | `result` (inherited from WinMLEPMonitor; default `None`; populated in `__exit__`) | `OpTraceResult` |
| `VitisAIMonitor` | `proof` (NEW typed `ProofOfExecution` class — out of scope for this lift; flagged as a follow-up) | `ProofOfExecution \| None` |
| `OpenVINOMonitor` | `proof` (same pattern as VitisAI) | `ProofOfExecution \| None` |
| `NullEPMonitor` | neither (both `result` and `proof` return `None`) | n/a |

The CLI's `commands/perf.py` JSON-output flow switches from a single `monitor.to_dict()` call to isinstance-based typed accessor dispatch:

```python
# commands/perf.py — replaces the previous unified ctx.monitor.to_dict() call
if isinstance(ctx.monitor, QNNMonitor) and ctx.monitor.result is not None:
    perf_record["op_trace"] = ctx.monitor.result.to_dict()
elif isinstance(ctx.monitor, (VitisAIMonitor, OpenVINOMonitor)):
    proof = ctx.monitor.proof  # follow-up: typed ProofOfExecution
    if proof is not None:
        perf_record["ep_proof"] = proof.to_dict()
```

This separation is honest: the JSON keys reflect what the data actually is. QNN's payload goes under `op_trace`; VitisAI/OpenVINO's payload goes under `ep_proof`. NullEPMonitor contributes nothing.

The `OpTraceResult.to_dict()` method on the dataclass itself is unchanged — it stays at `op_metrics.py` and report writers continue to consume it. Only the polymorphic `WinMLEPMonitor.to_dict()` ABC method is removed.

### 3.5 Identifier mapping reference

This section consolidates, in one place, how the three identifiers in the system relate to each other and which class owns which concern. The same picture is implicit in §3.2 (QNNMonitor design), §5 (fallback chain), and §6 (map construction); this is the assembled reference.

#### 3.5.1 The three identifiers

| Identifier | Source | Example |
|---|---|---|
| **ONNX `node.name`** | Set by exporter (HF, PyTorch, etc.) when producing `model.onnx`. Hierarchical path or leaf-only depending on exporter. | `/convnext/embeddings/layernorm/LayerNormalization` |
| **ONNX `node.op_type`** | The canonical ONNX operator name from the ONNX spec (every node has exactly one). | `LayerNormalization` |
| **QNN event ID** (the "EP op-tracing item name") | What QNN emits during profiling. Compiler augments `node.name` with `_token_N_M` suffixes to disambiguate runtime instances. Sometimes a synthetic glue-op string with no ONNX origin. | `/convnext/embeddings/layernorm/LayerNormalization_token_5_2` |

#### 3.5.2 The bridge: `node.name` connects QNN runtime to ONNX static view

`node.name` is the bridge between the QNN runtime view (event IDs) and the ONNX static view (`op_type`). The QNN compiler reads the ONNX file at compile time, generates HTP code, and at runtime emits event IDs that are derived from `node.name` augmented with `_token_N_M` suffixes encoding the runtime instance. To go from a QNN event ID to its ONNX op type the parser does two things in order: strip the `_token_N_M` suffix, then look up the cleaned string in a `dict[node.name, node.op_type]` built once at session setup. If the lookup misses, control falls through the chain — QHAS-`qnn_op_type` (when detail mode is active), then leaf-split heuristic, then the raw event ID as a last resort.

```
[model.onnx file]
  ↓ exported by HF/PyTorch
  node.name = "/encoder/conv1/Conv"  (set by exporter)
  node.op_type = "Conv"              (canonical ONNX spec name)
  ↓ ↓ ↓
  QNN compiler reads ONNX, augments node.name with runtime suffixes
  ↓ ↓ ↓
[runtime: model executes on QNN HTP]
  QNN profiling event ID = "/encoder/conv1/Conv_token_3_1"  (suffixed)
```

#### 3.5.3 Class responsibilities

| Class | Knows about | Does NOT know about |
|---|---|---|
| **`WinMLSession`** | ONNX file paths, `onnx.load()`, model graph traversal. Builds `dict[node.name, node.op_type]` at session setup and injects via `monitor.set_onnx_op_types(map)` unconditionally on every monitor. | QNN profiling output, op-tracing data formats |
| **`WinMLEPMonitor` ABC** | Benchmark lifecycle (`__enter__`/`__exit__`), session options, EP probe failures, `ep_name`. NEW (v2.0): concrete-default `set_onnx_op_types` (no-op) and `result` property (returns `None`). | Specific EP profiling formats (CSV, JSON, etc.); fallback chain semantics |
| **`QNNMonitor(WinMLEPMonitor)`** | EVERYTHING QNN-specific: CSV format, QHAS JSON format, `_token_N` regex, `_extract_samples` accumulator, `qnn_op_type` field semantics, the four-layer fallback chain (`_resolve_op_type`), the QNN heuristic (`_heuristic_op_type`). Single inheritance from WinMLEPMonitor; overrides `set_onnx_op_types` to actually store the map and populates `self._result` in `__exit__`. | ONNX file loading (only consumes the prebuilt map; never calls `onnx.load`) |
| **`OperatorMetrics` dataclass** | Just data fields (`name`, `op_path`, `samples_us`, etc.). Pure transport object. | Where any value came from |
| **Render layer (`report.py`)** | How to draw a Rich Table | All of the above — only sees `OperatorMetrics` |

#### 3.5.4 Worked walkthroughs

Four concrete cases trace a single op through the resolution chain end-to-end. Case 1 overlaps with the example at the end of §3.2 — refer there for the `_token_N` strip mechanics; the focus here is the resolver flow.

**Case 1: Happy path — token-strip then ONNX hit**

- QNN emits: `event_id = "/encoder/conv1/Conv_token_3_1"`
- `QNNMonitor` strips `_token_N` → `"/encoder/conv1/Conv"` (this becomes `op_path`)
- `_resolve_op_type("/encoder/conv1/Conv", ep_authoritative=None)`:
  - ONNX map lookup → HIT → returns `"Conv"`
- `OperatorMetrics(name="Conv", op_path="/encoder/conv1/Conv", ...)`
- Render shows: Node=`/encoder/conv1/Conv`, Type=`Conv`

**Case 2: ONNX miss → QHAS authoritative wins** (compiler-glue op, detail mode)

- QNN emits: `event_id = "_qnn_glue_Add_3"`
- After strip: `"_qnn_glue_Add_3"` (unchanged — no `_token` suffix)
- `_resolve_op_type("_qnn_glue_Add_3", ep_authoritative="ElementWiseAdd")`:
  - ONNX map lookup → MISS (this op doesn't exist in the original graph)
  - `ep_authoritative` is set (QHAS provides `qnn_op_type = "ElementWiseAdd"` for this row) → returns `"ElementWiseAdd"`
- `OperatorMetrics(name="ElementWiseAdd", op_path="_qnn_glue_Add_3", ...)`

**Case 3: Both upstream miss → leaf-split heuristic** (unusual; basic mode + non-glue op)

- QNN emits: `event_id = "/some/odd/path/SomethingCustom"`
- After strip: same
- `_resolve_op_type("/some/odd/path/SomethingCustom", ep_authoritative=None)`:
  - ONNX map lookup → MISS
  - `ep_authoritative` is None (basic mode has no QHAS data)
  - Heuristic `_heuristic_op_type` runs leaf-split → `"SomethingCustom"`
- `OperatorMetrics(name="SomethingCustom", op_path="/some/odd/path/SomethingCustom", ...)`

**Case 4: Bare event** (Gelu pattern observed in real T9 run)

- QNN emits: `event_id = "Gelu"` (no path, no token suffix)
- After strip: `"Gelu"`
- `_resolve_op_type("Gelu", ep_authoritative=None)`:
  - ONNX map lookup → likely MISS (most exports give Gelu a path key like `/encoder/.../Gelu`); could be HIT in some exporters
  - Heuristic: no `/`, leaf-split returns `"Gelu"` itself
- `OperatorMetrics(name="Gelu", op_path="Gelu", ...)` — degenerate but correct

#### 3.5.5 Three design notes worth preserving

1. **The token-suffix strip is the load-bearing bridge.** Without it, every path-style event ID would miss the ONNX lookup (because `node.name` doesn't carry `_token_N`) and we'd always fall through to fallbacks. The strip MUST happen inside the QNN monitor, before the ONNX lookup, and the cleaned form is what gets stored as `op_path` (so the user-visible Node column matches `node.name`, which is what users see in Netron).

2. **`OperatorMetrics` is genuinely format-agnostic.** A dataclass instance carrying `name="Conv"` could equally have been produced by a QNN monitor, a TensorRT monitor, or a hand-built test fixture. Nothing downstream (render, JSON serializer, top-K filter) can tell the difference. That's the architectural payoff: render layer doesn't import anything from `qnn/`, can be unit-tested with synthetic data, and a future TensorRT EP doesn't risk regressing the QNN output.

3. **`op_path` stores the cleaned event ID, not the raw one.** If a user wants to cross-reference a slow op against their model in Netron, the cleaned form `/encoder/conv1/Conv` matches `node.name` exactly. The raw form `/encoder/conv1/Conv_token_3_1` would be useless for that workflow. The cleaned form serves the user's most likely follow-up action — this is a deliberate UX choice, not just a side-effect of the strip.

## 4. Naming convention: ONNX `node.op_type` verbatim

This section addresses the question: when the ONNX graph reports `node.op_type = "LayerNormalization"`, do we display `"LayerNormalization"` or do we translate to a shorter `"LayerNorm"` (the PyTorch class name)?

**Decision: verbatim. We display exactly what `node.op_type` says. No translation table.**

Rationale, in order of weight:

1. **Cardinal Rule #1** (project `CLAUDE.md`): "Never hardcode model architecture names, node/operator names, input/output tensor names, layer naming patterns, or any model-specific logic. All solutions must be universal and architecture-agnostic." A translation map of the form `{"LayerNormalization": "LayerNorm", "Conv": "Conv2d", ...}` is a hardcoded operator-vocabulary translation. Adding it would be an explicit Cardinal Rule violation. There is no universal, architecture-agnostic version of such a map; it is by construction a bag of arbitrary opinions about how to shorten ONNX symbols.

2. **`LayerNormalization` is the ONNX spec literal.** The op was added in opset 17 (December 2022) under exactly that name. It is not a typo or a verbose alias — it is the canonical ONNX op type. Netron, ORT's profiling output, `onnx.checker`, and every tool that surfaces ONNX graph contents all use `LayerNormalization`. Translating to `LayerNorm` is translating *away from* the canonical name, not toward it.

3. **No upstream invariant to anchor on.** Different models, different opset versions, and different toolchains use different op-type strings for nominally the same operation. `Conv` vs `Conv2d` (ONNX vs PyTorch nn class), `Gemm` vs `Linear` vs `MatMul` (different lowerings of the same dense layer), `LayerNormalization` vs `LayerNorm` (ONNX spec vs informal short form). Picking a "canonical" rendering across all of these requires either a maintained translation table (the Cardinal Rule problem) or a runtime heuristic that will inevitably mis-translate some model. Doing nothing — passing the ONNX symbol through — is the only stable choice.

4. **Mental model alignment.** Users debugging an ONNX model already have a copy of the model open in Netron or have run `onnx.load(...).graph.node`. They will see `LayerNormalization` in those contexts. Showing the same string in our op-tracing report is the obvious, non-surprising behaviour. Users whose mental model is PyTorch's `nn.LayerNorm` will adapt once; the cost is a one-time vocabulary reconciliation, not a per-model translation maintenance burden.

5. **Render-layer is the right place to handle width problems, not the parser.** If `LayerNormalization` is too wide for the 12-cell `Type` column in the basic-mode mockup, the answer is one of: (a) widen the column; (b) accept Rich's `overflow="ellipsis"` truncation, which gives `LayerNormali…`; (c) introduce a separate user-facing setting (`--short-op-names`) if and only if a real user actually requests it. None of these requires the parser to maintain a translation table. Today the column is locked at width=12 (per the mockup spec) and accepts ellipsis-truncation for `LayerNormalization` and similarly long ops; if that turns out to be unacceptable, the fix is in `report.py`, not here.

The same rule extends to QHAS-authoritative `qnn_op_type` strings (`Conv2d`, `ElementWiseAdd`, `PoolMax2d`) when those reach Layer 2 of the chain. We do not translate `ElementWiseAdd` to `Add` to match the ONNX symbol either. Whatever the chain produces — ONNX op type from Layer 1, QNN op type from Layer 2, leaf-split guess from Layer 3 — is what the render shows. The Type column is a "best available op-type label" column, not a normalised-vocabulary column.

If a future requirement emerges for showing both ONNX and QNN op types side-by-side (e.g. to debug fusion behaviour), that is a render-layer feature: a separate column for the QNN-authoritative value, surfaced only in detail mode where it exists. The parser's contract does not change. See §9.

### 4.1 `OperatorMetrics.name` docstring

The current docstring on `OperatorMetrics.name` reads `# QNN op type ("Conv2d", "LayerNorm")` (`op_metrics.py:41`). After this spec lands, the field is no longer QNN-specific. The Step-4 migration (§7) updates the docstring to:

```python
name: str
"""Op type. Sourced from ONNX ``node.op_type`` when the model graph
is available; falls back to EP-specific labels (e.g. QNN's
``qnn_op_type``) when the graph lookup misses. Use ONNX naming
verbatim — no translation tables.

Examples (ONNX, primary): "Conv", "LayerNormalization", "MatMul", "Gelu"
Examples (QNN fallback):  "Conv2d", "ElementWiseAdd", "PoolMax2d"
"""
```

The rule v1.0 phrased as "best-available op-type label per fallback chain" is too abstract on its own; concrete examples beat abstract description. The docstring carries both the rule and a worked vocabulary so a reader can predict what the field will hold without chasing the chain definition.

## 5. Fallback chain (QNNMonitor-internal)

The four-layer fallback chain is implemented as the private `QNNMonitor._resolve_op_type` method. There is no ABC-level contract for it — the chain semantics are a QNN concern, and a future op-tracing EP will own its own resolution chain (which may have different layers, different ordering, or different fallback sources). The chain is documented here as the canonical reference for how QNN does it.

Source priority for `OperatorMetrics.name`:

| Layer | Source | Available in |
|---|---|---|
| 1 | ONNX graph: `onnx_op_types[op_path]` | basic + detail (when ONNX file is reachable at session setup) |
| 2 | EP-authoritative field, e.g. `qhas.qnn_op_type` | detail mode only (basic mode CSV has no op-type column) |
| 3 | EP heuristic, e.g. `_token_N`-strip + leaf-split | basic mode (last-mile fallback) |
| 4 | Raw `op_path` (verbatim) | always (last-resort) |

Each layer absorbs the failure modes of the one above. Walk-through:

| Scenario | L1 ONNX | L2 EP-auth | L3 heuristic | Wins | Type column shows |
|---|---|---|---|---|---|
| Detail mode, ONNX file loaded, op present in graph | hit | hit | n/a | L1 | ONNX symbol (e.g. `LayerNormalization`) |
| Basic mode, ONNX file loaded, op present in graph | hit | n/a | hit | L1 | ONNX symbol |
| Detail mode, ONNX file loaded, op NOT in graph (compiler-inserted glue) | miss | hit | n/a | L2 | QNN op type (e.g. `ElementWiseAdd`, `Convert`) |
| Basic mode, ONNX file loaded, op NOT in graph | miss | n/a | hit | L3 | leaf-split guess |
| Detail mode, ONNX file unavailable (empty map) | miss | hit | n/a | L2 | QNN op type — same as today |
| Basic mode, ONNX file unavailable | miss | n/a | hit | L3 | leaf-split guess — same as today |
| Pathological: bare event ID with no `/`, no ONNX, no QHAS | miss | n/a | "" or `op_path` | L4 | raw `op_path` |

### 5.1 Worked examples

**Match-via-token-strip** — QNN profiling event ID is the QNN-compiler-tagged form, but the cleaned form matches ONNX:

```
QNN event ID:           "/encoder/conv1/Conv_token_1_2"
After _token_N strip:   "/encoder/conv1/Conv"
ONNX map lookup:        hit  -> node.op_type = "Conv"
Result (L1):            name = "Conv"
```

**QNN-glue fallback** — ONNX has no node by that name, so the chain proceeds:

```
QNN event ID:           "_qnn_compiler_glue_Add_3"
After _token_N strip:   "_qnn_compiler_glue_Add_3"  (regex did not match)
ONNX map lookup:        miss
Detail mode:            qhas.qnn_op_type = "ElementWiseAdd"  -> L2 wins
Basic mode:             heuristic -> "_qnn_compiler_glue_Add_3"  -> L3 wins (degenerate, == L4)
```

Two facts worth calling out:

- **Layer 2 (EP-authoritative) does not exist in basic mode.** The QNN basic-mode CSV emits event IDs that are either bare op symbols (`"Gelu"`) or hierarchical paths (`"/encoder/layer/Conv"`); there is no separate column carrying the QNN op type. So in basic mode, after an ONNX miss, control falls directly to the leaf-split heuristic.
- **Layer 3 is a strict subset of Layer 4 in degenerate cases.** When `op_path` is bare (no `/`), the heuristic returns the same string and the chain effectively skips to L4 with the same value. This is fine — it is just a different way of saying "we have nothing better to offer than the input string itself."

The chain is **monotonic in quality**: each layer is at least as good as the one below it. If a higher layer hits, it always wins; if it misses, the lower layer's answer is strictly no worse than no answer at all. This is the property that makes the chain safe to walk top-down without re-validating downstream.

## 6. Map construction

The ONNX op-type map is built at **session-setup time** by `WinMLSession`, not in the monitor. The monitor receives a fully-formed `dict[str, str]` via `set_onnx_op_types()`. This separation matters because (a) loading ONNX is an I/O operation that should happen once per session, not once per parse call, and (b) the call is uniform across all monitors — non-op-tracing monitors inherit the WinMLEPMonitor no-op default and silently ignore the map.

### 6.1 Builder lives on `WinMLSession`

```python
class WinMLSession:
    def perf(self, monitor: WinMLEPMonitor, ...):
        # Inject the ONNX op-type map unconditionally on every monitor.
        # The WinMLEPMonitor base class provides a no-op default for set_onnx_op_types,
        # so non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor)
        # safely ignore the call. QNNMonitor overrides to actually store the map.
        if self._onnx_path is not None:
            monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))
        # ... rest of perf flow ...

    @staticmethod
    def _build_op_type_map(onnx_path: Path | None) -> dict[str, str]:
        """Build a node.name -> node.op_type map from an ONNX file.

        Loads only graph metadata (``load_external_data=False``) so the
        cost is ~milliseconds even for multi-GB models with separate
        weight files. Returns an empty dict when the path is None,
        missing, or unparseable — the parser falls through the chain
        in that case.
        """
        if onnx_path is None:
            return {}
        p = Path(onnx_path)
        if not p.is_file():
            return {}
        try:
            import onnx
            model = onnx.load(str(p), load_external_data=False)
        except Exception:
            return {}
        return {n.name: n.op_type for n in model.graph.node if n.name}
```

The `load_external_data=False` flag is essential. For models like Qwen3-0.6B-Q8_0 (≈600MB of weights in a sidecar `.bin`) we only need the graph topology, never the tensors. With the flag set, loading is roughly proportional to the protobuf graph size — typically tens of milliseconds.

The `if n.name` filter is defensive: ONNX permits anonymous nodes (`node.name == ""`), and we cannot key the map on an empty string (multiple anonymous nodes would collide). Anonymous nodes are rare in production-exported models but do occur in hand-authored graphs and some legacy converters; quietly excluding them is correct — the parser will fall through to Layer 2/3/4 for those ops.

### 6.2 Why unconditional injection is safe

The v1.x design used `isinstance(monitor, OpTraceParser)` to decide whether to inject. v2.0 drops that branch. The WinMLEPMonitor `set_onnx_op_types` default is a concrete no-op, so calling it on any monitor (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) is a safe no-op. QNNMonitor overrides to do real work. This is simpler than the isinstance branch and slightly more honest — every monitor sees the same call, every monitor decides for itself whether the call means anything.

The ONNX-map construction itself is gated only on `self._onnx_path is not None`; if the session was constructed without an ONNX path (rare), no map is built and no injection happens.

Constructor injection — `QNNMonitor(..., onnx_op_types=session._build_op_type_map(...))` — remains supported as an alternative for tests and standalone scripts that want to skip the WinMLSession layer entirely (see `QNNMonitor.parse_existing_artifacts` in §3.2).

### 6.3 Failure handling

`_build_op_type_map` never raises. If `onnx_path` is `None`, missing, or fails to load, it returns `{}`. The parser still works with an empty map — it just falls through to Layer 2/3/4 for every op. This preserves today's behaviour as the "no-ONNX" baseline. Critically, no part of the production path is allowed to *block* on ONNX-map availability: profiling must still produce a usable report when (e.g.) the ONNX file was deleted between export and benchmark.

Logging: a single `logger.debug("ONNX op-type map: %d entries from %s", len(m), p)` at construction time gives operators visibility without spamming the console. No warning when the map is empty; the parser's behaviour in that case is well-defined and documented.

## 7. Migration plan

### 7.1 Code-location migration

The principle: **strict information hiding around the QNN module**. Nothing about CSV/QHAS parsing leaks out. The `qnn/csv_parser.py` and `qnn/qhas_parser.py` modules — currently 283 and 122 lines respectively, both reachable as public-ish imports — are deleted in their current form. Their helpers either fold into private methods on `QNNMonitor` (option a) or migrate into a private sibling submodule `qnn/_internal.py` with no public exports (option b).

| Current location (`bb3e2a91`) | New home | Notes |
|---|---|---|
| `WinMLEPMonitor` ABC (existing) | **EXTENDED**: add concrete-default `set_onnx_op_types(map) -> None` (no-op) and `result` property (returns `getattr(self, "_result", None)`). | The two new members are concrete defaults — no abstract methods added. |
| `WinMLEPMonitor.to_dict` abstract method (existing) | **REMOVED** from the ABC contract. Concrete monitors expose data via typed accessors instead (`result` for op-tracing, `proof` for proof-of-execution). | See §3.4. |
| `NullEPMonitor.to_dict` (existing, returns `{}`) | **REMOVED** — the `result` property inherits the WinMLEPMonitor default and returns `None`, which is the honest answer. | NullEPMonitor exposes no data. |
| `QNNMonitor.to_dict` (existing, delegates to `result.to_dict()`) | **REMOVED** — callers go directly through `monitor.result.to_dict()` on the typed `OpTraceResult` accessor. | The `result` property already exists in production today (`qnn_monitor.py`); it stays. |
| `VitisAIMonitor.to_dict`, `OpenVINOMonitor.to_dict` (existing) | **KEEP for now, as transitional**. Document them as transitional. Follow-up PR introduces a typed `proof` property + a new `ProofOfExecution` class to replace these. Out of scope for this lift. | Flagged as OQ-6 in the PRD. |
| `commands/perf.py:542,549` (currently calls `ctx.monitor.to_dict()` to produce the `ep_proof` JSON field) | **REWRITE** to isinstance-based typed accessor dispatch (see §3.4 code sample). QNN's payload routes to `op_trace`; VitisAI/OpenVINO's payload routes to `ep_proof` (preserving the existing schema for those EPs while the typed `proof` follow-up lands). | This is the consumer-side fix for the to_dict god-method. |
| `src/winml/modelkit/session/monitor/qnn/csv_parser.py` (entire 283-line module) | **DELETED**. Helpers move to private methods on `QNNMonitor` or to `qnn/_internal.py` (private). | No public exports survive. |
| `src/winml/modelkit/session/monitor/qnn/qhas_parser.py` (entire 122-line module) | **DELETED**. Helpers move to private methods on `QNNMonitor` or to `qnn/_internal.py`. | Same. |
| `src/winml/modelkit/session/monitor/qnn/__init__.py` | **DELETED if empty** after the merge, or kept as a package marker with no public exports. The QNN viewer (`qnn/viewer.py`) is unaffected — it stays but remains an internal sibling, not re-exported. | |
| `qnn_monitor.py:309-319` (CSV → `OperatorMetrics` list-comp) | Body of `QNNMonitor._parse_basic` (private) | Identical arithmetic; only `name=` changes from `op["name"]` to `self._resolve_op_type(op["op_path"], None)`. |
| `qnn_monitor.py:403-419` (QHAS → `OperatorMetrics` list-comp, in `_try_qhas`) | Body of `QNNMonitor._parse_detail` (private) | Identical arithmetic; `name=` changes to `self._resolve_op_type(op["op_path"], ep_authoritative=op["qnn_op_type"])`. |
| `qnn/csv_parser.py::_split_op_event_id` (`csv_parser.py:39-77`) | Folded into `QNNMonitor._heuristic_op_type` (see §3.2 sketch) | The leaf-split is QNN business; no other EP needs it. |
| `qnn/csv_parser.py::_TOKEN_SUFFIX` regex / `_token_N` strip (`csv_parser.py:220`) | Stays QNN-private. Lives on `QNNMonitor` as a class attribute, applied inside `_heuristic_op_type` and inside the CSV row-reading helpers when building the L1 lookup key. | |
| `qnn/csv_parser.py::_aggregate_operators`, `_extract_samples`, `_extract_metadata`, `_parse_node_event` | Move to `qnn/_internal.py` as private (`_`-prefixed) module-level functions, OR onto `QNNMonitor` as private methods. | These are CSV-specific primitives; nothing outside the QNN module imports them after the refactor. |
| `qnn/qhas_parser.py::_transform_op` (reading `qnn_op_type`) | Moves to `qnn/_internal.py` as `_transform_qhas_op` (private), OR onto `QNNMonitor`. The "`name = qnn_op_type`" rule moves into `_resolve_op_type` (Layer 2). | The function keeps `qnn_op_type` in its output dict but the resolver is the authority on which field becomes `OperatorMetrics.name`. |
| `qnn_monitor.py::_parse_artifacts` mode dispatch (`if self._level == "detail":`) | Stays on the monitor as the `__exit__`-side dispatcher; calls the private `_parse_basic` / `_parse_detail`. | The mode dispatch is preserved; the dispatch *target* is now a single private method per mode. |
| `qnn_monitor.py::_try_qhas` (QHAS-viewer + parse) | Split: viewer-invocation stays on the monitor (it's device/SDK lifecycle), parsing migrates into `_parse_detail`. | Once `qhas_output.json` exists on disk, the path is handed to `_parse_detail` via `artifacts={"qhas": ...}`. |
| `QNNMonitor.parse_existing_artifacts` (NEW classmethod) | Public entry point on the QNN monitor for offline analysis of pre-existing CSV/QHAS files. | Replaces the v1.x abstract `parse_basic`/`parse_detail` interface; useful for tests and ad-hoc scripts. |
| `tests/unit/session/monitor/qnn/test_csv_parser_samples.py` (imports `_aggregate_operators`) | **DELETE** — test of a now-private helper. Replace with integration tests on `QNNMonitor.parse_existing_artifacts(level="basic", ...)` exercising the same fixtures end-to-end. | Architectural debt cleanup. |
| `tests/unit/session/monitor/qnn/test_event_id_split.py` (imports `_split_op_event_id`) | **DELETE** — test of a now-private helper. Coverage moves to `_heuristic_op_type` unit tests on `QNNMonitor`. | |
| `tests/unit/session/monitor/qnn/test_csv_parser.py` (imports `parse_qnn_profiling_csv`) | **DELETE or rewrite** — replace with `QNNMonitor.parse_existing_artifacts(level="basic", ...)` integration tests using the same CSV fixtures. | |
| `tests/unit/session/monitor/qnn/test_qhas_parser.py` (imports `parse_qhas`) | **DELETE or rewrite** — replace with `QNNMonitor.parse_existing_artifacts(level="detail", ...)` integration tests using the same QHAS fixtures. | |
| `tests/unit/session/monitor/test_qnn_monitor.py:549` (imports `parse_qhas` for ad-hoc parsing in a test) | Refactor to call `QNNMonitor.parse_existing_artifacts(level="detail", ...)` directly. | |

### 7.2 Option (a) vs. option (b) for helper placement

Two ways to honour "nothing leaks out":

- **Option (a): everything folds into `qnn_monitor.py`.** All CSV / QHAS / token-strip helpers become private methods on `QNNMonitor`. File grows from current ~700 lines to ~1000+ lines. One file to read.
- **Option (b): private sibling submodule `qnn/_internal.py`.** The ~378 lines of CSV/QHAS parsing move into a private submodule; `qnn_monitor.py` imports the helpers using a relative import. Submodule has no public exports (no `__all__`, no re-exports from `__init__.py`).

**Recommendation: option (b).** Folding 378 lines of CSV/QHAS parsing into the monitor file would push it past 1000 lines and conflate two concerns (device lifecycle + artifact parsing) in a single read. A private sibling module keeps the monitor file focused on lifecycle while still satisfying the information-hiding directive — the rule is "no module *outside the QNN package* imports parsing internals", not "no module at all". The submodule's privacy is enforced by convention (`_internal.py` filename, `_`-prefixed function names, no `__init__.py` re-exports) rather than by Python access control, which is the standard idiom in this codebase.

In either option, the architectural rule is the same and is testable by grep:

> **No module outside `src/winml/modelkit/session/monitor/qnn/` imports anything from the qnn parsing internals.** The single permitted importer of `qnn/_internal.py` (option b) is `qnn_monitor.py`.

A CI lint or a one-off `tests/unit/architecture/test_qnn_imports.py` regression test that scans the source tree and asserts this property would prevent regressions.

### 7.3 Refactor sequence (TDD-checkpointed)

The migration is a refactor: existing logic is preserved and relocated. Each step ends with `uv run pytest tests/` green.

| Step | Change | Tests |
|---|---|---|
| 1 | Extend `WinMLEPMonitor` ABC with concrete-default `set_onnx_op_types` (no-op) and `result` property (returns `getattr(self, "_result", None)`). Remove `to_dict` from the ABC. | New unit tests: defaults work for any subclass; `result` returns `None` when `_result` is unset; no abstract method added. |
| 2 | Add `WinMLSession._build_op_type_map` and the unconditional `monitor.set_onnx_op_types(...)` call in `WinMLSession.perf`. | New unit test: known node names from the resnet50 ONNX produce expected op types. Empty/missing path returns `{}`. The unconditional call is safe for NullEPMonitor/VitisAIMonitor/OpenVINOMonitor (no-op default). |
| 3 | In `QNNMonitor`: override `set_onnx_op_types` to actually store the map; add private `_resolve_op_type` and `_heuristic_op_type` methods; refactor existing `_parse_artifacts` to dispatch to private `_parse_basic` / `_parse_detail` methods that call `_resolve_op_type`. The two private parse methods initially still call into the existing `qnn/csv_parser.py` / `qnn/qhas_parser.py` modules — no behavioural change beyond the resolution call site. | All existing `test_qnn_monitor.py` tests pass unchanged. New unit tests: hand-built `onnx_op_types={}` + small fixtures → identical `OperatorMetrics` to the inline path; `_resolve_op_type` walks the chain correctly across the 8 hit/miss combinations. |
| 4 | Wire real ONNX-map construction in the CLI / session-setup paths. Update `OperatorMetrics.name` docstring per §4.1. Add the `QNNMonitor.parse_existing_artifacts` classmethod for standalone use. | New integration test: real ONNX + CSV + QHAS fixtures → ONNX-resolved `name` for nodes that exist in the graph; QHAS-resolved `name` for nodes that don't. |
| 5 | Move `_split_op_event_id` and `_TOKEN_SUFFIX` regex out of `qnn/csv_parser.py` and into `QNNMonitor._heuristic_op_type` (and the private CSV-reading helpers, where the strip happens before key-construction). | Add `_heuristic_op_type` unit tests on `QNNMonitor`. |
| 6 | Move CSV/QHAS reading primitives into `qnn/_internal.py` (option b) or onto `QNNMonitor` (option a). Delete `qnn/csv_parser.py` and `qnn/qhas_parser.py`. Empty `qnn/__init__.py` if no other public exports remain. | Existing integration tests on `QNNMonitor.parse_existing_artifacts` continue to pass. |
| 7 | Update `commands/perf.py:542,549` JSON-output flow from `ctx.monitor.to_dict()` to isinstance-based typed accessor dispatch (see §3.4 code sample). Remove `QNNMonitor.to_dict` and `NullEPMonitor.to_dict`. Leave `VitisAIMonitor.to_dict` / `OpenVINOMonitor.to_dict` as transitional. | New unit test: QNN's payload routes to `op_trace`; VitisAI/OpenVINO payloads route to `ep_proof`; NullEPMonitor contributes nothing. |
| 8 | Delete the four legacy unit-test files that import private helpers (`test_csv_parser.py`, `test_csv_parser_samples.py`, `test_event_id_split.py`, `test_qhas_parser.py`). Coverage is preserved by the integration tests added in Steps 4-6. Refactor `test_qnn_monitor.py:549` to call `QNNMonitor.parse_existing_artifacts` instead of `parse_qhas`. | Pytest still green; coverage report shows no regression on the QNN module. |
| 9 | Add architecture regression test: scan `src/` (excluding `qnn/`) and `tests/` and assert no import path matches `qnn.csv_parser`, `qnn.qhas_parser`, `qnn._internal` (or whichever forms remain post-merge). | The test fails if any future commit re-exposes a private QNN parsing helper. |

After Step 9, the migration is complete: all QNN parsing internals are private, the typed accessors replace the `to_dict` god-method, and the rule is enforced by CI.

### 7.4 What does NOT change

- `OperatorMetrics` and `OpTraceResult` dataclasses are unchanged. The parser's output type is exactly the same as today's `_parse_artifacts` output type (a `list[OperatorMetrics]`).
- The render layer (`report.py`) is unchanged. It already consumes `OperatorMetrics.name` as the Type-column value and is agnostic to which layer of the chain produced it.
- Public `wmk perf` CLI surface is unchanged. No new flags, no removed flags.
- `_LEVEL_TO_PROFILING` mapping (`qnn_monitor.py:45-48`) and the QNN provider-option contract (`profiling_level`, `profiling_file_path`) are unchanged.
- The QHAS viewer shell-out (`qnn/viewer.py`) is unchanged. Viewer invocation is a device/SDK lifecycle concern owned by the monitor; only artifact parsing moves into `_parse_detail`.

## 8. Testing

### 8.1 Strategy

Five test layers, each owning a different concern:

1. **WinMLEPMonitor ABC default tests** — `set_onnx_op_types(map)` is a no-op for any subclass that doesn't override; `result` property returns `None` when `_result` is unset and the populated value when set. Tests use a minimal `_TestMonitor(WinMLEPMonitor)` subclass.
2. **`QNNMonitor` private-method unit tests** — hand-built `onnx_op_types` dicts paired with small CSV/QHAS fixtures (no ONNX file, no QNN SDK). Asserts `OperatorMetrics.name` resolution at each chain layer for `_parse_basic` and `_parse_detail`. Calling these private methods directly is acceptable in unit tests because they live on the same class as the test target. The public `QNNMonitor.parse_existing_artifacts` classmethod gives a stable test entry point that doesn't require crossing private boundaries.
3. **`QNNMonitor` resolver + heuristic unit tests** — `_resolve_op_type` walks the chain correctly for all (L1 hit/miss) × (L2 hit/None) × (L3 hit/None) combinations. `_heuristic_op_type` covers the `_token_N` strip + leaf-split contract independently.
4. **Integration tests** — real ONNX file + real CSV + real QHAS fixtures (the resnet50 fixture already in the repo). Asserts that ONNX lookup wins over QHAS-authoritative when both have an entry, and that QHAS wins when ONNX misses.
5. **Architecture test** — scans the source tree and asserts no module outside `src/winml/modelkit/session/monitor/qnn/` imports any name from the QNN parsing internals. Pinned by Step 9 of the migration sequence.

The legacy primitive tests (`test_csv_parser.py` etc.) are deleted, not retained. Their coverage migrates to the integration tests in layers 2 and 4. This is intentional architectural cleanup: tests of private helpers were architectural debt, coupling the test suite to the private CSV vocabulary of one EP.

There are no `OpTraceParser` ABC contract tests in v2.0 — there is no `OpTraceParser` ABC. All resolver/heuristic tests are private-method tests on `QNNMonitor`.

### 8.2 Edge cases (must-have)

- **Empty `onnx_op_types`**: resolver still works, every op falls through to L2/L3/L4. No warnings, no crashes.
- **Empty `artifacts`**: `_parse_basic({})` and `_parse_detail({})` return `[]`. No exceptions.
- **Missing artifact files**: `artifacts={"csv": Path("/does/not/exist")}` returns `[]`. Logged at debug, not error.
- **Empty `op_path`**: `_resolve_op_type("", ...)` falls all the way through L4 and returns `""`. The render layer is responsible for displaying empty strings sensibly (today: it already handles em-dash fallback for empty cells).
- **Both ONNX and QHAS have entries for the same `op_path`** but they disagree: ONNX wins. Test asserts the rendered Type column shows the ONNX value.
- **Heuristic returns empty string**: chain treats empty as miss and falls through to L4.
- **`_token_N` cleaning**: a CSV event ID `"/encoder/conv1/Conv_token_1_2"` with an ONNX node `"/encoder/conv1/Conv"` resolves to `"Conv"` (L1 hit on the cleaned key).
- **`set_onnx_op_types` idempotence**: `set_onnx_op_types({})` followed by `set_onnx_op_types({"a": "Conv"})` results in the second map being authoritative.
- **`set_onnx_op_types` no-op default**: calling it on `NullEPMonitor`/`VitisAIMonitor`/`OpenVINOMonitor` does not raise and does not store anything visible.
- **`result` default**: `WinMLEPMonitor.result` returns `None` when `_result` was never set; returns the stored object when `_result` is populated.

### 8.3 Acceptance criteria

| AC | Description |
|---|---|
| P-1 | `WinMLEPMonitor.set_onnx_op_types({"a": "b"})` on a subclass that doesn't override is a no-op (no AttributeError, nothing stored visibly). |
| P-2 | `WinMLEPMonitor.result` is `None` for any subclass that doesn't set `self._result`; returns the value when set. |
| P-3 | `QNNMonitor._resolve_op_type` returns L1 value when L1 hits, regardless of L2/L3 availability. |
| P-4 | `QNNMonitor._resolve_op_type` returns L2 value when L1 misses and L2 is non-None. |
| P-5 | `QNNMonitor._resolve_op_type` returns L3 value when L1 misses and L2 is None and heuristic returns non-empty. |
| P-6 | `QNNMonitor._resolve_op_type` returns raw `op_path` when L1/L2/L3 all miss. |
| P-7 | `QNNMonitor._parse_basic` produces the same `OperatorMetrics` (modulo `name`) as today's `qnn_monitor.py:309-319` for the same CSV input. |
| P-8 | `QNNMonitor._parse_detail` produces the same `OperatorMetrics` (modulo `name`) as today's `qnn_monitor.py:403-419` for the same QHAS input. |
| P-9 | When the ONNX map has an entry for a node that QHAS also names, the rendered Type column shows the ONNX value. |
| P-10 | When the ONNX map misses a node that QHAS names (e.g. `ElementWiseAdd`/`Convert` glue), the rendered Type column shows the QHAS value. |
| P-11 | `WinMLSession._build_op_type_map(None)`, `_build_op_type_map(<missing>)`, and `_build_op_type_map(<corrupt>)` all return `{}` without raising. |
| P-12 | `WinMLSession._build_op_type_map(<resnet50.onnx>)` returns a non-empty dict whose keys include known node names from the fixture. |
| P-13 | `QNNMonitor._heuristic_op_type("/encoder/conv1/Conv_token_1_2")` returns `"Conv"`. |
| P-14 | All existing tests in `tests/unit/session/monitor/` and `tests/unit/commands/` pass unchanged after migration steps 1-4. (Steps 5-8 actively delete the four legacy primitive test files; coverage migrates to integration tests added in earlier steps.) |
| P-15 | Architecture test passes: no module outside `src/winml/modelkit/session/monitor/qnn/` imports any name from `qnn.csv_parser`, `qnn.qhas_parser`, or `qnn._internal` (whichever survives). |
| P-16 | `commands/perf.py` JSON output: QNN runs produce an `op_trace` key (sourced from `monitor.result.to_dict()`); VitisAI runs produce an `ep_proof` key; NullEPMonitor runs produce neither. |

## 9. Forward-compat: future EP subclasses

Sketches only; not implementations. Each future EP defines (a) what artifacts its monitor produces, (b) which levels it supports, and (c) what its EP-authoritative op-type field is named, if any. Each future EP's monitor is a plain `WinMLEPMonitor` subclass — single inheritance — that overrides `set_onnx_op_types` to store the map and populates `self._result` from `__exit__` so the typed `result` accessor works. Same information-hiding rule applies: nothing about TRT-specific JSON shape, OV-specific layer-type enums, etc., leaks out of the EP's containing module. There is no `OpTraceParser` ABC to mix in.

When a second op-tracing EP lands, the abstraction can be re-extracted from the two concrete implementers (TensorRTMonitor + QNNMonitor share resolver/heuristic shape). Until then, each EP owns its own resolver as private internals.

### 9.1 TensorRT (NVIDIA GPU)

```python
class TensorRTMonitor(WinMLEPMonitor):
    """TensorRT JSON profiler output -> OperatorMetrics via private resolver.

    TRT emits a JSON array per inference with ``layerName``,
    ``averageMs``, ``percentage``, ``tactic``. ``layerName`` is the
    TRT-internal kernel/layer name and is *not* the ONNX op type
    (TRT fuses aggressively). When the ONNX map has an entry for
    layerName, ONNX wins; otherwise we surface the TRT layer name.
    """

    def set_onnx_op_types(self, m: dict[str, str]) -> None:
        self._onnx_op_types = dict(m)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ops = self._parse_basic(...) if self._level == "basic" else self._parse_detail(...)
        self._result = OpTraceResult(operators=ops, ...)

    # Private internals (mirroring QNNMonitor's shape):
    #   _parse_basic, _parse_detail
    #   _resolve_op_type(layer_name, ep_authoritative=tactic_or_None)
    #   _heuristic_op_type — TRT-specific (or returns "" — chain falls through to L4)
```

### 9.2 OpenVINO (Intel CPU/GPU/NPU) — op-tracing variant

If/when OpenVINO gains op-tracing (separate from the proof-of-execution monitor introduced today), the same pattern applies:

```python
class OpenVinoOpTraceMonitor(WinMLEPMonitor):
    """OpenVINO benchmark_app -pcseq CSV -> OperatorMetrics via private resolver.

    OV emits ``layer_type`` directly as a column (``Convolution``,
    ``Eltwise``, ``ReduceMean``, etc.). Use it as ep_authoritative
    in the resolver chain.
    """

    def set_onnx_op_types(self, m: dict[str, str]) -> None:
        self._onnx_op_types = dict(m)

    # _parse_basic uses _resolve_op_type(layer_name, ep_authoritative=layer_type)
```

Both EPs fit the same shape: override `set_onnx_op_types`, populate `self._result` in `__exit__`, keep parsing internals private. Adding a new op-tracing EP is a contained change — no modifications to the ABC, the render layer, the CLI factory, or `OperatorMetrics`. The naming convention from §4 applies uniformly: whatever `node.op_type` says is what shows; whatever the EP's own op-type field says is what shows when ONNX misses; no translation tables ever.

## 10. Resolved questions (was: open questions)

The original four open questions in v1.0 are all resolved. v1.x carried specific resolutions that involved the `OpTraceParser` ABC; v2.0 supersedes those with the simpler "everything QNN-private" answer.

1. **(was open) Injection point: monitor or parser-as-dependency?** **Resolved: monitor only.** v1.x answer was "the monitor IS the parser via multiple inheritance"; v2.0 answer is simpler — there is no parser, only the monitor. The ONNX op-type map is injected via `WinMLEPMonitor.set_onnx_op_types(map)`, which has a concrete no-op default on the ABC and is overridden by op-tracing monitors. Constructor injection remains supported (`QNNMonitor(..., onnx_op_types=...)`) for tests and standalone scripts.

2. **(was open) Should the parser own samples-aggregation? / What about `csv_parser.py` and `qhas_parser.py` as public modules?** **Resolved: delete `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules.** Their helpers fold either onto `QNNMonitor` directly (option a) or into a private sibling submodule `qnn/_internal.py` (option b, recommended). Sample-aggregation is QNN-internal and lives wherever the helpers land. Strict information hiding: NO module outside `src/winml/modelkit/session/monitor/qnn/` imports any parsing internal after the refactor. Existing unit tests of private helpers (`test_csv_parser_samples.py`, `test_event_id_split.py`, `test_csv_parser.py`, `test_qhas_parser.py`) are deleted as architectural debt; coverage moves to integration tests on `QNNMonitor.parse_existing_artifacts`. See §7.1, §7.2, §8.1.

3. **(was open) Should `OperatorMetrics.name`'s docstring change?** **Resolved: yes, with concrete examples.** New docstring (per §4.1):

   ```python
   name: str
   """Op type. Sourced from ONNX ``node.op_type`` when the model graph
   is available; falls back to EP-specific labels (e.g. QNN's
   ``qnn_op_type``) when the graph lookup misses. Use ONNX naming
   verbatim — no translation tables.

   Examples (ONNX, primary): "Conv", "LayerNormalization", "MatMul", "Gelu"
   Examples (QNN fallback):  "Conv2d", "ElementWiseAdd", "PoolMax2d"
   """
   ```

   The original phrasing in v1.0 ("best-available op-type label per fallback chain") was too abstract; concrete examples beat abstract description. Updated as part of Step 4 of the migration (§7.3).

4. **(was open) What happens when the ONNX map exists but `op_path` is from a `_token_N` suffix-stripped CSV row?** **Resolved: `_token_N` stripping is a QNN-internal concern.** The regex `_TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")` and its `.sub()` call live inside `QNNMonitor` (or `qnn/_internal.py`). The token-stripped string is what gets used as the lookup key for both the L1 ONNX map and the L3 heuristic, so a QNN event `"…/Conv_token_1_2"` correctly matches an ONNX node `"…/Conv"`. Worked examples at the end of §3.2 and in §5.1.

### 10.1 New open questions (genuinely emergent from the v2.0 redesign)

These need a one-line decision before the migration starts:

1. **Option (a) or (b) for helper placement?** Recommend (b) — a private `qnn/_internal.py` submodule. See §7.2 for the trade-off. The implementing engineer can flip to (a) if the submodule ends up being called from only one place and the indirection feels unnecessary; both options satisfy the information-hiding rule.

2. **Architecture regression test: enforce by import-scan or by type-checking?** §7.3 Step 9 proposes a Python-level test that scans imports. An alternative is a `mypy` / `ruff` rule. Recommend the import-scan test for now — it's a single self-contained file, doesn't require touching the project's lint configuration, and is easy to extend when future EPs land.

3. **Should `WinMLSession._build_op_type_map` move to a free function for testability?** It's currently a `@staticmethod`. A free function in (say) `session/perf/op_type_map.py` would let unit tests import and exercise it without instantiating the session. Recommend: keep as `@staticmethod` — it has no state, is trivially testable as `WinMLSession._build_op_type_map(...)`, and breaking it out adds one more module without much benefit.

4. **(v2.0) Typed `proof` accessor follow-up.** v2.0 removes `WinMLEPMonitor.to_dict()` from the ABC contract but leaves `VitisAIMonitor.to_dict` and `OpenVINOMonitor.to_dict` in place as transitional. The follow-up PR introduces a typed `proof` property + a new `ProofOfExecution` dataclass to cover those EPs honestly. Out of scope for this lift; flagged so it doesn't get lost.
