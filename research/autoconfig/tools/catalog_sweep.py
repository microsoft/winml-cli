# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""catalog_sweep.py — unified, JSON-driven EP/device optimization sweep.

Single driver that replaces the per-EP ``catalog_{cpu,gpu,qnn}_sweep.py`` scripts.
Everything EP/device-specific is read from
``ep_device_knowledge/<ep>_<device>.json``:

  - ``sweep_config``  : quant/compile policy, screen/full bench protocol,
                        confirmation, effect-size gate, thermal-awareness,
                        accuracy eval, paired-A/B availability, timeouts.
  - ``hypotheses``    : the (id, label, opset, optim, guard) matrix.
  - ``models``        : the model catalog (id, task, model_type).
  - ``cross_checks``  : cross-hypothesis finding probes (opset_bypass /
                        catastrophic_regression / regression_probe).

Per-hypothesis guards:
  - ``skip_if_gemm``        : skip the hypothesis if the built model already has
                              Gemm nodes (cpu-002 — matmul_add_fusion is harmful).
  - ``conv_pct_regression`` : annotate the hypothesis as an expected regression
                              when Conv% of the baseline build exceeds a threshold
                              (npu-006 — FusedConv falls back to CPU on QNN NPU).

Bench protocol (config-driven):
  Phase A : screen (``screen.iters``); on thermal_aware EPs high CV is logged but
            never blocks Phase B (the multi-session cool-down is the thermal control).
  Phase B : ``full.sessions`` x ``full.iters`` with cool-down.
  Phase C : ``confirm_sessions`` extra sessions on the best hypothesis; CONFIRMED
            only when all session p50s fall strictly below the baseline range.

Usage:
    python tools/catalog_sweep.py --ep qnn --device npu
    python tools/catalog_sweep.py --ep cpu --device cpu --model microsoft/resnet-18
    python tools/catalog_sweep.py --ep qnn --device npu --only-hypotheses h6,h7 --paired-ab
    python tools/catalog_sweep.py --ep qnn --device gpu --list

Results: <results_dir>/<model_slug>/{results.json, report.html, champion_<ep>_<device>.json}, SUMMARY.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Agent package bootstrap: make the autoconfig root importable for sibling packages.
_AGENT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "ep_device_knowledge").is_dir()
)
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

try:
    from lib.gen_model_report import generate_model_report  # noqa: E402
except Exception:
    generate_model_report = None

try:
    from skills.optimizer.bench_utils import (  # noqa: E402
        adaptive_paired_ab_bench,
        run_perf_session,
    )
except Exception:
    adaptive_paired_ab_bench = None
    run_perf_session = None


sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

KB_DIR = _AGENT_ROOT / "ep_device_knowledge"
WINML = str(_AGENT_ROOT / ".venv" / "Scripts" / "winml.exe")

_OK = ("OK", "OK_HIGH_CV")


# ── small perf-json helpers ────────────────────────────────────────────────────


def _latency(perf_json: Path) -> tuple[float | None, float | None]:
    """Return (p50, cv) parsed from a winml perf JSON, or (None, None)."""
    try:
        d = json.loads(perf_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", d)
        p50 = float(lat.get("p50") or 0)
        std = float(lat.get("std") or 0)
        if p50 <= 0:
            return None, None
        return p50, std / p50
    except Exception:
        return None, None


def _median(values: list[float]) -> float:
    return float(sorted(values)[len(values) // 2])


def _session_cv(p50s: list[float]) -> float:
    """Session-to-session CV (std/mean) — the run-to-run noise floor."""
    n = len(p50s)
    if n < 2:
        return 0.0
    mean = sum(p50s) / n
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in p50s) / (n - 1)
    return (var**0.5) / mean


class CatalogSweep:
    """JSON-driven sweep driver for one (ep, device) combination."""

    def __init__(
        self, ep: str, device: str, paired_ab: bool = False, prune_artifacts: bool = False
    ) -> None:
        kb_path = KB_DIR / f"{ep}_{device}.json"
        if not kb_path.exists():
            raise SystemExit(f"ERROR: knowledge base not found: {kb_path}")
        self.ep = ep
        self.device = device
        self.kb = json.loads(kb_path.read_text(encoding="utf-8"))
        self.cfg = self.kb["sweep_config"]
        self.hyps: list[dict] = self.kb["hypotheses"]
        self.models: list[dict] = self.kb["models"]
        self.cross_checks: list[dict] = self.kb.get("cross_checks", [])

        self.results_dir = _AGENT_ROOT / self.cfg["results_dir"]
        self.screen = self.cfg["screen"]
        self.full = self.cfg["full"]
        self.timeouts = self.cfg["timeouts"]
        self.baseline_id = (self.cfg.get("baseline_priority") or ["h0"])[0]
        self.prune_artifacts = prune_artifacts
        self.paired_ab = paired_ab and self.cfg.get("paired_ab_available", False)
        if paired_ab and adaptive_paired_ab_bench is None:
            print(
                "  [warn] --paired-ab requested but bench_utils unavailable — disabled", flush=True
            )
            self.paired_ab = False

    # ── subprocess ─────────────────────────────────────────────────────────────

    def run_cmd(self, cmd: list[str], label: str = "", timeout: int = 300) -> tuple[int, str]:
        t0 = time.monotonic()
        print(f"  >> {label or ' '.join(cmd[:3])}", flush=True)
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.monotonic() - t0
            tag = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            print(f"     {elapsed:.0f}s [{tag}]", flush=True)
            if r.returncode != 0 and r.stderr.strip():
                print(f"     stderr: {r.stderr.strip()[:200]}", flush=True)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            print(f"     TIMEOUT after {timeout}s", flush=True)
            return -1, "TIMEOUT"

    # ── config / build ─────────────────────────────────────────────────────────

    def _patch_config(self, cfg: dict) -> dict:
        """Apply quant/compile policy from sweep_config to a base config."""
        cfg = copy.deepcopy(cfg)
        if not self.cfg.get("quant"):  # False/None => strip; "auto" => keep
            cfg["quant"] = None
        if not self.cfg.get("compile"):
            cfg["compile"] = None
        return cfg

    def get_base_config(self, model_id: str, task: str, model_type: str) -> dict | None:
        tmp = self.results_dir / "_tmp_base_config.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)

        def _try(extra: list[str]) -> dict | None:
            cmd = [
                WINML,
                "config",
                "-m",
                model_id,
                "-t",
                task,
                "--ep",
                self.ep,
                "--device",
                self.device,
            ]
            if not self.cfg.get("compile"):
                cmd += ["--no-compile"]
            cmd += ["-o", str(tmp)] + extra
            rc, out = self.run_cmd(
                cmd, label=f"winml config --ep {self.ep}", timeout=self.timeouts["config_s"]
            )
            if rc == 0 and tmp.exists():
                try:
                    cfg = json.loads(tmp.read_text(encoding="utf-8"))
                    tmp.unlink(missing_ok=True)
                    return cfg
                except Exception as e:
                    print(f"  [warn] config parse error: {e}", flush=True)
            # Fallback: some builds print the config as a JSON line on stdout.
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except Exception:
                        pass
            tmp.unlink(missing_ok=True)
            return None

        cfg = _try(["--model-type", model_type])
        if cfg is None:
            print("  [warn] config with --model-type failed, retrying without…", flush=True)
            cfg = _try([])
        return self._patch_config(cfg) if cfg is not None else None

    @staticmethod
    def make_hypothesis_config(base: dict, opset: int | None, optim: dict | None) -> dict:
        cfg = copy.deepcopy(base)
        if opset is not None and cfg.get("export"):
            cfg["export"]["opset_version"] = opset
        if optim:
            cfg["optim"] = {**(cfg.get("optim") or {}), **optim}
        return cfg

    def run_build(
        self,
        model_id: str,
        cfg_path: Path,
        out_dir: Path,
        build_flags: list[str] | None = None,
    ) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            WINML,
            "build",
            "-c",
            str(cfg_path),
            "-m",
            model_id,
            "-o",
            str(out_dir),
            "--ep",
            self.ep,
            "--device",
            self.device,
        ]
        if not self.cfg.get("quant"):
            cmd += ["--no-quant"]
        if not self.cfg.get("compile"):
            cmd += ["--no-compile"]
        if build_flags:
            cmd += list(build_flags)
        cmd += ["--rebuild"]
        rc, out = self.run_cmd(
            cmd, label=f"winml build [{out_dir.name}]", timeout=self.timeouts["build_s"]
        )
        return rc == 0, out

    # ── bench / eval ───────────────────────────────────────────────────────────

    def bench_screen(self, onnx: Path) -> tuple[float | None, float, bool]:
        out_json = onnx.parent / "screen_perf.json"
        rc, _ = self.run_cmd(
            [
                WINML,
                "perf",
                "-m",
                str(onnx),
                "--ep",
                self.ep,
                "--device",
                self.device,
                "--warmup",
                str(self.screen["warmup"]),
                "--iterations",
                str(self.screen["iters"]),
                "-o",
                str(out_json),
            ],
            label=f"perf screen ({self.screen['iters']} iters)",
            timeout=self.timeouts["bench_s"],
        )
        if rc != 0 or not out_json.exists():
            return None, 999.0, False
        p50, cv = _latency(out_json)
        if p50 is None:
            return None, 999.0, False
        stable = cv <= self.screen["cv_max"]
        if self.screen.get("thermal_aware") and not stable:
            tag = "HIGH-CV (DVFS noise — proceeding to Phase B)"
        else:
            tag = "stable" if stable else "high-CV"
        print(f"     screen: p50={p50:.2f}ms  CV={cv:.3f}  [{tag}]", flush=True)
        return p50, cv, stable

    def bench_full(self, onnx: Path) -> list[float]:
        p50s: list[float] = []
        n, cd = self.full["sessions"], self.full["cool_down_s"]
        for s in range(1, n + 1):
            out_json = onnx.parent / f"full_perf_s{s}.json"
            rc, _ = self.run_cmd(
                [
                    WINML,
                    "perf",
                    "-m",
                    str(onnx),
                    "--ep",
                    self.ep,
                    "--device",
                    self.device,
                    "--warmup",
                    str(self.full["warmup"]),
                    "--iterations",
                    str(self.full["iters"]),
                    "-o",
                    str(out_json),
                ],
                label=f"perf full s{s}/{n} ({self.full['iters']} iters)",
                timeout=self.timeouts["bench_s"],
            )
            p50, cv = _latency(out_json) if rc == 0 and out_json.exists() else (None, None)
            if p50 is not None:
                print(f"     full s{s}: p50={p50:.2f}ms  CV={cv:.3f}", flush=True)
                p50s.append(p50)
            else:
                print(f"     [warn] full bench s{s} failed", flush=True)
            if s < n:
                print(f"     cool-down {cd}s…", flush=True)
                time.sleep(cd)
        return p50s

    def run_eval(self, onnx: Path, model_id: str, task: str) -> float | None:
        out_json = onnx.parent / "eval_result.json"
        rc, _ = self.run_cmd(
            [
                WINML,
                "eval",
                "-m",
                str(onnx),
                "--model-id",
                model_id,
                "--task",
                task,
                "--ep",
                self.ep,
                "--device",
                self.device,
                "--samples",
                str(self.cfg["eval_samples"]),
                "-o",
                str(out_json),
            ],
            label="winml eval (accuracy gate)",
            timeout=self.timeouts["eval_s"],
        )
        if rc != 0 or not out_json.exists():
            return None
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
            acc = data.get("metrics", data).get("accuracy")
            if acc is not None:
                print(f"     eval accuracy: {float(acc):.4f}", flush=True)
            return float(acc) if acc is not None else None
        except Exception:
            return None

    # ── onnx introspection (guards) ─────────────────────────────────────────────

    @staticmethod
    def _model_has_gemm(onnx_path: Path) -> bool:
        try:
            import onnx  # noqa: PLC0415

            m = onnx.load(str(onnx_path))
            return any(n.op_type == "Gemm" for n in m.graph.node)
        except Exception:
            return False

    @staticmethod
    def _conv_pct(onnx_path: Path) -> tuple[float, int, int]:
        """Return (conv_pct, conv_count, total). (0.0, 0, 0) means UNKNOWN, not SAFE."""
        if not onnx_path.exists():
            return 0.0, 0, 0
        try:
            import onnx  # noqa: PLC0415

            ops = [n.op_type for n in onnx.load(str(onnx_path)).graph.node]
            total = len(ops)
            conv = sum(1 for o in ops if o == "Conv")
            return (round(conv / total * 100, 1) if total else 0.0), conv, total
        except Exception:
            return 0.0, 0, 0

    @staticmethod
    def _find_onnx(hyp_dir: Path) -> Path | None:
        for name in ("model.onnx", "quantized.onnx", "optimized.onnx"):
            if (hyp_dir / name).exists():
                return hyp_dir / name
        ctx = list(hyp_dir.glob("*_ctx*.onnx")) + list(hyp_dir.glob("model_npu*.onnx"))
        return ctx[0] if ctx else None

    @staticmethod
    def _op_signature(hyp_dir: Path) -> dict | None:
        """Read the post-optimize op inventory (total_operators + operator_counts +
        opset) that ``winml build`` emits to ``analyze_result.json``. This is the
        ground truth for whether an optim/fusion flag actually changed the graph, so
        a hypothesis can be diffed against the baseline build. Returns None when the
        analyze artifact is missing or malformed."""
        f = hyp_dir / "analyze_result.json"
        if not f.exists():
            return None
        try:
            md = (json.loads(f.read_text(encoding="utf-8")) or {}).get("metadata") or {}
        except Exception:
            return None
        total = md.get("total_operators")
        counts = md.get("operator_counts")
        if total is None or counts is None:
            return None
        return {
            "total_operators": total,
            "operator_counts": dict(counts),
            "opset": md.get("opset_version"),
        }

    @staticmethod
    def _same_graph(a: dict | None, b: dict | None) -> bool:
        """True when two op signatures are identical (same opset, total op count and
        per-op-type counts) — i.e. the flag under test was a NO-OP versus baseline."""
        if not a or not b:
            return False
        return (
            a.get("opset") == b.get("opset")
            and a.get("total_operators") == b.get("total_operators")
            and a.get("operator_counts") == b.get("operator_counts")
        )

    # ── per-model sweep ─────────────────────────────────────────────────────────

    def sweep_model(
        self,
        model_id: str,
        task: str,
        model_type: str,
        only_hyp_ids: set[str] | None = None,
        reuse_baseline_config: bool = False,
    ) -> dict:
        model_slug = model_id.replace("/", "--")
        model_dir = self.results_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        results_path = model_dir / "results.json"
        if only_hyp_ids and results_path.exists():
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
                print("  [resume] loaded existing results", flush=True)
            except Exception:
                results = {}
        else:
            results = {}

        results.update(
            {
                "model_id": model_id,
                "task": task,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ep": self.ep,
                "device": self.device,
            }
        )
        for k in ("hypotheses",):
            results.setdefault(k, {})
        for k in (
            "baseline_opset",
            "baseline_p50_ms",
            "best_hypothesis",
            "best_p50_ms",
            "best_gain_pct",
            "conv_pct",
            "best_gain_verdict",
        ):
            results.setdefault(k, None)
        results.setdefault("errors", [])
        results.setdefault("feature_gaps", [])

        print(f"\n{'=' * 64}\n  SWEEP [{self.ep}/{self.device}]: {model_id}  [{task}]", flush=True)
        if only_hyp_ids:
            print(f"  (delta — only: {sorted(only_hyp_ids)})", flush=True)
        print("=" * 64, flush=True)

        model_start = time.time()

        # Step 1: base config
        print("\n[1/3] Generating base config…", flush=True)
        base_config = None
        if reuse_baseline_config:
            bc = model_dir / self.baseline_id / "build_config.json"
            if bc.exists():
                try:
                    base_config = json.loads(bc.read_text(encoding="utf-8"))
                    print(f"  [reuse] loaded {self.baseline_id} config", flush=True)
                except Exception:
                    base_config = None
        if base_config is None:
            base_config = self.get_base_config(model_id, task, model_type)
        if base_config is None:
            results["errors"].append("base config generation failed")
            self._finalize(results, model_dir)
            return results

        results["baseline_opset"] = (base_config.get("export") or {}).get("opset_version", "?")
        base_quant = "kept" if self.cfg.get("quant") else "NONE"
        print(f"  baseline opset={results['baseline_opset']}  quant={base_quant}", flush=True)

        # Step 2: hypothesis loop
        print(f"\n[2/3] Running {len(self.hyps)} hypotheses…", flush=True)
        conv_pct = 0.0
        conv_risk = False
        has_conv_guard = any(
            (h.get("guard") or {}).get("type") == "conv_pct_regression" for h in self.hyps
        )
        gemm_known: bool | None = None
        baseline_onnx: Path | None = None
        baseline_sig: dict | None = None
        seen_sigs: dict[str, dict] = {}  # op_sig for every hypothesis built so far

        for hyp in self.hyps:
            hyp_id = hyp["id"]
            if only_hyp_ids is not None and hyp_id not in only_hyp_ids:
                continue
            model_to = self.timeouts.get("model_s")
            if model_to and time.time() - model_start > model_to:
                results["hypotheses"][hyp_id] = {"status": "TIMEOUT", "label": hyp["label"]}
                results["errors"].append(f"{hyp_id}: model timeout")
                continue

            label = hyp["label"]
            guard = hyp.get("guard") or {}
            sep = "─" * 56
            print(f"\n{sep}\n  {hyp_id}: {label}\n{sep}", flush=True)

            hyp_config = self.make_hypothesis_config(
                base_config, hyp.get("opset"), hyp.get("optim")
            )
            opset_used = (hyp_config.get("export") or {}).get("opset_version", "?")
            build_flags = hyp.get("build_flags") or []
            extra = f"  flags={build_flags}" if build_flags else ""
            print(f"  opset={opset_used}  optim={hyp.get('optim')}{extra}", flush=True)

            hyp_dir = model_dir / hyp_id
            hyp_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = hyp_dir / "build_config.json"
            cfg_path.write_text(json.dumps(hyp_config, indent=2), encoding="utf-8")

            build_ok, build_out = self.run_build(model_id, cfg_path, hyp_dir, build_flags)
            if not build_ok:
                is_to = "TIMEOUT" in build_out
                results["hypotheses"][hyp_id] = {
                    "status": "BUILD_TIMEOUT" if is_to else "BUILD_FAIL",
                    "label": label,
                    "opset": opset_used,
                    "build_error": ("build timed out" if is_to else build_out[-400:]),
                }
                results["errors"].append(f"{hyp_id}: build failed")
                if any(
                    k in build_out.lower() for k in ("unsupported", "not supported", "no handler")
                ):
                    results["feature_gaps"].append(f"{hyp_id} ({label}): EP/op unsupported")
                continue

            onnx_path = self._find_onnx(hyp_dir)
            if onnx_path is None:
                results["hypotheses"][hyp_id] = {"status": "NO_ONNX", "label": label}
                results["errors"].append(f"{hyp_id}: build OK but no ONNX produced")
                continue

            # Op-count / topology signature of the built graph (npu-011): the first
            # cut for benefit-gating is "did the flag actually change the graph?".
            op_sig = self._op_signature(hyp_dir)
            if hyp_id == self.baseline_id:
                baseline_sig = op_sig
            if op_sig is not None:
                seen_sigs[hyp_id] = op_sig

            # NO-OP short-circuit: a non-baseline hypothesis whose built graph is
            # byte-for-byte identical to the baseline (same opset + op counts) cannot
            # differ in perf — the flag never fired. Skip the expensive screen+full
            # bench and reuse the baseline numbers. This is what separates
            # "applied-but-no-benefit" from "never-applied" (see npu-011) and it
            # saves one screen + N full sessions per dead hypothesis.
            if (
                hyp_id != self.baseline_id
                and baseline_sig is not None
                and self._same_graph(op_sig, baseline_sig)
            ):
                base_bench = results["hypotheses"].get(self.baseline_id, {})
                noop = {
                    "status": "NOOP_SKIPPED",
                    "verdict": "NO_OP",
                    "label": label,
                    "opset": opset_used,
                    "optim": hyp.get("optim") or {},
                    "build_flags": build_flags,
                    "op_signature": op_sig,
                    "graph_changed": False,
                    "noop_note": (
                        "graph identical to baseline (opset + op counts unchanged) — "
                        "flag did not fire; bench skipped, baseline perf assumed"
                    ),
                }
                base_full = base_bench.get("full")
                if base_full:
                    noop["full_ref"] = self.baseline_id
                    noop["assumed_median_p50_ms"] = base_full.get("median_p50_ms")
                results["hypotheses"][hyp_id] = noop
                print(
                    f"  [no-op] {hyp_id} graph == baseline "
                    f"(ops={op_sig.get('total_operators')}, opset={op_sig.get('opset')}); "
                    "bench skipped",
                    flush=True,
                )
                if self.prune_artifacts:
                    self._prune_hyp_artifacts(hyp_dir)
                    self._prune_runnable_except_best(results, model_dir)
                continue

            # Duplicate-graph skip (npu-011 extension): a non-baseline hypothesis whose
            # graph matches a prior non-baseline build — e.g. opset21+fusion when the
            # fusion has no applicable ops → identical graph to opset21-alone.  The
            # build was already needed to discover this; only bench is spared.
            if hyp_id != self.baseline_id and op_sig is not None:
                dup_ref = next(
                    (
                        rid
                        for rid, rsig in seen_sigs.items()
                        if rid != hyp_id
                        and rid != self.baseline_id
                        and self._same_graph(op_sig, rsig)
                    ),
                    None,
                )
                if dup_ref is not None:
                    ref_bench = results["hypotheses"].get(dup_ref, {})
                    noop = {
                        "status": "NOOP_SKIPPED",
                        "verdict": "NO_OP",
                        "label": label,
                        "opset": opset_used,
                        "optim": hyp.get("optim") or {},
                        "build_flags": build_flags,
                        "op_signature": op_sig,
                        "graph_changed": False,
                        "noop_note": (
                            f"graph identical to {dup_ref} (opset + op counts match) — "
                            "flag did not fire; bench skipped, prior perf reused"
                        ),
                    }
                    ref_full = ref_bench.get("full")
                    if ref_full:
                        noop["full_ref"] = dup_ref
                        noop["assumed_median_p50_ms"] = ref_full.get("median_p50_ms")
                    results["hypotheses"][hyp_id] = noop
                    print(
                        f"  [no-op] {hyp_id} graph == {dup_ref} "
                        f"(ops={op_sig.get('total_operators')}, opset={op_sig.get('opset')}); "
                        "bench skipped",
                        flush=True,
                    )
                    if self.prune_artifacts:
                        self._prune_hyp_artifacts(hyp_dir)
                        self._prune_runnable_except_best(results, model_dir)
                    continue

            # Guard: skip_if_gemm (cpu-002)
            if guard.get("type") == "skip_if_gemm":
                if gemm_known is None:
                    opt = hyp_dir / "optimized.onnx"
                    gemm_known = self._model_has_gemm(opt) if opt.exists() else False
                if gemm_known:
                    print(f"  [{guard['finding']}] SKIP {hyp_id}: model has Gemm nodes", flush=True)
                    results["hypotheses"][hyp_id] = {
                        "status": "SKIPPED_GUARD",
                        "label": label,
                        "opset": opset_used,
                        "guard": guard["finding"],
                    }
                    continue

            # After baseline build: compute Conv% for conv_pct_regression guards (npu-006)
            if hyp_id == self.baseline_id:
                baseline_onnx = onnx_path
                if has_conv_guard:
                    conv_pct, conv_cnt, conv_total = self._conv_pct(onnx_path)
                    unknown = conv_pct == 0.0 and conv_total == 0
                    threshold = next(
                        (
                            h["guard"]["threshold_pct"]
                            for h in self.hyps
                            if (h.get("guard") or {}).get("type") == "conv_pct_regression"
                        ),
                        20.0,
                    )
                    conv_risk = unknown or conv_pct > threshold
                    results["conv_pct"] = None if unknown else conv_pct
                    print(
                        f"  [conv-guard] Conv%={'UNKNOWN' if unknown else conv_pct}"
                        f" ({conv_cnt}/{conv_total}) risk={conv_risk}",
                        flush=True,
                    )

            # Guard: conv_pct_regression annotation (npu-006)
            expected_regression = False
            if guard.get("type") == "conv_pct_regression" and conv_risk:
                expected_regression = True
                print(
                    f"  [{guard['finding']}] WARNING: {hyp_id} conv fusions on Conv-dense model"
                    f" (Conv%={conv_pct}) — expect catastrophic regression",
                    flush=True,
                )

            # Bench: Phase A screen + Phase B full
            p50_screen, cv_screen, stable = self.bench_screen(onnx_path)
            bench: dict = {
                "status": "PENDING",
                "label": label,
                "opset": opset_used,
                "optim": hyp.get("optim") or {},
                "op_signature": op_sig,
                "graph_changed": (
                    None
                    if (op_sig is None or baseline_sig is None or hyp_id == self.baseline_id)
                    else not self._same_graph(op_sig, baseline_sig)
                ),
                "screen": {"p50_ms": p50_screen, "cv": round(cv_screen, 4), "stable": stable},
            }
            if expected_regression:
                bench["expected_regression"] = True
                bench["regression_finding"] = guard["finding"]

            if p50_screen is None:
                bench["status"] = "SCREEN_FAIL"
                results["hypotheses"][hyp_id] = bench
                results["errors"].append(f"{hyp_id}: screen failed")
                continue

            full_p50s = self.bench_full(onnx_path)
            if not full_p50s:
                bench["status"] = "BENCH_FAIL"
                results["hypotheses"][hyp_id] = bench
                results["errors"].append(f"{hyp_id}: full bench failed")
                continue

            median = _median(full_p50s)
            bench["full"] = {
                "p50s_ms": [round(p, 3) for p in full_p50s],
                "median_p50_ms": round(median, 3),
            }
            bench["status"] = "OK" if stable else "OK_HIGH_CV"

            # Accuracy eval on the baseline build for image-classification models
            if (
                self.cfg.get("accuracy_eval")
                and hyp_id == self.baseline_id
                and task == "image-classification"
            ):
                bench["accuracy"] = self.run_eval(onnx_path, model_id, task)

            # Opt-in paired A/B (DVFS-cancelling) for non-baseline hypotheses
            if (
                self.paired_ab
                and hyp_id != self.baseline_id
                and baseline_onnx is not None
                and adaptive_paired_ab_bench is not None
                and run_perf_session is not None
            ):
                print("  [paired-A/B] interleaving baseline vs hypothesis…", flush=True)

                def _session(p: Path, _onnx=onnx_path) -> float | None:
                    return run_perf_session(
                        WINML,
                        p,
                        self.ep,
                        self.device,
                        iters=self.full["iters"],
                        warmup=self.full["warmup"],
                    )

                bench["paired_ab"] = adaptive_paired_ab_bench(
                    _session,
                    baseline_onnx,
                    onnx_path,
                    cool_down_s=self.full["cool_down_s"],
                )
                pa = bench["paired_ab"]
                print(f"  [paired-A/B] {pa['verdict']} mean={pa['mean_gain_pct']}%", flush=True)

            results["hypotheses"][hyp_id] = bench
            if self.prune_artifacts:
                self._prune_hyp_artifacts(hyp_dir)
                self._prune_runnable_except_best(results, model_dir)

        # Step 3: verdicts, cross-checks, confirmation
        # npu-011 roll-up: which flags fired vs were dead no-ops on this model.
        noop_ids = [
            hid for hid, h in results["hypotheses"].items() if h.get("status") == "NOOP_SKIPPED"
        ]
        if noop_ids:
            results["noop_hypotheses"] = noop_ids
            print(
                f"  [npu-011] {len(noop_ids)} no-op hypotheses (graph == baseline, bench skipped): "
                f"{', '.join(noop_ids)}",
                flush=True,
            )
        print("\n[3/3] Computing verdicts…", flush=True)
        self._compute_verdicts(results)
        self._run_cross_checks(results)
        self._confirm_pass(results, model_dir)
        self._finalize(results, model_dir)
        return results

    # ── verdicts ────────────────────────────────────────────────────────────────

    def _compute_verdicts(self, results: dict) -> None:
        hyps = results["hypotheses"]
        min_gain = self.cfg["min_improvement_pct"]

        # baseline: first OK hypothesis in baseline_priority
        baseline_p50: float | None = None
        baseline_h: dict = {}
        for hid in self.cfg.get("baseline_priority", ["h0"]):
            h = hyps.get(hid, {})
            if h.get("status") in _OK:
                baseline_p50 = h.get("full", {}).get("median_p50_ms")
                if baseline_p50:
                    baseline_h = h
                    h["verdict"] = "BASELINE"
                    break
        results["baseline_p50_ms"] = baseline_p50

        # regression-probe membership (per-hypothesis verdict override)
        probe_map: dict[str, dict] = {}
        for c in self.cross_checks:
            if c["type"] == "regression_probe":
                for hid in c["hypotheses"]:
                    probe_map[hid] = c

        best_p50: float | None = None
        best_h: str | None = None
        best_hyp: dict = {}
        for hid, h in hyps.items():
            if h.get("status") not in _OK:
                continue
            median = h.get("full", {}).get("median_p50_ms")
            if median is None:
                continue
            if baseline_p50 and h.get("verdict") != "BASELINE":
                gain = (baseline_p50 - median) / baseline_p50 * 100
                h["gain_vs_baseline_pct"] = round(gain, 2)
                if hid in probe_map and gain <= probe_map[hid]["gain_threshold_pct"]:
                    h["verdict"] = "REGRESSION"
                    h["regression_finding"] = probe_map[hid]["id"]
                elif gain >= min_gain:
                    h["verdict"] = "KEEP"
                elif gain > 0:
                    h["verdict"] = "MARGINAL"
                else:
                    h["verdict"] = "DISCARD"
            if best_p50 is None or median < best_p50:
                best_p50, best_h, best_hyp = median, hid, h

        results["best_hypothesis"] = best_h
        results["best_p50_ms"] = best_p50
        if baseline_p50 and best_p50 is not None:
            gain = (baseline_p50 - best_p50) / baseline_p50 * 100
            results["best_gain_pct"] = round(gain, 2)
            if self.cfg.get("effect_size_gate"):
                self._effect_size(results, baseline_h, best_hyp, best_h, gain)
            elif best_h and best_h != self.baseline_id and gain >= min_gain:
                results["best_gain_verdict"] = "KEEP"
            else:
                results["best_gain_verdict"] = "BASELINE_IS_BEST"

    def _effect_size(
        self, results: dict, baseline_h: dict, best_hyp: dict, best_h: str | None, gain: float
    ) -> None:
        mult = self.cfg["effect_size_cv_mult"]
        base_p50s = baseline_h.get("full", {}).get("p50s_ms", [])
        best_p50s = best_hyp.get("full", {}).get("p50s_ms", [])
        noise = max(_session_cv(base_p50s), _session_cv(best_p50s))
        noise_floor = round(mult * noise * 100, 2)
        ranges_sep = bool(best_p50s and base_p50s and max(best_p50s) < min(base_p50s))
        effect_ok = gain >= noise_floor
        reliable = bool(effect_ok and ranges_sep and best_h != self.baseline_id)
        results["best_gain_noise_floor_pct"] = noise_floor
        results["best_gain_ranges_separated"] = ranges_sep
        results["best_gain_reliable"] = reliable
        if best_h == self.baseline_id:
            results["best_gain_verdict"] = "BASELINE_IS_BEST"
        elif reliable:
            results["best_gain_verdict"] = "RELIABLE"
        elif not effect_ok:
            results["best_gain_verdict"] = "NEUTRAL_WITHIN_NOISE"
        else:
            results["best_gain_verdict"] = "UNRELIABLE_RANGES_OVERLAP"
        print(
            f"  [effect-size] best={best_h} gain={gain:+.1f}% noise_floor={noise_floor:.1f}%"
            f" ranges_sep={ranges_sep} -> {results['best_gain_verdict']}",
            flush=True,
        )

    # ── cross-model finding checks ──────────────────────────────────────────────

    def _run_cross_checks(self, results: dict) -> None:
        for c in self.cross_checks:
            if c["type"] == "opset_bypass":
                self._check_opset_bypass(results, c)
            elif c["type"] == "catastrophic_regression":
                self._check_catastrophic(results, c)
            # regression_probe is applied per-hypothesis in _compute_verdicts

    def _check_opset_bypass(self, results: dict, c: dict) -> None:
        """Generalized npu-001: candidate (opset21) must beat the explicit-opset
        stress reference AND the auto-config baseline by the effect-size gate, with
        non-overlapping session ranges. Guards against DVFS-inflated references.
        """
        cid = c["id"]
        hyps = results["hypotheses"]
        cand = hyps.get(c["candidate"], {})
        stress = hyps.get(c["stress_ref"], {})
        base = hyps.get(c.get("baseline_ref", ""), {})
        key = f"{cid}_generalized"
        rkey = f"{cid}_ranges_non_overlapping"
        results.setdefault(key, None)
        results.setdefault(rkey, None)

        if stress.get("status") not in _OK or cand.get("status") not in _OK:
            missing = [
                r
                for r, d in ((c["stress_ref"], stress), (c["candidate"], cand))
                if d.get("status") not in _OK
            ]
            results[key] = f"N/A ({', '.join(missing)} not OK)"
            return

        cand_p50 = cand["full"].get("median_p50_ms", float("inf"))
        stress_p50 = stress["full"].get("median_p50_ms", float("inf"))
        cand_p50s = cand["full"].get("p50s_ms", [cand_p50])
        stress_p50s = stress["full"].get("p50s_ms", [stress_p50])
        median_gain = cand_p50 < stress_p50 * 0.95
        median_loss = stress_p50 < cand_p50 * 0.95
        ranges_sep = max(cand_p50s) < min(stress_p50s) if cand_p50s and stress_p50s else None
        results[rkey] = ranges_sep

        # Guard 1: stress reference must be reliable (not high-CV / DVFS-thrashing).
        if stress.get("status") == "OK_HIGH_CV":
            results[key] = "N/A (high-CV stress reference)"
            print(f"  [{cid}] N/A: explicit-opset reference is HIGH-CV", flush=True)
            return

        # Guard 2: candidate must also beat the auto-config baseline by effect size.
        mult = self.cfg.get("effect_size_cv_mult", 2.0)
        beats_baseline: bool | None = None
        if base.get("status") in _OK:
            base_p50s = base["full"].get("p50s_ms", [])
            base_p50 = base["full"].get("median_p50_ms")
            if base_p50s and base_p50 and cand_p50s:
                gvb = (base_p50 - cand_p50) / base_p50 * 100
                floor = mult * max(_session_cv(base_p50s), _session_cv(cand_p50s)) * 100
                beats_baseline = gvb >= floor and max(cand_p50s) < min(base_p50s)

        # Guard 3: a decisive paired-A/B verdict overrides the sequential medians.
        pab = (cand.get("paired_ab") or {}).get("verdict")
        pab_rejects = pab in ("MARGINAL", "DISCARD")

        if beats_baseline is False or pab_rejects:
            results[key] = "neutral"
            print(f"  [{cid}] NEUTRAL vs auto-config baseline", flush=True)
        elif median_gain and ranges_sep:
            results[key] = True
            print(
                f"  [{cid}] CONFIRMED: {c['candidate']} beats {c['stress_ref']} + baseline",
                flush=True,
            )
        elif median_gain and not ranges_sep:
            results[key] = "median_only"
            print(f"  [{cid}] MARGINAL: median faster but ranges overlap (DVFS noise)", flush=True)
        elif median_loss:
            results[key] = False
            print(f"  [{cid}] NEGATIVE: {c['stress_ref']} faster than {c['candidate']}", flush=True)
        else:
            results[key] = "neutral"
            print(f"  [{cid}] NEUTRAL", flush=True)

    def _check_catastrophic(self, results: dict, c: dict) -> None:
        """npu-006: conv-fusion hypotheses regress >= ratio_threshold x baseline."""
        cid = c["id"]
        baseline_p50 = results.get("baseline_p50_ms")
        ratio = c.get("ratio_threshold", 5.0)
        hit = False
        for hid in c["hypotheses"]:
            h = results["hypotheses"].get(hid, {})
            if h.get("status") in _OK and baseline_p50:
                p50 = h.get("full", {}).get("median_p50_ms")
                if p50 and p50 >= baseline_p50 * ratio:
                    hit = True
                    print(
                        f"  [{cid}] CATASTROPHIC REGRESSION on {hid}:"
                        f" {p50:.1f}ms vs baseline {baseline_p50:.1f}ms"
                        f" ({p50 / baseline_p50:.0f}x)",
                        flush=True,
                    )
        results[f"{cid}_regression"] = hit

    # ── confirmation pass (Phase C) ─────────────────────────────────────────────

    def _confirm_pass(self, results: dict, model_dir: Path) -> None:
        best_h = results.get("best_hypothesis")
        baseline_p50 = results.get("baseline_p50_ms")
        if not best_h or best_h == self.baseline_id or not baseline_p50:
            return
        if (results.get("best_gain_pct") or 0) < self.cfg["min_improvement_pct"]:
            return
        n = self.cfg["confirm_sessions"]
        if n <= 0:
            return

        hyp_dir = model_dir / best_h
        onnx_path = self._find_onnx(hyp_dir)
        if onnx_path is None:
            return
        best_hyp = results["hypotheses"].get(best_h, {})
        print(f"\n  ── Phase C: confirming {best_h} ({n} extra sessions) ──", flush=True)

        cd = self.full["cool_down_s"]
        confirm: list[float] = []
        for s in range(1, n + 1):
            out_json = hyp_dir / f"confirm_s{s}.json"
            rc, _ = self.run_cmd(
                [
                    WINML,
                    "perf",
                    "-m",
                    str(onnx_path),
                    "--ep",
                    self.ep,
                    "--device",
                    self.device,
                    "--warmup",
                    str(self.full["warmup"]),
                    "--iterations",
                    str(self.full["iters"]),
                    "-o",
                    str(out_json),
                ],
                label=f"confirm s{s}/{n}",
                timeout=self.timeouts["bench_s"],
            )
            p50, _ = _latency(out_json) if rc == 0 and out_json.exists() else (None, None)
            if p50 is not None:
                print(f"     confirm s{s}: p50={p50:.2f}ms", flush=True)
                confirm.append(p50)
            if s < n:
                time.sleep(cd)
        if not confirm:
            return

        # Baseline session range for the non-overlap test
        base_h: dict = {}
        for hid in self.cfg.get("baseline_priority", ["h0"]):
            if results["hypotheses"].get(hid, {}).get("status") in _OK:
                base_h = results["hypotheses"][hid]
                break
        base_p50s = base_h.get("full", {}).get("p50s_ms", [baseline_p50])
        all_p50s = best_hyp.get("full", {}).get("p50s_ms", []) + confirm
        overall_median = _median(all_p50s)
        overall_gain = (baseline_p50 - overall_median) / baseline_p50 * 100
        confirmed = max(all_p50s) < min(base_p50s) if base_p50s else False

        best_hyp["confirm_p50s_ms"] = [round(p, 3) for p in confirm]
        best_hyp["all_p50s_ms"] = [round(p, 3) for p in all_p50s]
        best_hyp["overall_median_p50_ms"] = round(overall_median, 3)
        best_hyp["overall_gain_pct"] = round(overall_gain, 2)
        if confirmed:
            best_hyp["confirm_verdict"] = "CONFIRMED"
            results["best_gain_pct"] = round(overall_gain, 2)
            print(
                f"  [CONFIRMED] {best_h}: gain={overall_gain:+.1f}% (ranges non-overlapping)",
                flush=True,
            )
        else:
            best_hyp["confirm_verdict"] = "MARGINAL_UNCONFIRMED"
            print(f"  [MARGINAL_UNCONFIRMED] {best_h}: ranges overlap — DVFS noise", flush=True)

    # ── outputs ─────────────────────────────────────────────────────────────────

    def _finalize(self, results: dict, model_dir: Path) -> None:
        out = model_dir / "results.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Results: {out}", flush=True)
        self._emit_champion(results, model_dir)
        try:
            if generate_model_report is None:
                raise RuntimeError("gen_model_report unavailable")
            generate_model_report(results, model_dir / "report.html")
        except Exception as e:
            print(f"  [warn] report generation failed: {e}", flush=True)

    def _emit_champion(self, results: dict, model_dir: Path) -> None:
        """Copy the optimal build's winml_build_config.json into the model folder.

        The champion is the best hypothesis when its gain is reliable, otherwise the
        baseline (auto) config. The emitted file *is* the winml build config of that
        hypothesis — the fully-resolved ``winml_build_config.json`` winml writes into
        the build output dir — so it can be fed straight back to ``winml build -c``.
        Falls back to the input ``build_config.json`` if the build output config is
        unavailable (e.g. a results-only checkout). Lives in the per-model folder so
        all tuning products for a model stay together.
        """
        baseline_id = self.baseline_id
        best_h = results.get("best_hypothesis")
        reliable = bool(results.get("best_gain_reliable")) and best_h not in (None, baseline_id)
        champion_h = best_h if reliable else baseline_id
        if not champion_h:
            return

        build_config = self._load_winml_build_config(model_dir / champion_h)
        out_path = model_dir / f"champion_{self.ep}_{self.device}.json"
        if build_config is None:
            print(
                f"  [warn] champion config missing — no winml_build_config.json in"
                f" {model_dir / champion_h} (run a full sweep to materialize it)",
                flush=True,
            )
            return

        out_path.write_text(
            json.dumps(build_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        champ = results.get("hypotheses", {}).get(champion_h, {})
        champ_p50 = results.get("best_p50_ms") if reliable else results.get("baseline_p50_ms")
        print(
            f"  Champion config: {out_path}  "
            f"[{champion_h} {champ.get('label', '')!r}  p50={champ_p50}ms"
            f"  reliable_gain={reliable}]",
            flush=True,
        )

    @staticmethod
    def _load_winml_build_config(build_dir: Path) -> dict | None:
        """Return the build config from a hypothesis' build output dir.

        Prefers the fully-resolved ``winml_build_config.json`` winml persists after a
        build (also matches the ``<cache_key>_winml_build_config.json`` variant); falls
        back to the ``build_config.json`` the sweep passed as build input.
        """
        candidates = [build_dir / "winml_build_config.json"]
        candidates += sorted(build_dir.glob("*_winml_build_config.json"))
        candidates.append(build_dir / "build_config.json")
        for path in candidates:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return None

    @staticmethod
    def _prune_hyp_artifacts(hyp_dir: Path) -> None:
        """Delete bulky intermediate ONNX artifacts after a hypothesis is benched.

        Keeps the runnable ``model.onnx``/``quantized.onnx`` (+ ``.data``) — needed
        by the Phase-C confirm re-bench — and all small JSON/metadata. Removes the
        large export/optimized graphs (often hundreds of MB each) so long multi-model
        sweeps don't exhaust disk. Best-effort; failures are ignored.
        """
        freed = 0
        for pat in ("export.onnx", "export.onnx.data", "optimized.onnx", "optimized.onnx.data"):
            p = hyp_dir / pat
            if p.exists():
                try:
                    freed += p.stat().st_size
                    p.unlink()
                except OSError:
                    pass
        # Also remove shape-inference temp files that winml build writes to CWD
        # (the autoconfig root): sym_shape_infer_temp.onnx and UUID-named *.data
        # files produced by the ONNX external-data shape-inference pass.  These
        # accumulate across hypotheses since the per-hypothesis prune above only
        # covers files written inside the hypothesis output folder.
        for p in list(_AGENT_ROOT.glob("*.data")) + [_AGENT_ROOT / "sym_shape_infer_temp.onnx"]:
            if p.exists():
                try:
                    freed += p.stat().st_size
                    p.unlink()
                except OSError:
                    pass
        if freed:
            print(
                f"     [prune] freed {freed / 1024 / 1024:.0f} MB of build intermediates",
                flush=True,
            )

    def _prune_runnable_except_best(self, results: dict, model_dir: Path) -> None:
        """Keep only the running-best (and baseline) hypothesis' runnable ONNX.

        The Phase-C confirm only re-benches the single best non-baseline hypothesis,
        so retaining every hypothesis' ``model.onnx``/``quantized.onnx`` is wasteful —
        for large models (~1-2 GB each) that accumulation exhausts disk mid-sweep.
        After each hypothesis benches we therefore delete the runnable graphs of every
        hypothesis except the current lowest-median one and the baseline.
        """
        best_id: str | None = None
        best_med: float | None = None
        for hid, h in results["hypotheses"].items():
            if h.get("status") in _OK:
                med = h.get("full", {}).get("median_p50_ms")
                if med is not None and (best_med is None or med < best_med):
                    best_id, best_med = hid, med
        if best_id is None:
            return
        keep = {best_id, self.baseline_id}
        freed = 0
        for hyp_dir in model_dir.glob("h*"):
            if not hyp_dir.is_dir() or hyp_dir.name in keep:
                continue
            for pat in (
                "model.onnx",
                "model.onnx.data",
                "quantized.onnx",
                "quantized.onnx.data",
            ):
                p = hyp_dir / pat
                if p.exists():
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                    except OSError:
                        pass
        if freed:
            print(
                f"     [prune] freed {freed / 1024 / 1024:.0f} MB runnable onnx "
                f"(keeping best={best_id}, baseline={self.baseline_id})",
                flush=True,
            )

    def write_summary(self, all_results: list[dict]) -> None:
        lines = [
            f"# {self.ep.upper()} / {self.device.upper()} EP Optimization Sweep — Catalog Models",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
            f"EP: `{self.ep}` / device: `{self.device}`  ",
            f"Protocol: screen {self.screen['iters']} iters, full {self.full['iters']}"
            f"×{self.full['sessions']} sessions + {self.cfg['confirm_sessions']} confirm  ",
            "",
            "## Per-Model Results",
            "",
            "| Model | Baseline p50 | Best p50 | Best config | Gain% | Verdict | Notes |",
            "|-------|-------------|----------|-------------|-------|---------|-------|",
        ]
        for r in all_results:
            mid = r.get("model_id", "?")
            base = f"{r['baseline_p50_ms']:.1f} ms" if r.get("baseline_p50_ms") else "N/A"
            best = f"{r['best_p50_ms']:.1f} ms" if r.get("best_p50_ms") else "N/A"
            best_h = r.get("best_hypothesis") or "N/A"
            label = (
                r.get("hypotheses", {}).get(best_h, {}).get("label", "") if best_h != "N/A" else ""
            )
            gain = f"{r['best_gain_pct']:.1f}%" if r.get("best_gain_pct") is not None else "N/A"
            verdict = r.get("best_gain_verdict") or "—"
            notes = "; ".join(r.get("errors", []))[:60] or "none"
            lines.append(
                f"| `{mid}` | {base} | {best} | {best_h} ({label}) | {gain} | {verdict} | {notes} |"
            )

        # Cross-check section (data-driven from results keys the checks emit)
        if self.cross_checks:
            lines += ["", "## Cross-Model Finding Checks", ""]
            headers = ["Model"]
            check_keys: list[tuple[str, str]] = []
            for c in self.cross_checks:
                if c["type"] == "opset_bypass":
                    check_keys.append((c["id"], f"{c['id']}_generalized"))
                elif c["type"] == "catastrophic_regression":
                    check_keys.append((c["id"], f"{c['id']}_regression"))
                elif c["type"] == "regression_probe":
                    check_keys.append((c["id"], None))  # per-hypothesis; summarised below
            headers += [cid for cid, _ in check_keys]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for r in all_results:
                row = [f"`{r.get('model_id', '?')}`"]
                for cid, key in check_keys:
                    if key is None:
                        probe = [
                            h
                            for h, d in r.get("hypotheses", {}).items()
                            if d.get("regression_finding") == cid
                        ]
                        row.append(", ".join(probe) if probe else "no")
                    else:
                        row.append(str(r.get(key, "—")))
                lines.append("| " + " | ".join(row) + " |")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.results_dir / "SUMMARY.md"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n📄 Summary: {summary_path}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified JSON-driven EP/device sweep")
    parser.add_argument("--ep", required=True, help="execution provider (e.g. cpu, qnn, dml)")
    parser.add_argument("--device", required=True, help="device (e.g. cpu, npu, gpu)")
    parser.add_argument("--model", help="run a single model (HuggingFace model ID)")
    parser.add_argument("--task", help="task for single-model run")
    parser.add_argument("--model-type", dest="model_type", help="model type for single-model run")
    parser.add_argument(
        "--only-hypotheses", dest="only_hyp", help="comma-separated hypothesis IDs (e.g. h4,h5,h10)"
    )
    parser.add_argument(
        "--reuse-baseline-config",
        dest="reuse_base",
        action="store_true",
        help="reuse the baseline build_config.json instead of re-running winml config",
    )
    parser.add_argument(
        "--paired-ab",
        dest="paired_ab",
        action="store_true",
        help="enable opt-in paired A/B (if the EP supports it)",
    )
    parser.add_argument("--list", action="store_true", help="list catalog models and exit")
    parser.add_argument(
        "--prune-artifacts",
        dest="prune_artifacts",
        action="store_true",
        help="delete bulky export/optimized ONNX intermediates after each hypothesis"
        " is benched (keeps runnable model.onnx + JSONs); use for long disk-bound sweeps",
    )
    args = parser.parse_args()

    sweep = CatalogSweep(
        args.ep, args.device, paired_ab=args.paired_ab, prune_artifacts=args.prune_artifacts
    )

    if args.list:
        print(f"Catalog models for {args.ep}/{args.device}:")
        for m in sweep.models:
            print(f"  {m['id']:55s} {m['task']:24s} {m['model_type']}")
        return

    only_hyp_ids = set(args.only_hyp.split(",")) if args.only_hyp else None
    all_results: list[dict] = []

    if args.model:
        # task/model_type fall back to the catalog entry if present
        entry = next((m for m in sweep.models if m["id"] == args.model), None)
        task = args.task or (entry or {}).get("task")
        mtype = args.model_type or (entry or {}).get("model_type")
        if not task or not mtype:
            print("ERROR: --task and --model-type required (model not in catalog)", file=sys.stderr)
            sys.exit(1)
        all_results.append(
            sweep.sweep_model(
                args.model,
                task,
                mtype,
                only_hyp_ids=only_hyp_ids,
                reuse_baseline_config=args.reuse_base,
            )
        )
    else:
        for m in sweep.models:
            all_results.append(
                sweep.sweep_model(
                    m["id"],
                    m["task"],
                    m["model_type"],
                    only_hyp_ids=only_hyp_ids,
                    reuse_baseline_config=args.reuse_base,
                )
            )

    sweep.write_summary(all_results)
    print(
        f"\n{'=' * 64}\n  {args.ep.upper()}/{args.device.upper()} SWEEP COMPLETE\n{'=' * 64}",
        flush=True,
    )


if __name__ == "__main__":
    main()
