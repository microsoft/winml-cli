#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""promote_findings.py — Confidence-gated KB promotion (self-evolution-design Fix #4).

Reads every ``results.json`` produced by the catalog sweeps and applies the
L1 -> L4 confidence ladder from ``docs/self-evolution-design.html`` §2:

    L1  Observed   — one model, one run: median gain >= L1_GAIN_PCT.
    L2  Confirmed  — statistically robust on a single model: the hypothesis p50
                     range is strictly below the baseline range AND the gain
                     clears the effect-size floor (gain% >= EFFECT_SIZE_CV_MULT x
                     between-session CV). This is the same anti-DVFS gate the
                     sweep uses for ``best_gain_reliable``.
    L3  Generalized — the SAME (ep, flags) signature reaches L2 on >= 2 distinct
                      models of ONE architecture class (winml ``model_type``).
    L4  Cross-cutting — the same (ep, flags) signature reaches L2 across >= 3
                        architecture classes; scope broadens to EP-wide.

Output is written to ``ep_device_knowledge/_auto_promoted.json`` as a DRAFT sink — it
never clobbers the human-curated ``ep_device_knowledge/<ep>_<device>.json`` files. A human applies
the promotion checklist in ``ep_device_knowledge/README.md`` before merging anything into
the curated KB. This keeps "KB holds L3+ only" while protecting curated findings.

Architecture class == winml ``model_type`` (an architecture family such as
``vit`` / ``resnet`` / ``bert``), never a specific checkpoint — so grouping stays
universal and contains no hard-coded model logic.

Usage:
    uv run python promote_findings.py [--root .] [--out ep_device_knowledge/_auto_promoted.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Agent package bootstrap: make the autoconfig root importable for sibling packages.
_AGENT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "ep_device_knowledge").is_dir()
)
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from skills.optimizer.bench_utils import session_cv  # noqa: E402

# Effect-size multiplier — must match the `effect_size_cv_mult` in the sweep config
# (ep_device_knowledge/<ep>_<device>.json) used by catalog_sweep._effect_size.
EFFECT_SIZE_CV_MULT = 2.0
L1_GAIN_PCT = 5.0  # L1 (Observed) minimum median gain
L3_MIN_MODELS = 2  # distinct models of one arch class for L3
L4_MIN_ARCH_CLASSES = 3  # distinct arch classes for L4

OK_STATUSES = ("OK", "OK_HIGH_CV")


def _hyp_p50s(hyp: dict) -> list[float]:
    """Per-session p50 list for a hypothesis, tolerant of both sweep schemas.

    QNN sweep nests them under ``full.p50s_ms``; the GPU/CPU sweeps store a flat
    ``full_p50s_ms``. Returns an empty list when neither is present.
    """
    nested = (hyp.get("full") or {}).get("p50s_ms")
    if nested:
        return list(nested)
    flat = hyp.get("full_p50s_ms")
    return list(flat) if flat else []


def _flags_signature(hyp: dict) -> tuple[tuple[str, object], ...]:
    """Canonical, hashable signature of a hypothesis's config delta.

    Combines opset with the extra_optim flags so that, e.g., ``opset21`` and
    ``opset21 + bias_softmax_fusion`` are distinct signatures.
    """
    sig: dict[str, object] = {}
    opset = hyp.get("opset")
    if opset is not None:
        sig["opset"] = opset
    for k, v in (hyp.get("extra_optim") or {}).items():
        sig[k] = v
    return tuple(sorted(sig.items()))


def _flags_label(sig: tuple[tuple[str, object], ...]) -> str:
    if not sig:
        return "(baseline)"
    return " + ".join(f"{k}={v}" for k, v in sig)


def _baseline_p50s(hyps: dict) -> list[float]:
    """Per-session p50 list of the baseline hypothesis (prefer h0, fall back h1)."""
    for h_id in ("h0", "h1"):
        h = hyps.get(h_id, {})
        if h.get("status") in OK_STATUSES:
            p50s = _hyp_p50s(h)
            if p50s:
                return p50s
    return []


def classify_hypothesis(hyp: dict, base_p50s: list[float]) -> dict | None:
    """Return per-hypothesis L1/L2 classification vs a baseline, or None if not OK.

    Mirrors the effect-size gate in catalog_sweep._effect_size so promotion
    and the sweep agree on what "reliable" means.
    """
    if hyp.get("status") not in OK_STATUSES:
        return None
    p50s = _hyp_p50s(hyp)
    base_med = sorted(base_p50s)[len(base_p50s) // 2] if base_p50s else None
    hyp_med = sorted(p50s)[len(p50s) // 2] if p50s else None
    if not base_p50s or not p50s or not base_med or not hyp_med:
        return None

    gain_pct = (base_med - hyp_med) / base_med * 100
    noise_cv = max(session_cv(base_p50s), session_cv(p50s))
    noise_floor_pct = EFFECT_SIZE_CV_MULT * noise_cv * 100
    ranges_separated = max(p50s) < min(base_p50s)
    effect_size_ok = gain_pct >= noise_floor_pct
    reliable = bool(effect_size_ok and ranges_separated and gain_pct > 0)

    level = 0
    if reliable:
        level = 2
    elif gain_pct >= L1_GAIN_PCT:
        level = 1
    return {
        "gain_pct": round(gain_pct, 2),
        "noise_floor_pct": round(noise_floor_pct, 2),
        "ranges_separated": ranges_separated,
        "level": level,
    }


def collect(root: Path) -> list[dict]:
    """Walk every catalog-*-sweep/*/results.json and emit per-hypothesis records."""
    records: list[dict] = []
    for results_path in sorted(root.glob("catalog-*-sweep/*/results.json")):
        try:
            r = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] skipping {results_path}: {e}")
            continue
        hyps = r.get("hypotheses") or {}
        base_p50s = _baseline_p50s(hyps)
        if not base_p50s:
            continue
        model_id = r.get("model_id", results_path.parent.name)
        arch_class = r.get("model_type") or "unknown"
        ep = r.get("ep", "unknown")
        device = r.get("device", "unknown")
        for h_id, hyp in hyps.items():
            if h_id in ("h0", "h1"):  # baselines are not candidates
                continue
            cls = classify_hypothesis(hyp, base_p50s)
            if not cls or cls["level"] < 1:
                continue
            sig = _flags_signature(hyp)
            if not sig:
                continue
            records.append(
                {
                    "model_id": model_id,
                    "arch_class": arch_class,
                    "ep": ep,
                    "device": device,
                    "hyp_id": h_id,
                    "label": hyp.get("label", h_id),
                    "flags_sig": sig,
                    "flags": _flags_label(sig),
                    **cls,
                }
            )
    return records


def promote(records: list[dict]) -> dict:
    """Apply the L1->L4 ladder to per-hypothesis records."""
    l1 = [r for r in records if r["level"] >= 1]
    l2 = [r for r in records if r["level"] >= 2]

    # L3: same (ep, device, flags_sig, arch_class) reaching L2 on >= N distinct models.
    by_arch: dict[tuple, list[dict]] = defaultdict(list)
    for r in l2:
        by_arch[(r["ep"], r["device"], r["flags_sig"], r["arch_class"])].append(r)
    l3 = []
    for (ep, device, sig, arch), evs in by_arch.items():
        models = sorted({e["model_id"] for e in evs})
        if len(models) >= L3_MIN_MODELS:
            l3.append(
                {
                    "ep": ep,
                    "device": device,
                    "arch_class": arch,
                    "flags": _flags_label(sig),
                    "models": models,
                    "mean_gain_pct": round(sum(e["gain_pct"] for e in evs) / len(evs), 2),
                    "evidence": [
                        {
                            "model_id": e["model_id"],
                            "hyp_id": e["hyp_id"],
                            "gain_pct": e["gain_pct"],
                        }
                        for e in evs
                    ],
                }
            )

    # L4: same (ep, device, flags_sig) reaching L2 across >= M distinct arch classes.
    by_flags: dict[tuple, list[dict]] = defaultdict(list)
    for r in l2:
        by_flags[(r["ep"], r["device"], r["flags_sig"])].append(r)
    l4 = []
    for (ep, device, sig), evs in by_flags.items():
        arches = sorted({e["arch_class"] for e in evs})
        if len(arches) >= L4_MIN_ARCH_CLASSES:
            l4.append(
                {
                    "ep": ep,
                    "device": device,
                    "flags": _flags_label(sig),
                    "arch_classes": arches,
                    "models": sorted({e["model_id"] for e in evs}),
                    "mean_gain_pct": round(sum(e["gain_pct"] for e in evs) / len(evs), 2),
                }
            )

    def _public(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "flags_sig"}

    return {
        "L1_observed": [_public(r) for r in l1],
        "L2_confirmed_single_model": [_public(r) for r in l2],
        "L3_generalized_arch_rule": l3,
        "L4_cross_cutting_rule": l4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence-gated KB promotion (L1->L4).")
    parser.add_argument(
        "--root",
        type=Path,
        default=_AGENT_ROOT,
        help="autoconfig root containing catalog-*-sweep/ dirs (default: agent root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output draft file (default: <root>/ep_device_knowledge/_auto_promoted.json)",
    )
    args = parser.parse_args()
    root: Path = args.root
    out: Path = args.out or (root / "ep_device_knowledge" / "_auto_promoted.json")

    records = collect(root)
    ladder = promote(records)
    payload = {
        "_meta": {
            "generated_by": "promote_findings.py",
            "status": "draft",
            "note": (
                "Auto-generated promotion candidates. NOT curated KB. Apply the "
                "promotion checklist in ep_device_knowledge/README.md (paired A/B, clean "
                "baseline, effect-size > noise floor, independent reruns, "
                "baseline-drift check) before merging into <ep>_<device>.json."
            ),
            "gates": {
                "L1_gain_pct": L1_GAIN_PCT,
                "L2_effect_size_cv_mult": EFFECT_SIZE_CV_MULT,
                "L3_min_models": L3_MIN_MODELS,
                "L4_min_arch_classes": L4_MIN_ARCH_CLASSES,
            },
        },
        **ladder,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"promote_findings: scanned {len(records)} qualifying hypothesis record(s)")
    print(f"  L1 observed              : {len(ladder['L1_observed'])}")
    print(f"  L2 confirmed (1 model)   : {len(ladder['L2_confirmed_single_model'])}")
    print(f"  L3 generalized (arch)    : {len(ladder['L3_generalized_arch_rule'])}")
    print(f"  L4 cross-cutting         : {len(ladder['L4_cross_cutting_rule'])}")
    for r in ladder["L3_generalized_arch_rule"]:
        print(
            f"  [L3] {r['ep']}/{r['device']} {r['arch_class']}: {r['flags']} "
            f"on {len(r['models'])} models (+{r['mean_gain_pct']}%)"
        )
    for r in ladder["L4_cross_cutting_rule"]:
        print(
            f"  [L4] {r['ep']}/{r['device']}: {r['flags']} "
            f"across {len(r['arch_classes'])} arch classes (+{r['mean_gain_pct']}%)"
        )
    print(f"  draft written: {out}")


if __name__ == "__main__":
    main()
