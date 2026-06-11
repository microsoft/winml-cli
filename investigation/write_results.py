"""Step 7 — aggregate all phases into investigation/results.json.

Re-runs each phase in one process so results.json is internally consistent,
then writes the per-question blocks and the Q1/Q2/Q3 verdicts.
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

import run_cpu
import run_qnn_aot
import run_qnn_jit
import run_user_script
from qnn_common import INVEST, qnn_devices

# Recorded numeric tolerances (tightened/loosened from observed runs).
ATOL_CPU = 1e-5     # CPU EP is exact for this toy graph
ATOL_QNN = 1e-2     # QNN NPU/HTP runs fp32 at fp16 precision; observed max ~3.4e-3


def _arrs_to_lists(d: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def env_block() -> dict:
    devs = qnn_devices()
    lib = ""
    try:
        from windowsml import EpCatalog
        with EpCatalog() as cat:
            for p in cat.find_all_providers():
                if p.name == "QNNExecutionProvider":
                    lib = p.library_path
                    break
    except Exception:  # noqa: BLE001
        pass
    return {
        "ort": ort.__version__,
        "qnn_registration": "WinML EpCatalog + register_execution_provider_library "
                            "+ SessionOptions.add_provider_for_devices (NOT backend_path)",
        "qnn_ep_library": lib,
        "qnn_devices": sorted(str(t) for t in devs),
        "target_device": "NPU (QnnHtp; fp32 executed at fp16 precision)",
        "atol_cpu": ATOL_CPU,
        "atol_qnn_npu": ATOL_QNN,
    }


def main() -> None:
    cpu = run_cpu.main()
    jit = run_qnn_jit.main()
    q1 = run_qnn_aot.main()
    q3 = run_user_script.main()

    q2a = jit["q2_a"]
    q2b = jit["q2_b"]

    # ---- verdicts ----
    q2a_struct_ok = all(
        nc["EPContext"] >= 1 and nc["MatMul"] == 0 and not nc["initializers"]
        for nc in q2a["node_counts"].values()
    )
    q2a_num_ok = (
        q2a["max_abs_err"]["qnn_baked_vs_cpu_baked"] < ATOL_QNN
        and q2a["max_abs_err"]["qnn_baked2_vs_cpu_baked2"] < ATOL_QNN
        and q2a["max_abs_err"]["qnn_baked_vs_qnn_baked2"] > 1e-3
    )
    verdict_q2a = "CONFIRMED" if (q2a_struct_ok and q2a_num_ok) else "FALSIFIED"

    # Q2.b: every fresh build non-trivial (no free recompile on swap-back).
    builds = [s["build_s"] for s in q2b["sequence_log"]]
    verdict_q2b = "CONFIRMED" if min(builds) > 0.05 else "INCONCLUSIVE"

    # Q1: adapter could not change the precompiled EPContext output.
    verdict_q1 = "CONFIRMED no-op" if not q1["adapter_changed_output"] else "FALSIFIED"

    # Q3: CPU + QNN numeric match and no recompile on swap.
    q3c, q3d = q3["q3_c"], q3["q3_d"]
    q3_ok = (
        q3["q3_b"]["cpu_adapter1_vs_cpu_baked"] < ATOL_CPU
        and q3["q3_b"]["cpu_adapter2_vs_cpu_baked2"] < ATOL_CPU
        and q3c["max_abs_err"]["qnn_adapter1_vs_cpu_baked"] < ATOL_QNN
        and q3c["max_abs_err"]["qnn_adapter2_vs_cpu_baked2"] < ATOL_QNN
        and q3c["max_abs_err"]["qnn_adapter1_vs_adapter2"] > 1e-3
        and q3c["no_recompile_on_swap"]
        and q3c["compile_dominates"]
        and q3d["ctx_structure"]["EPContext"] >= 1
        and q3d["ctx_structure"]["MatMul"] == 0
        and len(q3d["ctx_structure"]["ctx_inputs"][0]) == 3
    )
    verdict_q3 = "CONFIRMED" if q3_ok else "FALSIFIED"

    results = {
        "env": env_block(),
        "cpu_refs": _arrs_to_lists(cpu),

        "q1__precompiled_plus_adapter_is_noop": {
            "adapter_api_used": q1["adapter_api_used"],
            "max_abs_err__adapter_vs_plain": q1["max_abs_err__adapter_vs_plain"],
            "adapter_changed_output": q1["adapter_changed_output"],
            "mechanism_note": q1["note"],
            "q1b_outside_epcontext": q1["q1b_outside_epcontext"],
        },

        "q2_a__jit_bake_works": {
            "node_counts": q2a["node_counts"],
            "compile_times_s": q2a["compile_times_s"],
            "max_abs_err": q2a["max_abs_err"],
        },
        "q2_b__swap_requires_recompile": {
            "compile_times_ms": q2b["compile_times_ms"],
            "median_build_s": q2b["median_build_s"],
            "sequence": q2b["sequence"],
            "sequence_log": q2b["sequence_log"],
        },

        "q3__graph_input_user_script": {
            "lora_inputs": q3["lora_inputs"],
            "node_counts": {"switchable_ctx": q3d["ctx_structure"]},
            "max_abs_err": {
                "cpu_adapter1_vs_cpu_baked": q3["q3_b"]["cpu_adapter1_vs_cpu_baked"],
                "cpu_adapter2_vs_cpu_baked2": q3["q3_b"]["cpu_adapter2_vs_cpu_baked2"],
                "qnn_jit_adapter1_vs_cpu_baked": q3c["max_abs_err"]["qnn_adapter1_vs_cpu_baked"],
                "qnn_jit_adapter2_vs_cpu_baked2": q3c["max_abs_err"]["qnn_adapter2_vs_cpu_baked2"],
                "qnn_aot_adapter1_vs_cpu_baked": q3d["max_abs_err"]["qnn_adapter1_vs_cpu_baked"],
                "qnn_aot_adapter2_vs_cpu_baked2": q3d["max_abs_err"]["qnn_adapter2_vs_cpu_baked2"],
            },
            "timings_ms": {"jit": q3c["timings_ms"], "aot": q3d["timings_ms"]},
            "no_recompile_on_swap": q3c["no_recompile_on_swap"] and q3d["no_recompile_on_swap"],
        },

        "verdict": {
            "Q1_precompiled_plus_adapter": verdict_q1,
            "Q2a_jit_bake": verdict_q2a,
            "Q2b_swap_requires_recompile": verdict_q2b,
            "Q3_graph_input_user_script": verdict_q3,
        },
    }

    out = INVEST / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[results] wrote {out}")
    print(json.dumps(results["verdict"], indent=2))
    return results


if __name__ == "__main__":
    main()
