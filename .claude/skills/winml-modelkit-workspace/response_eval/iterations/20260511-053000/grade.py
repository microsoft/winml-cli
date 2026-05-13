"""Write iter-3 with_skill gradings, including the commands-runnable assertion."""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent              # response_eval/iterations/iter-3/
RESPONSE_EVAL = HERE.parent.parent                   # response_eval/
ITER_DIR = HERE                                      # this iteration's directory
VERIFY = RESPONSE_EVAL / "verify_commands.py"

gradings = {
    "eval-snapdragon-resnet-build": [
        ("Recommends `winml inspect` (or equivalent inspection step) before kicking off the rest of the pipeline.", True, "Section 2 Inspect the model runs winml inspect -m microsoft/resnet-50 before any build step."),
        ("Identifies QNN as the correct EP for Snapdragon X Elite.", True, "Quote: 'Snapdragon X Elite means we are targeting the QNN execution provider on the Hexagon NPU.'"),
        ("Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path.", True, "Only QNN; no OpenVINO/VitisAI."),
        ("Walks through the build pipeline (export/optimize/quantize/compile or config+build).", True, "Uses config+build path; lists 'export to analyze to optimize to quantize to compile' as what build runs."),
        ("Includes a benchmark step (`winml perf` or equivalent) that produces latency numbers.", True, "Section 5 winml perf with --iterations 500 --warmup 50 --monitor, reports mean/p50/p95/p99."),
        ("Either directs the user to `winml --help` for current flags, OR uses only common/plausible flag patterns.", True, "Quote: 'If any step errors out, run winml <that-command> --help -- flags evolve, and the CLI is the source of truth.'"),
    ],
    "eval-ryzen-ai-quick-benchmark": [
        ("Identifies VitisAI as the correct EP for AMD Ryzen AI NPU.", True, "Quote: '--device npu on a Ryzen AI box routes to the VitisAI EP.'"),
        ("Recommends a single `winml perf` invocation (or equivalent one-shot benchmark) as the primary action.", True, "Two commands (sys + perf). Quote: 'That is it. perf will print mean/median/p95 latency'."),
        ("Does NOT instruct the user to run all of export, analyze, optimize, quantize, and compile as separate explicit steps.", True, "Quote: 'skip the build pipeline entirely and let winml perf do download + export + optimize + quantize + compile in one shot'."),
        ("Mentions running `winml inspect` first as a sanity check.", False, "Response uses prior knowledge ('ConvNeXT is a CNN -- in scope for ModelKit') and winml sys as sanity checks, but does NOT run winml inspect."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True, "All flags verified live via winml <cmd> --help."),
    ],
    "eval-llm-out-of-scope": [
        ("Identifies Phi-3 as a decoder-only LLM / generative model.", True, "Quote: 'Phi-3 is a generative decoder-only LLM, and that whole family (GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, seq2seq generators) is explicitly out of scope.'"),
        ("Clearly states this model is NOT supported / out of scope for the winml pipeline today (not hedged).", True, "Quote: 'Short version: WinML ModelKit can not do this yet.' + 'explicitly out of scope for the current pipeline.'"),
        ("Does NOT walk the user through export/optimize/quantize/compile/build.", True, "No winml build pipeline commands recommended."),
        ("Mentions LLM support is on the roadmap OR suggests a legitimate alternative.", True, "Both: 'late 2026' roadmap + OpenVINO GenAI / ONNX Runtime GenAI / DirectML alternatives. Notes 'None of those go through winml.'"),
    ],
    "eval-npu-vs-cpu-comparison": [
        ("Recommends `winml inspect` before any build/benchmark work.", True, "Step 1: 'uv run winml inspect -m google/vit-base-patch16-224' to confirm Supported."),
        ("Identifies QNN as the EP for Snapdragon X Elite NPU.", True, "Quote: 'QNN (Qualcomm's NPU EP on Snapdragon X Elite) should be a real option.'"),
        ("Recommends building once and running `winml perf` twice.", True, "Quote: 'The clean approach is to build once and then run perf twice, pointing each run at the right artifact for that EP.'"),
        ("Specifies that CPU run uses optimized (pre-compile) artifact, NOT QNN-compiled.", True, "Quote: 'The optimized one is your CPU benchmark target; the compiled one is the NPU target.'"),
        ("Warns EP-compiled artifacts are EP-locked.", True, "Quote: 'The QNN-compiled .onnx is EP-locked -- you cannot run it on CPU and get meaningful numbers.'"),
        ("Directs to `--help` or uses common patterns; no fabricated flags.", True, "All flags verified via --help."),
    ],
    "eval-is-model-supported": [
        ("Recommends `winml inspect` as the first, cheapest check (config-only).", True, "Step 1 winml inspect; 'reads only the model config (no weights, no GPU).'"),
        ("Recommends `winml analyze` after inspect.", True, "Step 2 winml analyze --ep ov --device npu."),
        ("Identifies OpenVINO as the EP for Intel NPU.", True, "Quote: 'For your Intel NPU, the EP is OpenVINO (--ep ov, --device npu).'"),
        ("Explains analyze's role: linter that classifies operators as supported/partial/unsupported per EP.", True, "Quote: 'analyze runs a static lint of every operator in the exported ONNX against the target EP and tells you which ops are supported / partial / unsupported.'"),
        ("Does NOT just say 'build the full pipeline and see what happens'.", True, "Explicit section 'Why not just run winml build and let it fail fast?' justifying the pre-flight."),
    ],
    "eval-optimize-failure-recovery": [
        ("Recommends running `winml analyze` against the exported ONNX as the first diagnostic step.", True, "Quote: 'Run analyze against your exported ONNX (not the HF id) and target QNN' with concrete command."),
        ("Explains analyze's linter will name the offending op pattern per EP.", True, "Quote: 'it lints every operator and pattern per EP and will name the offending op or pattern explicitly'."),
        ("Does NOT recommend hand-editing the ONNX graph.", True, "Quote: 'do not hand-edit the ONNX graph'."),
        ("Suggests trying different optim/quant config OR escalating to 'not a fit'.", True, "Both: --disable-... toggles + --optim-config flow + escalation path."),
        ("Encourages reading actual error message before guessing.", True, "Asks user to share analyze JSON / 'just the offending pattern name'; lists Swin patterns 'If the error mentions a specific op'."),
    ],
}

ASSERTION_RUNNABLE = "All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only."

eval_ids = {
    "eval-snapdragon-resnet-build": 1,
    "eval-ryzen-ai-quick-benchmark": 2,
    "eval-llm-out-of-scope": 3,
    "eval-npu-vs-cpu-comparison": 4,
    "eval-is-model-supported": 5,
    "eval-optimize-failure-recovery": 6,
}

for eval_name, items in gradings.items():
    resp = ITER_DIR / eval_name / "with_skill" / "run-1" / "outputs" / "response.md"
    r = subprocess.run(["python", str(VERIFY), str(resp)], capture_output=True, text=True)
    out = r.stdout
    if "FAILURES" in out:
        fails = [l.strip() for l in out.splitlines() if l.strip().startswith("- `")]
        run_passed = False
        run_evidence = f"verify_commands.py found {len(fails)} broken command(s)"
    elif "No winml commands" in out:
        run_passed = True
        run_evidence = "Response uses no winml commands (trivially passes)."
    else:
        m = re.search(r"All (\d+) commands verified OK", out)
        n = m.group(1) if m else "?"
        run_passed = True
        run_evidence = f"All {n} winml commands verified runnable against --help."

    expectations = [{"text": t, "passed": p, "evidence": e} for (t, p, e) in items]
    expectations.append({"text": ASSERTION_RUNNABLE, "passed": run_passed, "evidence": run_evidence})

    total = len(expectations)
    passed_n = sum(1 for e in expectations if e["passed"])

    data = {
        "eval_id": eval_ids[eval_name],
        "eval_name": eval_name.replace("eval-", ""),
        "config": "with_skill",
        "expectations": expectations,
        "summary": {"pass_rate": round(passed_n / total, 4), "passed": passed_n, "failed": total - passed_n, "total": total},
    }
    grading_path = ITER_DIR / eval_name / "with_skill" / "run-1" / "grading.json"
    grading_path.write_text(json.dumps(data, indent=2))
    print(f"{eval_name}/with_skill: {passed_n}/{total}")
