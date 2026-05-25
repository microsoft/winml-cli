"""Iter 20260520-133849 — clean team-submission baseline.

All 18 responses (9 with_skill + 9 baseline) are fresh; no reused baselines.
Same SKILL.md + cases.json as iter 20260520-131138.
"""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSE_EVAL = HERE.parent.parent
ITER_DIR = HERE
VERIFY = RESPONSE_EVAL / "verify_commands.py"

ws_gradings = {
    "eval-snapdragon-resnet-build": [
        ("Recommends `winml inspect` (or equivalent inspection step) before kicking off the rest of the pipeline.", True,
         "Section 1 'Inspect — confirm ModelKit knows the model' runs `winml inspect -m microsoft/resnet-50` before config/build."),
        ("Identifies QNN as the correct EP for Snapdragon X Elite.", True,
         "Quote: 'ResNet-50 on QNN is the textbook BYOM path for ModelKit ... Snapdragon X Elite NPU.'"),
        ("Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path.", True,
         "Only QNN mentioned."),
        ("Walks through the build pipeline (export/optimize/quantize/compile or config+build).", True,
         "Section 2 config; section 3 build runs export -> optimize -> quantize -> compile."),
        ("Includes a benchmark step (winml perf or equivalent) that produces latency numbers.", True,
         "Section 4 `winml perf -m ... --device npu --ep qnn --iterations 500 --warmup 20 -c resnet50-qnn.json --monitor`."),
        ("Either directs the user to `winml --help` for current flags, OR uses only common/plausible flag patterns.", True,
         "All flags real; recommends `winml analyze --help` for diagnostics path."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"just got a dev box\" so they likely don't have winml installed yet.", True,
         "Section 0 'Prereqs — install winml' with full uv venv + AITK wheel + verify."),
    ],
    "eval-ryzen-ai-quick-benchmark": [
        ("Identifies VitisAI as the correct EP for AMD Ryzen AI NPU.", True,
         "Skill body's hardware table specifies VitisAI for Ryzen AI; response uses --ep vitisai / VitisAIExecutionProvider verification."),
        ("Recommends a single `winml perf` invocation as the primary action.", True,
         "Single one-shot `winml perf -m facebook/convnext-tiny-224 --device npu --ignore-cache` shown as the 'one command' step."),
        ("Does NOT instruct the user to run export, analyze, optimize, quantize, and compile as separate steps.", True,
         "Response explicitly notes 'winml perf builds artifacts on the fly' and avoids manual chaining."),
        ("Mentions running `winml inspect` first as a sanity check.", True,
         "Inspect step included as the golden-rule preflight before perf."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "All flags real; references `winml <cmd> --help` for fallback diagnostic."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"I have a Ryzen AI laptop\" with no signal of prior winml usage, so the default-include-install rule applies.", True,
         "Prereq install block included with uv venv + AITK wheel + verify."),
    ],
    "eval-npu-vs-cpu-comparison": [
        ("Recommends `winml inspect` (or equivalent inspection step) before any build/benchmark work.", True,
         "Inspect step explicitly included as a golden-rule preflight."),
        ("Identifies QNN as the EP for Snapdragon X Elite NPU.", True,
         "Quote: 'Snapdragon X Elite ... QNN execution provider ...'"),
        ("Recommends building the model once and then running `winml perf` twice -- once on NPU, once on CPU -- rather than building two separate pipelines.", True,
         "Response explains 'one config, one build that produces both artifacts, two perf runs.'"),
        ("Specifies that the CPU run should use the optimized (pre-compile) artifact, NOT the QNN-compiled artifact.", True,
         "Quote: 'CPU run: optimized (pre-compile) ONNX, on the CPU EP. Same source model, same optimization passes ...'"),
        ("Warns that EP-compiled artifacts are EP-locked / can't be run on a different EP and produce nonsense numbers.", True,
         "Quote: EP-compiled artifacts are bound to the EP they were compiled for; trying CPU EP with a QNN-compiled .onnx is at best meaningless."),
        ("Either directs user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "All flags real; references `winml <cmd> --help` for flag verification."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user's prompt does not signal prior winml usage, so the default-include-install rule applies.", True,
         "Prereq install block included with uv venv + AITK wheel + verify."),
    ],
    "eval-is-model-supported": [
        ("Recommends `winml inspect` as the first, cheapest check (config-only, no weights download).", True,
         "Inspect framed as the cheapest preflight ('reads HF config without downloading weights')."),
        ("Recommends `winml analyze` after inspect -- explicitly to find op patterns that won't survive optimize/quantize for the target EP.", True,
         "Analyze step explicitly framed as 'the step that would have saved you half a day' — per-EP op-coverage check."),
        ("Identifies OpenVINO as the EP for Intel NPU.", True,
         "Uses --ep openvino --device NPU; mentions OpenVINOExecutionProvider in the EP-registration check."),
        ("Explains analyze's role: a linter that classifies operators as supported / partial / unsupported per EP (so the user knows WHY this answers their question).", True,
         "Linter behavior explained (classifies operators per EP) plus exit code 0/1/2 mapping."),
        ("Does NOT just say \"build the full pipeline and see what happens\" -- that's exactly the workflow the user already said burned them.", True,
         "Whole response framed as pre-flight before committing to the build."),
    ],
    "eval-optimize-failure-recovery": [
        ("Recommends running `winml analyze` against the exported ONNX as the first diagnostic step.", True,
         "Step 1 'Re-run analyze against the exported ONNX' with concrete `winml analyze -m <exported> --ep qnn --device NPU` command."),
        ("Explains that analyze's linter will name the offending op pattern per EP (i.e., tells the user what analyze will tell THEM).", True,
         "Pattern families (Gelu1..4, LayerNormPow/Mul, ReshapeTransposeReshape, etc.) mapped to optimizer rewrite flags."),
        ("Does NOT recommend hand-editing the ONNX graph.", True,
         "Explicit avoidance: hand-editing the ONNX graph desynchronizes from the toolkit and is called out as a thing to avoid."),
        ("Suggests either trying a different optim/quant config to dodge the unsupported pattern OR escalating to \"this model isn't a fit for this EP\". Avoids dead-ending.", True,
         "Step 3 'Dodge the specific pattern' lists targeted retries; escalation path included in 'if none of the above works'."),
        ("Encourages reading the actual error message (which names op/EP/stage) before guessing -- doesn't jump to generic advice.", True,
         "Step 1 explicitly tells the user to capture the offending pattern name from the optimize error output."),
    ],
    "eval-install-setup": [
        ("Recommends Python 3.10 specifically, and calls out that the user's existing 3.11 will NOT work (winml-cli wheel pins >=3.10,<3.11).", True,
         "Quote: 'ModelKit pins Python 3.10 exactly (>=3.10,<3.11). Your system Python 3.11 will not resolve the winml-cli wheel.'"),
        ("Uses `uv venv --python 3.10` (or an equivalent isolated venv command) rather than installing into system Python or a 3.11 environment.", True,
         "Step 2 `uv venv --python 3.10`."),
        ("Points to the AITK cache as the wheel source today (e.g., `$env:USERPROFILE\\.aitk\\bin\\model_lab_runtime\\cache\\winml_cli-*.whl`) and acknowledges PyPI is the planned/future path. Does NOT fabricate a generic `pip install winml` URL as if it works today.", True,
         "Step 3 uses the exact AITK cache path and notes 'When winml-cli is published to PyPI -- planned -- this step becomes a plain uv pip install winml-cli.'"),
        ("Includes a post-install verification step (e.g., `winml --help` or `winml sys --list-ep`) so the user can confirm the install worked before moving on.", True,
         "Step 4 'Verify the install' runs both `winml --help` and `winml sys --list-device --list-ep`."),
    ],
    "eval-local-onnx-file": [
        ("Identifies OpenVINO as the correct EP for Intel Core Ultra NPU.", True,
         "Response uses --ep openvino and references OpenVINOExecutionProvider for the Intel Core Ultra NPU path."),
        ("Recognizes that winml commands accept local `.onnx` files directly via `-m` / `--model` -- no re-export needed.", True,
         "Response notes `winml perf` accepts local .onnx via `-m` and routes through the local-ONNX benchmark path."),
        ("Does NOT walk the user through `winml export` or any HuggingFace download -- the model is already on disk.", True,
         "No `winml export`; response goes straight from sanity-check to `winml perf` against the local file."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "All flags real; references `winml perf --help` for flag verification."),
    ],
    "eval-config-build-for-ci": [
        ("Recommends the `winml config` -> `winml build` two-step (the shortcut/pipeline path), not chained primitives (export/optimize/quantize/compile).", True,
         "Quote: 'The setup you want maps cleanly onto ModelKit's config + build path.' Steps 1-2 use exactly that flow."),
        ("States or strongly implies the JSON output of `winml config` is the artifact to commit / version-control -- it's the single source of truth for the build.", True,
         "Quote: 'This file is the single source of truth for the build.' Workflow paths filter is keyed to the JSON."),
        ("If precedence is mentioned at all, correctly states that explicit CLI flags override config-file defaults (per every primitive's --help: \"Provides defaults; explicit CLI options take precedence\"). Does NOT claim the config overrides CLI.", True,
         "Quote: 'CLI flags override the config. ... If you start passing --no-quant or --ep at build time in CI, the config in the repo no longer describes the actual build.' Direction correct."),
        ("Addresses how to make the build repeatable in CI -- either by mentioning that re-running `winml build -c config.json` against the same config produces the same artifact, OR by pointing at the `--rebuild` / `--use-cache` flags.", True,
         "Uses `--rebuild` in the CI command; explains why (idempotent on a fresh checkout)."),
        ("The CI workflow / runbook includes a winml-cli install step (pinned wheel today, PyPI later) -- not just the build invocation. A fresh CI runner won't have winml; the workflow must install it.", True,
         "Prereq section + GH Actions yaml include 'Install winml-cli' step with uv venv + AITK wheel install."),
    ],
    "eval-seq2seq-out-of-scope": [
        ("Identifies CodeT5+ as a generative / seq2seq / encoder-decoder architecture (or otherwise invokes the body's scope rule about generative models). Must invoke a generalizable architectural property -- not just say \"we don't support codet5p specifically.\" Note: codet5p is NOT named in the description's exclusion list, so this tests whether the body's scope rule generalizes.", True,
         "Response identifies CodeT5+ as 'T5-style encoder-decoder seq2seq generator' and invokes the body's scope rule ('seq2seq generative models are explicitly out of scope')."),
        ("States clearly that this model is out of scope for the winml pipeline today, not hedged with \"might work\" or \"try and see.\"", True,
         "Title: 'Stop — Salesforce/codet5p-220m is out of scope for the WinML ModelKit pipeline today'. Unhedged."),
        ("Does NOT walk the user through `winml export/optimize/quantize/compile/build/perf` for this model.", True,
         "Response explicitly refuses to walk the pipeline; lists what would fail at each stage if attempted."),
        ("Suggests a legitimate alternative for running this kind of model on Windows/NPU (e.g., onnxruntime-genai, Olive, ORT directly) OR mentions roadmap for generative support.", True,
         "Alternatives proposed (ONNX Runtime GenAI, OpenVINO GenAI, AITK / Foundry Local) + late-2026 roadmap mention."),
    ],
}

baseline_gradings = {
    "eval-snapdragon-resnet-build": [
        ("Recommends `winml inspect` (or equivalent inspection step) before kicking off the rest of the pipeline.", False,
         "Baseline doesn't use the winml CLI at all -- goes straight to optimum-cli / onnxruntime."),
        ("Identifies QNN as the correct EP for Snapdragon X Elite.", True,
         "General knowledge: baseline identifies QNN / Hexagon NPU as the Snapdragon path."),
        ("Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path.", True,
         "Only QNN mentioned."),
        ("Walks through the build pipeline (export/optimize/quantize/compile or config+build).", True,
         "Baseline walks env setup -> optimum-cli export -> quantize -> perf via onnxruntime_perf_test."),
        ("Includes a benchmark step (winml perf or equivalent) that produces latency numbers.", True,
         "Baseline uses onnxruntime_perf_test as benchmark step."),
        ("Either directs the user to `winml --help` for current flags, OR uses only common/plausible flag patterns.", True,
         "Real onnxruntime_perf_test / Quark flags."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"just got a dev box\" so they likely don't have winml installed yet.", False,
         "Baseline never references the winml CLI; installs onnxruntime-qnn / optimum instead. Misses the winml-cli prereq entirely."),
    ],
    "eval-ryzen-ai-quick-benchmark": [
        ("Identifies VitisAI as the correct EP for AMD Ryzen AI NPU.", True,
         "General knowledge identifies VitisAI as the AMD NPU EP."),
        ("Recommends a single `winml perf` invocation as the primary action.", False,
         "Baseline walks several manual steps (install Ryzen AI software -> export -> quantize -> perf_test). Not a 'fastest path to the number' shape."),
        ("Does NOT instruct the user to run export, analyze, optimize, quantize, and compile as separate steps.", False,
         "Baseline explicitly walks export + quantize as separate steps before benchmarking."),
        ("Mentions running `winml inspect` first as a sanity check.", False,
         "Baseline never mentions winml. Uses Device-Manager-level NPU check instead, but that's environment-level not model-level."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Uses real Quark / onnxruntime_perf_test flag patterns."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"I have a Ryzen AI laptop\" with no signal of prior winml usage, so the default-include-install rule applies.", False,
         "Baseline installs Ryzen AI software + onnxruntime-vitisai instead of winml-cli. Misses the winml-cli prereq."),
    ],
    "eval-npu-vs-cpu-comparison": [
        ("Recommends `winml inspect` (or equivalent inspection step) before any build/benchmark work.", False,
         "Baseline doesn't use winml inspect; goes straight to optimum-cli export."),
        ("Identifies QNN as the EP for Snapdragon X Elite NPU.", True,
         "General knowledge identifies QNN / Hexagon NPU."),
        ("Recommends building the model once and then running `winml perf` twice -- once on NPU, once on CPU -- rather than building two separate pipelines.", False,
         "Baseline exports once but quantizes separately for NPU (separate QDQ file), then benchmarks. Doesn't have the 'shared build, two perf runs' shape."),
        ("Specifies that the CPU run should use the optimized (pre-compile) artifact, NOT the QNN-compiled artifact.", False,
         "Baseline has no notion of an EP-compiled artifact -- uses FP32 ONNX for CPU and a separately quantized QDQ ONNX for NPU."),
        ("Warns that EP-compiled artifacts are EP-locked / can't be run on a different EP and produce nonsense numbers.", False,
         "Baseline never warns about EP-locking."),
        ("Either directs user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Real ORT/QNN APIs."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user's prompt does not signal prior winml usage, so the default-include-install rule applies.", False,
         "Baseline installs onnxruntime-qnn / optimum, not the winml CLI."),
    ],
    "eval-is-model-supported": [
        ("Recommends `winml inspect` as the first, cheapest check (config-only, no weights download).", False,
         "Baseline doesn't use winml inspect -- recommends export + ONNX-level introspection as the pre-flight."),
        ("Recommends `winml analyze` after inspect -- explicitly to find op patterns that won't survive optimize/quantize for the target EP.", False,
         "Baseline doesn't recommend winml analyze -- uses ad-hoc try-and-see / netron inspection."),
        ("Identifies OpenVINO as the EP for Intel NPU.", True,
         "General knowledge identifies OpenVINO as the Intel NPU EP."),
        ("Explains analyze's role: a linter that classifies operators as supported / partial / unsupported per EP (so the user knows WHY this answers their question).", False,
         "No mention of an EP-coverage linter step. Pre-flight is much less specific."),
        ("Does NOT just say \"build the full pipeline and see what happens\" -- that's exactly the workflow the user already said burned them.", False,
         "Baseline largely recommends try-and-see / run-export-then-quantize approach without the cheap pre-flight that would catch op-pattern issues."),
    ],
    "eval-optimize-failure-recovery": [
        ("Recommends running `winml analyze` against the exported ONNX as the first diagnostic step.", False,
         "Baseline doesn't know about winml analyze. Recommends generic ORT-level debugging."),
        ("Explains that analyze's linter will name the offending op pattern per EP (i.e., tells the user what analyze will tell THEM).", False,
         "Baseline doesn't surface analyze's per-EP linter; suggests reading raw error or trying different opset."),
        ("Does NOT recommend hand-editing the ONNX graph.", True,
         "Baseline doesn't recommend graph hand-editing -- mostly suggests opset changes / re-export."),
        ("Suggests either trying a different optim/quant config to dodge the unsupported pattern OR escalating to \"this model isn't a fit for this EP\". Avoids dead-ending.", True,
         "Baseline does suggest config / quant changes as a recovery path -- partial credit; the recovery is generic but exists."),
        ("Encourages reading the actual error message (which names op/EP/stage) before guessing -- doesn't jump to generic advice.", True,
         "Baseline does encourage reading the raw error."),
    ],
    "eval-install-setup": [
        ("Recommends Python 3.10 specifically, and calls out that the user's existing 3.11 will NOT work (winml-cli wheel pins >=3.10,<3.11).", False,
         "Baseline endorses Python 3.11 / uses py -3.11 -m venv. Doesn't know about the 3.10 pin."),
        ("Uses `uv venv --python 3.10` (or an equivalent isolated venv command) rather than installing into system Python or a 3.11 environment.", False,
         "Baseline uses py -3.11 -m venv; uv only mentioned at the end as optional tooling."),
        ("Points to the AITK cache as the wheel source today (e.g., `$env:USERPROFILE\\.aitk\\bin\\model_lab_runtime\\cache\\winml_cli-*.whl`) and acknowledges PyPI is the planned/future path. Does NOT fabricate a generic `pip install winml` URL as if it works today.", False,
         "Baseline interprets 'winml' as Windows ML runtime + onnxruntime-qnn (pip). No mention of AITK cache or winml-cli."),
        ("Includes a post-install verification step (e.g., `winml --help` or `winml sys --list-ep`) so the user can confirm the install worked before moving on.", False,
         "Baseline's smoke test verifies onnxruntime-qnn, not the winml CLI."),
    ],
    "eval-local-onnx-file": [
        ("Identifies OpenVINO as the correct EP for Intel Core Ultra NPU.", True,
         "General knowledge identifies OpenVINO as the Intel NPU EP."),
        ("Recognizes that winml commands accept local `.onnx` files directly via `-m` / `--model` -- no re-export needed.", False,
         "Baseline doesn't use winml CLI -- uses raw onnxruntime InferenceSession with the local file path."),
        ("Does NOT walk the user through `winml export` or any HuggingFace download -- the model is already on disk.", True,
         "No re-export; uses local file directly."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "No fabricated winml flags. Recommends OpenVINO's benchmark_app, a real tool."),
    ],
    "eval-config-build-for-ci": [
        ("Recommends the `winml config` -> `winml build` two-step (the shortcut/pipeline path), not chained primitives (export/optimize/quantize/compile).", False,
         "Baseline writes its own build_model.py that manually chains download -> optimize -> quantize -> qnn-compile. Doesn't know about winml config / winml build."),
        ("States or strongly implies the JSON output of `winml config` is the artifact to commit / version-control -- it's the single source of truth for the build.", False,
         "Baseline commits a custom-schema model.config.json, not the output of winml config. Misses the canonical artifact."),
        ("If precedence is mentioned at all, correctly states that explicit CLI flags override config-file defaults (per every primitive's --help: \"Provides defaults; explicit CLI options take precedence\"). Does NOT claim the config overrides CLI.", True,
         "Baseline doesn't mention CLI vs config precedence at all -- vacuously passes (no reversal)."),
        ("Addresses how to make the build repeatable in CI -- either by mentioning that re-running `winml build -c config.json` against the same config produces the same artifact, OR by pointing at the `--rebuild` / `--use-cache` flags.", True,
         "Baseline goes hard on reproducibility (pinned deps, sha256, manifest, fixed seeds)."),
        ("The CI workflow / runbook includes a winml-cli install step (pinned wheel today, PyPI later) -- not just the build invocation. A fresh CI runner won't have winml; the workflow must install it.", False,
         "Baseline CI yaml uses `uv pip sync requirements.lock` with onnxruntime-qnn etc., not winml-cli."),
    ],
    "eval-seq2seq-out-of-scope": [
        ("Identifies CodeT5+ as a generative / seq2seq / encoder-decoder architecture (or otherwise invokes the body's scope rule about generative models). Must invoke a generalizable architectural property -- not just say \"we don't support codet5p specifically.\" Note: codet5p is NOT named in the description's exclusion list, so this tests whether the body's scope rule generalizes.", True,
         "Baseline correctly identifies CodeT5+ as T5-family encoder-decoder seq2seq; explains decoder/KV-cache implications."),
        ("States clearly that this model is out of scope for the winml pipeline today, not hedged with \"might work\" or \"try and see.\"", False,
         "Baseline is heavily hedged -- 'is the juice worth the squeeze' framing + walks the pipeline 'if you still want to try.'"),
        ("Does NOT walk the user through `winml export/optimize/quantize/compile/build/perf` for this model.", False,
         "Baseline explicitly walks the pipeline as an option."),
        ("Suggests a legitimate alternative for running this kind of model on Windows/NPU (e.g., onnxruntime-genai, Olive, ORT directly) OR mentions roadmap for generative support.", True,
         "Baseline suggests CPU fallback, smaller distilled model, retrieval/classification approach."),
    ],
}

ASSERTION_RUNNABLE = "All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only."

eval_ids = {
    "eval-snapdragon-resnet-build": 1, "eval-ryzen-ai-quick-benchmark": 2,
    "eval-npu-vs-cpu-comparison": 4, "eval-is-model-supported": 5,
    "eval-optimize-failure-recovery": 6, "eval-install-setup": 7,
    "eval-local-onnx-file": 8, "eval-config-build-for-ci": 9,
    "eval-seq2seq-out-of-scope": 10,
}


def runnable_check(resp_path: Path) -> tuple[bool, str]:
    r = subprocess.run(["python", str(VERIFY), str(resp_path)], capture_output=True, text=True)
    out = r.stdout
    if "FAILURES" in out:
        fails = [l.strip() for l in out.splitlines() if l.strip().startswith("- `")]
        return False, f"verify_commands.py found {len(fails)} broken command(s)"
    if "No winml commands" in out:
        return True, "Response uses no winml commands (trivially passes)."
    m = re.search(r"All (\d+) commands verified OK", out)
    n = m.group(1) if m else "?"
    return True, f"All {n} winml commands verified runnable against --help."


def write_grading(eval_name: str, config: str, items: list) -> None:
    resp = ITER_DIR / eval_name / config / "run-1" / "outputs" / "response.md"
    run_passed, run_evidence = runnable_check(resp)
    expectations = [{"text": t, "passed": p, "evidence": e} for (t, p, e) in items]
    expectations.append({"text": ASSERTION_RUNNABLE, "passed": run_passed, "evidence": run_evidence})
    total = len(expectations)
    passed_n = sum(1 for e in expectations if e["passed"])
    data = {
        "eval_id": eval_ids[eval_name],
        "eval_name": eval_name.replace("eval-", ""),
        "config": config,
        "expectations": expectations,
        "summary": {"pass_rate": round(passed_n / total, 4), "passed": passed_n, "failed": total - passed_n, "total": total},
    }
    (ITER_DIR / eval_name / config / "run-1" / "grading.json").write_text(json.dumps(data, indent=2))
    print(f"  {eval_name}/{config}: {passed_n}/{total}")


print("with_skill gradings:")
for eval_name, items in ws_gradings.items():
    write_grading(eval_name, "with_skill", items)
print()
print("without_skill (baseline) gradings:")
for eval_name, items in baseline_gradings.items():
    write_grading(eval_name, "without_skill", items)
