"""Iter 20260524-133637 — generalize SKILL.md from prescriptive to principle.

SKILL.md changes vs 20260524-120701:
  - Added new section "Outputs are explicit; cache is opaque" right after
    "The mental model". Teaches the general published-output-vs-cache rule.
  - Slimmed "Disambiguating 'I want to run X'" -> "Mapping 'I want to run X'
    to a command": kept the intent->command table (now framed in terms of
    each command's published output), dropped the "Order matters: build
    before perf, never perf before build" prescription and the trailing
    prose. Both derive from the new principle section.
  - Removed the perf-doesn't-expose-artifact caveat from the "Just benchmark"
    pattern. Also derivable from the new principle.

Goal: same coverage of the original failure modes (perf->build->perf,
"where did perf put the model"), but anchored on one teachable rule rather
than scattered prescriptions. Future similar failures (e.g. "use analyze
output as graph rewrite") should now be covered automatically.

with_skill: regenerated for all 11 cases (skill body changed).
baseline:   reused from 20260524-120701 (response.md copied during run-dir
            setup; gradings identical since baseline doesn't load skill).
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
         "Section 2 'Inspect the model (golden rule — always do this first)' runs `winml inspect -m microsoft/resnet-50` before config/build."),
        ("Identifies QNN as the correct EP for Snapdragon X Elite.", True,
         "Quote: 'Snapdragon X Elite is a first-class target (Qualcomm QNN EP, Ready)'."),
        ("Does NOT recommend OpenVINO, VitisAI, or other non-Qualcomm EPs as the primary path.", True,
         "Only QNN mentioned for the build/perf path."),
        ("Walks through the build pipeline (export/optimize/quantize/compile or config+build).", True,
         "Steps 4-5 use winml config + winml build (which runs export -> optimize -> quantize -> compile)."),
        ("Includes a benchmark step (winml perf or equivalent) that produces latency numbers.", True,
         "Step 6 'Benchmark latency on the NPU' runs `winml perf -m .\\out\\resnet50-qnn\\model.onnx --ep QNN -o .\\out\\resnet50-qnn\\perf.json`."),
        ("Either directs the user to `winml --help` for current flags, OR uses only common/plausible flag patterns.", True,
         "Explicit closing reminder: 'If any line above rejects a flag, run winml <that-command> --help and use what it prints — don't try to massage my command. The CLI's --help is always the source of truth.'"),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"just got a dev box\" so they likely don't have winml installed yet.", True,
         "Section 0 'Prereqs — install the winml CLI' with uv venv + AITK wheel + smoke test."),
    ],
    "eval-ryzen-ai-quick-benchmark": [
        ("Identifies VitisAI as the correct EP for AMD Ryzen AI NPU.", True,
         "Quote: 'For AMD Ryzen AI, the NPU execution provider is VitisAI (Ready as of 2026-05-20)'."),
        ("Recommends a single `winml perf` invocation as the primary action.", True,
         "Title 'Fastest path to a latency number on Ryzen AI NPU' + Step 3 is a single `winml perf -m facebook/convnext-tiny-224 -o perf-convnext-vitisai.json`."),
        ("Does NOT instruct the user to run export, analyze, optimize, quantize, and compile as separate steps.", True,
         "Quote: 'It builds whatever it needs on the fly (export -> optimize -> quantize -> compile) and writes a metrics JSON. No need to walk the pipeline manually.'"),
        ("Mentions running `winml inspect` first as a sanity check.", True,
         "Step 2 'Inspect (golden rule, cheap, no download)' runs `winml inspect -m facebook/convnext-tiny-224`."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Explicit 'check the real flag spellings on your installed version' + `winml perf --help`. Plus a 'Don't invent flags' gotcha."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user said \"I have a Ryzen AI laptop\" with no signal of prior winml usage, so the default-include-install rule applies.", True,
         "'Prereqs (skip if winml --help already works)' block with uv venv + AITK wheel + verify."),
    ],
    "eval-npu-vs-cpu-comparison": [
        ("Recommends `winml inspect` (or equivalent inspection step) before any build/benchmark work.", True,
         "Step 1 'inspect first' runs `winml inspect -m google/vit-base-patch16-224`."),
        ("Identifies QNN as the EP for Snapdragon X Elite NPU.", True,
         "Quote: 'Snapdragon X Elite means the QNN execution provider is Ready'."),
        ("Recommends building the model once and then running `winml perf` twice -- once on NPU, once on CPU -- rather than building two separate pipelines.", True,
         "Step 3 'build once, perf twice' — single config + build emits both compiled and optimized artifacts; perfs twice."),
        ("Specifies that the CPU run should use the optimized (pre-compile) artifact, NOT the QNN-compiled artifact.", True,
         "Quote: 'CPU number — perf the pre-compile optimized ONNX against the CPU EP. Do not point CPU at the QNN-compiled file.'"),
        ("Warns that EP-compiled artifacts are EP-locked / can't be run on a different EP and produce nonsense numbers.", True,
         "Quote: 'compiled artifacts are tied to the EP they were compiled for and the number you'd get back would be meaningless'."),
        ("Either directs user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Multiple 'confirm exact flag spelling with winml perf --help' nudges; closing line: 'Run winml perf --help and winml build --help before pasting commands to confirm current flag spellings'."),
        ("Includes an install/prereq block (or a clear pointer back to install steps) -- the user's prompt does not signal prior winml usage, so the default-include-install rule applies.", True,
         "'Prereq: install winml' block with uv venv + AITK wheel + verify."),
    ],
    "eval-is-model-supported": [
        ("Recommends `winml inspect` as the first, cheapest check (config-only, no weights download).", True,
         "Quote: 'inspect — does ModelKit know how to handle this architecture? It reads the Hugging Face config without downloading weights'."),
        ("Recommends `winml analyze` after inspect -- explicitly to find op patterns that won't survive optimize/quantize for the target EP.", True,
         "Quote: 'analyze — will the operators survive optimize/quantize/compile on the OpenVINO EP? This is the one that specifically addresses your \"died at optimize\" scar.' Plus explicit ordering: 'Always inspect first, then analyze.'"),
        ("Identifies OpenVINO as the EP for Intel NPU.", True,
         "Quote: 'Intel NPU goes through the OpenVINO EP, which is currently Ready'."),
        ("Explains analyze's role: a linter that classifies operators as supported / partial / unsupported per EP (so the user knows WHY this answers their question).", True,
         "Quote: 'The analyzer's linter walks every operator in the graph and classifies it as supported / partial / unsupported for the EP you're targeting'."),
        ("Does NOT just say \"build the full pipeline and see what happens\" -- that's exactly the workflow the user already said burned them.", True,
         "Whole response framed as cheap pre-flight ('Run them in order; if either one is unhappy, don't proceed to export') before committing to a full build."),
    ],
    "eval-optimize-failure-recovery": [
        ("Recommends running `winml analyze` against the exported ONNX as the first diagnostic step.", True,
         "Section 1 'Read the error literally, then run analyze against the exported ONNX' runs `winml analyze -m <path-to-your-exported-swin.onnx> --ep QNN`."),
        ("Explains that analyze's linter will name the offending op pattern per EP (i.e., tells the user what analyze will tell THEM).", True,
         "Quote: 'The linter classifies every operator as supported / partial / unsupported per EP. It will almost always either confirm what the optimize error already named, or surface a related pattern nearby in the graph.'"),
        ("Does NOT recommend hand-editing the ONNX graph.", True,
         "Quote: 'The skill's explicit guidance for this exact situation: don't hand-edit the ONNX graph. Try a different optim or quantization configuration to dodge the unsupported pattern.'"),
        ("Suggests either trying a different optim/quant config to dodge the unsupported pattern OR escalating to \"this model isn't a fit for this EP\". Avoids dead-ending.", True,
         "Section 2 dodges via different optim config; Section 3 escalates ('Two reasonable moves: fall back to DML/CPU, or swap to a smaller Swin variant')."),
        ("Encourages reading the actual error message (which names op/EP/stage) before guessing -- doesn't jump to generic advice.", True,
         "Section 1 opens with 'Read the error literally' and 'look at the actual op pattern the optimize step is complaining about and write it down'."),
    ],
    "eval-install-setup": [
        ("Recommends Python 3.10 specifically, and calls out that the user's existing 3.11 will NOT work (winml-cli wheel pins >=3.10,<3.11).", True,
         "Quote: 'ModelKit pins Python 3.10 exactly (>=3.10,<3.11). Your system 3.11 won't resolve the winml-cli wheel.'"),
        ("Uses `uv venv --python 3.10` (or an equivalent isolated venv command) rather than installing into system Python or a 3.11 environment.", True,
         "Step 2 runs `uv venv --python 3.10`."),
        ("Points to the AITK cache as the wheel source today (e.g., `$env:USERPROFILE\\.aitk\\bin\\model_lab_runtime\\cache\\winml_cli-*.whl`) and acknowledges PyPI is the planned/future path. Does NOT fabricate a generic `pip install winml` URL as if it works today.", True,
         "Step 3 uses exact AITK path + 'When winml-cli lands on PyPI (planned), the equivalent will just be uv pip install winml-cli.'"),
        ("Includes a post-install verification step (e.g., `winml --help` or `winml sys --list-ep`) so the user can confirm the install worked before moving on.", True,
         "Step 4 'Verify the install and confirm your NPU is visible' runs both `winml --help` and `winml sys --list-device --list-ep`."),
    ],
    "eval-local-onnx-file": [
        ("Identifies OpenVINO as the correct EP for Intel Core Ultra NPU.", True,
         "Quote: 'The Intel NPU goes through the OpenVINO execution provider, which is Ready on Core Ultra (Meteor Lake / Lunar Lake)'."),
        ("Recognizes that winml commands accept local `.onnx` files directly via `-m` / `--model` -- no re-export needed.", True,
         "Quote: 'winml accepts a local .onnx path anywhere a model is expected, so you can point it straight at the file on disk'."),
        ("Does NOT walk the user through `winml export` or any HuggingFace download -- the model is already on disk.", True,
         "Quote: 'Since you already have an ONNX file, you're entering the pipeline past export'. No export step."),
        ("Either directs the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Multiple 'Read winml perf --help first' nudges + 'I don't want to invent one'."),
    ],
    "eval-config-build-for-ci": [
        ("Recommends the `winml config` -> `winml build` two-step (the shortcut/pipeline path), not chained primitives (export/optimize/quantize/compile).", True,
         "Section 1 'Generate the config once, locally, then commit it' uses winml config; Section 2 'The reproducible build step' uses winml build. Explicit 'config + build pair is designed for' this."),
        ("States or strongly implies the JSON output of `winml config` is the artifact to commit / version-control -- it's the single source of truth for the build.", True,
         "Quote: 'Commit configs/resnet50-qnn.json. That file is now your reproducible build spec — version-controlled, diff-able in PRs, and the thing CI watches.'"),
        ("If precedence is mentioned at all, correctly states that explicit CLI flags override config-file defaults (per every primitive's --help: \"Provides defaults; explicit CLI options take precedence\"). Does NOT claim the config overrides CLI.", True,
         "Quote: '-c provides defaults; CLI flags override. Per every primitive's --help, anything you pass on the command line wins over the JSON.'"),
        ("Addresses how to make the build repeatable in CI -- either by mentioning that re-running `winml build -c config.json` against the same config produces the same artifact, OR by pointing at the `--rebuild` / `--use-cache` flags.", True,
         "Quote: 'winml build reads it and replays the full pipeline (export -> optimize -> quantize -> compile) deterministically' — reproducibility addressed."),
        ("The CI workflow / runbook includes a winml-cli install step (pinned wheel today, PyPI later) -- not just the build invocation. A fresh CI runner won't have winml; the workflow must install it.", True,
         "Section 2 step 1 is the install: 'Set up Python 3.10 + winml-cli (same as a fresh dev box)' with uv venv + AITK wheel install + comment '(swap to uv pip install winml-cli once it's on PyPI)'."),
    ],
    "eval-seq2seq-out-of-scope": [
        ("Identifies CodeT5+ as a generative / seq2seq / encoder-decoder architecture (or otherwise invokes the body's scope rule about generative models). Must invoke a generalizable architectural property -- not just say \"we don't support codet5p specifically.\" Note: codet5p is NOT named in the description's exclusion list, so this tests whether the body's scope rule generalizes.", True,
         "Quote: 'Salesforce/codet5p-220m is a T5-family encoder-decoder (seq2seq) model, and seq2seq / decoder-only / generative architectures are explicitly out of scope'. Architectural property invoked."),
        ("States clearly that this model is out of scope for the winml pipeline today, not hedged with \"might work\" or \"try and see.\"", True,
         "Title: 'Heads up: CodeT5+ 220M is out of scope for WinML ModelKit today'. Plus 'The honest answer is: don't run that model through winml today.' Unhedged."),
        ("Does NOT walk the user through `winml export/optimize/quantize/compile/build/perf` for this model.", True,
         "Quote: 'I don't want to walk you through inspect -> export -> analyze -> optimize -> quantize -> compile -> perf and have you hit a wall'. The pipeline shape shown at the end is explicitly for a 'replacement' in-scope model, not CodeT5+."),
        ("Suggests a legitimate alternative for running this kind of model on Windows/NPU (e.g., onnxruntime-genai, Olive, ORT directly) OR mentions roadmap for generative support.", True,
         "Four realistic paths listed: encoder model reframe, QNN SDK / AITK generative path, wait for late-2026 roadmap, smaller encoder + extractive."),
    ],
    "eval-ambiguous-run-intent": [
        ("Recognizes the prompt is ambiguous between 'just get a latency number' and 'produce a deployable artifact' -- either asks the user to clarify, OR explicitly presents both paths with a reasoned recommendation. Does NOT silently pick one path without acknowledging the other exists.", True,
         "Section 'Quick clarifier' explicitly enumerates three intents (artifact / number / both) with the right command for each. Picks 'a deployable artifact, then benchmark it' as the default while telling the user how to skip to just-perf."),
        ("Does NOT instruct the user to run `winml perf` first, then `winml build`, then `winml perf` again -- that sequence wastes the on-the-fly build that perf already did.", True,
         "Closing 'Just want the latency number?' section explicitly: 'If you later decide you want a deployable model, you'll have to run build separately (it won't cost double if cache is warm, but the artifact only becomes \"yours\" via build -o).' No perf->build->perf sequence anywhere."),
        ("If the response includes both a build step and a perf step, the order is `winml build` BEFORE `winml perf`, never perf before build.", True,
         "Section 'Quick clarifier' line: 'You want both -> winml build first to produce the artifact, then winml perf against the built artifact.' Order is build->perf."),
        ("Recommends `winml inspect` before other pipeline commands.", True,
         "Step 1 'Inspect first (always)' runs `winml inspect -m microsoft/resnet-50`."),
        ("Identifies QNN as the EP for Snapdragon X Elite.", True,
         "Quote: 'Snapdragon X Elite has a Qualcomm Hexagon NPU, which winml drives via the QNN execution provider'."),
        ("Either points the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Multiple 'check ... --help' nudges; closing troubleshooting bullet: 'Any flag I quoted above doesn't exist -> run winml <command> --help and use the real one.'"),
        ("Includes an install/prereq block -- the prompt gives no signal of prior winml usage.", True,
         "Section 'Prereq: install the winml CLI' with uv venv + AITK wheel + smoke test."),
    ],
    "eval-perf-output-recovery": [
        ("Correctly states that `winml perf`'s `-o` flag produces a perf metrics JSON, NOT the built model.", True,
         "Quote: 'the only published output of perf is the metrics JSON (whatever you passed to -o, or whatever it printed)'."),
        ("States that the on-the-fly artifact perf built lives in cache (or a temp folder with `--ignore-cache`) and is not directly exposed to the user as a deployable file.", True,
         "Quote: 'The build it used lives in an internal cache that isn't a supported output — don't try to fish it out, copy it, or hand it to your app. Cache layout is opaque and can change between releases.'"),
        ("Recommends running `winml build` with `-o <dir>` as the way to get a deployable artifact at a known location.", True,
         "Quote: 'Re-enter the pipeline at build, which writes the optimized + quantized + EP-compiled artifact to a directory you choose.' Followed by `winml build -c resnet50-qnn.json -o .\\dist\\resnet50-qnn`."),
        ("Notes that `winml build` needs a config file first (e.g., generated by `winml config`).", True,
         "Step 1 of the flow: '# 1. Generate a config that captures the same pipeline perf used' with `winml config -m microsoft/resnet-50 --device npu -o resnet50-qnn.json`."),
        ("Does NOT fabricate a phantom path like 'perf saves the model at /tmp/...' or invent a hidden flag on `winml perf` that exposes the model.", True,
         "Quote: 'Even if you found the cached build, it's not a contract — the path, naming, and whether the .bin sidecar is there are all implementation details.' No invented path or flag."),
        ("Either points the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Explicit 'confirm exact flags with --help — I'm not going to invent them' followed by three --help calls."),
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
    "eval-ambiguous-run-intent": [
        ("Recognizes the prompt is ambiguous between 'just get a latency number' and 'produce a deployable artifact' -- either asks the user to clarify, OR explicitly presents both paths with a reasoned recommendation. Does NOT silently pick one path without acknowledging the other exists.", False,
         "Baseline silently picks one path (raw onnxruntime-qnn Python integration) -- never acknowledges the benchmark-only vs deployable-artifact fork."),
        ("Does NOT instruct the user to run `winml perf` first, then `winml build`, then `winml perf` again -- that sequence wastes the on-the-fly build that perf already did.", True,
         "Baseline never uses winml perf/build at all (trivially passes)."),
        ("If the response includes both a build step and a perf step, the order is `winml build` BEFORE `winml perf`, never perf before build.", True,
         "Vacuous -- baseline includes neither winml build nor winml perf, so the order constraint doesn't trigger."),
        ("Recommends `winml inspect` before other pipeline commands.", False,
         "Baseline never uses winml inspect (or any winml CLI command). Goes straight to onnxruntime-qnn + raw Python."),
        ("Identifies QNN as the EP for Snapdragon X Elite.", True,
         "Quote: 'ONNX Runtime with the QNN Execution Provider (which is how you actually reach the Snapdragon NPU)'."),
        ("Either points the user to `winml --help`, OR uses only common/plausible flag patterns.", True,
         "Baseline doesn't quote any winml commands -- no fabricated winml flags. Uses real ORT QNN provider options."),
        ("Includes an install/prereq block -- the prompt gives no signal of prior winml usage.", False,
         "Baseline installs `onnxruntime-qnn` + `onnx` + `numpy` -- not the winml CLI. User explicitly said 'with winml' in the prompt, which the baseline ignored."),
    ],
    "eval-perf-output-recovery": [
        ("Correctly states that `winml perf`'s `-o` flag produces a perf metrics JSON, NOT the built model.", False,
         "Baseline never says perf's -o is metrics JSON. Says 'where (or whether) it leaves a redistributable ONNX on disk depends on how it ran under the hood' -- hedged and wrong-shaped."),
        ("States that the on-the-fly artifact perf built lives in cache (or a temp folder with `--ignore-cache`) and is not directly exposed to the user as a deployable file.", False,
         "Baseline actively encourages cache-hunting with a Get-ChildItem PowerShell command to find recently-written .onnx files. Treats the cache as a discoverable artifact source rather than opaque."),
        ("Recommends running `winml build` with `-o <dir>` as the way to get a deployable artifact at a known location.", False,
         "Baseline recommends chaining `winml optimize` + `winml quantize` + `winml compile` primitives manually. Never mentions `winml build`."),
        ("Notes that `winml build` needs a config file first (e.g., generated by `winml config`).", False,
         "Baseline never mentions `winml config`."),
        ("Does NOT fabricate a phantom path like 'perf saves the model at /tmp/...' or invent a hidden flag on `winml perf` that exposes the model.", False,
         "Baseline fabricates multiple cache locations (%LOCALAPPDATA%\\winml\\, %USERPROFILE%\\.winml\\, %USERPROFILE%\\.cache\\winml\\) and speculates about flags like --save-model / --keep-artifacts that don't exist on perf."),
        ("Either points the user to `winml --help`, OR uses only common/plausible flag patterns.", False,
         "Baseline speculates about non-existent flags (--save-model, --keep-artifacts). Even though hedged with 'if it does', the speculation itself is the fabrication this assertion guards against."),
    ],
}

ASSERTION_RUNNABLE = "All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only."

eval_ids = {
    "eval-snapdragon-resnet-build": 1, "eval-ryzen-ai-quick-benchmark": 2,
    "eval-npu-vs-cpu-comparison": 4, "eval-is-model-supported": 5,
    "eval-optimize-failure-recovery": 6, "eval-install-setup": 7,
    "eval-local-onnx-file": 8, "eval-config-build-for-ci": 9,
    "eval-seq2seq-out-of-scope": 10, "eval-ambiguous-run-intent": 11,
    "eval-perf-output-recovery": 12,
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
