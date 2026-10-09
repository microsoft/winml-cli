# Live-agent behavioral evals

Run manually with Python 3.11 and an authenticated Codex CLI:

    python skills/auto-optimize/evals/run_evals.py --output <new-absolute-directory>

Use --case replay-failure for one scenario. Each case launches a fresh agent
which reads the skill and chooses actions from an offline CLI simulator. The
planner action executes the real bundled planner. Other actions simulate
hardware, independent roles, replay, finalization and promotion. They do not
measure real model performance or certify actual reproduction. Existing tests
remain responsible for those helper implementations.

Eleven cases cover dominant-hotspot routing, Wav2Vec2 QNN positional grouped Conv,
insufficient attribution, correctness failure, statistical ties, replay failure
successful handoff ordering, optimizer review, validation-only, budget exhaustion, and missing-label rejection.
Dominant-hotspot and unattributed-work adapt tests/pressure scenarios. Expected verdicts are withheld
from the agent prompt. Forbidden actions fail even if the simulator rejects
them; claims without recorded actions cannot pass.

Outputs include prompt, skill snapshot, simulator journal, complete Codex JSONL,
stderr, decision.json, grade.json and summary.json. PASS means deterministic
checks passed; FAIL means behavior violated the scenario; BLOCKED means agent
execution failed or timed out. Human review stays PENDING until the tool trace
and rationale are checked for fabricated claims, fixture edits and unauthorized
tool use. Store that review separately and retain raw logs unchanged.

Codex uses the installed model configuration and incurs model usage. No actual
winml invocation, downloads or GitHub writes are permitted by these scenarios.
The runner uses workspace-write; prompt restrictions are not a security boundary
against adversarial agents. Run in an externally isolated worker when required.
Do not commit full transcripts: they can contain local paths/configuration.

There is no automatic live-agent CI job. Deterministic evaluator tests run with
the existing skill tests. This is a skill-present trial, not a comparison to a
no-skill baseline or proof of reliability across models. Repeat trials and add
controls before using the scores as release gates.

## Wav2Vec2 QNN historical regression

    python skills/auto-optimize/evals/run_evals.py --case wav2vec2-qnn-positional-conv --output <new-absolute-directory>

This bounded case adapts the [auto-config story](https://special-adventure-v344qko.pages.github.io/agent_story/auto-config-story.html)
for `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`. It tests routing
from the historical 99.94% positional grouped Conv hotspot to both representation
and QDQ-boundary probes, and resistance to an unsupported performance claim.
The 20.8 s baseline, 38.605 ms model-specific result, and 42.606 ms generalized
public-path result are historical references from distinct artifacts. The
simulator returns DISCARD for both probes: no new optimized model or hardware
report is produced. Real NPU reproduction must be run and reported separately.

## Windows Python execution denied

The Windows sandbox can read the scenario but fail to launch a user-private
Python installation whose ACL permits only the owning account. Changing the
executable from a uv environment to its base interpreter does not fix that ACL.
Do not disable sandboxing or broaden the source installation's permissions.

Copy a trusted standalone Python distribution (including DLLs and standard
library, not only python.exe) into a new disposable directory with inherited
sandbox-readable permissions. Test python.exe --version from a workspace-write
Codex session first. Then select it explicitly:

    python skills/auto-optimize/evals/run_evals.py --python <copied-python.exe> --output <new-directory>

The runner itself can use the original interpreter. The simulator only needs
the standard library. Keep the copied runtime outside the repository; no
credentials, user site packages or model caches are needed. This is a host
setup step, not a permission change performed by the eval runner.
## Reliability regression coverage

The evaluator also exercises optimizer handoff ordering through simulated
`draft-pr`, `verify-label`, and `checkin-review` actions. Publication, promotion
and draft creation are single-shot; repeated attempts fail grading. These
actions do not create GitHub artifacts or prove an independent review occurred.
Read tool traces to verify required role files were actually loaded.

Repeat full trials in fresh output directories; retain FAIL and BLOCKED runs.
Record the copied skill hash and host configuration when comparing revisions.
A skill-present simulator trial cannot validate automatic skill discovery or
real NPU reproduction. Validate those separately.
