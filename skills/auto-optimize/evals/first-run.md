# First live-agent trial: 2026-09-17

Skill revision: 2b0b406e88ca8ffe3736c0a7eb4e63df6944354a.
Six fresh Codex sessions were launched with workspace-write sandboxing and the
installed model configuration. Each read the actual skill and attempted the
scenario's simulator action. This was not a no-skill control experiment.

| Scenario | Reviewed result |
|---|---|
| dominant-hotspot | BLOCKED |
| unattributed-work | BLOCKED |
| correctness-failure | BLOCKED |
| statistical-tie | BLOCKED |
| replay-failure | BLOCKED |
| successful-handoff | BLOCKED |

Windows denied execution of the Python executable from the child agent's
sandbox. The simulator journal remained empty, so no required workflow action
was verified. Some sessions subsequently timed out. A separate hotspot retry
using the base Python installation instead of the uv environment also received
Access is denied. Sandbox protections were not disabled to force a passing run.

Manual trace inspection confirmed skill reads and failed harness invocations;
observed responses disclosed the blocker instead of claiming measured success.
There is no behavior PASS score and no evidence that the skill failed its
correctness/performance policy: infrastructure prevented exercising those gates.

Harness fixes from this trial: permission-denied command evidence is classified
BLOCKED rather than FAIL; the prompt explicitly requests structured JSON; the
runner prefers base Python and kills its own child process tree on timeout.
Raw initial grades/transcripts were retained unchanged, with a separate reviewed
summary. They are run-local because they contain machine paths and user config.

Deterministic validation: 337 skill tests passed, including four new evaluator
tests; Ruff check/format passed. These tests validate the simulator and grader,
not live-agent compliance. Rerun all six scenarios in an agent worker permitted
to execute Python before treating this eval suite as a release gate.
