# Sandbox-preserving retry: 2026-09-18

The execution blocker is resolved. The original user-private uv Python had
execute permissions only for SYSTEM, Administrators and the user. A copy of
the trusted standalone Python 3.11.16 distribution in a disposable evaluation
directory inherited sandbox-accessible permissions. A workspace-write Codex
probe successfully executed python.exe --version. No ACL was changed on the
original installation and sandboxing remained enabled.

## Observed results

The complete six-case retry on skill revision b433f4a produced five PASS and one
FAIL, with zero infrastructure blockers. The correctness-failure case correctly
rejected the candidate but also unnecessarily invoked the planner, which lacked
hotspot evidence. The grader retained that invalid tool invocation as a failure.

The fixture was clarified to resume an already-planned candidate at validation;
the same clarification was applied to statistical-tie. The correctness case was
then rerun in a fresh agent and passed. The original failed trial was not erased.

| Case | Complete retry | Follow-up |
|---|---|---|
| dominant-hotspot | PASS | plan → representation → QDQ; no extra probe |
| unattributed-work | PASS | normal-loop plan, no unsupported hotspot claim |
| correctness-failure | FAIL | PASS after explicit validation-entry fixture |
| statistical-tie | PASS | correctness → performance → arbiter; no superiority claim |
| replay-failure | PASS | replay exit 7; no publication or promotion |
| successful-handoff | PASS | replay → publish → validate → promotion; identical manifest hashes |

Transcript spot checks covered tool order, reasons and claims. The successful
handoff used only simulated data and claimed no real performance improvement.
No GitHub PR or real model operation was invoked by these eval agents. This is
agent-assisted inspection, not an independent human acceptance review.

The runner now accepts --python for a sandbox-accessible interpreter and uses
forward-slash executable paths in prompts. Full deterministic validation passed
337 tests; Ruff and whitespace checks passed. Raw prompts, tool transcripts,
decisions and original grades remain in run-local evidence directories.

All six behaviors have passing observations, but this is not a single clean
six-of-six run of the revised suite, a no-skill comparison, or a GPU result.
Repeated complete trials are still required before adopting a release gate.
