# Resume routing skill change: 2026-09-18

The skill now routes supplied candidates to the first unverified gate rather
than restarting experiment planning. references/resume.md defines evidence
matching, invalidation, validation-only boundaries and failure handling.

Removed the extra already-planned hint previously added to the correctness and
tie prompts. Both scenarios now explicitly grade any planner invocation as a
failure. This tests the skill correction rather than relying on a leading prompt.

One complete live-agent trial returned six deterministic PASS results:

| Case | Recorded actions |
|---|---|
| dominant-hotspot | plan, representation, QDQ |
| unattributed-work | plan (normal loop) |
| correctness-failure | correctness (failed); candidate rejected |
| statistical-tie | correctness, performance, arbiter |
| replay-failure | replay (failed); no publication |
| successful-handoff | replay, publish, validate, promotion, validate |

The successful handoff retained the same manifest hash. The correctness case
did not call the planner despite receiving the original, less-leading prompt.
All actions are simulated except the bundled planning helper. No real model
speedup or independent human review is claimed.

Agent-assisted transcript inspection noted that the replay-failure rationale
reported shell exit 1 while the fixture payload/journal records failure code 7;
the required stop behavior was correct. This detail is not checked by the
current deterministic grader and remains an evidence-reporting limitation.

The first two cases used the initial wording of the new resume link; the
dispatcher was subsequently shortened to satisfy its existing 800-word budget,
without changing the routing rule. Later cases used the shortened dispatcher.
This is one full trial of the change, not repeatability evidence or a no-skill
comparison. Raw prompts, copied skill files, journals and decisions remain in
the run-local auto-optimize-resume-skill-eval directory.

All 337 deterministic skill tests passed after the final edits; Ruff and
whitespace checks passed. Previous failing and infrastructure-blocked trial
records remain unchanged.
