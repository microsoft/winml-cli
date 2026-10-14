# Reliability changes: 2026-09-27

The entry dispatcher now classifies new search, supplied-candidate validation
and resume before baseline measurements. Graph Scout has an explicit required
role link. Feature Gap Engineer returns public-path implementation evidence;
main-agent promotion owns eligible optimizer PR creation after publication.
User stop, search exhaustion and incomplete delivery have distinct outcomes.
Repeated reproduction has a separate contract covering pinned calibration,
independent builds, correctness and between-build performance evidence.

Deterministic verification: 346 skill tests passed. The public finalizer already
checks final report closure; an added regression confirms unfinished closure
blocks publication, so no duplicate finalizer logic was introduced. Independent
code review found a new PR action could escape the old stop boundary. The
regression failed before correction; non-PR scenarios now forbid PR actions.

Live-agent observations: original seven scenarios passed once, an eight-case
suite including optimizer handoff passed twice, and three additional boundary
cases passed once. Wav2Vec2 hotspot routing passed in all three main trials.
The dispatcher/evaluator evolved between trials; this is not three complete
runs of the final eleven-case suite, a no-skill control, or evidence of
automatic skill discovery. Raw journals and snapshots remain run-local.
Simulated role actions do not certify actual independent role execution.

Real-device rebuild/correctness evidence is separate from these simulations.
Model identities, artifacts and device measurements remain run-local and
are not added to reusable knowledge.
