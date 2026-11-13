# Report delivery: facts, computation, acceptance

This is the standard output contract for every auto-optimize run. Apply it
without requiring report-format instructions in the user prompt. A user may
request an additional audience-specific view, but it does not replace this
report or relax any evidence, validation, or publication requirement.

1. Inspect actual performance JSON keys before declaring metrics unavailable.
   Explicitly select compatible, uncontaminated sessions for one artifact.
   Verify model/input hashes, provider options, profiling mode, batch size and
   timing protocol. Do not mix baseline/candidate or independent artifacts.
2. If raw_samples_ms exists, run the linked scripts/aggregate_perf.py with
   explicit session paths and --output to a new run-local JSON. Its receipt
   binds source file hashes. It computes p50/p90/p99 from the SAME pooled raw
   timed samples (linear interpolation), and serial batch-1 inference rate as
   1000 * sample_count / sum(milliseconds). This is not concurrent or sustained
   service throughput. Keep paired gain/CI separate and label its estimator.
3. Copy the helper metrics, sample count and method into the report, retaining
   source receipts. Never average session percentiles into a pooled percentile.
   With only summary data, label a selected session or statistic explicitly;
   do not invent raw samples. Before missing_reasons, inspect all recorded
   evidence and explain the actual absence, not a presumed absence.
4. The first three report sections are fixed:
   - Benchmark: matched reference/baseline/champion rows with target identity,
     latency distribution, serial throughput, memory, quality, size and protocol.
   - Optimizations Applied: only accepted final-recipe changes, each with its
     purpose, exact change, evidence and measured impact. Rejected experiments
     remain in the later lineage table.
   - Outcome: champion ONNX, resolved and rebuild configs, config-validation
     receipt, manifest and reproduction entry point.
5. Map analyzer coverage/opportunities and native trace units into supported
   fields. Refresh diagnosis after the last completed gate. Missing metrics
   display N/A with a small explanatory note, not oversized metric text.
6. Run validate_config.py after clean replay with the exact final serialized
   rebuild config, confirmed champion and frozen inputs. A passing candidate
   step or replay statement is not a substitute. Add its hash-bound receipt to
   the Outcome only after clean build, public I/O, target load and correctness
   all pass.
7. Run render_report.py report.json report.html --final. Nonzero exit means
   incomplete, never success. Inspect the generated report against its source
   facts. Schema acceptance, model evidence and visual review are different
   claims; none implies another.
8. Publish only through workflow.py deliver, which invokes finalize_output.py and validates the bundle,
   creates/validates its promotion handoff; validate the wrapper separately. Freeze that bundle. Corrections
   require a new versioned bundle/handoff; retain earlier evidence.

Resolve both scripts from this skill directory, not the current directory.
The aggregate helper checks serialized artifact/provider identity and batch-1
only; it cannot prove equal input hashes or absence of profiling/contamination.
Those checks remain prerequisites, recorded in run-local evidence.
