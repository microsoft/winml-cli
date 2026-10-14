# Repeated reproduction

Read this contract when the user requests stable reproduction. Freeze the
source revision or exported ONNX plus external-data hashes, public I/O,
calibration dataset/seed/effective defaults, evaluation inputs, provider options,
CLI build, runtime/provider versions and device. Pin dependency overlays as
explicit dependencies; do not hide them behind an unversioned PYTHONPATH.

Use at least two independent clean builds through the public CLI and exact
serialized config. Never substitute a historical prototype, transform script
or stale compiled context for the public-path result. Record every attempt,
including failures. For each build: check graph/I/O, compare every output on
the frozen evaluation set, then compile and validate target execution before
performance. Declare thresholds before examining candidate results.

Measure alternating A/B and B/A on an otherwise idle target with the same
inputs and explicit provider options. Report per-build latency distribution
and between-build spread, not only the best sample. Concurrent target work
invalidates a stability claim; retain the incident and repeat those pairs.
A repeated-build claim requires both builds to pass correctness and the
predeclared performance criterion. Distinguish replaying an exported source
from repeating HF export and from an autonomous tuning search.

For quantized baselines, report equivalence against that baseline separately
from FP32/task quality. Synthetic inputs do not certify task quality. Missing
provider trace/closure evidence still blocks full skill bundle completion.
Historical timings are context, never an acceptance result for new artifacts.

Persist a runnable replay with hash verification, config and inputs in the
run directory. A replay must reject modified inputs and existing output
directories, propagate native failures, and stop before perf after a failed
correctness gate. The script and lock must contain all required dependencies
or name their pinned locations honestly. Validate the published bundle through
the existing finalizer; a hand-written diagnostic report is not certification.
