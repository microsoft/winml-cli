# Capability Closure Pressure Scenario

Act as Graph Scout. There is no reference model and no desired winner.

The current leader leaves one static tensor partitioned by two sibling Slice
routes on the same axis. Their immutable unit-step intervals are ordered,
non-overlapping, and cover the full static axis. Exact output names, shapes,
dtypes, order, and consumers can be preserved; one output is public. A prior
route rewrite made this topology newly visible. The matched trace still shows
material routing work.

`winml optimize --check-optim` reports only unrelated capabilities. The verbose registry
contains `--enable-gather-slice-to-split-fusion`, described as fusing
compatible Gather/Slice routes into Split while preserving output tensors. The
prior ledger has not tested it. A selected generic case says equivalent routing
representations can lower differently and expose downstream matchers.

Return the capability closure ledger and coverage verdict. Success requires:

- the unreported registered capability is `PROBE_REQUIRED`;
- the exact enable flag and a fresh explicit build probe are named;
- analyzer silence is treated only as detector evidence;
- graph and trace deltas are predicted without claiming a latency gain;
- public-output preservation is a safety check, not an automatic rejection;
- the verdict is `MATERIAL_OMISSION_FOUND`, never `NO_MATERIAL_OMISSION`.

Then evaluate an unrelated future registry entry absent from selected knowledge
cases and analyzer output:

```text
--enable-static-pad-into-conv
Fold a single-consumer, constant-zero Pad with immutable nonnegative spatial
pads into the following Conv while preserving the Conv output tensor.
```

The leader contains exactly that residual `Pad -> Conv` neighborhood. Pad mode
is constant, value is zero, pads are immutable nonnegative spatial values, Conv
attributes are compatible, Pad has one consumer, and public I/O is unaffected.
The matched provider trace shows separate Pad memory traffic. The prior ledger
has not tested this capability.

Apply the same registry-plus-topology reasoning. Success also requires a second
`PROBE_REQUIRED` entry naming `--enable-static-pad-into-conv`, its cheapest
explicit build probe, expected Pad/trace removal without a latency claim, and
the same `MATERIAL_OMISSION_FOUND` verdict. Do not require a production code change,
knowledge-case match, or model comparison to propose it.