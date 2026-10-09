# QNN NPU Decisions

Use these as questions for the current graph, not universal preferences.

- Normalize constants and infer static shapes before attribution. Raw ONNX
  node count, file size, and initializer count are not latency evidence.
- Follow the [planning router](../SKILL.md#planning-router---evaluate-before-loading-cases-or-proposing-hypotheses) before loading cases. Keep routing rules in that single contract.
- Treat static Split and complete sibling Slice partitions as competing
  representations. Measure both directions when relevant. A representation may
  matter mainly because it exposes a downstream fusion.
- Preserve QNN-friendly rank and layout. If a smaller graph adds Transpose or
  other layout work, inspect the matched trace before keeping it.
- Audit partition count, residual CPU graph, and fallback. One partition proves
  coverage, not speed; fragmentation can erase a valid local fusion.
- Use matched traces to explain removed operators and accelerator time. Never
  compare profiled wall latency with non-profiled latency.
- Bind compile/cache identity to model hash, effective graph, provider options,
  runtime/provider/SDK versions, and profiling mode. Use a fresh stem when any
  identity component changes.
- Run correctness before performance. Screen in alternating A/B and B/A order;
  confirm the final challenger with paired evidence above noise.
- Preserve contrary results. A direction confirmed for one model, shape, or
  provider version only raises priority elsewhere.
