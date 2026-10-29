# DXGI steady memory measurements

Use `winml perf --memory-backend dxgi --etw-tracing <new-directory>` for a
separate Runtime CGC or WinMLCG EP GPU memory run. Existing performance runs
keep the default PDH memory backend.

```powershell
winml perf -m model.onnx --runtime winml-ort --ep winmlcg --device gpu --device-luid 0x00000000_0x00000001 --memory-backend dxgi --etw-tracing memory-trace --iterations 4000
```

Replace the LUID with the selected device from `winml sys --list-device`. ETW
permissions and a compatible native Runtime/EP payload are required.

The mode enforces >=10 warmup calls and >=2 elapsed warmup seconds, then
>=200 measured calls and >=10 elapsed inference seconds. GPU timestamps are
validated after capture: less than 2 accumulated GPU seconds rejects the
result (increase `--iterations` and use a fresh output directory). It does
not replace missing GPU evidence with host elapsed time.

`--warmup-seconds` is also available for unmonitored single-model inference;
it supplements the warmup count. Extra warmup calls remain excluded from
GPU and CPU measurement samples. Unsupported raw CLI usage is rejected.

## Counters

- Working set/RSS: psutil's native Windows process working-set counter.
- Private commit: Windows process PrivateUsage, not resident RAM.
- GPU local/non-local: in-process IDXGIAdapter3 QueryVideoMemoryInfo CurrentUsage
  on the explicitly selected adapter. No device/queue is created by the reader.
- Steady medians, sampled phase peaks and sample counts are in `process_memory`.
- `steady_bytes` uses working_set/private_commit/local_usage/nonlocal_usage,
  matching the core counters in the native dxcgc-run memory experiment.
- Query failures remain unavailable and invalidate steady results; zeros are
  accepted only when reported successfully. Non-local GPU memory is not added
  to RAM. OS lifetime peak remains separate from sampled phase peaks.

## Limitations

This is not the entire native dxcgc-run resource-instrumentation contract.
Sampling is on a Python thread (requested 10 ms, observed intervals reported),
so GIL contention and short spikes remain limitations. Phases are API-visible;
internal initialization subphases, heap lifetime and resource-release attribution
are unavailable, not inferred. No promise of identical sampling cadence or
statistically greater stability is made. Benchmark memory separately from
formal performance acceptance and repeat in fresh processes.

The initial implementation has counter smoke tests and unit/CLI regression
coverage. Full hardware steady-window validation remains required before
using this mode for release gating.
