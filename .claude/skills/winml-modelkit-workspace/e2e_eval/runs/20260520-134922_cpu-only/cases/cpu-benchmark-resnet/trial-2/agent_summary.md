Benchmark complete. The artifact was saved to `C:\repo\WinML-ModelKit\.claude\skills\winml-modelkit-workspace\e2e_eval\scratch\cpu-benchmark-resnet\trial-2\resnet50_perf.json` and includes a `latency_ms` field.

**`microsoft/resnet-50` on CPU: 27.28 ms average latency** (P50 26.55 ms, throughput 36.66 samples/sec).

Note: this dev box has no NPU registered (only CPU, DirectML, and OpenVINO-CPU EPs), so the benchmark ran on the CPU EP. If your AI PC has an NPU, you'd see significantly lower latency by re-running with the NPU EP.
