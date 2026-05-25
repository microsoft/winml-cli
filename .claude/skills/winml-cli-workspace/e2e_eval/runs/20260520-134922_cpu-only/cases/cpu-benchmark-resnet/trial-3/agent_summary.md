Done. Benchmarked `microsoft/resnet-50` on CPU EP with the default 100 iterations / 10 warmup.

**Average latency: 34.25 ms** (P50 31.08 ms, throughput 29.2 samples/sec)

Artifact saved at: `C:\repo\WinML-ModelKit\.claude\skills\winml-modelkit-workspace\e2e_eval\scratch\cpu-benchmark-resnet\trial-3\resnet50_perf.json`

Note: this dev box is CPU-only (no NPU registered). On an actual Snapdragon X Elite / Intel Core Ultra / Ryzen AI box, you'd get a much faster number by adding `--device npu` to route through QNN / OpenVINO / VitisAI.
