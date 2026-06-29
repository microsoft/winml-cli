# Run estimation

`python experiments/ov_estimation/run.py`

# Run benchmark

`benchmark_app -pc -d NPU -hint latency -report_type average_counters -report_folder experiments/ov_estimation -m experiments/ov_estimation/model.onnx`
