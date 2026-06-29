# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import openvino as ov
from openvino import properties


os.chdir("experiments/ov_estimation")

model_file = r"model.onnx"

# Initialize OpenVINO Core
core = ov.Core()

# Load the IR model
model = core.read_model(model_file)

# Define custom compile properties
config = {
    # properties.inference_num_threads(): 7,
    properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY,
    # properties.hint.num_requests(): 1,
    properties.hint.execution_mode(): properties.hint.ExecutionMode.PERFORMANCE,
    "NPU_COMPILATION_MODE_PARAMS": "enable-schedule-trace=true",
}

# Compile the model for CPU with custom settings
compiled_model = core.compile_model(model, "NPU", config)


# Verify runtime configuration
def print_property(key):
    """Print a compiled-model property, ignoring keys that aren't supported."""
    try:
        print(f"{key}: {compiled_model.get_property(key)}")
    except Exception:
        pass


for key in [
    "INFERENCE_NUM_THREADS",
    "NUM_STREAMS",
    "PERFORMANCE_HINT",
    "PERFORMANCE_HINT_NUM_REQUESTS",
    "EXECUTION_MODE_HINT",
    "EXECUTION_DEVICES",
    "INFERENCE_PRECISION_HINT",
]:
    print_property(key)
