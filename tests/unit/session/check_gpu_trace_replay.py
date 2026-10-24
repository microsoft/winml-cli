# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Replay a captured trace through analysis, JSON and console reporting."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

from winml.modelkit.commands.perf import BenchmarkConfig, BenchmarkResult, display_console_report
from winml.modelkit.session.gpu_capture import GpuCapture, analyze_trace


folder = Path(sys.argv[1])
result = analyze_trace(folder)
assert result["valid_runs"] > 0
assert result["events_lost"] == 0
report = BenchmarkResult(config=BenchmarkConfig(model_id="trace-replay"))
report.gpu_execution = result
assert report.to_dict()["gpu_execution"] == result
console = Mock()
display_console_report(report, console)
assert any("GPU execution time" in str(call) for call in console.print.call_args_list)
capture = GpuCapture(folder)
capture.command = Mock()
assert capture.finish() == result
assert len(capture.command.call_args_list) == 2
assert all(
    not ("-rts" in call.args and "-summary" in call.args) for call in capture.command.call_args_list
)
print(json.dumps(result, indent=2))
