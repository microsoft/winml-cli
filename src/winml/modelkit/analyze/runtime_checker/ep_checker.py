import tempfile
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from ... import winml


# TODO: allow test case iter to take dtypes as inputs
# TODO: define dataclass for result

# Notes:
# - Paasing only the inference session object would not suffice for tester,
# as sess.get_session_options() may return a modified version of session options


class EPChecker:
    """Test execution provider compilation and runtime behavior."""

    # EPs that require a file path (not in-memory bytes) for compilation.
    # VitisAI EP fails with "ep.context_file_path and model_path are both empty"
    # when given in-memory model bytes.
    EPS_REQUIRING_FILE_PATH = {"VitisAIExecutionProvider"}

    def __init__(
        self,
        ep_name: str,
        device_type: ort.OrtHardwareDeviceType,
        provider_options: Sequence[dict[Any, Any]] | None = None,
    ) -> None:
        self.device_type = device_type
        self.ep_name = ep_name
        self._provider_options = provider_options

    def _get_sess_options(self) -> ort.SessionOptions:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        winml.add_ep_for_device(sess_options, self.ep_name, self.device_type)
        return sess_options

    def _needs_file_path(self) -> bool:
        """Check if this EP requires a file path instead of in-memory bytes."""
        return self.ep_name in self.EPS_REQUIRING_FILE_PATH

    def check_compile(
        self,
        path_or_bytes: str | bytes | PathLike[Any],
        input_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Test model compilation with execution provider."""
        sess_options = self._get_sess_options()
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        # Some EPs (e.g. VitisAI) require a file path for compilation.
        # Write bytes to a temp file if needed.
        if isinstance(path_or_bytes, bytes) and self._needs_file_path():
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp.write(path_or_bytes)
                tmp_path = Path(tmp.name)
            try:
                return self._do_compile(sess_options, str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            return self._do_compile(sess_options, path_or_bytes)

    def _do_compile(
        self,
        sess_options: ort.SessionOptions,
        path_or_bytes: str | bytes | PathLike[Any],
    ) -> dict[str, Any]:
        """Execute the actual compilation step."""
        compiler = ort.ModelCompiler(
            sess_options,
            path_or_bytes,
            flags=ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED,
        )
        # TODO: run compiled model with same inputs as run test
        try:
            model_bytes = compiler.compile_to_bytes()
            model = onnx.load_from_string(model_bytes)
            nodes = model.graph.node
            assert len(nodes) == 1, (
                f"Expected single EPContext node of compiled model, got {len(nodes)} nodes."
            )
            assert nodes[0].op_type == "EPContext", (
                f"Expected single EPContext node, got {nodes[0].op_type}"
            )
        except Exception as e:
            return {"success": False, "reason": str(e)}
        else:
            return {"success": True, "reason": None}

    def check_run(
        self,
        path_or_bytes: str | bytes | PathLike[Any],
        input_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Test model execution with execution provider."""
        session = ort.InferenceSession(
            path_or_bytes,
            self._get_sess_options(),
            provider_options=self._provider_options,
        )
        # inputs = self._generate_inputs(session)
        graph_input_names = {inp.name for inp in session.get_inputs()}
        inputs = {k: v for k, v in input_args.items() if k in graph_input_names}
        # TODO: return outputs?
        try:
            outputs = session.run(None, inputs)
            print(f"Run outputs: {outputs}")
        except Exception as e:
            return {"success": False, "reason": str(e)}
        else:
            return {"success": True, "reason": None}
