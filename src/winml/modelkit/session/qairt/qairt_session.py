"""WinMLQairtSession - QAIRT SDK session for QNN compilation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ...utils.python_env import ensure_venv
from ..session import SessionState, WinMLSession


if TYPE_CHECKING:
    from ...compiler.configs import EPConfig

logger = logging.getLogger(__name__)

# QAIRT SDK dependencies for venv-wmk virtual environment
QAIRT_DEPENDENCIES = [
    "onnx>=1.14.0,<1.17",
    "torch==2.4.1",
    "numpy>=1.24.0,<2.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "aenum>=3.1.0",
    "paramiko>=3.0.0",
    "jsonschema>=4.0.0",
    "typing_extensions>=4.0.0",
    "packaging>=21.0",
]

# Path to compile_qairt_bin.py script (in same directory)
COMPILE_QAIRT_BIN_SCRIPT = Path(__file__).parent / "compile_qairt_bin.py"


class WinMLQairtSession(WinMLSession):
    """Session that compiles and runs models using Qualcomm QAIRT SDK.

    Overrides compile() to use the QAIRT SDK pipeline instead of
    ort.ModelCompiler. The SDK runs in an isolated Python 3.10 venv
    via subprocess.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "qnn",
        ep_config: EPConfig | None = None,
    ) -> None:
        # Initialize parent WinMLSession
        super().__init__(onnx_path, device=device, ep_config=ep_config)

        # QAIRT-specific paths
        self._bin_path = self._onnx_path.parent / f"{self._onnx_path.stem}_qnn_ctx_qnn.bin"
        self._bin_info_path = self._onnx_path.parent / f"{self._onnx_path.stem}_cache_info.json"
        self._ctx_path = self._onnx_path.parent / f"{self._onnx_path.stem}_qnn_ctx.onnx"

        self._qnn_sdk_root = (
            ep_config.qnn_sdk_root if ep_config else None
        ) or self._resolve_sdk_path()

        logger.info("WinMLQairtSession initialized: %s", onnx_path)

    def compile(self) -> None:
        """Compile model using QAIRT SDK.

        Pipeline:
            1. Ensure venv-wmk in SDK directory
            2. Run compile_qairt_bin.py subprocess → .bin
            3. Generate cache_info.json
            4. Wrap binary into EPContext ONNX model
            5. Create ORT InferenceSession from EPContext model
        """
        # If already compiled, ignore (idempotent)
        if self._session is not None:
            if self._is_verbose():
                logger.info("Already compiled for %s", self._device)
            return

        logger.info("Compiling via QAIRT SDK: %s", self._onnx_path)

        # Step 1: Set up venv with QAIRT dependencies
        venv_python = ensure_venv(
            root_path=self._qnn_sdk_root,
            venv_name="venv-wmk",
            python_version="3.10",
            requirements=QAIRT_DEPENDENCIES,
        )
        logger.info("Virtual environment ready: %s", venv_python)

        # Step 2: Compile to QNN binary
        self._compile_to_qnn_bin(venv_python)
        logger.info("QNN bin compiled: %s", self._bin_path)

        # Step 3: Generate cache_info.json
        self._create_context_bin_info()
        logger.info("Cache info ready: %s", self._bin_info_path)

        # Step 4: Wrap binary into ONNX with EPContext
        self._wrap_bin_to_onnx()
        logger.info("EPContext ONNX created: %s", self._ctx_path)

        # Step 5: Create ORT InferenceSession from EPContext model
        self._create_inference_session()
        logger.info("Session created from EPContext model")

    def _resolve_sdk_path(self) -> Path:
        """Resolve QAIRT SDK path from environment variables."""
        for var in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT"):
            value = os.environ.get(var)
            if value:
                path = Path(value)
                if path.exists():
                    return path
        raise FileNotFoundError(
            "QAIRT SDK path not found. Provide --qnn-sdk-root or set QNN_SDK_ROOT."
        )

    def _compile_to_qnn_bin(self, venv_python: Path) -> None:
        """Run compile_qairt_bin.py subprocess to produce .bin."""
        result = subprocess.run(
            [
                str(venv_python),
                str(COMPILE_QAIRT_BIN_SCRIPT),
                "--qairt-root", str(self._qnn_sdk_root),
                "--model", str(self._onnx_path),
                "--output-dir", str(self._onnx_path.parent),
            ],
            text=True,
            timeout=600,
            stdout=None if logger.getEffectiveLevel() <= logging.DEBUG else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"QAIRT compilation failed (exit code {result.returncode}): {result.stderr}"
            )

        # Rename generated bin to _bin_path
        generated_bin = self._onnx_path.parent / f"{self._onnx_path.stem}.bin"
        if generated_bin != self._bin_path:
            generated_bin.rename(self._bin_path)

    def _create_context_bin_info(self) -> None:
        """Generate cache_info.json using qnn-context-binary-utility.exe.

        Some SDK versions generate this file automatically during compilation,
        so we only run the utility if the file doesn't exist.
        """
        if self._bin_info_path.exists():
            logger.info("Cache info already exists: %s", self._bin_info_path)
            return

        utility_exe = (
            self._qnn_sdk_root / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-utility.exe"
        )
        if not utility_exe.exists():
            raise FileNotFoundError(f"qnn-context-binary-utility.exe not found: {utility_exe}")

        result = subprocess.run(
            [
                str(utility_exe),
                "--context_binary", str(self._bin_path),
                "--json_file", str(self._bin_info_path),
            ],
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"qnn-context-binary-utility failed (exit code {result.returncode})"
            )

    def _wrap_bin_to_onnx(self) -> None:
        """Wrap QNN bin file into ONNX model with EPContext node."""
        from onnxruntime.tools.qnn import gen_qnn_ctx_onnx_model

        with open(self._bin_info_path) as f:
            qnn_json_obj = json.load(f)

        if "info" not in qnn_json_obj or "graphs" not in qnn_json_obj["info"]:
            raise RuntimeError("Unrecognized bin info JSON format")

        qnn_version = qnn_json_obj["info"]["buildId"]
        for qnn_graph in qnn_json_obj["info"]["graphs"]:
            qnn_input_tensor_dic = {}
            qnn_output_tensor_dic = {}
            graph_name = gen_qnn_ctx_onnx_model.parse_qnn_graph(
                qnn_graph, qnn_input_tensor_dic, qnn_output_tensor_dic
            )

            # Fix: parse_qnn_graph doesn't set id field, extract from raw JSON
            for raw_input in qnn_graph["info"]["graphInputs"]:
                tensor_name = raw_input["info"]["name"]
                if tensor_name in qnn_input_tensor_dic:
                    qnn_input_tensor_dic[tensor_name].id = raw_input["info"]["id"]
            for raw_output in qnn_graph["info"]["graphOutputs"]:
                tensor_name = raw_output["info"]["name"]
                if tensor_name in qnn_output_tensor_dic:
                    qnn_output_tensor_dic[tensor_name].id = raw_output["info"]["id"]

            gen_qnn_ctx_onnx_model.generate_wrapper_onnx_file(
                grap_name=graph_name,
                model_file_name=str(self._ctx_path),
                qnn_input_tensor_dic=qnn_input_tensor_dic,
                qnn_output_tensor_dic=qnn_output_tensor_dic,
                disable_embed_mode=not self._embed_context,
                qnn_ctx_file=str(self._bin_path) if self._embed_context else f"./{self._bin_path.name}",
                quantized_IO=False,
                qnn_sdk_version=qnn_version,
            )

            break  # Only process first graph


    def _create_inference_session(self) -> None:
        """Create ORT InferenceSession from EPContext model."""
        import onnxruntime as ort

        sess_options = self._build_session_options(self._device)
        self._session = ort.InferenceSession(str(self._ctx_path), sess_options=sess_options)
        self._state = SessionState.COMPILED

        actual_providers = self._session.get_providers()
        logger.info("Session created with providers: %s", actual_providers)
