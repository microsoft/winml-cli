# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Keep heavy numerical dependencies out of GGUF and CGC perf and memory checkpoints.

The test process's conftest imports ORT, so these checks use clean interpreters
and fail on attempted imports even if production code catches the exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "src"
# GGUF helper modules that only an evidence run loads: the evidence writer and the log capture.
EVIDENCE_MODULES = [
    "winml.modelkit.commands._gguf_evidence",
    "winml.modelkit.commands._gguf_native_log",
]
# Blocked on top of the GUARD list for --backend cgc, which must load none of these
# before Windows ML: ONNX and the modules that import ONNX or ORT (another
# DirectML.dll breaks the windowsml.runtime import), comtypes (imported first, it
# skips the COM apartment setup of windowsml) and the llama.cpp binding. It also
# never creates a D3D12 device or a DXGI reader, the two memory modules last.
CGC_BLOCKED = (
    "onnx",
    "comtypes",
    "winml.modelkit.export",
    "winml.modelkit.compiler",
    "winml.modelkit.session.runtime_session",
    "winml.modelkit.session.ep_device",
    "winml.modelkit.session.ep_registry",
    "winml.modelkit.session.llama_session",
    "winml.modelkit.commands._perf_gguf",
    "winml.modelkit.session.monitor._gpu_baseline",
    "winml.modelkit.session.monitor._dxgi_memory",
)
GUARD = """
import importlib.abc
import json
import sys

blocked = (
    "numpy",
    "onnxruntime",
    "onnxruntime_genai",
    "torch",
    "transformers",
    "winml.modelkit.commands._perf_genai",
) + EXTRA_BLOCKED
attempts = []

def is_blocked(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)

assert not [name for name in sys.modules if is_blocked(name)]

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if is_blocked(fullname):
            attempts.append(fullname)
            raise AssertionError("GGUF imported unrelated runtime: " + fullname)
        return None

sys.meta_path.insert(0, RejectHeavyImports())
"""


def _run_isolated(code: str, *arguments: str, extra_blocked: tuple[str, ...] = ()) -> dict:
    """Run ``code`` in a clean interpreter after ``GUARD``, which also blocks ``extra_blocked``."""
    script = (
        f"import sys; sys.path.insert(0, {str(SOURCE)!r})\n"
        f"EXTRA_BLOCKED = {extra_blocked!r}\n" + GUARD + dedent(code)
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned source
        [sys.executable, "-I", "-c", script, *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_gguf_module_and_perf_help_do_not_import_unrelated_runtimes() -> None:
    outcome = _run_isolated(
        """
        from types import SimpleNamespace
        from click.testing import CliRunner
        from winml.modelkit.commands import _perf_gguf
        from winml.modelkit.cli import main
        from winml.modelkit.telemetry.telemetry import Telemetry

        # Keep the command test local without initializing telemetry or consent.
        Telemetry.get_or_init = classmethod(lambda cls: SimpleNamespace(disabled=True))
        result = CliRunner().invoke(main, ["perf", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--memory" in result.output and ".gguf" in result.output
        assert not attempts, attempts
        assert not [name for name in sys.modules if is_blocked(name)]
        print(json.dumps({"help": True, "imports": attempts}))
        """
    )
    assert outcome == {"help": True, "imports": []}


@pytest.mark.parametrize(
    ("device", "memory", "ignored_config", "evidence"),
    [
        ("cpu", False, False, False),
        ("gpu", True, False, False),
        ("cpu", True, True, False),
        # The benchmark skill's flag set: exact IDs, a fixed capacity, no EOS stop.
        ("gpu", False, False, True),
    ],
)
def test_gguf_command_runs_without_heavy_dependency_imports(
    tmp_path: Path, device: str, memory: bool, ignored_config: bool, evidence: bool
) -> None:
    outcome = _run_isolated(
        """
        from pathlib import Path
        from types import SimpleNamespace
        from click.testing import CliRunner
        from winml.modelkit.cli import main
        from winml.modelkit.commands import _perf_gguf
        from winml.modelkit.session import LlamaGenerationTiming
        from winml.modelkit.telemetry.telemetry import Telemetry

        Telemetry.get_or_init = classmethod(lambda cls: SimpleNamespace(disabled=True))
        root, device, memory, ignored_config, evidence = sys.argv[1:]
        memory = memory == "True"
        root = Path(root)
        model = root / "placeholder.gguf"
        model.write_bytes(b"GGUF")
        output = root / "report.json"
        sessions = []

        class FakeSession:
            def __init__(self, model_path, *, device):
                assert model_path == model
                self.device = device
                self.context_length = None
                self.effective_device = None
                self.chat_template_applied = None
                self.task_default_eos_token_ids = None
                self.load_timings_ms = {}
                self.payload = None
                self.generated = 0
                self.closed = False
                sessions.append(self)

            def open_tokenizer(self):
                pass

            def encode_prompt(self, prompt, *, apply_template):
                assert evidence != "True", "exact token IDs are never encoded"
                assert prompt == "test prompt"
                self.chat_template_applied = apply_template
                return [1, 2, 3, 4]

            def load(self, *, context_length, eos_token_ids=None):
                self.context_length = context_length
                self.effective_device = self.device
                eos = (2,) if eos_token_ids is None else tuple(eos_token_ids)
                self.task_default_eos_token_ids = eos

            def generate_timed(self, token_ids, *, max_new_tokens, clock):
                assert token_ids == [1, 2, 3, 4] and max_new_tokens == 3
                self.generated += 1
                submit, *token_times, completed = (clock() for _ in range(max_new_tokens + 2))
                return LlamaGenerationTiming(
                    input_tokens=4, generated_tokens=3, reset_s=0.001,
                    stream_create_s=0.001, first_token_s=0.01,
                    decode_s=[0.002, 0.003], completion_s=0.001,
                    result_fetch_s=0.001, generation_s=0.017,
                    response_text="abc", stop_reason="MAX_TOKENS",
                    submit_time=submit, token_times=tuple(token_times),
                    completed_time=completed, token_ids=(10, 11, 12),
                )

            def close(self):
                self.closed = True

        # Only the native session is replaced; dispatch, aggregation, memory
        # checkpoint readers, the native log capture, the evidence writes and
        # report persistence execute their real code.
        _perf_gguf.WinMLLlamaSession = FakeSession
        args = ["perf", "-m", str(model), "--device", device, "--max-new-tokens", "3",
                "--warmup", "1", "--iterations", "2", "--output", str(output)]
        if evidence == "True":
            # As the skill runs it: no memory option, since --evidence turns sampling off.
            ids = root / "prompt-ids.json"
            ids.write_text("[1, 2, 3, 4]", encoding="utf-8")
            args.extend(["--prompt-ids-file", str(ids), "--context-length", "256",
                         "--ignore-eos", "--evidence", str(root / "task-events.json")])
        else:
            args.extend(["--prompt", "test prompt"])
            if not memory:
                args.append("--no-memory")
        if ignored_config == "True":
            config = root / "ignored-config.json"
            config.write_text("{}", encoding="utf-8")
            args.extend(["--config", str(config)])
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not attempts, attempts
        assert not [name for name in sys.modules if is_blocked(name)]
        assert "windowsml" not in sys.modules
        assert len(sessions) == 1 and sessions[0].generated == 3 and sessions[0].closed
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["schema_version"] == 3
        assert data["benchmark_info"]["api"] == "windowsml.tasks"
        assert data["benchmark_info"]["effective_device"] == device
        assert data["aggregate"]["warmup_request_count"] == 1
        assert data["aggregate"]["timed_request_count"] == 2
        assert len(data["requests"]) == 3
        assert data["response_text"] == "abc"
        assert ("memory" in data) == memory
        if memory:
            assert data["memory"]["rss_baseline_mb"] > 0
            assert data["memory"]["rss_after_compile_mb"] > 0
            assert data["memory"]["rss_after_inference_mb"] > 0
            assert not any(key.startswith("vram") for key in data["memory"])
        rows = None
        if evidence == "True":
            document = json.loads((root / "task-events.json").read_text(encoding="utf-8"))
            assert document["status"] == "complete", document
            rows = len(document["requests"])
        info = data["benchmark_info"]
        print(json.dumps({
            "device": device, "memory": memory, "context_length": info["context_length"],
            "ignore_eos": info["ignore_eos"], "prompt": info["prompt"], "rows": rows,
            "gguf_modules": sorted(
                name for name in sys.modules if name.startswith("winml.modelkit.commands._gguf_")
            ),
            "imports": attempts,
        }))
        """,
        str(tmp_path),
        device,
        str(memory),
        str(ignored_config),
        str(evidence),
    )
    assert outcome == {
        "device": device,
        "memory": memory,
        "context_length": 256 if evidence else 7,
        "ignore_eos": evidence,
        "prompt": None if evidence else "test prompt",
        "rows": 3 if evidence else None,
        "gguf_modules": EVIDENCE_MODULES if evidence else [],
        "imports": [],
    }


def test_extra_blocked_names_are_enforced() -> None:
    # Control: a name passed in extra_blocked is refused like the built-in ones.
    outcome = _run_isolated(
        """
        try:
            import winml.modelkit.session.cgc_session
        except AssertionError as exc:
            error = str(exc)
        else:
            error = None
        print(json.dumps({"imports": attempts, "error": error}))
        """,
        extra_blocked=("winml.modelkit.session.cgc_session",),
    )
    assert outcome == {
        "imports": ["winml.modelkit.session.cgc_session"],
        "error": "GGUF imported unrelated runtime: winml.modelkit.session.cgc_session",
    }


@pytest.mark.parametrize(
    ("kind", "memory", "ignored_config"),
    [("gguf", True, False), ("folder", False, True), ("folder", True, False)],
)
def test_cgc_command_runs_without_heavy_dependency_imports(
    tmp_path: Path, kind: str, memory: bool, ignored_config: bool
) -> None:
    outcome = _run_isolated(
        """
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from click.testing import CliRunner
        from winml.modelkit.cli import main
        from winml.modelkit.commands import _perf_cgc
        from winml.modelkit.session import TaskGenerationTiming
        from winml.modelkit.telemetry.telemetry import Telemetry

        Telemetry.get_or_init = classmethod(lambda cls: SimpleNamespace(disabled=True))
        root, kind, memory, ignored_config = sys.argv[1:]
        memory = memory == "True"
        root = Path(root)
        if kind == "gguf":
            model = root / "placeholder.gguf"
            model.write_bytes(b"GGUF")
        else:
            # The report and the --config file stay outside the checkpoint folder.
            model = root / "ckpt"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"")
        output = root / "report.json"
        sessions = []

        class FakeSession:
            # The whole WinMLCGCSession surface, so a missing attribute never
            # reads as an import failure.
            def __init__(self, model_path, *, model_format, device_luid, program_cache_dir):
                assert model_path == model
                self.model_path = model_path
                self.model_format = model_format
                self.adapter_name = "Fake GPU"
                self.adapter_luid = "0x00000000_0x0000BEEF"
                self.adapter_selection = "high-performance"
                self.vocabulary_size = None
                self.context_length = None
                self.effective_device = None
                self.chat_template_applied = None
                self.task_default_eos_token_ids = None
                self.load_timings_ms = dict.fromkeys(
                    ("runtime_create_duration_ms", "target_create_duration_ms",
                     "model_create_duration_ms", "set_model_duration_ms",
                     "session_create_duration_ms"),
                    1.0,
                )
                self.generated = 0
                self.closed = False
                sessions.append(self)

            def open_tokenizer(self):
                pass

            def encode_prompt(self, prompt, *, apply_template=True):
                assert prompt == "test prompt"
                self.chat_template_applied = apply_template
                return [1, 2, 3, 4]

            def load(self, *, context_length=None, eos_token_ids=None):
                self.context_length = context_length
                self.effective_device = "gpu"
                eos = (2,) if eos_token_ids is None else tuple(eos_token_ids)
                self.task_default_eos_token_ids = eos

            def generate_timed(self, token_ids, *, max_new_tokens, clock=time.perf_counter):
                assert token_ids == [1, 2, 3, 4] and max_new_tokens == 3
                self.generated += 1
                submit, *token_times, completed = (clock() for _ in range(max_new_tokens + 2))
                return TaskGenerationTiming(
                    input_tokens=4, generated_tokens=3, reset_s=0.001,
                    stream_create_s=0.001, first_token_s=0.01,
                    decode_s=[0.002, 0.003], completion_s=0.001,
                    result_fetch_s=0.001, generation_s=0.017,
                    response_text="abc", stop_reason="MAX_TOKENS",
                    submit_time=submit, token_times=tuple(token_times),
                    completed_time=completed, token_ids=(10, 11, 12),
                )

            def close(self):
                self.closed = True

        # Only the native session is replaced; routing, aggregation, the memory
        # checkpoints and report persistence execute their real code.
        _perf_cgc.WinMLCGCSession = FakeSession
        args = ["perf", "-m", str(model), "--backend", "cgc", "--prompt", "test prompt",
                "--max-new-tokens", "3", "--warmup", "1", "--iterations", "2",
                "--output", str(output)]
        if not memory:
            args.append("--no-memory")
        if ignored_config == "True":
            config = root / "ignored-config.json"
            config.write_text("{}", encoding="utf-8")
            args.extend(["--config", str(config)])
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not attempts, attempts
        assert not [name for name in sys.modules if is_blocked(name)]
        assert "windowsml" not in sys.modules
        assert len(sessions) == 1 and sessions[0].generated == 3 and sessions[0].closed
        data = json.loads(output.read_text(encoding="utf-8"))
        assert ("memory" in data) == memory
        if memory:
            # The pinned GPU's VRAM is recorded next to process RAM.
            assert "vram_local_after_compile_mb" in data["memory"]
        info = data["benchmark_info"]
        print(json.dumps({
            "kind": kind, "memory": memory, "backend": info["backend"],
            "model_format": info["model_format"], "imports": attempts,
        }))
        """,
        str(tmp_path),
        kind,
        str(memory),
        str(ignored_config),
        extra_blocked=CGC_BLOCKED,
    )
    assert outcome == {
        "kind": kind,
        "memory": memory,
        "backend": "cgc",
        "model_format": "gguf" if kind == "gguf" else "safetensors",
        "imports": [],
    }
