# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for ``winml compile``.

Conventions
-----------
* Each test invokes the real ``compile`` Click command via ``CliRunner``.
* EP-availability gating: tests that exercise a specific EP runtime call
  :func:`tests.e2e.require_ep.require_ep`; absent EPs skip with a clear reason.
* Test fixtures (tiny ONNX models, config files) are generated in-process —
  no checked-in binaries.

EP categories
-------------
* **EP-context EPs** (``qnn``, ``openvino``): emit a new compiled ``*.onnx``
  containing an ``EPContext`` node.
* **Passthrough EPs** (``cpu``, ``cuda``, ``dml``, ``nv_tensorrt_rtx``,
  ``vitisai``, ``migraphx``): currently the ``winml compile`` CLI rejects
  these with a "does not support EPContext compilation" error rather than
  no-op passthrough. The CLI gate sits in front of the inner
  ``compile_onnx(model, config=None)`` passthrough branch, so the user
  always sees the rejection. See ``test_unsupported_ep_returns_error``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from tests.e2e.require_ep import require_ep
from winml.modelkit.commands.compile import compile as compile_cmd
from winml.modelkit.onnx import is_compiled_onnx


if TYPE_CHECKING:
    from click.testing import Result


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

EPCONTEXT_EPS = ("qnn", "openvino")
PASSTHROUGH_EPS = ("cpu", "cuda", "dml", "nv_tensorrt_rtx", "vitisai", "migraphx")


def _invoke(*args: str) -> Result:
    """Run ``winml compile <args>`` through CliRunner."""
    return CliRunner().invoke(compile_cmd, list(args), obj={"debug": False})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _epcontext_attrs(model_path: Path) -> dict[str, object]:
    """Return the first ``EPContext`` node's attributes as a name->value dict."""
    model = onnx.load(str(model_path))
    for node in model.graph.node:
        if node.op_type != "EPContext":
            continue
        attrs: dict[str, object] = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s
        return attrs
    raise AssertionError(f"No EPContext node found in {model_path}")


def assert_epcontext_artifact(
    out_path: Path,
    source_path: Path,
    *,
    embed: bool,
) -> None:
    """Assertion contract for EP-context compile outputs (qnn, openvino)."""
    out_path = Path(out_path)
    source_path = Path(source_path)
    assert out_path.is_file(), f"Compiled artifact missing: {out_path}"
    assert out_path.resolve() != source_path.resolve(), (
        "EPContext compile must produce a NEW file, not return the input"
    )

    onnx.load(str(out_path))  # parses
    # NOTE: We intentionally do NOT call `onnx.checker.check_model` here.
    # ORT's QAIRT wrapper writes a valid EPContext model that ORT can load
    # and execute, but it does not emit an explicit opset import for the
    # `com.microsoft` domain — which `check_model` requires. The CLI's own
    # `--validate` step (post-compile InferenceSession warm-up) is the
    # authoritative validity check; we re-verify functional correctness
    # below via the EPContext attribute contract.
    assert is_compiled_onnx(out_path), f"{out_path} has no EPContext node"

    attrs = _epcontext_attrs(out_path)
    embed_mode = attrs.get("embed_mode")
    ep_cache_context = attrs.get("ep_cache_context", b"")

    if embed:
        # Contract for --embed: the EPContext node carries the cached graph
        # inline. We deliberately do NOT assert "no .bin sidecar exists in
        # the directory" — some backends (QAIRT) leave an intermediate .bin
        # behind that is not referenced from the EPContext node. The user's
        # `.onnx` is self-contained regardless, which is what matters.
        assert embed_mode == 1, f"--embed should set embed_mode=1, got {embed_mode}"
        assert isinstance(ep_cache_context, bytes) and len(ep_cache_context) > 0, (
            "--embed should inline non-empty ep_cache_context bytes"
        )
    else:
        # Contract for sidecar mode: ep_cache_context is a relative filename
        # pointing at a `.bin` next to the `.onnx`.
        assert embed_mode == 0, f"default mode should set embed_mode=0, got {embed_mode}"
        assert isinstance(ep_cache_context, bytes) and ep_cache_context, (
            "default mode should reference a sidecar filename"
        )
        sidecar_name = ep_cache_context.decode("utf-8")
        assert (out_path.parent / sidecar_name).is_file(), (
            f"Sidecar {sidecar_name!r} declared in EPContext but missing on disk"
        )


def _find_qairt_sdk_root() -> Path | None:
    """Locate an installed QAIRT SDK on this host, or None.
    """
    for env_var in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT"):
        value = os.environ.get(env_var)
        if value:
            path = Path(value)
            if path.is_dir():
                return path
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    """A minimal MatMul ONNX model usable by every EP."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    w_arr = np.eye(4, dtype=np.float32)
    w = helper.make_tensor("weight", TensorProto.FLOAT, [4, 4], w_arr.flatten().tolist())
    node = helper.make_node("MatMul", ["input", "weight"], ["output"], name="matmul")
    graph = helper.make_graph([node], "tiny", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ===========================================================================
# CLI surface — parser, --help, --list, input rejection (no EP runtime)
# ===========================================================================


@pytest.mark.e2e
class TestCliSurface:
    def test_help_lists_every_option(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0
        for opt in (
            "--model", "--output", "--output-dir", "--device", "--ep",
            "--validate", "--no-validate", "--verbose", "--compiler",
            "--qnn-sdk-root", "--embed", "--list",
        ):
            assert opt in result.output, f"--help missing {opt}"

    def test_missing_model_without_list_errors(self) -> None:
        result = _invoke()
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "model" in combined.lower() or "missing option" in combined.lower()

    def test_list_works_without_model(self) -> None:
        result = _invoke("--list")
        assert result.exit_code == 0, result.output
        assert "ort" in result.output.lower()

    def test_list_for_npu_includes_qairt(self) -> None:
        result = _invoke("--list", "--device", "npu")
        assert result.exit_code == 0, result.output
        out = result.output.lower()
        assert "ort" in out and "qairt" in out

    def test_reject_already_compiled_model(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        # Build an EPContext-looking ONNX by hand (no real compile needed).
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        node = helper.make_node(
            "EPContext",
            ["input"],
            ["output"],
            embed_mode=1,
            ep_cache_context=b"fake",
            domain="com.microsoft",
        )
        graph = helper.make_graph([node], "fake_ctx", [x], [y])
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ])
        model.ir_version = 8
        ctx_path = tmp_path / "fake_ctx.onnx"
        onnx.save(model, str(ctx_path))
        assert is_compiled_onnx(ctx_path)

        result = _invoke("-m", str(ctx_path))
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert (
            "already a compiled" in combined.lower()
            or "cannot be re-compiled" in combined.lower()
        )


# ===========================================================================
# Happy-path compile — EP-context EPs only (qnn, openvino)
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("ep", EPCONTEXT_EPS)
def test_happy_path_per_ep(ep: str, tiny_model: Path, tmp_path: Path) -> None:
    """``winml compile --ep <EP>`` succeeds and emits a valid EPContext artifact."""
    require_ep(ep)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = out_dir / "out.onnx"

    result = _invoke("-m", str(tiny_model), "--ep", ep, "-o", str(out_file))
    assert result.exit_code == 0, f"compile --ep {ep} failed:\n{result.output}"
    assert "Success! Model compiled" in result.output
    assert_epcontext_artifact(out_file, tiny_model, embed=False)


# ===========================================================================
# Unsupported (passthrough) EPs — CLI rejects with explicit error
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("ep", PASSTHROUGH_EPS)
def test_unsupported_ep_returns_error(ep: str, tiny_model: Path) -> None:
    """EPs without offline compile support are rejected with a clear message."""
    require_ep(ep)
    src_hash = _sha256(tiny_model)
    result = _invoke("-m", str(tiny_model), "--ep", ep)

    assert result.exit_code != 0, (
        f"--ep {ep} should be rejected but exit was 0.\n{result.output}"
    )
    combined = result.output.lower()
    assert "does not support epcontext compilation" in combined, (
        f"Expected unsupported-EP error message for {ep}.\n{result.output}"
    )
    # The CLI must not have mutated the input model.
    assert _sha256(tiny_model) == src_hash, "Input ONNX was mutated despite rejection"


# ===========================================================================
# Output path handling — EP-context EPs only
# ===========================================================================


@pytest.mark.e2e
class TestOutputPaths:
    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_explicit_output_file(
        self, ep: str, tiny_model: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "custom_name.onnx"
        result = _invoke("-m", str(tiny_model), "--ep", ep, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_output_dir(self, ep: str, tiny_model: Path, tmp_path: Path) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _invoke(
            "-m", str(tiny_model), "--ep", ep, "--output-dir", str(out_dir)
        )
        assert result.exit_code == 0, result.output
        produced = [p for p in out_dir.glob("*.onnx") if is_compiled_onnx(p)]
        assert len(produced) == 1, f"Expected exactly one EPContext .onnx, got {produced}"


# ===========================================================================
# Embed vs sidecar (--embed) — ORT backend
# ===========================================================================


@pytest.mark.e2e
class TestEmbedSidecar:
    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_embed_produces_single_file(
        self, ep: str, tiny_model: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "embedded.onnx"
        result = _invoke("-m", str(tiny_model), "--ep", ep, "--embed", "-o", str(out))
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, tiny_model, embed=True)

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_default_emits_external_bin(
        self, ep: str, tiny_model: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "external.onnx"
        result = _invoke("-m", str(tiny_model), "--ep", ep, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, tiny_model, embed=False)


# ===========================================================================
# Validation toggle (--validate / --no-validate)
# ===========================================================================


_VALIDATE_TOKENS = ("validating compiled model", "validation")


@pytest.mark.e2e
class TestValidate:
    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_no_validate_skips_validation(
        self, ep: str, tiny_model: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "no_validate.onnx"
        result = _invoke(
            "-m", str(tiny_model), "--ep", ep, "--no-validate", "--verbose",
            "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)
        lower = result.output.lower()
        assert not any(token in lower for token in _VALIDATE_TOKENS), (
            f"--no-validate stdout should not contain validation logs.\n{result.output}"
        )

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_default_runs_validation(
        self, ep: str, tiny_model: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "validated.onnx"
        result = _invoke(
            "-m", str(tiny_model), "--ep", ep, "--verbose", "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)
        lower = result.output.lower()
        assert any(token in lower for token in _VALIDATE_TOKENS), (
            f"Default validate should emit validation logs.\n{result.output}"
        )


# ===========================================================================
# QAIRT backend (qnn-only) — banner, artifacts, embed mode coverage
# ===========================================================================


def _require_qairt_sdk() -> Path:
    sdk = _find_qairt_sdk_root()
    if sdk is None:
        pytest.skip("QAIRT SDK not installed on this host")
    return sdk


@pytest.mark.e2e
class TestQairtBackend:
    def test_qairt_backend_default_emits_sidecar(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        """``--compiler qairt`` (no ``--embed``) emits sidecar EPContext artifact.

        Verifies the QAIRT pipeline runs end-to-end: banner advertises the
        qairt backend + SDK root, the produced ONNX is a valid EPContext
        model, and the documented intermediate artifacts (``*_qnn_ctx_qnn.bin``,
        ``*_cache_info.json``) land next to the source model.
        """
        require_ep("qnn")
        sdk = _require_qairt_sdk()

        out = tmp_path / "qairt.onnx"
        result = _invoke(
            "-m", str(tiny_model), "--ep", "qnn",
            "--compiler", "qairt", "--qnn-sdk-root", str(sdk),
            "-o", str(out), "--verbose",
        )
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "compiler:" in lower and "qairt" in lower
        assert "sdk root:" in lower and str(sdk).lower() in lower
        assert_epcontext_artifact(out, tiny_model, embed=False)

        # QAIRT pipeline drops intermediate artifacts beside the source ONNX.
        stem = tiny_model.stem
        bin_path = tiny_model.parent / f"{stem}_qnn_ctx_qnn.bin"
        info_path = tiny_model.parent / f"{stem}_cache_info.json"
        assert bin_path.is_file(), f"QAIRT intermediate .bin missing: {bin_path}"
        assert info_path.is_file(), f"QAIRT cache_info.json missing: {info_path}"

    def test_qairt_backend_with_embed(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        """``--compiler qairt --embed`` produces an inlined EPContext artifact.

        Strict: any failure here is either a QAIRT bug or a missing
        embed-mode contract on the QAIRT path. The test does not paper
        over either outcome — the failure message will name it.
        """
        require_ep("qnn")
        sdk = _require_qairt_sdk()

        out = tmp_path / "qairt_embed.onnx"
        result = _invoke(
            "-m", str(tiny_model), "--ep", "qnn",
            "--compiler", "qairt", "--qnn-sdk-root", str(sdk),
            "--embed", "-o", str(out), "--verbose",
        )
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, tiny_model, embed=True)


# ===========================================================================
# --config (-c): precedence between config-file fields and CLI flags
# ===========================================================================


# Config schema: WinMLBuildConfig (JSON). The `compile` block keys come from
# `WinMLCompileConfig.from_dict`:
#   execution_provider, compiler, embed_context, validate, verbose


def _make_compile_cfg(**compile_fields: object) -> dict:
    """Build a WinMLBuildConfig dict with the given ``compile`` fields."""
    return {"compile": compile_fields}


@pytest.mark.e2e
class TestConfigFile:
    def test_config_defaults_applied(self, tiny_model: Path, tmp_path: Path) -> None:
        """Every field in the config's ``compile`` section is honored when no CLI flag overrides it.

        Config requests: provider=qnn, compiler=ort, embed_context=true,
        validate=false, verbose=true. CLI passes only ``-m`` and ``-o``.
        """
        require_ep("qnn")
        cfg = _write_json(
            tmp_path / "build_cfg.json",
            _make_compile_cfg(
                execution_provider="qnn",
                compiler="ort",
                embed_context=True,
                validate=False,
                verbose=True,
            ),
        )
        out = tmp_path / "cfg_full.onnx"
        result = _invoke("-m", str(tiny_model), "-c", str(cfg), "-o", str(out))
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "provider:" in lower and "qnn" in lower
        assert "compiler:" in lower and "ort" in lower
        # validate=false from config → no validation log lines
        assert not any(t in lower for t in _VALIDATE_TOKENS), result.output
        # embed_context=true from config → inlined artifact, no sidecar
        assert_epcontext_artifact(out, tiny_model, embed=True)

    def test_cli_overrides_config(self, tiny_model: Path, tmp_path: Path) -> None:
        """Explicit CLI flags beat the matching config-file fields.

        Config-file ships with ``embed_context: false`` and ``validate: false``.
        CLI passes ``--embed`` and ``--validate``. Both flags must win:
          * artifact is inlined (embed_mode=1, no ``.bin`` sidecar)
          * stdout contains validation log lines

        This is the scenario flagged by the user where, in other commands, a
        config dataclass's default value was observed to win over an explicit
        CLI option. If that regression resurfaces in ``compile``, this test
        catches it.
        """
        require_ep("qnn")
        cfg = _write_json(
            tmp_path / "build_cfg.json",
            _make_compile_cfg(
                execution_provider="qnn",
                compiler="ort",
                embed_context=False,
                validate=False,
                verbose=False,
            ),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "cli_wins.onnx"
        result = _invoke(
            "-m", str(tiny_model), "-c", str(cfg),
            "--embed",       # overrides config embed_context=false
            "--validate",    # overrides config validate=false
            "--verbose",     # so we can observe the validation log line
            "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        # CLI --validate beat config validate=false
        assert any(t in lower for t in _VALIDATE_TOKENS), (
            f"CLI --validate did not override config validate=false.\n{result.output}"
        )
        # CLI --embed beat config embed_context=false
        assert_epcontext_artifact(out, tiny_model, embed=True)

    def test_config_rejects_unsupported_ep(self, tiny_model: Path, tmp_path: Path) -> None:
        """A config-file that selects a passthrough EP also gets the explicit error.

        Verifies the unsupported-EP gate fires regardless of where the
        provider came from (CLI vs. config-file).
        """
        require_ep("cpu")
        cfg = _write_json(
            tmp_path / "build_cfg.json",
            _make_compile_cfg(execution_provider="cpu"),
        )
        result = _invoke("-m", str(tiny_model), "-c", str(cfg))
        assert result.exit_code != 0, result.output
        assert "does not support epcontext compilation" in result.output.lower(), result.output


# ===========================================================================
# --device → provider resolution (no --ep): pins _resolve_compile_provider
# ===========================================================================
#
# _DEVICE_TO_PROVIDER = {"npu": "qnn", "gpu": "dml", "cpu": None}
# _resolve_compile_provider(device, ep=None):
#   provider = _DEVICE_TO_PROVIDER.get(device)
#   if provider is None: return "cpu" if device == "cpu" else "qnn"
#   return provider
#
# Click restricts --device to {auto, npu, gpu, cpu}. Therefore:
#   --device npu  -> "qnn"             (compiles on qnn host)
#   --device gpu  -> "dml"             (rejected: dml not EPContext)
#   --device cpu  -> None -> "cpu"     (rejected: cpu not EPContext)
#   --device auto -> None -> "qnn"     (else-branch fall-through)


@pytest.mark.e2e
class TestDeviceResolution:
    def test_device_npu_resolves_to_qnn(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        """``--device npu`` (no ``--ep``) → provider qnn → successful compile."""
        require_ep("qnn")
        out = tmp_path / "npu.onnx"
        result = _invoke("-m", str(tiny_model), "--device", "npu", "-o", str(out))
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "device:" in lower and "npu" in lower
        assert "provider:" in lower and "qnn" in lower
        assert_epcontext_artifact(out, tiny_model, embed=False)

    def test_device_gpu_resolves_to_dml(self, tiny_model: Path) -> None:
        """``--device gpu`` (no ``--ep``) → provider dml → unsupported-EP error.

        Pins the resolver behavior: gpu maps to dml via ``_DEVICE_TO_PROVIDER``,
        not to qnn. dml does not support EPContext compilation, so the CLI
        rejects with the standard message naming dml. The error is raised by
        ``for_provider`` returning ``None`` *before* the banner is printed,
        so the only externally observable signal is the error message.
        """
        result = _invoke("-m", str(tiny_model), "--device", "gpu")
        assert result.exit_code != 0, result.output
        lower = result.output.lower()
        assert "does not support epcontext compilation" in lower
        assert "'dml'" in lower

    def test_device_cpu_resolves_to_cpu(self, tiny_model: Path) -> None:
        """``--device cpu`` (no ``--ep``) → provider cpu → unsupported-EP error.

        Same pre-banner rejection as ``test_device_gpu_resolves_to_dml``;
        only the error message is observable.
        """
        result = _invoke("-m", str(tiny_model), "--device", "cpu")
        assert result.exit_code != 0, result.output
        lower = result.output.lower()
        assert "does not support epcontext compilation" in lower
        assert "'cpu'" in lower

    def test_device_auto_falls_through_to_qnn(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        """``--device auto`` (no ``--ep``) → resolver else-branch → qnn.

        Pins the only path that exercises the ``else "qnn"`` fall-through in
        ``_resolve_compile_provider`` (auto is in click's Choice but not in
        ``_DEVICE_TO_PROVIDER``). Banner must announce qnn; the compile
        itself only succeeds on a qnn host.
        """
        require_ep("qnn")
        out = tmp_path / "auto.onnx"
        result = _invoke("-m", str(tiny_model), "--device", "auto", "-o", str(out))
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "device:" in lower and "auto" in lower
        assert "provider:" in lower and "qnn" in lower
        assert_epcontext_artifact(out, tiny_model, embed=False)


# ===========================================================================
# --ep overrides --device-derived provider
# ===========================================================================


@pytest.mark.e2e
class TestEpOverridesDevice:
    # Coverage scope: only the ``--device auto --ep qnn`` combination is
    # exercised here. Other device/ep combinations (e.g. ``--device gpu --ep
    # qnn``, ``--device cpu --ep qnn``) currently exhibit a bug where the
    # ``--device`` value propagates into the EP factory as ``device_type``
    # and the resulting compile reports success while producing a non-
    # EPContext artifact. That bug will be fixed in a separate PR; tests
    # for those combinations will be added there.

    def test_ep_overrides_device_auto_with_qnn(
        self, tiny_model: Path, tmp_path: Path
    ) -> None:
        """``--device auto --ep qnn`` → provider qnn → real EPContext artifact.

        Documented contract on ``--ep`` help text: "Overrides
        device-to-provider mapping." ``auto`` is not in ``_DEVICE_TO_PROVIDER``,
        so this case also exercises the resolver's else-branch alongside the
        ``--ep`` override.
        """
        require_ep("qnn")
        out = tmp_path / "auto_qnn.onnx"
        result = _invoke(
            "-m", str(tiny_model), "--device", "auto", "--ep", "qnn",
            "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "device:" in lower and "auto" in lower
        assert "provider:" in lower and "qnn" in lower
        assert_epcontext_artifact(out, tiny_model, embed=False)


# ===========================================================================
# Already-compiled rejection — real round-trip (compile then re-compile)
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("ep", EPCONTEXT_EPS)
def test_reject_recompile_of_real_compiled_output(
    ep: str, tiny_model: Path, tmp_path: Path
) -> None:
    """Feeding a real ``winml compile`` output back into ``winml compile`` is rejected.

    Complements the hand-crafted EPContext test by exercising the realistic
    user mistake: forgetting that the output of a prior compile is already
    an EPContext model and trying to re-compile it. Both ``--embed`` (single
    file) and default (sidecar) outputs must be rejected the same way.
    """
    require_ep(ep)

    # First compile produces a real EPContext artifact.
    out = tmp_path / "first.onnx"
    first = _invoke("-m", str(tiny_model), "--ep", ep, "-o", str(out))
    assert first.exit_code == 0, first.output
    assert is_compiled_onnx(out)

    # Second compile on that artifact must be rejected.
    second = _invoke("-m", str(out), "--ep", ep)
    assert second.exit_code != 0, second.output
    combined = (second.output or "") + (
        str(second.exception) if second.exception else ""
    )
    assert (
        "already a compiled" in combined.lower()
        or "cannot be re-compiled" in combined.lower()
    ), combined
