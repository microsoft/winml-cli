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
* **Passthrough EPs** (``cpu``, ``dml``; full set also includes ``cuda``,
  ``nv_tensorrt_rtx``, ``vitisai``, ``migraphx``): currently the
  ``winml compile`` CLI rejects these with a "does not support EPContext
  compilation" error rather than no-op passthrough. The CLI gate sits in
  front of the inner ``compile_onnx(model, config=None)`` passthrough
  branch, so the user always sees the rejection. We parametrize the
  rejection test over ``cpu`` and ``dml`` only — those are the EPs
  commonly available on test hosts; the others hit the same gate.
  See ``test_unsupported_ep_returns_error``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import pytest
from click.testing import CliRunner

from tests.e2e.require_ep import require_ep, require_not_ep
from winml.modelkit.commands.compile import compile as compile_cmd
from winml.modelkit.onnx import is_compiled_onnx
from winml.modelkit.utils import normalize_ep_name
from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES


if TYPE_CHECKING:
    from click.testing import Result


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

EPCONTEXT_EPS = ("qnn", "openvino")
PASSTHROUGH_EPS = ("cpu", "dml")


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


def assert_banner_matches_artifact(result: Result, out_path: Path) -> None:
    """Banner ``Provider:`` line must match the artifact's ``EPContext.source``.

    Catches "lying success" where the CLI announces one EP but ORT silently
    produced the artifact via a different EP.
    """
    banner_ep: str | None = None
    for line in result.output.splitlines():
        if line.strip().lower().startswith("provider:"):
            banner_ep = line.split(":", 1)[1].strip()
            break
    assert banner_ep is not None, f"No 'Provider:' line in banner:\n{result.output}"

    source = _epcontext_attrs(out_path).get("source", b"")
    assert isinstance(source, bytes) and source, (
        f"Artifact at {out_path} has no EPContext.source attribute"
    )
    artifact_ep = source.decode("utf-8")
    assert banner_ep == artifact_ep, (
        f"Banner provider={banner_ep!r} but artifact source={artifact_ep!r}. "
        f"Lying success: ORT used a different EP than the resolver chose."
    )


def assert_by_run_inference(
    out_path: Path,
    *,
    device: str,
    ep: str,
    sample_input: dict,
) -> None:
    """Bind ``ep`` + ``device`` and run one inference call on the compiled artifact.

    Unlike the CLI's ``--validate`` step (which lets ORT pick any registered
    EP), this asserts the artifact specifically loads and runs on the
    requested ``(device, ep)`` pair. Catches the case where the compile
    succeeded against a different EP/device than the user asked for.
    """
    from winml.modelkit.session import WinMLSession

    session = WinMLSession(out_path, device=device, ep=ep)
    outputs = session.run(sample_input)
    assert outputs, "Inference produced no outputs"


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

# Note: the input ONNX model is provided by the conftest's `simple_matmul_onnx`
# fixture (1x4 MatMul, opset 13). Compile tests don't depend on the specific
# weight values or input names, so we reuse it instead of defining a local
# `simple_matmul_onnx` fixture.


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
        require_ep("qnn")
        result = _invoke("--list", "--device", "npu", "--ep", "qnn")
        assert result.exit_code == 0, result.output
        out = result.output.lower()
        assert "ort" in out and "qairt" in out

    def test_reject_already_compiled_model(
        self, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        # Build an EPContext-looking ONNX by hand (no real compile needed).
        x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
        y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])
        node = onnx.helper.make_node(
            "EPContext",
            ["input"],
            ["output"],
            embed_mode=1,
            ep_cache_context=b"fake",
            domain="com.microsoft",
        )
        graph = onnx.helper.make_graph([node], "fake_ctx", [x], [y])
        model = onnx.helper.make_model(graph, opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
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
def test_happy_path_per_ep(ep: str, simple_matmul_onnx: Path, tmp_path: Path) -> None:
    """``winml compile --ep <EP>`` succeeds and emits a valid EPContext artifact."""
    require_ep(ep)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = out_dir / "out.onnx"

    result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep, "-o", str(out_file))
    assert result.exit_code == 0, f"compile --ep {ep} failed:\n{result.output}"
    assert "Success! Model compiled" in result.output
    assert_epcontext_artifact(out_file, simple_matmul_onnx, embed=False)


# ===========================================================================
# Unsupported (passthrough) EPs — CLI rejects with explicit error
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("ep", PASSTHROUGH_EPS)
def test_unsupported_ep_returns_error(ep: str, simple_matmul_onnx: Path) -> None:
    """EPs without offline compile support are rejected with a clear message."""
    require_ep(ep)
    src_hash = _sha256(simple_matmul_onnx)
    result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep)

    assert result.exit_code != 0, (
        f"--ep {ep} should be rejected but exit was 0.\n{result.output}"
    )
    combined = result.output.lower()
    assert "does not support epcontext compilation" in combined, (
        f"Expected unsupported-EP error message for {ep}.\n{result.output}"
    )
    # The CLI must not have mutated the input model.
    assert _sha256(simple_matmul_onnx) == src_hash, "Input ONNX was mutated despite rejection"


# ===========================================================================
# Output path handling — EP-context EPs only
# ===========================================================================


@pytest.mark.e2e
class TestOutputPaths:
    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_explicit_output_file(
        self, ep: str, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "custom_name.onnx"
        result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_output_dir(self, ep: str, simple_matmul_onnx: Path, tmp_path: Path) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _invoke(
            "-m", str(simple_matmul_onnx), "--ep", ep, "--output-dir", str(out_dir)
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
        self, ep: str, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "embedded.onnx"
        result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep, "--embed", "-o", str(out))
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=True)

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_default_emits_external_bin(
        self, ep: str, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "external.onnx"
        result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep, "-o", str(out))
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=False)


# ===========================================================================
# Validation toggle (--validate / --no-validate)
# ===========================================================================


# The single emitter for the validation log line is
# `src/winml/modelkit/compiler/stages/compile.py:153` ("Validating compiled
# model..."). Use the exact phrase to avoid false positives like
# "skipping validation" or any error message containing the bare word
# "validation".
_VALIDATE_LOG = "validating compiled model"


@pytest.mark.e2e
class TestValidate:
    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_no_validate_skips_validation(
        self, ep: str, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "no_validate.onnx"
        result = _invoke(
            "-m", str(simple_matmul_onnx), "--ep", ep, "--no-validate", "--verbose",
            "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)
        lower = result.output.lower()
        assert _VALIDATE_LOG not in lower, (
            f"--no-validate stdout should not contain validation logs.\n{result.output}"
        )

    @pytest.mark.parametrize("ep", EPCONTEXT_EPS)
    def test_default_runs_validation(
        self, ep: str, simple_matmul_onnx: Path, tmp_path: Path
    ) -> None:
        require_ep(ep)
        out = tmp_path / "validated.onnx"
        result = _invoke(
            "-m", str(simple_matmul_onnx), "--ep", ep, "--verbose", "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        assert out.is_file() and is_compiled_onnx(out)
        lower = result.output.lower()
        assert _VALIDATE_LOG in lower, (
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
        self, simple_matmul_onnx: Path, tmp_path: Path
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
            "-m", str(simple_matmul_onnx), "--ep", "qnn",
            "--compiler", "qairt", "--qnn-sdk-root", str(sdk),
            "-o", str(out), "--verbose",
        )
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "compiler:" in lower and "qairt" in lower
        assert "sdk root:" in lower and str(sdk).lower() in lower
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=False)

        # QAIRT pipeline drops intermediate artifacts beside the source ONNX.
        stem = simple_matmul_onnx.stem
        bin_path = simple_matmul_onnx.parent / f"{stem}_qnn_ctx_qnn.bin"
        info_path = simple_matmul_onnx.parent / f"{stem}_cache_info.json"
        assert bin_path.is_file(), f"QAIRT intermediate .bin missing: {bin_path}"
        assert info_path.is_file(), f"QAIRT cache_info.json missing: {info_path}"

    def test_qairt_backend_with_embed(
        self, simple_matmul_onnx: Path, tmp_path: Path
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
            "-m", str(simple_matmul_onnx), "--ep", "qnn",
            "--compiler", "qairt", "--qnn-sdk-root", str(sdk),
            "--embed", "-o", str(out), "--verbose",
        )
        assert result.exit_code == 0, result.output
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=True)


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
    def test_config_defaults_applied(self, simple_matmul_onnx: Path, tmp_path: Path) -> None:
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
        result = _invoke("-m", str(simple_matmul_onnx), "-c", str(cfg), "-o", str(out))
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        assert "provider:" in lower and "qnn" in lower
        assert "compiler:" in lower and "ort" in lower
        # validate=false from config → no validation log lines
        assert _VALIDATE_LOG not in lower, result.output
        # embed_context=true from config → inlined artifact, no sidecar
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=True)

    def test_cli_overrides_config(self, simple_matmul_onnx: Path, tmp_path: Path) -> None:
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
            "-m", str(simple_matmul_onnx), "-c", str(cfg),
            "--embed",       # overrides config embed_context=false
            "--validate",    # overrides config validate=false
            "--verbose",     # so we can observe the validation log line
            "-o", str(out),
        )
        assert result.exit_code == 0, result.output
        lower = result.output.lower()
        # CLI --validate beat config validate=false
        assert _VALIDATE_LOG in lower, (
            f"CLI --validate did not override config validate=false.\n{result.output}"
        )
        # CLI --embed beat config embed_context=false
        assert_epcontext_artifact(out, simple_matmul_onnx, embed=True)

    def test_config_rejects_unsupported_ep(self, simple_matmul_onnx: Path, tmp_path: Path) -> None:
        """A config-file that selects a passthrough EP also gets the explicit error.

        Verifies the unsupported-EP gate fires regardless of where the
        provider came from (CLI vs. config-file).
        """
        require_ep("cpu")
        cfg = _write_json(
            tmp_path / "build_cfg.json",
            _make_compile_cfg(execution_provider="cpu"),
        )
        result = _invoke("-m", str(simple_matmul_onnx), "-c", str(cfg))
        assert result.exit_code != 0, result.output
        assert "does not support epcontext compilation" in result.output.lower(), result.output


# ===========================================================================
# Already-compiled rejection — real round-trip (compile then re-compile)
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("ep", EPCONTEXT_EPS)
def test_reject_recompile_of_real_compiled_output(
    ep: str, simple_matmul_onnx: Path, tmp_path: Path
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
    first = _invoke("-m", str(simple_matmul_onnx), "--ep", ep, "-o", str(out))
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


# ===========================================================================
# Device/EP good inputs
# ===========================================================================

_EPCONTEXT_CAPABLE_EPS = ("qnn", "openvino", "vitisai", "nv_tensorrt_rtx")
EP_DEVICE_SUPPORT: dict[str, tuple[str, ...]] = {
    alias: EP_SUPPORTED_DEVICES[normalize_ep_name(alias)]
    for alias in _EPCONTEXT_CAPABLE_EPS
}


def _expand_good_inputs(
    support: dict[str, tuple[str, ...]],
) -> list[tuple[str | None, str | None, str]]:
    """Cartesian-expand ``EP_DEVICE_SUPPORT`` into ``(device, ep, require_ep)`` rows."""
    rows: list[tuple[str | None, str | None, str]] = []
    for require_ep_name, devices in support.items():
        rows.extend(
            (device, ep, require_ep_name)
            for device in (*devices, "auto", None)
            for ep in (require_ep_name, None)
        )
    return rows


_GOOD_INPUT_PARAMS = _expand_good_inputs(EP_DEVICE_SUPPORT)


@pytest.mark.e2e
@pytest.mark.parametrize("device,ep,require_ep_name", _GOOD_INPUT_PARAMS)
def test_good_input_compiles_and_runs(
    device: str | None,
    ep: str | None,
    require_ep_name: str,
    simple_matmul_onnx: Path,
    sample_input: dict,
    tmp_path: Path,
) -> None:
    """Supported ``(device, ep)`` compiles to a real EPContext that ORT
    can load and run on the requested EP+device.
    """
    require_ep(require_ep_name)
    # Skip e2e for VitisAI due to Windows Access violation in model compilation for some models
    require_not_ep("vitisai")

    out = tmp_path / f"{device or 'nodev'}_{ep or 'noep'}.onnx"
    cmd = ["-m", str(simple_matmul_onnx), "-o", str(out)]
    if device is not None:
        cmd.extend(["--device", device])
    if ep is not None:
        cmd.extend(["--ep", ep])
    result = _invoke(*cmd)

    assert result.exit_code == 0, f"compile failed:\n{result.output}"
    assert "Success! Model compiled" in result.output, result.output
    assert_epcontext_artifact(out, simple_matmul_onnx, embed=False)
    assert_banner_matches_artifact(result, out)
    # When ``--ep`` is explicit, bind that EP. When it's omitted, the resolver
    # picks an EP based on host registry order, which is not deterministic
    # across hosts — read the actual EP back from the artifact's
    # ``EPContext.source`` so the inference binding matches what was compiled.
    if ep is not None:
        runtime_ep = ep
    else:
        source = _epcontext_attrs(out)["source"]
        assert isinstance(source, bytes) and source, (
            f"Artifact at {out} has no EPContext.source attribute"
        )
        runtime_ep = source.decode("utf-8")
    assert_by_run_inference(
        out,
        device=device if device is not None else "auto",
        ep=runtime_ep,
        sample_input=sample_input,
    )


# ===========================================================================
# Device/EP bad inputs
# ===========================================================================


def _assert_rejected(
    result: Result,
    error_phrase: str | tuple[str, ...],
    src_hash: str,
    src_path: Path,
) -> None:
    assert result.exit_code != 0, f"Expected rejection but exit was 0:\n{result.output}"
    # Some rejections raise an uncaught exception (e.g. ``sysinfo`` raises
    # ``ValueError`` from device resolution) rather than emitting a Click
    # ``UsageError`` whose text reaches ``result.output``. Search both.
    haystack = result.output + ("" if result.exception is None else f"\n{result.exception}")
    phrases = (error_phrase,) if isinstance(error_phrase, str) else error_phrase
    assert any(p in haystack for p in phrases), (
        f"Expected one of {phrases!r} in error output, got:\n{haystack}"
    )
    assert _sha256(src_path) == src_hash, "Input ONNX was mutated despite rejection"


def _expand_conflict_inputs(
    support: dict[str, tuple[str, ...]],
) -> list[tuple[str, str]]:
    """Derive ``(device, ep)`` pairs where the EP does not support the device."""
    return [
        (device, ep)
        for ep, devices in support.items()
        for device in {"npu", "gpu", "cpu"} - set(devices)
    ]


_BAD_INPUT_CONFLICT_PARAMS = _expand_conflict_inputs(EP_DEVICE_SUPPORT)


@pytest.mark.e2e
@pytest.mark.parametrize("device,ep", _BAD_INPUT_CONFLICT_PARAMS)
def test_bad_input_device_ep_conflict(
    device: str, ep: str, simple_matmul_onnx: Path
) -> None:
    """``--device X --ep Y`` is rejected when Y does not support X.

    Gated on ``ep`` being registered on this host so the ``sysinfo``
    device-resolution preflight has a non-empty intersection to evaluate
    and surfaces the device/EP incompatibility rather than the empty-
    registry warning.
    """
    require_ep(ep)
    src_hash = _sha256(simple_matmul_onnx)
    result = _invoke("-m", str(simple_matmul_onnx), "--device", device, "--ep", ep)
    # Two valid rejection paths depending on whether ORT enumerates ``ep``
    # on ``device``:
    #   * Yes (e.g. QNN on the cpu node) -> ``_resolve_compile_provider``
    #     policy check raises ``--ep X cannot run on --device Y``.
    #   * No (e.g. NvTensorRTRTX is GPU-only) -> ``resolve_device`` raises
    #     ``Device 'X' requested but no compatible EP is available``.
    _assert_rejected(
        result,
        (f"cannot run on --device {device}", "no compatible EP is available"),
        src_hash,
        simple_matmul_onnx,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("ep", ("dml", "cuda", "migraphx"))
def test_bad_input_unsupported_ep(ep: str, simple_matmul_onnx: Path) -> None:
    """``--ep X`` is rejected when X does not produce EPContext.

    Requires the EP to be registered on this host so the earlier
    registration check does not short-circuit the rejection we want to assert.
    """
    require_ep(ep)
    src_hash = _sha256(simple_matmul_onnx)
    result = _invoke("-m", str(simple_matmul_onnx), "--ep", ep)
    _assert_rejected(
        result, "does not support EPContext compilation", src_hash, simple_matmul_onnx
    )


# Pair each EP with a device it supports. With no ``--device``, ``sysinfo``
# falls back to CPU when the requested EP isn't registered, which then
# trips the policy check ("cannot run on --device cpu") instead of the
# host-state rejection this test targets.
_EP_NOT_REGISTERED_PARAMS = [
    (ep, EP_DEVICE_SUPPORT[ep][0]) for ep in _EPCONTEXT_CAPABLE_EPS
]


@pytest.mark.e2e
@pytest.mark.parametrize("ep,device", _EP_NOT_REGISTERED_PARAMS)
def test_bad_input_ep_not_registered(
    ep: str, device: str, simple_matmul_onnx: Path
) -> None:
    """``--ep X`` on a host without X is rejected.

    With the EP unavailable on host, ``sysinfo``'s device-resolution
    preflight raises ``ValueError`` ("Requested EP 'X' is not available
    on this system"), which the ``compile`` command converts to a
    ``UsageError`` at the CLI boundary.
    """
    require_not_ep(ep)
    src_hash = _sha256(simple_matmul_onnx)
    result = _invoke(
        "-m", str(simple_matmul_onnx), "--device", device, "--ep", ep
    )
    _assert_rejected(
        result, "is not available on this system", src_hash, simple_matmul_onnx
    )


@pytest.mark.e2e
def test_bad_input_no_ep_covers_device(simple_matmul_onnx: Path) -> None:
    """``--device cpu`` (no ``--ep``) on a host with no EPContext-capable EP
    covering cpu is rejected.

    ``CPUExecutionProvider`` is always implicitly available, so
    ``sysinfo``'s preflight resolves to it rather than raising. The
    rejection then comes from the capability check in
    ``commands/compile.py``: CPU cannot produce EPContext models.
    """
    require_not_ep("openvino")
    src_hash = _sha256(simple_matmul_onnx)
    result = _invoke("-m", str(simple_matmul_onnx), "--device", "cpu")
    _assert_rejected(
        result,
        "does not support EPContext compilation",
        src_hash,
        simple_matmul_onnx,
    )
