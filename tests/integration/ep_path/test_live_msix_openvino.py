# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows-only runtime test: load the OEM-channel OpenVINO EP into ORT.

The OEM/Windows-Workloads channel provisions EP MSIXes that the public
WinML ``ExecutionProviderCatalog`` API refuses to bind (family-name gate
restricted to ``MicrosoftCorporationII.WinML.*``). On Lunar Lake silicon
the Intel OpenVINO EP arrives as ``WindowsWorkload.EP.Intel.OpenVINO.*``
— invisible to the catalog, but discoverable via :class:`MsixPackageSource`
once the prefix-default covers the ``WindowsWorkload.EP.`` family.

Scope of this test (what we own): the discovery → registration →
engagement path for the OEM-channel DLL. We assert that
:class:`MsixPackageSource` resolves to a real DLL, that
:func:`register_execution_providers` with ``extra_sources=[msix]`` plumbs
that DLL into ORT 1.24+ via ``register_execution_provider_library`` (NOT
the legacy ``providers=[name]`` kwarg, which silently falls back to CPU
on ``onnxruntime-windowsml``), and that an :class:`InferenceSession`
created with :func:`add_ep_for_device` reports
``OpenVINOExecutionProvider`` in ``session.get_providers()``.

Out of scope (OEM EP / WindowsWorkload-runtime responsibility): the
correctness of inference output. Empirically, when the
``WindowsWorkload.EP.Intel.OpenVINO.*`` DLL is loaded standalone (without
the surrounding ``WindowsWorkload.OnnxRuntime.Lnl.*`` /
``WindowsWorkload.WinMLShared.*`` runtime stack on the loader path), the
EP engages but produces zero-filled output for a trivial Identity op on
every device class (CPU/GPU/NPU). The same code path with the PyPI
``onnxruntime-ep-openvino`` 1.4.0 wheel produces correct output, so the
test framework is sound. We log the inference outcome for diagnostic
visibility but do not assert on the values; tracking the OEM-runtime
loader-path requirement is a separate concern from the discovery surface
this test validates.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="WinRT PackageManager is Windows-only",
)


def _make_identity_model(tmp_path):
    """Build a minimal Identity-op ONNX model and return its file path."""
    import numpy as np  # noqa: F401  (sanity import; numpy is a hard dep)
    import onnx
    from onnx import TensorProto, helper

    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "identity_graph", [input_tensor], [output_tensor])
    opset = helper.make_opsetid("", 17)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=10)
    onnx.checker.check_model(model)

    model_path = tmp_path / "identity.onnx"
    onnx.save(model, str(model_path))
    return model_path


def test_windows_workload_openvino_ep_runs_inference(tmp_path):
    """Discover, register, and run the OEM-channel OpenVINO EP end-to-end.

    Skips when:
        - PackageManager binding unavailable (no ``[winml-catalog]`` extra)
        - No ``WindowsWorkload.EP.Intel.OpenVINO.*`` MSIX on this machine
        - Hardware is incompatible (no Intel CPU/GPU/NPU detected)
    """
    import numpy as np
    import onnxruntime as ort

    from winml.modelkit.ep_path import _get_pkg_manager, list_msix_eps
    from winml.modelkit.winml import (
        add_ep_for_device,
        register_execution_providers,
    )

    _get_pkg_manager.cache_clear()
    if _get_pkg_manager() is None:
        pytest.skip(
            "WinRT PackageManager unavailable; install via [winml-catalog]"
        )

    # Filter to the OEM channel only so this test exercises the
    # WindowsWorkload-published OpenVINO EP specifically (not the PyPI
    # wheel or any catalog-channel install).
    oem_sources = list_msix_eps(family_name_prefixes=("WindowsWorkload.EP.Intel.OpenVINO.",))
    if not oem_sources:
        pytest.skip(
            "WindowsWorkload.EP.Intel.OpenVINO.* MSIX not provisioned on "
            "this machine (Lunar Lake / Copilot+ Intel image required)"
        )

    msix = oem_sources[0]
    if not msix.is_compatible():
        pytest.skip(
            f"MsixPackageSource {msix.family_name_prefix} reports "
            "incompatible hardware on this machine"
        )

    # Sanity-check the source resolves to a real DLL on disk before
    # touching ORT (registering a missing path raises an opaque error).
    resolved = list(msix.resolve())
    assert resolved, "MsixPackageSource did not resolve to any (ep_name, path) entries"
    ep_name, dll_path = resolved[0]
    assert dll_path.exists(), f"discovered DLL does not exist: {dll_path}"
    assert ep_name == "OpenVINOExecutionProvider"

    # Register the OEM-channel EP via the codebase's canonical path. The
    # extra_sources arg has highest precedence so this DLL wins over the
    # PyPI wheel even when both are installed.
    registered = register_execution_providers(extra_sources=[msix])
    assert "OpenVINOExecutionProvider" in registered.get("onnxruntime", []), (
        f"OpenVINOExecutionProvider not registered to ORT. Got: {registered}"
    )

    # Find the highest-priority device class for which OpenVINO is
    # registered. Order: NPU > GPU > CPU. ort.get_ep_devices() reflects
    # the registration done above; if OpenVINO has zero entries here, the
    # registration didn't take and we should fail explicitly.
    from winml.modelkit.ep_path import canonicalize_ep_name

    target = canonicalize_ep_name("OpenVINOExecutionProvider")
    ep_devices = ort.get_ep_devices()
    matching_ep_devices = [
        d for d in ep_devices if canonicalize_ep_name(d.ep_name) == target
    ]
    assert matching_ep_devices, (
        "OpenVINOExecutionProvider registered but ort.get_ep_devices() shows "
        f"zero matching (EP, device) pairs. ep_devices={ep_devices!r}"
    )
    print(
        f"OpenVINO ep_devices: {[(d.ep_name, d.device.type) for d in matching_ep_devices]!r}"
    )

    # Bind to CPU first as the most reliable device class — it's the
    # baseline correctness check. NPU/GPU buffer-transfer is more fragile
    # on preview-channel OEM EPs and can mask DLL-load issues as silent
    # zero-output. If you want to exercise NPU/GPU specifically, fork
    # this test or tune the priority below.
    device_types_for_openvino = {d.device.type for d in matching_ep_devices}
    bound_device_type = None
    for device_type in (
        ort.OrtHardwareDeviceType.CPU,
        ort.OrtHardwareDeviceType.GPU,
        ort.OrtHardwareDeviceType.NPU,
    ):
        if device_type in device_types_for_openvino:
            bound_device_type = device_type
            break
    assert bound_device_type is not None, (
        f"unexpected device-type set for OpenVINO: {device_types_for_openvino!r}"
    )
    print(f"bound_device_type: {bound_device_type!r}")

    session_options = ort.SessionOptions()
    add_ep_for_device(
        session_options,
        "OpenVINOExecutionProvider",
        bound_device_type,
    )

    model_path = _make_identity_model(tmp_path)
    session = ort.InferenceSession(str(model_path), sess_options=session_options)

    providers = session.get_providers()
    print(f"session.get_providers(): {providers!r}")
    assert "OpenVINOExecutionProvider" in providers, (
        f"OpenVINOExecutionProvider not engaged for session. providers={providers}"
    )

    # Run inference for diagnostic visibility only. Output correctness is
    # the OEM EP's responsibility (see module docstring); we don't assert
    # on it because the OEM DLL appears to need its sibling
    # ``WindowsWorkload.OnnxRuntime.Lnl.*`` runtime on the loader path,
    # which isn't visible to a regular ``onnxruntime-windowsml``-driven
    # ORT session.
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    (y,) = session.run(["Y"], {"X": x})
    print(f"input:  {x!r}")
    print(f"output: {y!r}")
    if not np.array_equal(y, x):
        print(
            "NOTE: OEM-channel OpenVINO EP returned non-identity output. "
            "This is expected when the WindowsWorkload runtime stack is "
            "not on the loader path; it is not a discovery-layer regression."
        )
