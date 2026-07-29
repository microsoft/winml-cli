# Export Device Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic export-time compatibility policy system that can apply EP/device-specific model config overrides, with QNN requiring eager Hugging Face attention export.

**Architecture:** Add a pure-data policy resolver under `winml.modelkit.export.policy`, store the resolved policy on `WinMLExportConfig`, and make `HTPExporter` apply compatibility contexts from that config. Config generation will use an explicit resolved EP/device target when the caller requested one, or the full supported EP/device catalog when no target was specified.

**Tech Stack:** Python dataclasses, pytest, torch ONNX export tests, existing WinML EP/device catalog (`EP_DEVICE_SPECS`), existing `WinMLBuildConfig`/`WinMLExportConfig` serialization.

## Global Constraints

- Do not hardcode model architecture names, ONNX node names, tensor names, layer naming patterns, or model-specific logic.
- Use pytest only; run tests with `uv run`.
- Add no new dependencies.
- Keep `winml sys`/help lightweight: the new policy module must not import `torch` or `transformers`.
- Preserve existing compile/quant runtime target selection behavior.
- Export policy defaults to the full WinML supported EP/device catalog when no EP/device is explicitly specified.
- Include resolved export compatibility in serialized config and cache keys.
- Use relative imports in `src\`.
- Run `uv run ruff check --fix ...` after Python changes.

---

## File Structure

- Create `src\winml\modelkit\export\policy.py`: pure-data compatibility config, target model, rule registry, resolver, and helper to convert explicit requested EP/device into a policy target.
- Modify `src\winml\modelkit\export\config.py`: add `WinMLExportConfig.compatibility`, serialize/deserialize it, and validate it.
- Modify `src\winml\modelkit\export\__init__.py`: re-export policy types for package-level imports.
- Modify `src\winml\modelkit\config\build.py`: apply export compatibility policy during config generation, preserve it in export merges, inherit it for submodule configs, and expose a helper for loaded config files.
- Modify `src\winml\modelkit\commands\build.py`: distinguish explicit target flags from auto-resolved runtime target and apply export policy to config-file builds.
- Modify `src\winml\modelkit\commands\config.py`: pass explicit policy targets only when `--ep` or `--device` was supplied.
- Modify `src\winml\modelkit\commands\perf.py`: pass explicit policy targets for module build generation when perf resolved a user-requested target.
- Modify `src\winml\modelkit\models\auto.py`: preserve portable default when the API auto-resolves `ep_device`, but use the concrete target when the caller supplied `ep_device`, `ep`, or `device`.
- Modify `src\winml\modelkit\export\htp\exporter.py`: replace unconditional eager attention override with policy-driven application.
- Modify `tests\unit\export\test_htp_exporter_attention_compat.py`: assert eager attention only happens when compatibility requests it.
- Create `tests\unit\export\test_export_policy.py`: pure policy resolver tests.
- Modify `tests\unit\export\test_config_validation.py`: export config serialization tests.
- Modify `tests\unit\config\test_build.py`: build config cache key, policy application, and submodule inheritance tests.

---

### Task 1: Add the Export Policy Resolver

**Files:**
- Create: `src\winml\modelkit\export\policy.py`
- Modify: `src\winml\modelkit\export\__init__.py`
- Test: `tests\unit\export\test_export_policy.py`

**Interfaces:**
- Consumes: `winml.modelkit.utils.constants.normalize_ep_name`, `EP_DEVICE_SPECS` lazily from `winml.modelkit.session.ep_device`.
- Produces:
  - `ExportCompatibilityConfig(transformers_attention: Literal["eager"] | None = None)`
  - `ExportPolicyTarget(ep: str, device: str)`
  - `ExportCompatibilityRule(ep: str, device: str | None, compatibility: ExportCompatibilityConfig, reason: str)`
  - `resolve_export_compatibility(targets: Sequence[object] | None = None, *, rules: Sequence[ExportCompatibilityRule] = EXPORT_COMPATIBILITY_RULES) -> ExportCompatibilityConfig`
  - `export_policy_targets_for_request(*, ep: str | None, device: str | None, target_was_explicit: bool) -> tuple[ExportPolicyTarget, ...] | None`

- [ ] **Step 1: Write failing policy tests**

Add `tests\unit\export\test_export_policy.py`:

```python
from __future__ import annotations

import pytest

from winml.modelkit.export.policy import (
    ExportCompatibilityConfig,
    ExportCompatibilityRule,
    ExportPolicyTarget,
    export_policy_targets_for_request,
    resolve_export_compatibility,
)


def test_qnn_gpu_target_requires_eager_transformers_attention() -> None:
    cfg = resolve_export_compatibility([ExportPolicyTarget(ep="qnn", device="gpu")])

    assert cfg.transformers_attention == "eager"


def test_non_qnn_target_does_not_force_transformers_attention() -> None:
    cfg = resolve_export_compatibility([ExportPolicyTarget(ep="DmlExecutionProvider", device="gpu")])

    assert cfg.transformers_attention is None
    assert cfg.to_dict() == {}


def test_no_targets_uses_supported_catalog_and_includes_qnn_requirement() -> None:
    cfg = resolve_export_compatibility()

    assert cfg.transformers_attention == "eager"


def test_export_policy_targets_for_request_keeps_portable_default_when_not_explicit() -> None:
    targets = export_policy_targets_for_request(
        ep="QNNExecutionProvider",
        device="gpu",
        target_was_explicit=False,
    )

    assert targets is None


def test_export_policy_targets_for_request_resolves_explicit_alias() -> None:
    targets = export_policy_targets_for_request(
        ep="qnn",
        device="gpu",
        target_was_explicit=True,
    )

    assert targets == (ExportPolicyTarget(ep="QNNExecutionProvider", device="gpu"),)


def test_conflicting_rules_raise_clear_error() -> None:
    rules = (
        ExportCompatibilityRule(
            ep="QNNExecutionProvider",
            device="gpu",
            compatibility=ExportCompatibilityConfig(transformers_attention="eager"),
            reason="first rule",
        ),
        ExportCompatibilityRule(
            ep="QNNExecutionProvider",
            device="gpu",
            compatibility=ExportCompatibilityConfig(transformers_attention="sdpa"),  # type: ignore[arg-type]
            reason="second rule",
        ),
    )

    with pytest.raises(ValueError, match="Conflicting export compatibility"):
        resolve_export_compatibility(
            [ExportPolicyTarget(ep="qnn", device="gpu")],
            rules=rules,
        )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
uv run pytest tests\unit\export\test_export_policy.py -q
```

Expected: FAIL because `winml.modelkit.export.policy` does not exist.

- [ ] **Step 3: Implement `src\winml\modelkit\export\policy.py`**

Create the file with this structure:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..utils.constants import normalize_ep_name

TransformersAttentionPolicy = Literal["eager"]


@dataclass(frozen=True)
class ExportCompatibilityConfig:
    """Resolved export-time compatibility knobs."""

    transformers_attention: TransformersAttentionPolicy | None = None

    def __bool__(self) -> bool:
        return self.transformers_attention is not None

    def to_dict(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.transformers_attention is not None:
            result["transformers_attention"] = self.transformers_attention
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExportCompatibilityConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(
                f"export.compatibility must be an object, got {type(data).__name__}"
            )
        unknown = set(data) - {"transformers_attention"}
        if unknown:
            raise ValueError(f"Unknown export.compatibility field(s): {sorted(unknown)}")
        attention = data.get("transformers_attention")
        if attention is not None and attention != "eager":
            raise ValueError(
                "export.compatibility.transformers_attention must be 'eager' or null, "
                f"got {attention!r}"
            )
        return cls(transformers_attention=attention)


@dataclass(frozen=True)
class ExportPolicyTarget:
    """One EP/device target used by export compatibility policy."""

    ep: str
    device: str

    def __post_init__(self) -> None:
        normalized_ep = normalize_ep_name(self.ep)
        object.__setattr__(self, "ep", normalized_ep if normalized_ep is not None else self.ep)
        object.__setattr__(self, "device", self.device.lower())


@dataclass(frozen=True)
class ExportCompatibilityRule:
    """One EP/device compatibility rule."""

    ep: str
    device: str | None
    compatibility: ExportCompatibilityConfig
    reason: str

    def matches(self, target: ExportPolicyTarget) -> bool:
        return target.ep == normalize_ep_name(self.ep) and (
            self.device is None or target.device == self.device
        )


EXPORT_COMPATIBILITY_RULES: tuple[ExportCompatibilityRule, ...] = (
    ExportCompatibilityRule(
        ep="QNNExecutionProvider",
        device=None,
        compatibility=ExportCompatibilityConfig(transformers_attention="eager"),
        reason="QNN does not reliably support SDPA-exported attention guard paths.",
    ),
)


def export_policy_targets_for_request(
    *,
    ep: str | None,
    device: str | None,
    target_was_explicit: bool,
) -> tuple[ExportPolicyTarget, ...] | None:
    """Return explicit policy targets, or None for the portable catalog default."""
    if not target_was_explicit:
        return None

    from ..session import EPDeviceTarget, resolve_device

    resolved = resolve_device(
        EPDeviceTarget(ep=ep or "auto", device=(device or "auto").lower())
    )
    return (ExportPolicyTarget(ep=resolved.ep, device=resolved.device),)


def resolve_export_compatibility(
    targets: Sequence[object] | None = None,
    *,
    rules: Sequence[ExportCompatibilityRule] = EXPORT_COMPATIBILITY_RULES,
) -> ExportCompatibilityConfig:
    """Resolve export compatibility for explicit targets or the portable catalog."""
    resolved_targets = _catalog_targets() if targets is None else tuple(_coerce_target(t) for t in targets)

    transformers_attention: str | None = None
    transformers_attention_source: str | None = None

    for target in resolved_targets:
        for rule in rules:
            if not rule.matches(target):
                continue
            incoming = rule.compatibility.transformers_attention
            if incoming is None:
                continue
            if transformers_attention is None:
                transformers_attention = incoming
                transformers_attention_source = f"{rule.ep}/{rule.device or '*'}"
            elif transformers_attention != incoming:
                raise ValueError(
                    "Conflicting export compatibility for transformers_attention: "
                    f"{transformers_attention!r} from {transformers_attention_source} vs "
                    f"{incoming!r} from {rule.ep}/{rule.device or '*'}"
                )

    return ExportCompatibilityConfig(
        transformers_attention=transformers_attention,  # type: ignore[arg-type]
    )


def _catalog_targets() -> tuple[ExportPolicyTarget, ...]:
    from ..session.ep_device import EP_DEVICE_SPECS

    return tuple(ExportPolicyTarget(ep=spec.ep, device=spec.device) for spec in EP_DEVICE_SPECS)


def _coerce_target(target: object) -> ExportPolicyTarget:
    if isinstance(target, ExportPolicyTarget):
        return target
    ep = getattr(target, "ep", None)
    device = getattr(target, "device", None)
    if not isinstance(ep, str) or not isinstance(device, str):
        raise TypeError(
            "export policy target must expose string 'ep' and 'device' attributes, "
            f"got {type(target).__name__}"
        )
    return ExportPolicyTarget(ep=ep, device=device)
```

- [ ] **Step 4: Re-export policy types from `export.__init__`**

Modify `src\winml\modelkit\export\__init__.py`:

```python
from .config import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    resolve_export_config,
)
from .policy import (
    ExportCompatibilityConfig,
    ExportCompatibilityRule,
    ExportPolicyTarget,
    export_policy_targets_for_request,
    resolve_export_compatibility,
)
```

Add these names to `__all__`:

```python
    "ExportCompatibilityConfig",
    "ExportCompatibilityRule",
    "ExportPolicyTarget",
    "export_policy_targets_for_request",
    "resolve_export_compatibility",
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
uv run pytest tests\unit\export\test_export_policy.py -q
uv run ruff check --fix src\winml\modelkit\export\policy.py tests\unit\export\test_export_policy.py src\winml\modelkit\export\__init__.py
uv run pytest tests\unit\export\test_export_policy.py -q
```

Expected: all tests PASS.

Commit:

```powershell
git add src\winml\modelkit\export\policy.py src\winml\modelkit\export\__init__.py tests\unit\export\test_export_policy.py
git commit -m "feat(export): add export compatibility policy resolver" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Persist Compatibility in Export Config and Cache Keys

**Files:**
- Modify: `src\winml\modelkit\export\config.py`
- Modify: `src\winml\modelkit\config\build.py`
- Test: `tests\unit\export\test_config_validation.py`
- Test: `tests\unit\config\test_build.py`

**Interfaces:**
- Consumes: `ExportCompatibilityConfig` from Task 1.
- Produces: `WinMLExportConfig.compatibility: ExportCompatibilityConfig`.

- [ ] **Step 1: Write failing export config serialization tests**

Append to `tests\unit\export\test_config_validation.py`:

```python
from winml.modelkit.export.policy import ExportCompatibilityConfig


class TestExportCompatibilitySerialization:
    def test_compatibility_round_trips_when_present(self) -> None:
        cfg = WinMLExportConfig(
            compatibility=ExportCompatibilityConfig(transformers_attention="eager")
        )

        data = cfg.to_dict()
        round_tripped = WinMLExportConfig.from_dict(data)

        assert data["compatibility"] == {"transformers_attention": "eager"}
        assert round_tripped.compatibility.transformers_attention == "eager"

    def test_empty_compatibility_is_omitted_from_export_dict(self) -> None:
        cfg = WinMLExportConfig()

        assert "compatibility" not in cfg.to_dict()

    def test_invalid_compatibility_value_raises(self) -> None:
        with pytest.raises(ValueError, match="transformers_attention"):
            WinMLExportConfig.from_dict(
                {"compatibility": {"transformers_attention": "sdpa"}}
            )
```

Add `pytest` import if the file section does not already have it in scope.

- [ ] **Step 2: Write failing cache key and merge tests**

Append to `tests\unit\config\test_build.py`:

```python
from winml.modelkit.export.policy import ExportCompatibilityConfig


class TestExportCompatibilityBuildConfig:
    def test_export_compatibility_changes_cache_key(self) -> None:
        default_config = WinMLBuildConfig(export=WinMLExportConfig())
        eager_config = WinMLBuildConfig(
            export=WinMLExportConfig(
                compatibility=ExportCompatibilityConfig(transformers_attention="eager")
            )
        )

        assert default_config.generate_cache_key() != eager_config.generate_cache_key()

    def test_registered_export_merge_preserves_override_compatibility(self) -> None:
        from winml.modelkit.config.build import _merge_export_config

        base = WinMLExportConfig()
        override = WinMLExportConfig(
            compatibility=ExportCompatibilityConfig(transformers_attention="eager")
        )

        merged = _merge_export_config(base, override)

        assert merged.compatibility.transformers_attention == "eager"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```powershell
uv run pytest tests\unit\export\test_config_validation.py::TestExportCompatibilitySerialization tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig -q
```

Expected: FAIL because `WinMLExportConfig.compatibility` does not exist.

- [ ] **Step 4: Add compatibility to `WinMLExportConfig`**

Modify imports in `src\winml\modelkit\export\config.py`:

```python
from dataclasses import InitVar, dataclass, field
from .policy import ExportCompatibilityConfig
```

Add the field after `dynamo`:

```python
    compatibility: ExportCompatibilityConfig = field(default_factory=ExportCompatibilityConfig)
```

Add to `__post_init__` before validation that reads `self.compatibility`:

```python
        if isinstance(self.compatibility, dict):
            self.compatibility = ExportCompatibilityConfig.from_dict(self.compatibility)
```

Add to `to_dict()` after dynamic axes handling:

```python
        compatibility = self.compatibility.to_dict()
        if compatibility:
            result["compatibility"] = compatibility
```

Add to `from_dict()` constructor:

```python
            compatibility=ExportCompatibilityConfig.from_dict(data.get("compatibility")),
```

- [ ] **Step 5: Preserve compatibility in registered export merges**

Modify `_merge_export_config()` in `src\winml\modelkit\config\build.py` so the returned `WinMLExportConfig(...)` includes:

```python
        compatibility=(
            copy.deepcopy(override.compatibility)
            if override.compatibility
            else copy.deepcopy(base.compatibility)
        ),
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
uv run pytest tests\unit\export\test_config_validation.py::TestExportCompatibilitySerialization tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig -q
uv run ruff check --fix src\winml\modelkit\export\config.py src\winml\modelkit\config\build.py tests\unit\export\test_config_validation.py tests\unit\config\test_build.py
uv run pytest tests\unit\export\test_config_validation.py::TestExportCompatibilitySerialization tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig -q
```

Expected: all targeted tests PASS.

Commit:

```powershell
git add src\winml\modelkit\export\config.py src\winml\modelkit\config\build.py tests\unit\export\test_config_validation.py tests\unit\config\test_build.py
git commit -m "feat(export): persist export compatibility config" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Apply Export Policy During Build Config Generation

**Files:**
- Modify: `src\winml\modelkit\config\build.py`
- Test: `tests\unit\config\test_build.py`

**Interfaces:**
- Consumes:
  - `ExportCompatibilityConfig`
  - `ExportPolicyTarget`
  - `resolve_export_compatibility(targets)`
- Produces:
  - `apply_export_compatibility_policy(config: WinMLBuildConfig, export_policy_targets: Sequence[object] | None) -> None`
  - `generate_hf_build_config(..., export_policy_targets: Sequence[object] | None = None, ...)`
  - `generate_build_config(..., export_policy_targets: Sequence[object] | None = None, ...)`

- [ ] **Step 1: Write failing config generation tests**

Append to `tests\unit\config\test_build.py`:

```python
from winml.modelkit.export.policy import ExportPolicyTarget


class TestGeneratedExportCompatibilityPolicy:
    def test_generated_hf_config_uses_portable_policy_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_loader_config: WinMLLoaderConfig,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.config.build.resolve_loader_config",
            lambda *args, **kwargs: (
                mock_loader_config,
                mock_hf_config,
                mock_model_class,
                MagicMock(),
            ),
        )
        monkeypatch.setattr(
            "winml.modelkit.config.build._resolve_export_config_from_specs",
            lambda *args, **kwargs: mock_export_config,
        )

        cfg = generate_hf_build_config(
            "local-model",
            device="gpu",
            ep="DmlExecutionProvider",
            policy_overrides_config=True,
            export_policy_targets=None,
        )

        assert cfg.export is not None
        assert cfg.export.compatibility.transformers_attention == "eager"

    def test_generated_hf_config_respects_explicit_non_qnn_export_policy_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_loader_config: WinMLLoaderConfig,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.config.build.resolve_loader_config",
            lambda *args, **kwargs: (
                mock_loader_config,
                mock_hf_config,
                mock_model_class,
                MagicMock(),
            ),
        )
        monkeypatch.setattr(
            "winml.modelkit.config.build._resolve_export_config_from_specs",
            lambda *args, **kwargs: mock_export_config,
        )

        cfg = generate_hf_build_config(
            "local-model",
            device="gpu",
            ep="DmlExecutionProvider",
            policy_overrides_config=True,
            export_policy_targets=(
                ExportPolicyTarget(ep="DmlExecutionProvider", device="gpu"),
            ),
        )

        assert cfg.export is not None
        assert cfg.export.compatibility.transformers_attention is None

    def test_submodule_config_inherits_export_compatibility(self) -> None:
        parent = WinMLBuildConfig(
            loader=WinMLLoaderConfig(model_type="bert", task="fill-mask"),
            export=WinMLExportConfig(
                compatibility=ExportCompatibilityConfig(transformers_attention="eager")
            ),
        )
        sub_info = SubmoduleInfo(
            class_name="Linear",
            module_path="encoder.layer.0.output.dense",
            input_shapes=[[1, 4]],
            output_shapes=[[1, 4]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
            input_names=["hidden_states"],
        )

        sub_cfg = _build_submodule_config(sub_info, parent)

        assert sub_cfg.export is not None
        assert sub_cfg.export.compatibility.transformers_attention == "eager"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
uv run pytest tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy -q
```

Expected: FAIL because the config generation functions do not accept `export_policy_targets`.

- [ ] **Step 3: Add policy application helper to `config\build.py`**

Add imports under `TYPE_CHECKING`:

```python
    from collections.abc import Sequence
```

Add runtime imports near other export config imports:

```python
from ..export.policy import resolve_export_compatibility
```

Add helper near the device/precision policy helpers:

```python
def apply_export_compatibility_policy(
    config: WinMLBuildConfig,
    export_policy_targets: Sequence[object] | None,
) -> None:
    """Populate export compatibility when the config has an export stage."""
    if config.export is None:
        return
    if config.export.compatibility:
        return
    config.export.compatibility = resolve_export_compatibility(export_policy_targets)
```

- [ ] **Step 4: Extend config generation signatures**

Add `export_policy_targets: Sequence[object] | None = None` to:

- both `generate_hf_build_config` overloads
- `generate_hf_build_config` implementation
- both `generate_build_config` overloads
- `generate_build_config` implementation

Forward it from `generate_build_config(...)` to `generate_hf_build_config(...)`:

```python
        export_policy_targets=export_policy_targets,
```

After override merging, target policy application, and `no_compile` handling in `generate_hf_build_config`, call:

```python
    apply_export_compatibility_policy(parent_config, export_policy_targets)
```

Place the call before the `if module:` branch so submodule configs inherit it.

- [ ] **Step 5: Inherit compatibility in submodule configs**

In `_build_submodule_config(...)`, add this argument to the `WinMLExportConfig(...)` constructor:

```python
            compatibility=copy.deepcopy(parent_config.export.compatibility)
            if parent_config.export is not None
            else WinMLExportConfig().compatibility,
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
uv run pytest tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig -q
uv run ruff check --fix src\winml\modelkit\config\build.py tests\unit\config\test_build.py
uv run pytest tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig -q
```

Expected: all targeted tests PASS.

Commit:

```powershell
git add src\winml\modelkit\config\build.py tests\unit\config\test_build.py
git commit -m "feat(config): resolve export compatibility policy" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Propagate Explicit vs Portable Targets From Call Sites

**Files:**
- Modify: `src\winml\modelkit\commands\build.py`
- Modify: `src\winml\modelkit\commands\config.py`
- Modify: `src\winml\modelkit\commands\perf.py`
- Modify: `src\winml\modelkit\models\auto.py`
- Test: `tests\unit\config\test_build.py`

**Interfaces:**
- Consumes:
  - `export_policy_targets_for_request(ep, device, target_was_explicit)`
  - `apply_export_compatibility_policy(config, export_policy_targets)`
- Produces consistent target semantics across CLI and API:
  - explicit `--ep` or `--device` uses a single resolved target
  - no explicit EP/device uses portable full-catalog policy

- [ ] **Step 1: Write failing tests for loaded config policy defaults**

Append to `tests\unit\config\test_build.py`:

```python
class TestLoadedConfigExportCompatibilityPolicy:
    def test_apply_export_policy_populates_loaded_config_without_compatibility(self) -> None:
        from winml.modelkit.config.build import apply_export_compatibility_policy

        cfg = WinMLBuildConfig(export=WinMLExportConfig())

        apply_export_compatibility_policy(cfg, None)

        assert cfg.export is not None
        assert cfg.export.compatibility.transformers_attention == "eager"

    def test_apply_export_policy_preserves_serialized_compatibility(self) -> None:
        from winml.modelkit.config.build import apply_export_compatibility_policy

        cfg = WinMLBuildConfig(
            export=WinMLExportConfig(
                compatibility=ExportCompatibilityConfig(transformers_attention="eager")
            )
        )

        apply_export_compatibility_policy(
            cfg,
            (ExportPolicyTarget(ep="DmlExecutionProvider", device="gpu"),),
        )

        assert cfg.export is not None
        assert cfg.export.compatibility.transformers_attention == "eager"
```

- [ ] **Step 2: Run tests and verify current gaps**

Run:

```powershell
uv run pytest tests\unit\config\test_build.py::TestLoadedConfigExportCompatibilityPolicy -q
```

Expected: PASS only after Task 3 helper exists; if it fails, finish Task 3 first.

- [ ] **Step 3: Update `commands\build.py`**

Import helpers inside `build(...)` after `ep_value` is computed:

```python
    from ..export.policy import export_policy_targets_for_request
```

Before the current `if ep_value is None:` block, compute:

```python
    export_target_was_explicit = cli_utils.is_cli_provided(ctx, "ep") or cli_utils.is_cli_provided(ctx, "device")
```

After the existing auto-resolution block has concrete `ep_value` and `device`, compute:

```python
    export_policy_targets = export_policy_targets_for_request(
        ep=cast("str | None", ep_value),
        device=device,
        target_was_explicit=export_target_was_explicit,
    )
```

Pass `export_policy_targets=export_policy_targets` to the HF `generate_build_config(...)` call. Do not pass it to the ONNX file path because no export happens there.

After loading config-file configs and applying `export_overrides`, apply the default policy:

```python
                from ..config.build import apply_export_compatibility_policy

                config_list = (
                    config_or_configs
                    if isinstance(config_or_configs, list)
                    else [config_or_configs]
                )
                for cfg in config_list:
                    apply_export_compatibility_policy(cfg, export_policy_targets)
```

Place this after the `export_overrides` merge so an explicitly serialized `export.compatibility` is preserved.

- [ ] **Step 4: Update `commands\config.py`**

At each `generate_hf_build_config(...)` call, compute:

```python
            from ..export.policy import export_policy_targets_for_request

            export_policy_targets = export_policy_targets_for_request(
                ep=ep_name,
                device=device,
                target_was_explicit=(
                    cli_utils.is_cli_provided(ctx, "ep")
                    or cli_utils.is_cli_provided(ctx, "device")
                ),
            )
```

Pass:

```python
                export_policy_targets=export_policy_targets,
```

Do this for both normal config generation and helper generation paths in the file.

- [ ] **Step 5: Update `commands\perf.py`**

In `_run_module_perf(...)`, after `resolved_target = resolve_device(...)` and after `resolved_device = resolved_target.device` / `ep = cast("EPName", resolved_target.ep)`, compute:

```python
        from ..export.policy import export_policy_targets_for_request

        export_policy_targets = export_policy_targets_for_request(
            ep=ep,
            device=resolved_device,
            target_was_explicit=ep is not None or device is not None,
        )
```

Pass:

```python
            export_policy_targets=export_policy_targets,
```

- [ ] **Step 6: Update `models\auto.py`**

Before auto-resolving `ep_device`, record whether the API caller supplied a target:

```python
        export_target_was_explicit = ep_device is not None or ep is not None or device is not None
```

After `ep_device` is concrete and before `generate_hf_build_config(...)`, compute:

```python
        from ..export.policy import export_policy_targets_for_request

        export_policy_targets = export_policy_targets_for_request(
            ep=_resolved_ep_short_name(ep_device),
            device=ep_device.device.device_type.lower(),
            target_was_explicit=export_target_was_explicit,
        )
```

Pass:

```python
            export_policy_targets=export_policy_targets,
```

- [ ] **Step 7: Run call-site tests and commit**

Run:

```powershell
uv run pytest tests\unit\config\test_build.py::TestLoadedConfigExportCompatibilityPolicy tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy -q
uv run ruff check --fix src\winml\modelkit\commands\build.py src\winml\modelkit\commands\config.py src\winml\modelkit\commands\perf.py src\winml\modelkit\models\auto.py tests\unit\config\test_build.py
uv run pytest tests\unit\config\test_build.py::TestLoadedConfigExportCompatibilityPolicy tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy -q
```

Expected: all targeted tests PASS.

Commit:

```powershell
git add src\winml\modelkit\commands\build.py src\winml\modelkit\commands\config.py src\winml\modelkit\commands\perf.py src\winml\modelkit\models\auto.py tests\unit\config\test_build.py
git commit -m "feat(export): propagate export policy targets" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Make HTP Exporter Apply Compatibility Policy

**Files:**
- Modify: `src\winml\modelkit\export\htp\exporter.py`
- Modify: `tests\unit\export\test_htp_exporter_attention_compat.py`

**Interfaces:**
- Consumes: `export_config.compatibility.transformers_attention`.
- Produces: policy-driven application of `use_eager_attention_for_export(model)`.

- [ ] **Step 1: Update attention exporter tests to require policy**

Replace `tests\unit\export\test_htp_exporter_attention_compat.py` with tests that cover both policy states:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from winml.modelkit.export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.export.htp import HTPExporter
from winml.modelkit.export.policy import ExportCompatibilityConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _AttentionConfig:
    model_type = "fake"

    def __init__(self, implementation: str = "sdpa") -> None:
        self._attn_implementation = implementation


class _NestedAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _AttentionConfig()
        self.proj = nn.Linear(2, 2)
        self.proj.config = _AttentionConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _export_config(*, eager_attention: bool) -> WinMLExportConfig:
    return WinMLExportConfig(
        input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 2))],
        output_tensors=[OutputTensorSpec(name="y")],
        compatibility=ExportCompatibilityConfig(
            transformers_attention="eager" if eager_attention else None
        ),
    )


def test_htp_exporter_uses_eager_attention_when_policy_requests_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _NestedAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["root"] = model.config._attn_implementation
        captured["child"] = model.proj.config._attn_implementation

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    HTPExporter()._convert_model_to_onnx(
        model,
        str(tmp_path / "model.onnx"),
        {"x": torch.ones(1, 2)},
        _export_config(eager_attention=True),
        task=None,
    )

    assert captured == {"root": "eager", "child": "eager"}
    assert model.config._attn_implementation == "sdpa"
    assert model.proj.config._attn_implementation == "sdpa"


def test_htp_exporter_leaves_attention_unchanged_without_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _NestedAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["root"] = model.config._attn_implementation
        captured["child"] = model.proj.config._attn_implementation

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    HTPExporter()._convert_model_to_onnx(
        model,
        str(tmp_path / "model.onnx"),
        {"x": torch.ones(1, 2)},
        _export_config(eager_attention=False),
        task=None,
    )

    assert captured == {"root": "sdpa", "child": "sdpa"}
    assert model.config._attn_implementation == "sdpa"
    assert model.proj.config._attn_implementation == "sdpa"
```

- [ ] **Step 2: Run tests and verify the negative test fails**

Run:

```powershell
uv run pytest tests\unit\export\test_htp_exporter_attention_compat.py -q
```

Expected: FAIL because exporter still applies eager unconditionally.

- [ ] **Step 3: Make `HTPExporter` policy-driven**

Modify `src\winml\modelkit\export\htp\exporter.py`.

Add a small helper method near `_convert_model_to_onnx`:

```python
    @staticmethod
    def _export_compatibility_context(model: nn.Module, export_config: WinMLExportConfig):
        if export_config.compatibility.transformers_attention == "eager":
            return use_eager_attention_for_export(model)
        return contextlib.nullcontext()
```

Update the export context:

```python
        with (
            self._get_optimum_patcher(model, task),
            self._export_compatibility_context(model, export_config),
        ):
```

Do not mention QNN in this file.

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
uv run pytest tests\unit\export\test_htp_exporter_attention_compat.py -q
uv run ruff check --fix src\winml\modelkit\export\htp\exporter.py tests\unit\export\test_htp_exporter_attention_compat.py
uv run pytest tests\unit\export\test_htp_exporter_attention_compat.py -q
```

Expected: all tests PASS.

Commit:

```powershell
git add src\winml\modelkit\export\htp\exporter.py tests\unit\export\test_htp_exporter_attention_compat.py
git commit -m "feat(export): apply attention compatibility by policy" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Validate the Integrated Behavior

**Files:**
- Modify only if preceding tasks reveal a missed wiring issue:
  - `src\winml\modelkit\config\build.py`
  - `src\winml\modelkit\commands\build.py`
  - `src\winml\modelkit\export\htp\exporter.py`
- Test: targeted unit tests and local QNN E2E commands.

**Interfaces:**
- Consumes all interfaces from Tasks 1-5.
- Produces verified generic export policy behavior.

- [ ] **Step 1: Run the complete targeted unit test set**

Run:

```powershell
uv run pytest tests\unit\export\test_export_policy.py tests\unit\export\test_config_validation.py::TestExportCompatibilitySerialization tests\unit\config\test_build.py::TestExportCompatibilityBuildConfig tests\unit\config\test_build.py::TestGeneratedExportCompatibilityPolicy tests\unit\config\test_build.py::TestLoadedConfigExportCompatibilityPolicy tests\unit\export\test_htp_exporter_attention_compat.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run ruff on all touched Python files**

Run:

```powershell
uv run ruff check --fix src\winml\modelkit\export\policy.py src\winml\modelkit\export\config.py src\winml\modelkit\export\__init__.py src\winml\modelkit\config\build.py src\winml\modelkit\commands\build.py src\winml\modelkit\commands\config.py src\winml\modelkit\commands\perf.py src\winml\modelkit\models\auto.py src\winml\modelkit\export\htp\exporter.py tests\unit\export\test_export_policy.py tests\unit\export\test_config_validation.py tests\unit\config\test_build.py tests\unit\export\test_htp_exporter_attention_compat.py
```

Expected: command exits 0.

- [ ] **Step 3: Verify generated config records portable eager policy**

Run:

```powershell
uv run winml config -m openai/clip-vit-base-patch32 --task feature-extraction --output temp\export-policy-config.json --overwrite
Select-String -Path temp\export-policy-config.json -Pattern '"compatibility"|"transformers_attention"'
```

Expected: output contains `compatibility` and `transformers_attention` with value `eager`.

- [ ] **Step 4: Verify explicit non-QNN target does not force eager**

Run:

```powershell
uv run winml config -m openai/clip-vit-base-patch32 --task feature-extraction --ep dml --device gpu --output temp\export-policy-dml-config.json --overwrite
Select-String -Path temp\export-policy-dml-config.json -Pattern '"compatibility"|"transformers_attention"'
```

Expected: no match for `transformers_attention`, unless a future DML policy rule was intentionally added.

- [ ] **Step 5: Run the representative local QNN E2E checks**

Run the smallest local QNN checks that previously reproduced the issue:

```powershell
uv run pytest tests\e2e\test_perf_e2e.py --run-perf-e2e --model-id openai/clip-vit-base-patch32 --task feature-extraction --ep-device qnn_gpu -q
uv run pytest tests\e2e\test_perf_e2e.py --run-perf-e2e --model-id openai/clip-vit-base-patch32 --task zero-shot-image-classification --ep-device qnn_gpu -q
uv run pytest tests\e2e\test_perf_e2e.py --run-perf-e2e --model-id google-bert/bert-base-multilingual-cased --task masked-lm --ep-device qnn_gpu -q
uv run pytest tests\e2e\test_perf_e2e.py --run-perf-e2e --model-id google/flan-t5-base --task text2text-generation --ep-device qnn_gpu -q
```

Expected: each command PASS or reports an existing provider limitation that is unrelated to SDPA attention export.

- [ ] **Step 6: Commit any validation fixes**

If Step 1-5 required code changes, commit them:

```powershell
git add src\winml\modelkit tests\unit
git commit -m "fix(export): complete export policy integration" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

If Step 1-5 required no code changes, do not create an empty commit.
