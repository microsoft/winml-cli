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
