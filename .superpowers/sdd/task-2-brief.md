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
