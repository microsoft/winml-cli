# src/winml/modelkit/sysinfo/sysinfo.py

## TL;DR
Removes the `WindowsAppRuntimeVersion` class (24 lines) and the `SysInfo.windows_app_runtime_version` property + dict entry that exposed it. The class scraped the Windows App Runtime version from the pip package whose name ended in `-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap` and applied a chain of string substitutions to normalize it. With the WinAppSDK initialization no longer owned by `winml.py` (moved into `session/`) and no in-tree consumer of the runtime-version string, the class is dead weight and is excised. `import re` is also removed because no other use remains.

## Diff metrics
- Lines: +0 / -38 (net -38)
- Hunks: 4 (import + class def + property + `to_dict` entry + `__init__` line)
- Symbols removed: 1 class (`WindowsAppRuntimeVersion`), 1 instance attribute (`SysInfo._windows_app_runtime_version`), 1 property (`SysInfo.windows_app_runtime_version`), 1 dict key (`"windowsAppRuntimeVersion"` from `to_dict()`).

## Role before vs after
- Before: `sysinfo.SysInfo` collected system info **including** a normalized Windows App Runtime version, scraped from `pip list` output (any package whose name ended with `-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap`). Used a 3-step regex chain: strip `^N!` epoch prefix, convert `.devN` → `-experimentalN`, strip trailing `0` from `-experimental0` to match the upstream tag.
- After: `SysInfo` no longer includes the WinAppSDK runtime version. The pip package is presumably still installed, but its version is no longer scraped or exposed. `to_dict()` no longer carries a `windowsAppRuntimeVersion` key.

## Symbol-level changes
- **Removed** `import re` (line 5 of pre-commit). No other regex use in this file.
- **Removed** class `WindowsAppRuntimeVersion` (24 lines, pre-commit lines 8–32):
  - `_package_name_suffix` class constant
  - `__init__(self, pip_packages)` — scans for the suffix, raises `ValueError` if not found
  - `_version` instance attribute
  - `@property version` — returns the normalized version string
- **Removed** in `SysInfo.__init__`: the line `self._windows_app_runtime_version = WindowsAppRuntimeVersion(self._pip_packages)`.
- **Removed** `@property def windows_app_runtime_version(self) -> WindowsAppRuntimeVersion: return self._windows_app_runtime_version`.
- **Removed** the `"windowsAppRuntimeVersion": self._windows_app_runtime_version.version,` entry from `SysInfo.to_dict()`.

## Behavior / contract changes
- `SysInfo()` no longer raises `ValueError` on a host where no `*-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap` pip package is installed. Before, this was a hard failure on every `SysInfo()` construction on a non-WinAppSDK host (e.g. a Linux CI runner or a fresh Python env). After, `SysInfo()` succeeds in those environments.
- `SysInfo.to_dict()` is missing the `"windowsAppRuntimeVersion"` key. Any external tool (e.g. telemetry exporter, diagnostic JSON consumer) reading that key will get `KeyError` or whatever default the consumer uses.
- The `pip_packages` data is still collected and exposed via `SysInfo.pip_packages` / `to_dict()["pipPackages"]`. Consumers can still derive the WinAppSDK version themselves by filtering the pip list — only the pre-normalized convenience accessor is gone.

## Cross-file impact
- No test in `tests/` references `WindowsAppRuntimeVersion`, `windows_app_runtime_version`, or `windowsAppRuntimeVersion` (verified). So the test suite is silent on the removal — no test break, but also no test that *guards* the removal from being silently reintroduced.
- No source-tree consumer in `src/` references the symbol either.
- Telemetry payloads (`telemetry/library/exporter.py`) likely consume `SysInfo.to_dict()`. The disappearance of `windowsAppRuntimeVersion` is a schema change on the wire if telemetry forwards the dict verbatim — confirm against the telemetry schema (likely `telemetry/library/schema.py` if it exists).

## Risks / subtleties
- **Telemetry schema drift.** If the `OneCollectorLogExporter` or any other downstream consumer of `SysInfo.to_dict()` ships a versioned schema declaring `windowsAppRuntimeVersion` as a required field, this commit silently breaks that schema. The diff doesn't include a telemetry-schema update, so either (a) the schema is flexible (extra/missing keys are OK), or (b) the schema is broken and this commit didn't notice.
- **Loss of CI-host failure signal.** Before, `SysInfo()` raised on hosts without the WinAppSDK package. That was sometimes used as a sanity check that the WinAppSDK toolchain was installed. Now `SysInfo()` succeeds silently on such hosts.
- **The regex chain documented a real upstream encoding quirk** (`.dev0` → `-experimental` strip-trailing-0). That institutional knowledge is now lost from the source tree. Any future re-introduction will need to re-derive it.

## Simplification opportunities
- The remaining `SysInfo` class is still moderately large with one property per inventory category. Could be reduced to a dataclass with `@dataclass` and `to_dict = dataclasses.asdict`. Not done in this commit.
- The `to_dict()` method hand-builds the camelCase keys; a small helper or marshmallow-style schema would compress this. Out of scope for this commit.
- `__init__` still hard-eagerly constructs every inventory (CPU, GPU, NPU, RAM, OS, pip, ep) — each touches WMI and is slow. A lazy-property pattern would defer the expensive collection. Out of scope.

## Open questions / TODOs surfaced
- Confirm no telemetry schema declares `windowsAppRuntimeVersion` as required; if it does, either re-introduce the property or update the schema.
- Should `SysInfo` expose a generic `pip_package_version(name_suffix: str) -> str | None` helper so callers can scrape the WinAppSDK version themselves without the legacy regex chain?
- Was the loss of the regex normalization deliberate, or was the runtime-version-string consumer migrated to read the raw `pip_packages` entry? Need to verify there's no downstream caller that depended on the `^N!` epoch strip.
