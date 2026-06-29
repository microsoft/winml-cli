# src/winml/modelkit/commands/sys.py

## TL;DR
`_gather_ep_info()` is rewritten as the **broad-enumeration (Path B / Tier 2)
inventory pass** for the `--list-ep` renderer. The pre-state's flat `list[dict]`
keyed on `(ep_name, device)` from the `ep_device_map` merge is gone. The
post-state walks `WinMLEPRegistry.all_discovered()` (now including built-ins
via `BuiltinSource`), calls `register_ep` per entry, queries
`EP_CATALOG.is_compatible()` for the L2 vendor verdict, and emits a nested
`dict[ep_name, {entries: [...]}]` shape with per-source rows carrying the
**L1/L2 status taxonomy** from `2_coreloop.md` §7.1.1 (`failed` /
`incompatible` / `primary` / `shadowed`). The text renderer (`_output_ep_text`,
`_format_devices_from_handles`, `_describe_source`) and a new `_gather`
dispatcher + `_RENDERERS` table collapse the previous 100-line per-(format,
section) `if`-ladder in `sysinfo()` into a single table-driven flow. The
`--list-device` block also gains a `WinMLDevice.device_facts()` enrichment pass
that merges Architecture into the per-device rendering when an EP successfully
registered against that device.

## Diff metrics
- 618 lines changed (489 insertions / 129 deletions per `--stat`).
- Largest single-file diff in the commands batch (alongside `perf.py`'s 620).
- New top-level helpers: `_describe_source`, `_format_devices_from_handles`,
  `_gather`, `_render_text`, `_render_json`, `_render_compact`.
- New module-level data: `_INDENT_L2 / _INDENT_L3 / _INDENT_L4`,
  `_SOURCE_KIND_LABEL`, `_RENDERERS`.
- Symbols deleted: `get_ep_device_map` import (the catalog hardcode it
  fronted is gone — the catalog is now the only source of EP↔device truth).

## Role before vs after
Role of the command unchanged: `winml sys [--list-device] [--list-ep]`
surfaces system info / device list / EP inventory in text, JSON, or compact
form. What changed is the EP inventory contract:

- **Before**: at most one row per EP name, with `device` chosen by
  `ep_device_map.get(name, "unknown").upper()`. Two-pass merge: WinML
  registry entries (with `path`) first, then ORT-only providers (with
  `path=None`). The status field did not exist — rows were either
  "present" or absent.
- **After**: nested `{ep_name: {entries: [...]}}` shape; each entry is a
  per-`EPEntry` row carrying source kind (PyPI / MSIX / NuGet /
  WinMLCatalog / Directory / Builtin), version, dll_path, status
  (`failed` / `incompatible` / `primary` / `shadowed`), per-device facts
  list, and (failures only) an `error` field. Same EP appearing in
  multiple sources (e.g., OpenVINO via PyPI and MSIX) gets multiple rows
  under the same key; the §7.1.2 precedence-winner is tagged
  `primary`, later rows `shadowed`.

The default text/json/compact split is preserved; what changed is which
sections are appended. Pre-state: text+verbose hard-coded "devices then EPs"
fallthrough. Post-state: `_gather` builds a partial `info` dict gated on
flags, and `_RENDERERS[fmt]` dispatches — text adds blank lines between
sub-sections, compact joins one line per aspect, JSON dumps as-is.

## Symbol-level changes

### Imports
- Removed: `get_ep_device_map` (sole external reference to the legacy
  flat catalog map deleted in this commit).
- Added top-level: `EPEntry`, `PyPISource`, `MSIXPackageSource`,
  `NuGetSource`, `DirectorySource`, `WinMLCatalogSource` from `..ep_path`;
  `WinMLEP`, `WinMLEPRegistry` from `..session`. `TYPE_CHECKING` guard for
  `Callable` and `Path`.
- Late imports preserved: `EP_CATALOG` and `WinMLEPRegistrationFailed`
  inside `_gather_ep_info()` (keeps CLI startup cheap when --list-ep
  isn't used).

### `_INDENT_L2 / _L3 / _L4` constants
Single source of truth for the four indent levels in `--list-ep` text
output. Comment explicitly calls out the prior drift: `_INDENT_L3 = " " *
14` is computed as `_INDENT_L2 + len("[status] ") + width 9 + sep`,
making it obvious which renderer column they align under.

### `_describe_source(entry: EPEntry) -> dict[str, Any]`
Pure projector — `EPEntry` → `{source_kind, version?, distribution? |
nuget_id? | family_name_prefix? | catalog_name? | root?}`. **Reads
`entry.version` as the single source-of-truth** for version metadata
(populated per-EPSource at `.resolve()` time per the dispatch table in
`_describe_source`'s doc comment). The version-recovery logic that used to
live in `commands/sys.py` is gone.

### `_gather_ep_info() -> dict[str, dict[str, Any]]`
Total rewrite. Status derivation **per 2_coreloop.md §7.1.1 in strict
precedence**:
1. `register_ep` raised → `status="failed"`, carries
   `error=f"{type(err).__name__}: {err}"`. The catalog L2 check was never
   evaluated (`compatible=None`).
2. `EP_CATALOG.is_compatible(entry.ep_name)` returned `False` →
   `status="incompatible"`.
3. First successful, vendor-compatible row per ep_name →
   `status="primary"`; subsequent ok rows → `status="shadowed"`.

Three concerns:
- Built-ins (CPU/Dml/Azure) are now indistinguishable from plugin EPs at
  this layer — they flow through `all_discovered()` via `BuiltinSource`.
  The pre-state's "two loops" (WinML registry → ORT fallback) collapses
  to one.
- Failure capture: both `WinMLEPRegistrationFailed` (registration-layer)
  and the broad `Exception` (catalog WMI failure) are recorded as
  `compatible=None` rows so the build loop's `err is not None`
  short-circuit handles them before the compatibility check fires (a
  defensive `False` would falsely tag the row as L2-incompatible).
- `catalog_default_paths` is a set of `entry.dll_path` values whose
  origin is `WinMLCatalogSource`. Rows in OTHER sources (e.g., a PyPI
  install) that happen to resolve to the same DLL the Catalog row points
  at also get the `(catalog default)` tag. This ensures the user sees
  "this row is what `Catalog` would pick" even when the user's package
  came via a different source-kind.

### `_SOURCE_KIND_LABEL`
Maps `PyPISource → "PyPI"`, `MSIXPackageSource → "MSIX"`, etc., with
`BuiltinSource → "bundled"` — the L2 column label users see in text
output. Matches the `_entry_source_tag` dispatcher recognised
in `session.ep_device` (commit body: "completing the `--ep cpu@bundled`
round-trip").

### `_format_devices_from_handles(devices: list[dict[str, Any]]) -> list[str]`
Renders the device-level lines (L3 + L4) under each entry. Comment notes
the `vendor` field is kept on the dict for JSON consumers but is no
longer printed: ORT reports "Intel" vs "Intel Corporation" for the same
vendor_id 0x8086 (a known inconsistency).

### `_output_ep_text(eps: dict[str, dict[str, Any]]) -> None`
Walks the nested dict. EP-level `compat_tag` is computed from per-row
statuses: `[bold red]\[incompatible][/bold red]` shown at the EP header
**only when no row is `primary` or `shadowed`** — collapses L1-failed
and L2-incompatible into one header tag, matching §7.1.1 spec. Status
colors per §7.1.2:

| status         | color  | meaning                                                |
|----------------|--------|--------------------------------------------------------|
| `primary`      | green  | this EP's precedence-winner                            |
| `shadowed`     | yellow | registered cleanly; not Scenario A's pick              |
| `failed`       | red    | `register_ep` raised; carries `error` field            |
| `incompatible` | red    | vendor rule overrides a successful register            |

MSIX `family_name_prefix` is shortened by `rpartition("_")` to drop the
trailing publisherId (e.g. `8wekyb3d8bbwe`) — compact display only;
the full string remains in JSON.

### `_gather_device_info()` — new enrichment loop
After hardware queries (NPU > GPU > CPU priority) populate
`result[].details`, a new block walks `registry._registered.values()`
and merges `WinMLDevice.device_facts()` (only Architecture per the doc
comment — Driver/Manufacturer are sourced from sysinfo). Fuzzy
hardware-name matching (substring in either direction) handles the
case where ORT/OpenVINO appends `(iGPU)` to FULL_DEVICE_NAME that the
WMI query doesn't include. `details.setdefault(label.lower(), value)`
guarantees the sysinfo-sourced fact wins on key collision.

**TODO surfaced inline**: `reaches into ``registry._registered``,
which is registry-internal. Consider exposing a small public accessor
like ``registered_eps()``` — same smell as the existing `_discovered`
reach.

### `_gather` / `_render_*` / `_RENDERERS`
Collapses the previous `sysinfo()` callback's 100-line `if use_json: …
elif compact: … else: …` block into a table-driven dispatch.

Three knobs to `_gather`: `system`, `devices`, `eps` (which sections to
include) plus `tolerant` (per-section error handling — explicit-pin
mode raises `ClickException`, default mode logs and emits an empty
container). Crucially, EPs run **before** devices because
`_gather_device_info`'s enrichment loop reads `registry._registered`,
which only populates as `_gather_ep_info` calls `register_ep`.

### `sysinfo()` Click callback
Same Click options as before. The body shrinks from ~100 lines to ~15:

```python
if list_device or list_ep:
    info = _gather(devices=list_device, eps=list_ep, ..., tolerant=False)
else:
    include_sections = fmt != "compact"
    info = _gather(system=True, devices=include_sections, eps=include_sections,
                   ..., tolerant=True)
_RENDERERS[fmt](info, verbose)
```

Pre-state had three orthogonal axes per format × per section nested
in `if use_json: … elif compact: …` ladder; each section had its
own try/except. Post-state: one collection pass + one render dispatch.

## Behavior / contract changes

### (a) Row schema: flat → nested with status taxonomy
Before:
```python
[{"name": "QNNExecutionProvider", "device": "NPU", "path": "..."},
 {"name": "CPUExecutionProvider", "device": "CPU", "path": None}, ...]
```

After:
```python
{
  "QNNExecutionProvider": {
    "entries": [
      {"source_kind": "PyPISource", "distribution": "onnxruntime-qnn",
       "version": "1.20.0", "status": "primary", "dll_path": "...",
       "devices": [{"device_type": "NPU", "hardware_name": "...",
                    "vendor": "...", "facts": [...]}]},
      {"source_kind": "MSIXPackageSource", "family_name_prefix": "...",
       "version": "1.18.0", "status": "shadowed", "dll_path": "...",
       "devices": [...]},
    ]
  },
  ...
}
```

This breaks `winml sys --list-ep --format json` consumers downstream.
The commit body documents this as part of "v2.7 splits the §7.1
`--list-ep` status taxonomy into two independent layers"; integration
tests in `tests/integration/ep_path/test_live_msix.py` (which the
git status shows as modified in the same commit) need to assert
the new shape.

### (b) Status L1 vs L2 — single source of truth
The L1 (`failed`) / L2 (`incompatible`) split is computed only inside
`_gather_ep_info`. The text renderer and the JSON consumer both read
the resulting `entry["status"]` string. There's no duplicate
"compatible" field on the EP record (the code comment makes this
explicit: *"EP-level 'compatible' is derived at render time from
entry[status]"*). This is the right single-source-of-truth shape — but
the renderer's `compat_tag` recomputation could itself be folded onto
the record as `record["any_usable"]` to skip the per-row scan; it's
deferred as a micro-optimization.

### (c) `BuiltinSource dll_path=None`
Comment explicitly addresses the brief's check: BuiltinSource entries
carry a sentinel `Path()` (i.e. `Path(".")`); the descriptor emits
`dll_path=None` via `entry.is_filesystem_backed()` test rather than
`str(entry.dll_path)`. This matches the design's
"`BuiltinSource dll_path=None`" requirement — verified.

### (d) `device_types` is per-source, not static
The text renderer's L3 device line emits `device_type` + `hardware_name`
from the **actual `WinMLEP.devices` tuple** that ORT reported for THIS
source's registration. There is no fallback to a static EP→device
catalog claim. Matches the design check in the brief: "no
device_types static claim". The `EP_DEVICE_SPECS` import that
characterized the previous refactor is **gone** from this file —
appropriate because the renderer's truth is the live `WinMLEP.devices`,
not the catalog.

### (e) Devices populated only for primary/shadowed
`if winml_ep is not None and compatible: desc["devices"] = [...]`. The
inline comment is explicit: an L2-incompatible row may have a CPU
fallback in `winml_ep.devices`, but showing that under the EP's name
"would mislead readers into thinking the EP is running on real
hardware". Failed rows have `winml_ep=None` so the guard handles them
implicitly. This is a small but important UX fix vs. naively dumping
all device tuples regardless of compatibility.

### (f) Default mode is now `tolerant=True`
Per-section failures (e.g. WMI unavailable on a headless server) no
longer blow up the full `winml sys` report — they log a WARNING and
fill an empty `{}` or `[]`. The pre-state had inline `try/except`
swallows per section that printed a yellow notice and dropped the
section. Explicit-pin mode (`--list-ep` / `--list-device`) still raises
`ClickException` so the user knows their pin produced nothing.

## Cross-file impact
- **Hard dependency on the unified `ep_path` public surface**:
  `EPEntry`, `PyPISource`, `MSIXPackageSource`, `NuGetSource`,
  `DirectorySource`, `WinMLCatalogSource` from `..ep_path`. The
  commit's `__init__.py` re-exports must include all six.
- **Hard dependency on `WinMLEPRegistry.all_discovered()`**: this is the
  new public iterator over `_discovered`. The commit body confirms
  `_discovered` is renamed from `_entries` and now includes the
  synthesized built-ins.
- **Reaches into `registry._registered` directly** in
  `_gather_device_info` — same smell as pre-state's
  `WinMLEPRegistry.get_available_eps()` reach into the registry's
  caches. Flagged inline as a TODO.
- `WinMLEPRegistry.instance()` (not `get_instance()`) is the singleton
  accessor. Commit body: "WinMLEPRegistry.instance() is the sole
  singleton entry point" — verified.
- `EP_CATALOG.is_compatible(ep_name)` is the L2 vendor verdict. Catches
  `RuntimeError` from headless servers per commit body's "graceful
  CPU fallback on headless servers" note for `auto_detect_device`.

## Risks / subtleties
- **JSON shape break**: any external consumer of
  `winml sys --list-ep --format json` is broken. There's no
  back-compat field. Per CLAUDE.md `feedback_no_back_compat` this is
  intended — but downstream README examples / CI scrapers need updating.
  Not visible in the diff itself.
- **`primary_seen` is per-`ep_name`**: a `failed` or `incompatible`
  row does NOT consume a "primary slot". So an EP with rows
  `[failed, ok, ok]` correctly marks the second row as `primary` and
  the third as `shadowed`. Verified by re-reading the loop. Inline
  comment makes this explicit.
- **`registry._registered` is registry-internal**: `_gather_device_info`
  uses it directly. If `WinMLEPRegistry` later switches to a lazy
  registration model where `_registered` is empty until consumers ask,
  the enrichment loop silently degrades to "no Architecture facts in
  the devices section." Worth gating on a public accessor (the TODO is
  explicit).
- **`registry.all_discovered()` is called twice on a `--list-device
  --list-ep` invocation**: once by `_gather_ep_info` (which populates
  `_registered` as a side effect) and once by `_gather_device_info`
  (reads `_registered`). The `_gather` dispatcher's ordering (EPs
  first) is load-bearing. If somebody ever flips the order in a
  refactor, the Architecture row in `--list-device` silently
  disappears.
- **Format-level coupling between `compact` text and `_gather`'s
  `include_sections`**: compact mode skips device/EP sections by
  design (compact is sysinfo-overview). If we later want
  `--format compact --list-ep`, the existing branch already handles
  it (since `list_ep=True` forces `_gather(..., devices=False,
  eps=True, ...)`). But `_render_compact` then prints only EP names
  (no devices) — verified consistent with the renderer code.
- **`_describe_source` returns `dict[str, Any]`**: the type signature
  doesn't constrain which subset of optional keys appear. Each
  consumer's `entry.get(...)` defaulting handles it, but a TypedDict
  would catch shape drift earlier.
- **The `error` field is rendered raw**: `f"{type(err).__name__}:
  {err}"` which can include file paths or exception args. Per the
  design that's appropriate for a power-user diagnostic; on machines
  with bare-DLL failures the renderer will print "DLL load failed
  while importing onnxruntime_providers_qnn: The specified module
  could not be found." — useful but long.

## Simplification opportunities
- **`compat_tag` recomputation in `_output_ep_text`**: the renderer
  walks all entries to derive `any_usable`. `_gather_ep_info` could
  stash that boolean on the record (`record["any_usable"] = True`)
  to drop the inner loop. Cost: one boolean per EP. Win: renderer no
  longer recomputes the same fact.
- **Pre-state's `Path("")` sentinel for `BuiltinSource`**: the
  `entry.is_filesystem_backed()` predicate is a clean boolean test —
  but the sentinel still lives. A `BuiltinSource.dll_path` typed as
  `Path | None` would make this self-documenting; the
  `is_filesystem_backed()` indirection then collapses to `is not None`.
- **`_RENDERERS` table**: three entries; a `match` statement on `fmt`
  would inline cleanly given the small set. The table is preferable
  if there's a chance of dynamic registration (no current need).
- **`_format_devices_from_handles` argument type**: takes
  `list[dict[str, Any]]` but only reads `device_type`, `hardware_name`,
  `facts`. Either a TypedDict at the boundary or pass the original
  `WinMLDevice` objects through. Today the loop dehydrates
  `WinMLDevice` → dict in `_gather_ep_info` and the renderer reads
  the dict; the round-trip via JSON-shape dicts is justified for the
  `--format json` path but is overhead for text rendering.
- **The whole `_gather_device_info` enrichment block** is bolted on
  after the hardware queries. Could be folded into the per-hardware
  loop where we already iterate `result`, with an early-exit on the
  first matching `WinMLEP`. The current shape walks `result` twice;
  the diff has it as a separate `try/except` block which is
  defensible for diagnostic isolation but is a code smell.

## Open questions / TODOs surfaced
- **TODO inline at `_gather_device_info`**: *"reaches into
  `registry._registered`, which is registry-internal. Consider exposing
  a small public accessor like `registered_eps()`."* Same smell exists
  for `_discovered` reach. A single `WinMLEPRegistry.registered_eps()`
  / `discovered_entries()` pair would clean both up.
- **JSON format consumers**: no `--json-version` field on the new
  payload. If we break the schema again the diff will be silent.
  Worth a `"_schema": "list-ep@v2"` row for canary purposes.
- **Built-in EPs and `device_facts`**: the comment says built-ins flow
  through `_gather_ep_info`'s main loop, but the renderer code for
  CPU shows just "(bundled)" — there's no test in the visible diff
  that confirms CPU's `dll_path` correctly suppresses the L3 Path
  row. The `entry.is_filesystem_backed()` test should return False
  for `BuiltinSource`; if `BuiltinSource` ever gains a real DLL path
  (e.g., for `CPUExecutionProvider`'s actual DLL location), the
  suppression breaks silently.
- **Vendor name drift**: the comment on `_format_devices_from_handles`
  documents the "Intel" vs "Intel Corporation" inconsistency.
  Suppressing the field in text but keeping in JSON is a reasonable
  short-term call; long-term, an EP-side normalization layer would
  fix the underlying drift.
- **Test coverage**: the `git status` shows
  `tests/unit/commands/test_cli.py`, `tests/unit/session/test_ep_registry.py`
  modified. Worth verifying the new nested schema is asserted there;
  flat-list assertions on the old payload would silently break.
