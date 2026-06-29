# Batch 00 — Verification Report

Scope: 13 per-file docs under `commit-analysis/774f121d/per-file/`.
Method: every file:line claim cross-checked against `git show 774f121d:<path>` (post-squash) and `git diff 7a66c024..774f121d` (mergebase → tip).

---

## `__init__.md`
**Source file:** `src/winml/modelkit/__init__.py`
**Total claims checked:** 11
**Result:** 8 verified, 2 overstated, 0 false, 1 unverified

### Verified claims
- "adds a second side-effecting subpackage import (`from . import _transformers_compat`)" — line 39: `from . import _transformers_compat`.
- "explanatory comment + `# noqa: I001` to lock the import order" — lines 36-38 hold the 2-line preamble; line 38 ends `from . import _warnings  # noqa: I001`.
- "Removed: bare `from . import _warnings  # Configure warning filters before importing subpackages` (the inline comment)" — pre-commit confirms exact deleted line.
- "+4 / -1 (net +3)" — verified via diff: 4 insertions, 1 deletion.
- "logging.getLogger / __version__ / __all__ / __getattr__ / __dir__ untouched" — verified by reading both versions.
- "Order matters and is now documented" — confirmed in lines 36-37 comment.
- "No public API addition or removal at this module level" — `__all__` unchanged.
- "Lazy loader (`_LAZY_IMPORTS`, `__getattr__`) preserves the existing pep562 lazy mechanism" — unchanged.

### Overstated / corrected
- "Import time grows by however long transformers takes to import (typically 1–3 s on cold cache). For very lean callers (e.g. `from winml.modelkit import __version__`) this is now mandatory overhead that wasn't there before." — **partly wrong**: `__version__` is set at module top by `importlib.metadata`, not lazy. But the file already had a `__getattr__` lazy loader for heavy exports that explicitly avoided "torch/transformers/optimum (~30s)". Now `_transformers_compat` makes transformers an eager import at *every* `import winml.modelkit`. The doc's claim about overhead is true; what's overstated is the framing — the lazy loader exists for a reason and this commit partially regresses it. Doc does not note the existing lazy mechanism.
- "Tests that mock `transformers` at module level need to mock *before* `import winml.modelkit`" — true in principle, but unverifiable; no existing test depends on this and the doc didn't check.

### Unverified
- "Author labels the entire module a temporary band-aid: ..." quote — verified in `_transformers_compat.py` lines 21-22, but no commit message confirms author intent.

---

## `_transformers_compat.md`
**Source file:** `src/winml/modelkit/_transformers_compat.py`
**Total claims checked:** 24
**Result:** 19 verified, 4 overstated, 0 false, 1 unverified

### Verified claims
- "+304 / -0 (entire file new)" — verified by `git diff` (file did not exist pre-commit).
- "`CLIPFeatureExtractor` (lines 62-74)" — actual: class spans lines 60-74.
- "`MT5Tokenizer` (lines 91-114)" — actual: spans lines 91-115.
- "`_top_objects.setdefault("AutoModelForVision2Seq", AutoModelForImageTextToText)` (line 132)" — verified at line 132.
- "`is_offline_mode()` (lines 146-148)" — verified, function body lines 145-148 within `if not hasattr(...)` block.
- "`get_parameter_dtype(parameter)` (lines 158-169)" — verified.
- "`_CAN_RECORD_REGISTRY = {}` (line 180)" — verified at line 180.
- "`OutputRecorder` (lines 190-207)" — verified.
- "`_sdpa_mask_without_vmap_tf5(...)` (lines 249-302)" — verified, spans 249-302.
- "wrapped in `try/except ImportError` so the patch is a no-op when optimum is not installed" — verified at lines 232-235.
- "Uses `_top_objects.setdefault(...)` not assignment" — verified at lines 119-131.
- "plain setattr for submodules (`transformers.utils`, etc.)" — verified at lines 150, 173, 184, 211.
- "Captures `_top_objects = sys.modules['transformers']._objects` after pre-loading from transformers" — verified at lines 45-46.
- "MT5Tokenizer blocks `__new__`, `__init__`, and `from_pretrained` all raising `RuntimeError`" — verified at lines 103-114.
- "CLIPFeatureExtractor is a subclass of `CLIPImageProcessor` emitting `UserWarning` on construction" — verified at lines 60-74.
- "AutoModelForVision2Seq aliased to AutoModelForImageTextToText" — verified at line 132.
- "Optimum patch replaces `optimum.exporters.onnx.model_patcher.sdpa_mask_without_vmap`" — verified at line 302.
- "_torch fallback (final `return torch.float32`)" — verified at lines 165-169.
- "`MT5Tokenizer` error message duplicated as both `cls._ERROR` and `self._ERROR` references" — verified at lines 105, 109, 114.

### Overstated / corrected
- "Symbols defined: 2 module-level classes (`CLIPFeatureExtractor`, `MT5Tokenizer`), 3 inner classes/funcs conditionally defined" — actual: `CLIPFeatureExtractor`, `MT5Tokenizer`, `OutputRecorder` are classes; `is_offline_mode`, `get_parameter_dtype`, `_sdpa_mask_without_vmap_tf5` are functions. Three module-top symbols not "two" if `OutputRecorder` is counted as module-top (it is, conditionally).
- "Body identical to optimum's 0.1.0 implementation modulo the q_indices derivation and `prepare_padding_mask` call (drops the dropped-in-5.x `_slice=False` kwarg)" — claim about `_slice=False` being a 4.x-only kwarg cannot be verified without optimum-onnx 0.1.0 source; treat as author assertion.
- "the recorder branch never fires for any model" — speculative; depends on optimum's `_traceable_decorator.py` behavior not verified.
- "Pre-loading `AutoModelForImageTextToText` and `CLIPImageProcessor` from transformers triggers transformers' full lazy-loader chain" — verified that those imports exist (lines 39-42), but "full lazy-loader chain" overstates the side-effect surface without runtime measurement.

### Unverified
- "Optimum-onnx 0.1.0 (last PyPI release as of 2026-04-30)" — date in docstring at line 8, not independently checked.

---

## `analyze__analyzer.md`
**Source file:** `src/winml/modelkit/analyze/analyzer.py`
**Total claims checked:** 14
**Result:** 9 verified, 4 overstated, 1 false, 0 unverified

### Verified claims
- "7 insertions / 6 deletions (net +1)" — diff confirms 7+/6-.
- "all in one hunk near line 667" — verified, hunk starts at line 667.
- "Touches one method only: `ONNXStaticAnalyzer.analyze_from_proto` (the `ep_normalized is None` branch)" — verified.
- "Adds one lazy import: `from ..session import eps_for_device` (function-scoped)" — verified at line 671.
- "branch `if ep_normalized is None` no longer instantiates a literal list" — verified.
- "imports `eps_for_device` lazily and assigns `eps_to_analyze = sorted(eps_for_device('npu'))`" — verified at line 673.
- "logger.info message string changed from 'all supported EPs' to 'all NPU-capable EPs'" — verified at line 674.
- "Default EP set is now `sorted(eps_for_device('npu'))` — currently the same three EPs (QNN, OpenVINO, VitisAI) but in alphabetic order" — verified; pre-commit list was QNN/OpenVINO/VitisAI in that order.
- "`eps_for_device` is exported in `session/__init__.py`'s `__all__`" — verified.

### Overstated / corrected
- "around line 716–717" (VitisAI carve-out location) — actual line: `if current_ep == "VitisAIExecutionProvider":` is at **line 742**, not 716-717. Correction: the carve-out is ~25 lines further down.
- "the special-case `if current_ep == 'VitisAIExecutionProvider': run_unknown_op_for_ep = False` carve-out further down (around line 716–717) is untouched — VitisAI is still string-matched here" — kernel of truth, wrong line numbers.
- "Default device unchanged (`'NPU'`)" — partial: doc later says "Default-device behaviour is still a string literal `'NPU'` on line 687" — line is wrong; actual `device_to_use = device if device is not None else "NPU"` is at **line 688**. Off by one.
- "This hunk is byte-identical to the same hunk in commit `a509a67` — the v2.9 squash carries it forward unchanged" — checked: a509a67 has the same `eps_for_device("npu")` swap, but the surrounding context now differs (774f121d added an "auto" device-resolution branch between the EP-selection and device-resolution lines that a509a67 lacks). The *hunk content* for the EP-selection block is identical; the *file at that line* differs. Overstated.

### False
- "The `if ep_normalized is None / else` split could collapse to `eps_to_analyze = [ep_normalized] if ep_normalized else sorted(eps_for_device('npu'))`" — this is a simplification suggestion, not a claim about the file; technically valid. Marking as not false, removing.
- "Default-device behaviour is still a string literal `'NPU'` on line 687" — **line 687 is `if device is not None and device.lower() == "auto":`** (verified at line 681 in actual file; line 688 holds the `device_to_use = device if device is not None else "NPU"` fallback). **CONTRADICTS** the cited line.

---

## `analyze__core__doc_constraint_checker.md`
**Source file:** `src/winml/modelkit/analyze/core/doc_constraint_checker.py`
**Total claims checked:** 6
**Result:** 6 verified, 0 overstated, 0 false, 0 unverified

### Verified claims
- "1 insertion / 1 deletion (net 0), one token" — diff verified.
- "Touches one `except Exception as e:` line inside the per-op-constraint loader loop near line 127" — actual: line 127, exactly matches.
- "`# noqa: PERF203` removed, no other change" — verified.
- "`logger.error(f'Failed to load constraints for {op_type}: {e}')` preserved" — verified at line 128.
- "behavioural change: none" — verified, single comment removal.
- "coupled with similar `# noqa: PERF203` deletion in `model_validator_manager.py`" — verified.

---

## `analyze__core__model_validators__model_validator_manager.md`
**Source file:** `src/winml/modelkit/analyze/core/model_validators/model_validator_manager.py`
**Total claims checked:** 5
**Result:** 5 verified, 0 overstated, 0 false, 0 unverified

### Verified claims
- "1 insertion / 1 deletion (net 0)" — verified.
- "Touches the `except Exception as e:` line near line 145 of the validator loop" — actual line 145, matches exactly.
- "noqa: PERF203 removed" — verified.
- "logger.exception with f-string format" — verified at line 146.
- "pairs with the identical noqa removal in `analyze/core/doc_constraint_checker.py`" — verified.

---

## `analyze__core__runtime_checker_query.md`
**Source file:** `src/winml/modelkit/analyze/core/runtime_checker_query.py`
**Total claims checked:** 22
**Result:** 17 verified, 4 overstated, 1 false, 0 unverified

### Verified claims
- "28 insertions / 18 deletions (net +10)" — verified via diff.
- "Single hunk in `_is_ep_available_locally` plus the module-top `import onnxruntime as ort` deletion (line 18)" — verified at original line 18.
- "Module-top: `import onnxruntime as ort` (line 18, pre-commit) — deleted" — verified.
- "No other `ort.` reference survives in the file" — verified by grep.
- "Lazy import block changed from `from ... import winml` + `from ...utils.constants import DEVICE_TO_DEVICE_TYPE` to a 7-symbol tuple import from `...session`" — verified.
- "`winml.register_execution_providers(ort=True)` call → removed" — verified.
- "`device_type_enum = DEVICE_TO_DEVICE_TYPE.get(self.device_type)` early-out → removed" — verified.
- "`ort.get_ep_devices()` + `any(...)` membership test → removed" — verified.
- "New construction: `EPDeviceTarget(ep=short_ep_name(self.ep_name), device=self.device_type.lower())` followed by `resolve_device(target)` and `WinMLEPRegistry.instance().auto_device(resolved)`" — verified.
- "`except Exception as e:` → narrowed to `except (WinMLEPNotDiscovered, WinMLEPRegistrationFailed, DeviceNotFound, ValueError) as e:`" — verified.
- "Log message changed to 'EP %s on %s not available locally: %s'" — verified.
- "`DEVICE_TO_DEVICE_TYPE` still appears in `_get_ep_checker` at line 1337" — verified at line 1337.
- "Caller-observable contract on return value unchanged" — verified (still returns bool, still memoised).
- "The `_is_ep_available_locally()` callsite at line 1512" — verified at line 1512.
- "7 symbols imported from `..session`: `DeviceNotFound`, `EPDeviceTarget`, `WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, `WinMLEPRegistry`, `resolve_device`, `short_ep_name`" — verified.
- "all seven exported in `session/__init__.py`'s `__all__`" — verified.
- "A new docstring paragraph added to `_is_ep_available_locally` explaining the new probe strategy" — verified at lines 1287-1292.

### Overstated / corrected
- "Single hunk in `_is_ep_available_locally` (lines 1285–1328)" — actual: `_is_ep_available_locally` definition starts at line 1285, but the diff hunk spans lines 1285-1328 in the new file; verified bounds roughly match (the diff context ends at line 1328 in the new file). OK as approximation.
- "exception list omits `WinMLEPMonitorMismatch`. The session package raises that during monitor binding. `_is_ep_available_locally` doesn't go through the monitor path" — assertion about `auto_device` not exercising monitor — would need verification of `auto_device` body.
- "session/__init__.py `__all__`. (verified: lines 11, 13, 17, 18, 26, 27, 44, 61, 62, 66, 73, 74)" — the line numbers are approximations; the symbols are exported but the specific line citations weren't traceable to those exact lines. Verified existence in `__all__`; specific line numbers are imprecise.
- "Other call sites in the package (`analyze/runtime_checker/ep_checker.py`, `analyze/runtime_checker/check_ops.py`, `analyze/pattern/check_patterns.py`) still call `winml.register_execution_providers(ort=True)` or `winml.add_ep_for_device(...)`" — **partially wrong**: `check_ops.py` and `check_patterns.py` had their module-level `winml.register_execution_providers(ort=True)` calls **removed in this same commit** (verified in diffs). `ep_checker.py` no longer calls this either; it builds the registration via `WinMLEPRegistry.instance().auto_device(resolved)`. So the claim that "the `winml` module dependency survives at the package level" because of these three call sites is wrong for two of them.

### False
- "`EPChecker` itself (constructed by `_get_ep_checker`) still does its own `winml.register_execution_providers(ort=True)` call at module-import time (`runtime_checker/check_ops.py:41`)" — **CONTRADICTS**. In commit 774f121d the `winml.register_execution_providers(ort=True)` call at the module top of `check_ops.py` (originally at lines 38-41) was **removed**. The "probe and the actual check now use two different registration paths" framing is wrong — both go through the registry now (via `EPChecker._get_sess_options()`).

---

## `analyze__models__ihv_type.md`
**Source file:** `src/winml/modelkit/analyze/models/ihv_type.py`
**Total claims checked:** 8
**Result:** 4 verified, 1 overstated, 3 false, 0 unverified

### Verified claims
- "2 insertions / 2 deletions (net 0)" — verified.
- "import + class declaration" — verified.
- "`class IHVType(str, Enum):` → `class IHVType(StrEnum):`" — verified.
- "`from enum import Enum` → `from enum import StrEnum`" — verified.

### Overstated / corrected
- "`StrEnum` is `enum.StrEnum`, formally `str` + `Enum` with `__str__` returning the value (the pre-commit form returned `'IHVType.QC'` for `str(IHVType.QC)`, while `StrEnum` returns `'QC'`)" — the `__str__` behavior claim is accurate, but the inline characterization "formally `str` + `Enum`" is slightly imprecise. Marking as overstated for precision.

### False
- "Other unmigrated string-enums in the codebase (per Grep: `runtime_checks.py`, `support_level.py`, `pattern/models.py`, `onnx/domains.py`, `telemetry/deviceid/deviceid.py`) were not touched in this commit, so the codebase is now mixed-idiom." — **CONTRADICTS**: `runtime_checks.py` and `support_level.py` **WERE** migrated in this same commit (verified via `git diff`). `pattern/models.py` (actual path: `src/winml/modelkit/pattern/models.py`, not `analyze/pattern/models.py`), `onnx/domains.py`, and `telemetry/deviceid/deviceid.py` all **already use `StrEnum`** at commit 774f121d (verified by reading files). The "mixed-idiom" claim is entirely false — the codebase is uniformly `StrEnum`.
- "Why only 3 of the 8 string-enum classes in the codebase?" — **CONTRADICTS** the same false premise.
- "The 5 untouched sites are listed above" — **CONTRADICTS**; none of those 5 sites use `(str, Enum)`; all use `StrEnum`.

---

## `analyze__models__information.md`
**Source file:** `src/winml/modelkit/analyze/models/information.py`
**Total claims checked:** 7
**Result:** 5 verified, 0 overstated, 2 false, 0 unverified

### Verified claims
- "2 insertions / 2 deletions (net 0)" — verified.
- "`ActionLevel` class base: `(str, Enum)` → `(StrEnum)`" — verified.
- "Module import: `Enum` → `StrEnum`" — verified.
- "`REQUIRED`, `RECOMMENDED` etc. all unchanged" — actual members are `REQUIRED`, `OPTIONAL`, `WARNING`. `REQUIRED` matches; `RECOMMENDED` is not a member. Doc says "etc." so technically not falsified, but member name `RECOMMENDED` is wrong if interpreted as a citation.
- "`Information` Pydantic model unchanged" — verified.

### False
- "`from .runtime_checks import PatternRuntime` and `from .support_level import SupportLevel` still import the legacy `(str, Enum)` versions — codebase is now mixed-idiom" — **CONTRADICTS**: both `runtime_checks.py` and `support_level.py` use `StrEnum` at commit 774f121d.
- "`PatternRuntime` and `SupportLevel` (imported in this file) still use the legacy form — follow-up sweep candidate" — **CONTRADICTS**; both already use `StrEnum` (migrated in this same commit).

---

## `analyze__models__onnx_model.md`
**Source file:** `src/winml/modelkit/analyze/models/onnx_model.py`
**Total claims checked:** 7
**Result:** 5 verified, 1 overstated, 1 false, 0 unverified

### Verified claims
- "2 insertions / 2 deletions (net 0)" — verified.
- "`ModelTag` class base: `(str, Enum)` → `(StrEnum)`" — verified.
- "Module import: `Enum` → `StrEnum`" — verified.
- "`from ...onnx import ONNXDomain` is preserved" — verified.
- "`ONNXModel` Pydantic model unchanged" — verified.

### Overstated / corrected
- "Members like `INFERRED_SHAPES`, `EXTERNAL_DATA`, etc. are unchanged" — **wrong member names**: actual `ModelTag` members are `INVALID_PATTERN_MATCHER_MODEL` and `MISSING_NODE_NAMES`. Neither `INFERRED_SHAPES` nor `EXTERNAL_DATA` exists. Correction: members are unchanged from pre-commit but the cited member names are invented.

### False
- "The sibling `ONNXDomain` enum imported here (`onnx/domains.py`) was *not* migrated — inconsistency worth flagging" — **CONTRADICTS**: `src/winml/modelkit/onnx/domains.py` uses `StrEnum` at commit 774f121d (verified: `class ONNXDomain(StrEnum):` at line 142).

---

## `analyze__models__runtime_checks.md`
**Source file:** `src/winml/modelkit/analyze/models/runtime_checks.py`
**Total claims checked:** 8
**Result:** 8 verified, 0 overstated, 0 false, 0 unverified

### Verified claims
- "+3 / -3" — verified.
- "Imports: `from enum import Enum` → `from enum import StrEnum`" — verified.
- "Class declarations updated: `NodeTag`, `AlternativeType`" — verified.
- "`NodeTag` members unchanged: `ALL_INPUTS_CONSTANT = 'all_inputs_constant'`, `MISSING_SHAPE_INFERENCE = 'missing_shape_inference'`" — verified at lines 22-23.
- "`AlternativeType` member shown in the diff: `EQUIVALENT = 'equivalent'`" — verified at line 30.
- "`support_level.py` was migrated in this same commit" — verified.
- "Migration changes `str(member)` to return the value" — accurate `StrEnum` behavior.
- "PEP 663 / Python 3.11+" — correct (project floor verified at 3.11).

---

## `analyze__models__support_level.md`
**Source file:** `src/winml/modelkit/analyze/models/support_level.py`
**Total claims checked:** 5
**Result:** 5 verified, 0 overstated, 0 false, 0 unverified

### Verified claims
- "+2 / -2" — verified.
- "`from enum import Enum` → `from enum import StrEnum`" — verified.
- "`class SupportLevel(str, Enum)` → `class SupportLevel(StrEnum)`" — verified.
- "Members unchanged (`SUPPORTED = 'supported'`, ... — diff shows the first; remainder presumed identical)" — verified: `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, `UNKNOWN`.
- "`runtime_checks.py` got the same `StrEnum` migration in this commit, confirming a sweep" — verified.

---

## `analyze__pattern__check_patterns.md`
**Source file:** `src/winml/modelkit/analyze/pattern/check_patterns.py`
**Total claims checked:** 11
**Result:** 11 verified, 0 overstated, 0 false, 0 unverified

### Verified claims
- "+4 / -4" — verified.
- "Removed import: `from ... import winml`" — verified at original line 23.
- "Removed module-level call: `winml.register_execution_providers(ort=True)`" — verified.
- "Added: 4 comment lines inside the `--ep` argparse definition" — verified at lines 273-276.
- "`build_parser` parser definition touched" — verified.
- "`from ..runtime_checker.ep_checker import EPChecker` import is preserved" — verified at line 32.
- "argparse `choices=` list unchanged: `['QNNExecutionProvider', 'OpenVINOExecutionProvider']`" — verified at line 277.
- "Companion `check_ops.py` got the same module-level surgery" — verified.
- "carve-out comment names `eps_for_device('npu')` and `EP_DEVICE_SPECS`" — verified in comment text.
- "Both symbols (`eps_for_device`, `EP_DEVICE_SPECS`) exist in `session/__init__.py` `__all__`" — verified.
- "Sibling tool (`check_ops.py`) still has a 5-EP allowlist (QNN, OpenVINO, VitisAI, MIGraphX, NvTensorRtRtx)" — verified at `check_ops.py` lines 324-330.

---

## `analyze__runtime_checker__check_ops.md`
**Source file:** `src/winml/modelkit/analyze/runtime_checker/check_ops.py`
**Total claims checked:** 14
**Result:** 13 verified, 1 overstated, 0 false, 0 unverified

### Verified claims
- "Removed import: `from ... import winml`" — verified at original line 24.
- "Removed module-level call + comment block (4 lines): `# Register WinML EPs at module level...`" — verified, lines 38-41 of pre-commit removed.
- "All four occurrences of `NvTensorRTRTXExecutionProvider` (all-caps RTRTX) renamed to `NvTensorRtRtxExecutionProvider`" — verified via diff (4 changes).
- "`RTXChecker.__init__`: error message rename and `super().__init__(ep_name=...)` rename" — verified at lines 258, 261.
- "`get_ep_checker` mapping key rename" — verified at line 283.
- "`build_parser` argparse `choices=` list + `help=` text renamed" — verified at lines 329, 337.
- "`get_ep_checker` not-found path raises `ValueError`" — verified at lines 287-291; error message includes "Available: {...}" listing the dict keys.
- "EP catalog uses `NvTensorRtRtxExecutionProvider` as the canonical form" — verified in `session/ep_device.py` line 291 and short-name map line 90 (`"nvtensorrtrtx": "NvTensorRtRtxExecutionProvider"`).
- "`EPChecker._get_sess_options()` calls `short_ep_name(self.ep_name)`" — verified at `ep_checker.py` line 55.
- "Misplaced docstring (`'''Initialize RTX checker.'''` line appears after the `raise ValueError`, so unreachable string statement)" — verified at lines 257-260 of `check_ops.py`.
- "argparse parser `choices=` list contains 5 EPs" — verified.
- "registration is lazy via `EPChecker._get_sess_options()`" — verified at `ep_checker.py`.
- "`get_ep_checker` is called from `build_parser` flow" — verified via codepath.

### Overstated / corrected
- "Lines changed: ~8 / ~8 (16 total)" — actual: 7 insertions / 8 deletions in the diff. Approximate; OK as "~8".

---

## Batch 00 — Overall

- Total claims across all docs: ~142
- Verified: ~115
- Overstated: ~17
- False: ~9
- Unverified: ~2

### Most concerning false claims (top 3 by blast radius)

1. **`analyze__models__ihv_type.md`, `information.md`, `onnx_model.md`** — all three docs claim that other string-enum classes in the codebase (`runtime_checks.PatternRuntime`, `SupportLevel`, `pattern.models.*`, `onnx.domains.ONNXDomain`, `telemetry/deviceid`) were "not migrated" and "the codebase is now mixed-idiom." In reality, `runtime_checks.py` and `support_level.py` **were migrated in the same commit** (their own diffs confirm), and all other cited files were **already using `StrEnum` before this commit**. The premise of the "follow-up sweep candidate" recommendation in all three docs is wrong — the sweep is already complete. This affects any reader using these docs to plan future cleanups.

2. **`analyze__core__runtime_checker_query.md`** — claims that `check_ops.py` still does `winml.register_execution_providers(ort=True)` at module-import time at "line 41" and that "the probe and the actual check now use two different registration paths." This is contradicted by `check_ops.py`'s own diff in the same commit, which **removed** that module-level call (and `check_patterns.py` similarly). The narrative that the v2.9 refactor left half the call sites on the old path is false.

3. **`analyze__analyzer.md`** — VitisAI carve-out is claimed to live "around line 716-717" and the default-device fallback "on line 687." Actual lines are 742 and 688 respectively. Anyone using these as line citations to navigate the file will land in the wrong place (off by ~25 lines for the VitisAI carve-out).

### Surprising verifications worth highlighting

- The `_transformers_compat.py` review's structural claims (line ranges for each shim, idempotent `setdefault` pattern, lazy-module ordering rationale, `_top_objects` capture) are highly accurate — line numbers match to within 1 line for every shim cited.
- The `check_ops.md` "misplaced docstring after `raise ValueError`" observation is correct and is a real bug in the source — `"""Initialize RTX checker."""` is an unreachable string statement at line 260, not the function docstring.
- The `runtime_checker_query.md` mechanical migration claims (deleted import, replaced lookup, narrowed exception tuple, new log message format) are perfectly accurate; only the cross-file ripple-effect claims are wrong.

### Note on doc quality pattern

Docs do well on **what was changed in the file under review** (diff-grounded claims are mostly accurate). They do poorly on **cross-file context** — claims about other files' state ("X still uses Y", "Z hasn't been migrated") are repeatedly wrong because they weren't grounded by checking those other files. The 3 StrEnum docs and the runtime_checker_query "cross-file impact" sections share this failure mode.
