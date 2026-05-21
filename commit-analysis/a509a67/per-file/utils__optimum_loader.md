# src/winml/modelkit/utils/optimum_loader.py

## TL;DR

A single, comment-only change: inserted a 5-line "CARVE-OUT" rationale above the hardcoded `provider="CUDAExecutionProvider"` argument inside `OptimumONNXModel.from_onnx`, explicitly forbidding any future refactor from replacing it with `default_ep_for_device("gpu")`. No runtime behaviour changes. This is a guardrail against a likely-but-wrong unification with the EPDeviceSpec catalog that landed in the same commit.

## Diff metrics

`+5 / -0`. One hunk, inside `OptimumONNXModel.from_onnx` immediately before the `provider=` kwarg.

## Role before vs after

Role is identical: thin wrapper that exports an ONNX model from disk into a HuggingFace Optimum `ORTModelFor…` instance using the matching task class. The "do not unify" comment now nails down the contract that this file's EP-selection logic is intentionally separate from the WinML EPDeviceSpec catalog — Optimum runs cross-platform (Linux + cuDNN included) and its `provider` argument semantics differ from the Windows-ML provider taxonomy.

## Symbol-level changes

- `OptimumONNXModel.from_onnx` — added inline comment block, no code change:
  ```python
  model = ort_model_class.from_pretrained(
      temp_path,
      # CARVE-OUT: HF Optimum's ORTModel uses CUDA as the generic non-CPU GPU EP.
      # This is a cross-platform HF Optimum codepath and does NOT use the
      # Windows-ML catalog's default GPU EP (DmlExecutionProvider on Windows).
      # Do NOT replace with default_ep_for_device("gpu") — that breaks the Optimum
      # integration on Linux + cuDNN setups.
      provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
      **kwargs,
  )
  ```

No other symbol was touched. `_detect_task`, `_get_ort_model_class`, the module-level `load_optimum_model` convenience function, and the imports are unchanged.

## Behavior / contract changes

None. The selected `provider` string is byte-identical: `"CPUExecutionProvider"` when `device == "cpu"`, otherwise `"CUDAExecutionProvider"`. The carve-out is purely documentary.

It is worth noting what is *not* preserved through this carve-out:

- The comment hard-codes the assumption that Optimum on this codepath equates "non-CPU device" with CUDA. Users on macOS or AMD Windows hitting this path with `device="cuda"` would still see CUDA selected. The comment is a deliberate "leave it alone for now" — not a guarantee that the behaviour is correct everywhere.
- The `device` argument here is the HF-Optimum-style lowercase string (`"cpu"` or `"cuda"`), not a WinML `EPDevice` descriptor. This file therefore lives outside the (EP, device)-pair refactor.

## Cross-file impact

- No imports added or removed. The file does **not** import from `..session`, intentionally — the comment exists precisely so that the import boundary stays one-way (the rest of utils is migrating onto `session`, but this loader stays decoupled).
- One downstream caller in the package: `load_optimum_model` is exported from this same module and used by HuggingFace-loading codepaths. None of those callers changed in this commit.
- Indirect: any future "unify all EP selection through `default_ep_for_device`" refactor must now read the comment first.

## Risks / subtleties

- **Comment-only fix has no enforcement.** Nothing prevents a future contributor from running a global find-and-replace and missing the carve-out. A small unit test asserting that the string `"CUDAExecutionProvider"` (or the equivalent literal) appears unmodified in this function would catch a regression; none was added.
- **Hardcoded EP string survives the Cardinal Rule "no hardcoded model architecture / operator / EP names" only because the carve-out is explicit.** The comment is the justification — it puts this file in a documented "exception" status that future code reviews need to honour.
- **`device == "cpu"` is the only branch.** Any non-`"cpu"` value (e.g. `"gpu"`, `"npu"`, `"cuda:0"`, `"dml"`) collapses to CUDA. Callers who pass new lowercase device strings from the EPDevice refactor would silently land on CUDA. Not addressed in this commit, but the comment implicitly endorses this status quo.

## Open questions / TODOs surfaced

- If WinML wants to support non-CUDA GPU for Optimum loading (DML on Windows is the obvious case), this function will need a per-OS branch. The comment's stance is "we do not currently want that"; if the policy changes, both the comment and the conditional must be updated together.
- Should there be an architectural test (similar to the qnn `_internal` regression test added elsewhere in this commit) that asserts `optimum_loader.py` does not import from `..session.ep_device`? The carve-out lives only in a comment today.
- The `device` parameter typing is still `str` (no validation against `{"cpu", "cuda"}`). A `Literal["cpu", "cuda"]` annotation would lock the contract that this loader speaks Optimum's device vocabulary, not WinML's.
