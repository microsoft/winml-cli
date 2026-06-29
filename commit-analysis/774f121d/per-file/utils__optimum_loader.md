# src/winml/modelkit/utils/optimum_loader.py

## TL;DR
Adds a five-line **carve-out comment** above the `provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"` line in `OptimumONNXModel`. No code change. The comment documents that HF Optimum's `ORTModel` uses CUDA as the generic non-CPU GPU EP (cross-platform HF Optimum codepath), so this site deliberately does *not* call `default_ep_for_device("gpu")` from the new session catalog — that would break HF Optimum on Linux + cuDNN setups. Pure documentation against future "consistency refactor" temptation.

## Diff metrics
- Lines: +5 / -0 (net +5)
- Hunks: 1 (comment block immediately above existing `provider=` line)
- Symbols touched: 0

## Role before vs after
- Before: `OptimumONNXModel.from_pretrained_with_metadata` constructed an ORT-backed model via Optimum's `ort_model_class.from_pretrained(temp_path, provider="CPUExecutionProvider" | "CUDAExecutionProvider", **kwargs)`. The hard-coded CUDA-on-GPU branch was a bare CLI string with no rationale documented.
- After: same line; same behavior; now has a five-line comment block warning future maintainers not to replace the hard-coded EP names with `default_ep_for_device("gpu")` from the session catalog.

## Symbol-level changes
- Added five new lines of comment (all `# CARVE-OUT: …`) immediately above the existing `provider=` kwarg. Nothing else.

## Behavior / contract changes
- None. The runtime behavior is byte-equivalent.
- Net architectural: a soft contract is added — "do not refactor this to use session.default_ep_for_device". Future refactor PRs that attempt to "unify" this will have a built-in explanation of why they shouldn't.

## Cross-file impact
- None. This is the only file that hardcodes `"CUDAExecutionProvider"` as the GPU EP — everywhere else in the codebase has been migrated to use `session.default_ep_for_device(...)` (which on Windows would return `"DmlExecutionProvider"`). The comment explains the asymmetry.
- The comment implicitly documents an architectural boundary: HF Optimum operates outside the WinML EP catalog and has its own EP-naming conventions.

## Risks / subtleties
- The comment is the only enforcement mechanism. A future maintainer using a grep tool to find "remaining `CUDAExecutionProvider` literals" and refactor them would have to read this comment to understand the carve-out — easy to miss in a sweep.
- The comment's claim that `default_ep_for_device("gpu")` returns `"DmlExecutionProvider"` on Windows is reasonable but unverified by a test. If the catalog's GPU default ever changes (e.g. `WebGPU` becomes the default), this comment's premise weakens.

## Simplification opportunities
- **Lift the carve-out into a typed constant.** Something like `_OPTIMUM_GPU_EP = "CUDAExecutionProvider"  # see CARVE-OUT below` would make the constant grep-able, and the carve-out narrows to one location. Today the literal `"CUDAExecutionProvider"` is in the conditional expression.
- **Add a test that verifies the carve-out.** A test that asserts `default_ep_for_device("gpu")` is *not* what gets passed to `ort_model_class.from_pretrained` would protect against silent regressions in either direction.
- **Move the comment to a docstring on the function** so it shows in autocomplete and `help()` output. The current inline comment is hidden from any introspection.

## Open questions / TODOs surfaced
- Is there a way to test the carve-out end-to-end without actually running Optimum on a Linux + cuDNN host? A mock-based test that asserts the literal `"CUDAExecutionProvider"` is passed when `device="gpu"` would be a cheap proxy.
- Should `OptimumONNXModel` factor the EP-name choice into a separate helper so the carve-out lives in one named place rather than as a comment inside a conditional expression? Worth a small follow-up.
