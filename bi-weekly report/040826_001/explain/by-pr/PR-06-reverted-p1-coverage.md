# PR-06: REVERTED — pickup p1_coverage (#20 / #21)

## Commit Metadata
| Field | Value |
|-------|-------|
| Original Commit | `a31ffd5` |
| Revert Commit | `a50b037` |
| Date | 2026-03-30 |
| Original Author | Fangyang Ci (fangyangci) |
| Revert Author | Zhipeng Wang (timenick) |
| PR Numbers | #20 (original), #21 (revert) |

## Summary
PR #20 ("pickup p1_coverage") landed on 2026-03-30 and was reverted the same day in PR #21. The original change expanded QNN NPU rule zips for opsets 12-22 (all significantly larger) and updated `result_processor.py`, `indexing_input_generator.py`, `slice_input_generator.py`, and `squeeze_input_generator.py`, as well as adding `test_input_generators.py`. PR #21 fully reverted all these changes.

The content of this work was subsequently re-landed in a corrected form as PR #22 (commit `603309d`).

> **Note**: This PR pair is excluded from module change tracking. See PR-08 for the final accepted version.
