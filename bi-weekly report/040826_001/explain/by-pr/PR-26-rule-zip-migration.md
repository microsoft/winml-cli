# PR-26: Move Runtime Check Rule Zips to External Repo (#213)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `c9c3a88` |
| Date | 2026-04-03 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #213 |
| Files Changed | 25 |
| Insertions | +144 |
| Deletions | -1 |

## Summary
Removed all runtime check rule zip files from git tracking to reduce repository size. The 22 zip files (spanning QNN NPU opset11–22, QNN GPU, OpenVINO CPU/GPU/NPU, WinML NPU, QNN NPU com.microsoft, VitisAI NPU) are now hosted in the external `gim-home/ModelKitArtifacts` repository. Added `scripts/download_rules.py` to automate fetching the zips via git sparse checkout, added a `runtime_check_rules/README.md` with setup instructions, updated `.gitignore` to exclude `*.zip` from the rules directory, and improved `runtime_checker_query.py` warning messages to point to the download script when zips are missing.

## Files Changed
- `scripts/download_rules.py` — new download automation script (+104)
- `src/winml/modelkit/analyze/rules/runtime_check_rules/README.md` — developer setup guide (+25)
- `src/winml/modelkit/analyze/core/runtime_checker_query.py` — missing-zip warnings improved (+11/-1)
- `.gitignore` — added `runtime_check_rules/*.zip` exclusion
- `pyproject.toml` — minor update
- All 22 rule zip files — removed from git tracking (binary, 0 bytes)
