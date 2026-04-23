# PR-16: Add NvTensorRTRTX to EP Device Map (#188)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `5c5b873` |
| Date | 2026-03-31 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #188 |
| Files Changed | 2 |
| Insertions | +7 |
| Deletions | -12 |

## Summary
Added `NvTensorRTRTXExecutionProvider` to the `_EP_DEVICE_MAP` in `sysinfo/device.py`. This WinML-specific NVIDIA EP name was missing, causing `winml sys --list-ep` to report UNKNOWN as the device type instead of GPU. Also simplified the device mapping logic in the same file.

## Files Changed
- `src/winml/modelkit/sysinfo/device.py` — added NvTensorRTRTXExecutionProvider to device map; simplified mapping logic
- `tests/unit/sysinfo/test_device.py` — updated test assertions for new mapping
