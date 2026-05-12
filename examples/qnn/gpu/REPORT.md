# QNN (Qualcomm) GPU Test Report

## Summary

- **Models tested**: 56
- **Configs tested**: 192
- **Perf pass rate**: 162/192 (84%)
- **Eval pass rate**: 105/192 (55%)
- **Non-pass results**: 53 errors, 65 timeouts

## Notes

- This report is generated from current artifacts under `examples/qnn/gpu/`.
- Result file types used for summary:
  - `*_perf.json`
  - `*_eval.json`
  - `*.error.txt`
  - `*.timeout`
- Latest resumed completion segment reported: `PASS=27, FAIL=3, TIMEOUT=2, SKIP=160`.
