# E2E run — 20260513-220000_cpu-only

**Started:** 2026-05-13T14:00:08+00:00  
**Host:** DEVBOX-CPU-TEST  
**Skill commit:** `passk-demo`  
**winml:** 0.4.x  
**Registered EPs:** CPUExecutionProvider, DmlExecutionProvider, OpenVINOExecutionProvider

## Summary

- Cases attempted: 3
- Fully-passing cases (Pass@K = K/K): **3/3**
- Skipped: 1

| Case | Pass@K | Avg assertion pass rate | Trials |
|------|--------|-------------------------|--------|
| `cpu-benchmark-resnet` | **3/3** | 100% | trial-1=5/5, trial-2=5/5, trial-3=5/5 |
| `llm-refusal-phi3` | **3/3** | 100% | trial-1=5/5, trial-2=5/5, trial-3=5/5 |
| `cpu-full-build-resnet` | **3/3** | 100% | trial-1=4/4, trial-2=4/4, trial-3=4/4 |

## Skipped cases

- `qnn-benchmark-resnet` — QNNExecutionProvider not registered

## Per-case detail

### `cpu-benchmark-resnet`
- Pass@3: **3/3**
  - `trial-1` — PASS (5/5, 7 tool calls, 172.3s)
  - `trial-2` — PASS (5/5, 7 tool calls, 172.3s)
  - `trial-3` — PASS (5/5, 7 tool calls, 172.3s)

### `llm-refusal-phi3`
- Pass@3: **3/3**
  - `trial-1` — PASS (5/5, 1 tool calls, 16.3s)
  - `trial-2` — PASS (5/5, 1 tool calls, 16.3s)
  - `trial-3` — PASS (5/5, 1 tool calls, 16.3s)

### `cpu-full-build-resnet`
- Pass@3: **3/3**
  - `trial-1` — PASS (4/4, 9 tool calls, 220.1s)
  - `trial-2` — PASS (4/4, 9 tool calls, 220.1s)
  - `trial-3` — PASS (4/4, 9 tool calls, 220.1s)
