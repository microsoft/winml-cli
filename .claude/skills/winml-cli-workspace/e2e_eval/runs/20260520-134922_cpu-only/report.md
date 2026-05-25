# E2E run — 20260520-134922_cpu-only

**Started:** 2026-05-20T13:38:49Z  
**Host:** CPC-shzhe-JK9DU  
**Skill commit:** `5820645fd751ae08fcfbcbcb8602e7c7122c652a`  
**winml:** winml, version 0.0.2  
**Registered EPs:** CPUExecutionProvider, DmlExecutionProvider, OpenVINOExecutionProvider

## Summary

- Cases attempted: 3
- Fully-passing cases (Pass@K = K/K): **3/3**
- Skipped: 3

| Case | Pass@K | Avg assertion pass rate | Trials |
|------|--------|-------------------------|--------|
| `cpu-benchmark-resnet` | **3/3** | 100% | trial-1=5/5, trial-2=5/5, trial-3=5/5 |
| `llm-refusal-phi3` | **3/3** | 100% | trial-1=5/5, trial-2=5/5, trial-3=5/5 |
| `cpu-full-build-resnet` | **3/3** | 100% | trial-1=4/4, trial-2=4/4, trial-3=4/4 |

## Skipped cases

- `qnn-benchmark-resnet` — QNNExecutionProvider not registered
- `qnn-full-build-resnet` — QNNExecutionProvider not registered
- `vitisai-benchmark-resnet` — VitisAIExecutionProvider not registered

## Per-case detail

### `cpu-benchmark-resnet`
- Pass@3: **3/3**
  - `trial-1` — PASS (5/5, 5 tool calls, 160.9s)
  - `trial-2` — PASS (5/5, 4 tool calls, 132.9s)
  - `trial-3` — PASS (5/5, 5 tool calls, 142.5s)

### `llm-refusal-phi3`
- Pass@3: **3/3**
  - `trial-1` — PASS (5/5, 1 tool calls, 25.4s)
  - `trial-2` — PASS (5/5, 1 tool calls, 23.6s)
  - `trial-3` — PASS (5/5, 1 tool calls, 23.0s)

### `cpu-full-build-resnet`
- Pass@3: **3/3**
  - `trial-1` — PASS (4/4, 8 tool calls, 152.6s)
  - `trial-2` — PASS (4/4, 8 tool calls, 141.5s)
  - `trial-3` — PASS (4/4, 8 tool calls, 131.6s)
