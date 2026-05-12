# Analyzer Performance Investigation

**Date**: 2026-03-23
**Branch**: `mvp/analyzer`

## Problem

bert-base-uncased (442 nodes) takes 2x longer to analyze than DETR (724 nodes):

| Model | Nodes | QNN | All 3 EPs |
|-------|-------|-----|-----------|
| bert  | 442   | 21s | 44s       |
| DETR  | 724   | 8.5s | 23s      |

## Root Cause

**`make_hashable()` in `runtime_checker_query.py` is called 29 million times**, consuming 39s (75% of total time).

### Profile (bert, QNN, run_unknown_op=False)

```
28,982,211 calls  20.7s  make_hashable()      — recursive tuple conversion
87,399,578 calls  10.5s  isinstance()          — called inside make_hashable
   288,946 calls   0.2s  from pandas.apply     — _sanitize_df table loading
     9,380 calls   0.01s from dictcomp line 481 — per-node conditions
```

### Call Chain

1. `_sanitize_df()` runs `df[col].apply(make_hashable)` on every column of every operator's rule table
2. Rule tables are loaded from 627 MB JSON files (compressed to 9 MB in zip)
3. Tables are loaded **per EP** — 3 EPs × same data = 3x cost
4. `make_hashable` recurses into deeply nested list/dict structures in rule data

### Why bert is slower than DETR

Not about node count — about which ops are used:
- bert has `Gather`, `Where`, `Erf`, `Cast` — these have larger/more complex rule tables
- DETR has `Conv`, `Relu` — simpler rules, faster lookup

## Attempted Fix

Skipping `make_hashable(arr)` for weight initializers (line 406) — **no improvement** because the 29M calls come from `_sanitize_df` on rule tables, not from weight data. The initializer fix saved only 9,380 calls out of 29M.

## Proposed Fixes (separate PR)

### Option A: Cache tables across EPs (quick win)
Same `LazyDomainTables` data can be shared when EPs use the same opset tables. Currently each `RuntimeCheckerQuery` creates new instances.

### Option B: Pre-sanitize at build time (permanent fix)
Run `_sanitize_df` once during rule generation (offline), save hashable tables to disk. Runtime loads pre-sanitized data — no `make_hashable` at query time.

### Option C: Skip unnecessary columns (medium effort)
Only sanitize columns that appear in the query conditions. Most cells are already hashable (strings, ints, bools). Only list/dict cells need conversion.

### Option D: Replace pandas with dict lookup (architectural)
The DataFrame + `query_table_exact_match` pattern is slow. A nested dict keyed by condition values would be O(1) lookup instead of O(n) DataFrame scan.

## Files

- `modelkit/static_analyzer/core/runtime_checker_query.py:71-79` — `_sanitize_df`
- `modelkit/static_analyzer/core/runtime_checker_query.py:82-103` — `LazyDomainTables`
- `modelkit/static_analyzer/utils/model_utils.py:205-222` — `make_hashable`
- `modelkit/static_analyzer/rules/runtime_check_rules/*.zip` — 627 MB JSON rule tables
