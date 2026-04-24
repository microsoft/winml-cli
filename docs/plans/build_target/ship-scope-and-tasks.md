# Ship Scope & Core Tasks

## Ship Scope

### Repo

| Item | Target |
|------|--------|
| Repo | Public-ready open-source release |
| SDK | 1.8 & 2.0 |

### Built-in Models

50+

### EP + Device — 8 combinations, above quality bar

| EP | Hardware |
|----|----------|
| QNN | NPU, GPU * |
| Intel OpenVINO | NPU, CPU, GPU |
| AMD VitisAI | NPU * |
| NVIDIA TensorRT | GPU * |
| MLAS | CPU |

\* Out of SA scope

### Commands

| Required (12) | Optional (3) |
|----------------|--------------|
| winml analyze | winml catalog (list) |
| winml build | winml run |
| winml compile | winml serve |
| winml config | |
| winml eval | |
| winml export | |
| winml help | |
| winml inspect | |
| winml optimize | |
| winml perf | |
| winml quantize | |
| winml sys | |

---

## Check List

### Release (2w)

| Task | Pri | Owner | Notes |
|------|-----|-------|-------|
| Telemetry + privacy review | P0 | @Zhipeng | |
| Documentation | P0 | @Zheng, @Brenda | |
| Pypi release (version 0.1.0) | P0 | @Zhipeng | |
| Launch comms | P0 | @Brenda, @Lu | homepage |

### AITK

| Task | Pri | Owner |
|------|-----|-------|
| UI | P0 | @Shiyi |

---

## Feature Scale

### WinML 2.0 (1w + 1w)

| Task | Pri | Owner | Notes |
|------|-----|-------|-------|
| Update ModelKit dependency to WinML 2.0 | P0 | | |
| Test all models with WinML 2.0 | P0 | | |
| SA: min-delta dataset support for new EPs | P1 | | Leverage run unknown |
| Profiling + op tracing: verification on all EPs | P1 | @Hualiang, @Zheng | Depends on IHV EP quality |

### New Features (2w)

| Task | Pri | Owner | Notes |
|------|-----|-------|-------|
| First run experience | P0 | @Brenda | 1w dev time needed |
| Core metrics & dashboard | P0 | @Qiong | SA status, perf/memory, summary |
| ModelKit skills | P1 | | Install skill (env check?) |
| Feature flag support | P2 | | |

---

## EP Scale

| Area | Task | Pri | Owner |
|------|------|-----|-------|
| ModelKit | E2E verified with all 8 EPs | P0 | @Qiong |
| SA | Dataset prep — Intel CPU/GPU, TRT RTX, QNN GPU | P0 | @Fangyang |
| SA | Enable MLAS for CPU without dataset | P0 | |
| SA | Disable AMD NPU * | P0 | |

---

## Model Scale

| Task | Pri | Target | Owner |
|------|-----|--------|-------|
| Built-in models x 8 EPs | P0 | >= 50 | |

---

## Quality

See [ship-quality-plan.md](ship-quality-plan.md) for detailed action items per quality area.

| Area | Driver | Pri | Key Metric |
|------|--------|-----|------------|
| Functionality | Zhipeng | P0 | 0 open P0 bugs at ship |
| Performance & Memory | TBD | P0 | All components meet defined targets |
| Documentation | Brenda | P0 | Spec review meeting sign-off |
| EP Data Quality | TBD | P1 | >= 90% P0 & built-in; >= 80% Top 200 |
| Code Quality | TBD | P2 | >= 80% coverage; 0 ruff violations; mypy clean |

---

## Appendix: Component Owners

| Component | Owners |
|-----------|--------|
| Load & Export | @Yi, @Chao |
| Analyzer | @Yi, @Chao, @Fangyang |
| Optimizer | @Yi, @Yue |
| Eval | @Zhenchao, @Qiong |
| Compile & Sys | @Zhenchao, @Chao |
| Quantize | @Zhenchao, @Hualiang |
| Perf | @Hualiang, @Zheng |
| Catalog | @Qiong |
| Other | @Zheng |
| Repository | @Zhipeng, @Yue |
| Inspect | @Zheng |
