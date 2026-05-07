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

| Task | Pri | Owner | Status | Notes |
|------|-----|-------|--------|-------|
| Telemetry + privacy review | P0 | @Zhipeng | In-progress | |
| Documentation | P0 | @Zheng, @Brenda | | |
| PyPI release (version 0.1.0) | P0 | @Zhipeng | | |
| Launch comms | P0 | @Brenda, @Lu | | homepage |
| Regular E2E eval & report pipeline | P1 | @Yue | | |

### AITK

| Task | Pri | Owner | Status |
|------|-----|-------|--------|
| UI | P0 | @Shiyi | In-progress |

---

## Feature Scale

### WinML 2.0 (1w + 1w)

| Task | Pri | Owner | Status | Notes |
|------|-----|-------|--------|-------|
| Update ModelKit dependency to WinML 2.0 | P0 | @Chao | In-progress | |
| Test all models with WinML 2.0 | P0 | @Chao | | |
| SA: min-delta dataset support for new EPs | P1 | @Chao | | Leverage run unknown |
| Profiling + op tracing: verification on all EPs | P1 | @Hualiang, @Zheng | | Depends on IHV EP quality |

### New Features (2w)

| Task | Pri | Owner | Status | Notes |
|------|-----|-------|--------|-------|
| First run experience | P0 | @Brenda | | 1w dev time needed |
| Core metrics & dashboard | P0 | @Qiong | In-progress | SA status, perf/memory, summary |
| ModelKit skills | P1 | @Shiyi | | Install skill (env check?) |
| Feature flag support | P2 | | | |

---

## EP Scale

| Area | Task | Pri | Owner | Status | Notes |
|------|------|-----|-------|--------|-------|
| ModelKit | E2E verified with all 8 EPs | P0 | @Qiong | In-progress | CPU (146 pass) |
| SA | Dataset prep — Intel CPU/GPU, TRT RTX, QNN GPU | P0 | @Fangyang | In-progress | TRT RTX in PR, others done |
| SA | Enable MLAS for CPU without dataset | P0 | @Qiong | Done | |
| SA | Disable AMD NPU * | P0 | @Fangyang | Done | |

---

## Model Scale

### E2E Perf — Inference Ability (Driver: @Hualiang)

**P0 Models**

| Model | Owner | Status |
|-------|-------|--------|
| Sam2 | @Chao | Done |
| ESRGAN | @Zhenchao | In-progress |

**Top 200 Models**

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Multi model support & KV cache | @Yi, @Zhenchao | In-progress | |
| 1/3, 2/3 models & default configs | @Chao, @Shiyi | In-progress | |
| Top 200 model list adjustment | @Hualiang, @Yi, @Zhenchao, all | In-progress | |

### E2E Eval — Accuracy (Driver: @Zhenchao)

| Task | Owner | Status |
|------|-------|--------|
| Accuracy evaluation for non-generation models | @Zhenchao | Done |
| Accuracy evaluation for generation models (blip, tocr) | @Zhenchao | In-progress |

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
