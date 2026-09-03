# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## WinML CLI v0.3.1

This hotfix restores compatibility with current Transformers and Windows runtime dependencies, fixes VitisAI cache permissions, and stabilizes Hugging Face model export and evaluation.

### 🐛 Fixes

- **Hugging Face export and evaluation** — stabilized SDPA and attention-mask handling, decoder wrappers, BLIP export, QNN/HTP tracing, and processor resolution for Transformers 4.57 (#1372, #1379, #1384).
- **VitisAI compilation** — moved the compilation cache under the user-writable WinML cache root instead of installation-relative protected paths (#1378).
- **Runtime dependencies** — constrained `plotext` to the compatible 5.x series, required the published Windows ML runtime build, and prevented QNN from installing a conflicting standard ONNX Runtime distribution (#1372).

### 📦 Assets

- `winml_cli-0.3.1-py3-none-any.whl`
- `rules-v0.3.1.zip`

## WinML CLI v0.3.0

This cycle expands **model preparation and evaluation** across the CLI: precision-driven quantization, composite-model and dynamic-axis workflows, real-input perf/eval, optimization previews, and opt-in Dynamo export. It also introduces one-command Qwen3 onnxruntime-genai bundles, GenAI benchmarking, broader model recipes, and more reliable EP discovery, compilation, and monitoring. See the behavior changes below.

### ⚠️ Behavior changes

- **Output-producing commands** now refuse to replace existing files or non-empty directories unless `--overwrite` is passed; `winml build` retains its existing incremental `--rebuild` behavior (#970).
- `winml quantize` renames `--model-name` to `--model-id`, including the corresponding quantization-config field (#984).
- **Compile configuration** no longer silently defaults a missing execution provider to QNN; incomplete configurations now fail validation instead (#1026).
- `winml inspect` / `winml perf` hide third-party and native warning noise by default; use `-v`, `-vv`, or `WINMLCLI_SHOW_ALL_WARNINGS=1` to restore diagnostics (#1232, #1246).

### ✨ Improvements

- **Quantization** — `--precision` selects FP16 conversion, RTN INT4, static QDQ, or calibration-free dynamic INT8; `winml quantize` can compose multiple precision passes such as INT4 followed by FP16 (#872, #985, #1047).
- `winml build` — `--export-type optimized` produces a complete Qwen3 onnxruntime-genai NPU/QNN bundle, including prefill/decode, embeddings, LM head, tokenizer, and manifest files (#836, #996, #1008, #1081, #1104).
- `winml perf --runtime winml-genai` — benchmarks prebuilt or automatically cached GenAI bundles with TTFT, token throughput, prompt-template controls, EP overrides, and isolated pre-compilation (#1015, #1042, #1046, #1054, #1109).
- **Composite models** — `export` and `build` automatically fan out pipeline components; `export` / `build` / `perf` support `--submodel`, and explicit composite tasks such as summarization and translation are accepted (#1031, #1037, #1058, #1071, #1089).
- **Export controls** — dynamic axes and symbolic input dimensions are supported while static TorchScript export remains the default; `build`, `config`, `perf`, and `eval` expose matching shape/input/export overrides (#1074, #1083, #1106, #1141, #1156, #1188).
- `winml perf` — real `.npz` inputs, time-budgeted `--duration` runs, cached per-module builds, actual dynamic dimensions, and QNN profiler ONNX metrics (#1004, #1055, #1066, #1102, #1168).
- **QNN op tracing** — per-model tracing can be enabled automatically; basic traces exclude warmup samples and null fields, while detail tracing accepts compile EP options and auto-compiles raw ONNX inputs when required (#1006, #1032, #1249, #1252).
- `winml eval --mode compare` — compares two ONNX models directly or uses real `.npz` samples against a Hugging Face reference; Qwen3 adds perplexity evaluation (#1139, #1209, #1221).
- `winml optimize` / `winml analyze` — `--check-optim` previews applicable rewrites and verifies their produced operators against the target EP; new rewrites cover static Split-to-Slice and Conv affine/BatchNormalization folding (#1142, #1167, #1171, #1238, #1257).
- **EP discovery and monitoring** — registration is isolated and failures are structured, startup remains lazy, op-tracing dispatch is unified, and provider-download progress is restored (#1019, #1239).
- **CLI quality of life** — EP/device and pipeline-stage flags are consistent across commands; `--no-color` disables ANSI output for one invocation (#923, #978, #992).
- **Hub-hosted ONNX** — commands accept `<org>/<repo>/<path>.onnx` references from Hugging Face Hub, enabling SAM 3 encoder/decoder workflows (#582).
- **Keypoint detection** — ViTPose supports `config`, `build`, and `perf`, plus COCO OKS-AP evaluation (#905, #949).
- **Vision and document recipes** — refreshed coverage adds DINOv2, SwinV2, OWL-ViT/OWLv2, BEiT, SegFormer, YOLOS, ViTPose/SynthPose, LayoutLM/LayoutLMv3, and document/question-answering models (#925, #1064, #1088, #1093, #1100, #1101, #1123, #1125, #1145, #1155, #1173, #1174, #1178, #1187, #1201, #1202, #1205, #1208).
- **Language recipes** — expanded BART, BERT, DeBERTa, DistilBERT, KoELECTRA, MiniLM, MPNet, Marian/OPUS, GTE reranker, feature-extraction, and entity-linking coverage (#1068, #1080, #1112, #1115, #1116, #1117, #1118, #1120, #1121, #1124, #1134, #1143, #1144, #1153, #1169, #1170, #1179, #1200, #1214).
- **Audio recipes** — expanded Wav2Vec2, HuBERT, AST, MMS, language/gender/music classification, forced alignment, and multilingual ASR coverage (#1094, #1095, #1114, #1131, #1148, #1154, #1176, #1186, #1206, #1207, #1211, #1225).

### 🐛 Fixes

- **`winml perf`** — throughput uses the batch size actually executed; analyzer EP resolution and op-trace paths now match the runtime target (#930, #941, #1000).
- **`winml build`** — honors explicit `--ep`, supports non-compiling cross-target builds, keeps ONNX caches distinct by resolved path and configuration, preserves configured model classes, reports disk-full failures clearly, and keeps CPU/GPU automatic precision at FP32 (#856, #947, #987, #997, #998).
- **GenAI and composite export** — fixed component export/build failures, compile fallback paths, accelerator selection, isolated EPContext preparation, and final Qwen3 bundle assembly (#1037, #1051, #1103, #1138, #1248).
- **Task and model resolution** — reconciled the task registry, corrected model-specific task listings and Hub `pipeline_tag` fallback, accepted composite tasks, and resolved CTC-based ASR model classes correctly (#724, #986, #1070, #1071, #1113, #1154).
- **Depth and keypoint evaluation** — fixed inference-time evaluator failures (#1023).
- **Analyzer and optimizer rules** — corrected coverage counting, aligned pattern checks with node support, consolidated recommendation metadata, and fixed dtype constraints and unknown-pattern handling (#922, #1020, #1063, #1130, #1162).
- **EP / device resolution** — WindowsML catalog providers register correctly; device listings retain hardware details without duplicate aliases; analyzer auto-selection prefers the strongest exact target; CPU bridge providers resolve safely; invalid EP/device pairs fail early (#1076, #1220, #1227, #1228, #1231, #1237).
- **Native EP execution** — hardened spawned-provider progress, prevented compiler-output deadlocks, released native sessions before process exit, and replaced pipe-backed warning capture with a bounded file spool to avoid EP compiler hangs (#1017, #1223, #1230, #1266, release cherry-pick #1267).
- **Export and quantization** — standalone quantization suppresses duplicate ORT preprocessing warnings; decoder KV-cache dimensions survive tracing; large external-data models can convert to FP16; and EPs that quantize internally no longer receive redundant WinML quantization (#956, #1176, #1235, #1242).
- **QNN evaluation and tracing** — repaired evaluation failures, detail-trace DLL/summary handling, and compile-time provider options (#1247, #1249).
- `winml eval` — default text datasets and sentiment recipes use fully qualified Hugging Face dataset IDs (#1262, release cherry-pick #1263).
- **CLI help** — `winml --help` shows the correct `sys` summary and concise, untruncated `build` / `quantize` descriptions (#1254).
- **Telemetry** — local `Path` model references no longer cause successful commands to fail during telemetry scrubbing (#1273).

### 🔧 Internals & CI

- **Release pipelines** — E2E aligns ModelKitArtifacts with the matching release branch, stable GitHub releases receive CHANGELOG notes and “Latest” status, and the official-build toolchain is pinned for reproducibility (#940, #967, #1268).
- **Evaluation CI** — recipe-driven build/eval supports per-EP matrices, pre-exported ONNX, reliable resume behavior, actual applied-precision reporting, unquantized-track EPs, and broader MIGraphX/TensorRT RTX coverage (#845, #902, #1009, #1039, #1086, #1160, #1163, #1226, #1243, #1286).
- **Telemetry** — action events record scrubbed model identifiers, while error events retain scrubbed root-cause details for diagnosis (#1108, #1111).
- **Development environment** — expanded type checking, added a tracked `uv` lockfile, selected CPU-only PyTorch wheels, and consolidated development dependencies (#932, #957, #1105, #1251, #1255).
- **Documentation publishing** — added and published the version-stamped model accuracy report from the current documentation site (#974, #975, #979, #1203).

### 📦 Assets

- `winml_cli-0.3.0-py3-none-any.whl`
- `rules-v0.3.0.zip`

## WinML CLI v0.2.0

This cycle unifies **task detection** across the CLI (modality- and architecture-aware) and expands the eval and perf surfaces — new depth-estimation and tensor-similarity evaluators, a full SA eval pipeline with an HTML report, `winml perf --memory` / `--ep-options`, and `--format json` on `eval` / `analyze` / `perf`. `winml compile` gains a multi-model shared EP context, `winml build` gains `--precision`, and timm image-classification is supported. See the behavior changes below.

### ⚠️ Behavior changes

- `winml perf` no longer compiles by default — added `--compile/--no-compile`, defaulting to no-compile (#879).
- Boolean CLI options are now `--flag/--no-flag` pairs (#844).
- Telemetry is enabled in the shipped wheel; consent reworded as "unlinked pseudonymized" (#810).

### ✨ Improvements

- **Task detection** — modality- and architecture-aware `detect_task`, unified across commands via `resolve_task` / `TaskResolution` (#807, #841, #878).
- `winml perf` — `--memory` reports RAM/VRAM per phase (#861); `--ep-options` passes runtime EP options (#865, #889); output now shows the model path and precision (#875).
- `winml compile` — multi-model shared EP context with a selectable backend (#871).
- `winml build` — added `--precision` (#914).
- `winml inspect` — renders composite (pipeline-led) model structure (#903).
- `winml analyze` — `--ep` / `--device` auto resolves to a single best target (#919); faster re-runs plus a `--debug` rule locator (#906).
- `winml eval` — new SA eval pipeline with per-stage perf and an HTML report (#599); depth-estimation (#326, #437) and tensor-similarity (#805) evaluators; scripts track ONNX size and sanitize output (#755).
- Cross-command — `--format json` on `eval` / `analyze` / `perf` (#855); `--allow-unsupported-nodes` on `perf` / `build` / `eval` / `run` (#821).
- Quality of life — timm image-classification via library routing (#790); `~` expanded in paths (#815); progress bar during EP warmup (#788); refreshed `--list-device` coloring (#812).

### 🐛 Fixes

- **`winml perf`** — declared `psutil` as a runtime dependency, fixing a crash on clean install (#937); composite (dual-encoder) models supported (#866); HF and ONNX paths unified through `PerfBenchmark` (#659); `--monitor` live chart in `--module` mode (#654, #920); `rich` Live thread crashes (#832).
- **`winml analyze`** — coverage-counting bugs (#922); analyzer API EP list matches the CLI (#803); Pad / Gemm rule conflicts (#906).
- **Task / config validation** — fill-mask heads detected as `text2text-generation` (#851); vision feature-extraction model-task inconsistency (#786); model task validated in config (#723); full encoder-decoder composite built for no-task seq2seq (#850, #862); device/EP combination validated without a system check (#780).
- **`winml export`** — `.data` files written to the output dir, not the cwd (#853); timm `image_size` from `pretrained_cfg` (#806).
- **`winml inspect` / `winml catalog`** — `--task` validated at parse time (#546, #771); `catalog -t` short flag aligned (#541, #772); VitisAI EP ordered last, catalog table width fixed (#763).
- **Feature extraction** — `last_hidden_state` now populated in the output (#863).
- **`winml optimize`** — untie batched constant `MatMul` for OpenVINO GPU (#817).
- **`winml eval`** — fixed failures on AMD hosts (#783); cleanup runs on `SKIP_*` / exception paths (#890).
- **CLI output** — quieted `optimum` logger noise (#904); unified verbosity, logger routed to stderr (#566, #793).

### 📦 Assets

- `winml_cli-0.2.0-py3-none-any.whl`
- `rules-v0.2.0.zip`

## WinML CLI v0.1.0

First **public preview** release. With the Windows ML 2.0 baseline now in place, this release shifts focus to polishing the CLI surface: faster `winml inspect` / `winml eval`, more accurate device & EP resolution, a real PyPI release pipeline, and a meaningful pass over sysinfo and quantization behavior.

### 🎉 Public preview

- Promoted to `Development Status :: 4 - Beta` in `pyproject.toml`.
- First release published to PyPI via the new ESRP-signed release pipeline (#473).

### ✨ Improvements

- `winml inspect`: banner + spinner during HF metadata fetch (#718, hidden in JSON mode #745); `--list-tasks` <500 ms (#717); processor `Auto*` lookups gated (#719, #746).
- `winml eval`: lazy module loading drops cold-start latency (#711); inputs validated up-front with friendlier errors and a structured `--schema` output (#694).
- `winml export`: `model-id` and `task` validated before the export runs (#714).
- `winml analyze`: cleaner EP/device selection, clearer "op-check skipped" UI, merged optimization config (#702).
- `winml perf`: estimated model precision (QDQ / block-wise quant / dominant float dtype) is now reported by `WinMLSession` (#706); expanded perf e2e coverage across EPs and devices (#698).
- `winml monitor`: queries all NPU/GPU engines and reports the max utilization (#716).
- CLI-wide: did-you-mean suggestions on mistyped subcommands (#699); consistent option-vs-config-file value priority across commands (#720); `op_tracing` hidden from the public surface (#738).
- Adopted the official `windowsml` usage example — removed the redundant `WinML` singleton, fixing a benign "library already registered" traceback on `winml perf --device npu` (#729).

### 🐛 Fixes

- **Quantization (P0)** — `--precision` now rejects invalid values instead of silently falling back to `uint8/uint8`; default image calibration dataset streams rather than downloading ~5 GB; DETR-family object detection supports `pixel_mask` padding (#680).
- **`winml eval`** — pinned `pyarrow <24` to avoid an EP DLL load-order crash (#750).
- **`winml perf`** — QDQ precision detection fix (#753); NPU monitoring adds `3D` engine, device line shows requested vs. actual (#747).
- **EP / device resolution** — `resolve_device`/`resolve_eps` now use `get_registered_ep_devices` (#712); dropped misleading `ov`/`vitis`/`trtrtx` aliases (#690); `winml sys` raises when an EP isn't available on the host (#686); per-provider `ensure_ready` failures demoted to debug (#703); analyze regression caught during compile e2e (#740).
- **Native ORT / WinML** — suppressed ORT native stderr, fixed a HANDLE leak (#709); nulled the EP catalog handle after enumeration to prevent a QNN NPU crash on exit (#701); fixed the `onnxruntime` DLL search path (#689).
- **`winml sys`** — diagnostic sections gated behind `-v`, json-mode logs routed to stderr (#737); CPU/Mem scoped to the current process and PDH percent counters no longer artificially capped (#715); host arch reported via `IsWow64Process2` on Windows ARM64 (#705).
- **OpenVINO** — `is_npu` detection updated (#722).

### 🔧 Internals & CI

- Added a `winml-cli` Copilot skill (#733).

### 📦 Assets

- `winml_cli-0.1.0-py3-none-any.whl`
- `rules-v0.1.0.zip`

## WinML CLI v0.0.4

### 🚀 Platform upgrades

- **Windows ML → 2.0** (#441)
- **Python → 3.11** (`requires-python = ">=3.11,<3.12"`)

### ⚠️ Behavior changes

- Incompatible `--ep` / `--device` pairs are now rejected instead of silently overridden (#641, #661).
- `winml config/build --device npu` exits non-zero when no compatible NPU EP is available (#660).
- `winml analyze --ep cpu` resolves to CPU instead of falling back to NPU (#641).
- `trust_remote_code` now prints a bold-red stderr warning whenever it is honoured (#641).

### ✨ Improvements

- `winml build` writes `analyze_result.json` to the output folder (#673) and validates the config up-front (#675).
- Exported ONNX is auto-normalized via `optimize_onnx()` (#681).
- `winml inspect` distinguishes local-path-not-found from network errors (#679).

### 🐛 Fixes

- `--run-unknown-op` compile=false regression (#662).
- `winml build --device npu` failing with `quant.task is required` (#673).
- HF build path dropped explicit `--ep` on compile-less paths (#678).
- `run_eval.py` not forwarding `--device` to `winml build` (#674).
- Seq2seq decoder calibration crash on image-to-text models (#671).

### 🔧 Internals & tests

- Strong-typed EP parameters across analyze/compiler/optracing (#632).
- `EP_SUPPORTED_DEVICES` as single source of truth (#641).
- Expanded E2E / CLI surface tests for `analyze`, `compile`, `inspect`, `catalog`, and perf (#645, #652, #661, #665, #669, #672, #676).

### 📦 Assets

- `winml_cli-0.0.4-py3-none-any.whl`
- `rules-v0.0.4.zip`

## WinML CLI v0.0.3

### ⚠️ Breaking — Runtime rule artifacts

The format and packaging of the analyzer runtime rule artifacts changed in v0.0.3. Anyone who scripted against the v0.0.2 release assets, or who points the analyzer at an external rules directory, needs to update.

**1. Release asset layout: many per-EP/opset ZIPs → one versioned ZIP of Parquet files**

- v0.0.2 published dozens of individual rule archives, one per EP × device × opset (e.g. `QNNExecutionProvider_NPU_ai.onnx_opset17.zip`, `OpenVINOExecutionProvider_GPU_ai.onnx_opset20.zip`, …).
- v0.0.3 publishes a single `rules-v0.0.3.zip` containing Parquet rule files. The filename is version-qualified (`rules-v<version>.zip`).
- Inside the archive, rule data is now stored as `*.parquet` rather than the previous ZIP-wrapped JSON. Old ZIP-expansion tooling has been removed.

**2. Environment variable rename: `MODELKIT_RULES_DIR` → `WINMLCLI_RULES_DIR`**

The override for additional runtime-rule lookup directories was renamed as part of the broader **ModelKit → WinML CLI** product rename. There is no compatibility shim — the old name is silently ignored.

### Migration

If you build from source or otherwise need to fetch rules manually:

```bash
gh release download v0.0.3 --repo microsoft/winml-cli --pattern 'rules-v0.0.3.zip' --dir .
```

```powershell
Expand-Archive -Path .\rules-v0.0.3.zip -DestinationPath src\winml\modelkit\analyze\rules\runtime_check_rules -Force
```

`gh release download` skips pre-releases unless you pass `--tag`, so the explicit `v0.0.3` is required.

If you set `MODELKIT_RULES_DIR` anywhere (shell profile, CI pipeline, user env), rename it to `WINMLCLI_RULES_DIR`. It points to a single rules directory (not split on `os.pathsep`); relative paths still resolve from `src/winml/modelkit/analyze/utils/`.

Related PRs: #411 (Parquet migration), #600 (rules zip in release), #627 (versioned filename), #587 (env var rename as part of ModelKit → WinML CLI Wave 1).

### 📦 Assets

- `winml_cli-0.0.3-py3-none-any.whl`
- `rules-v0.0.3.zip`
