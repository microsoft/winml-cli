# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## WinML CLI v0.4.0

This cycle adds CGIR and Windows ML Runtime workflows, native PyTorch evaluation, physical-adapter selection, and broader operator profiling. It improves pattern-aware analysis, targeted optimization, model-loading memory measurements, and release evaluation. The release also carries forward the fixes already published in v0.3.1. See the behavior changes below.

### ⚠️ Behavior changes

- **Runtime selection** - runtime names are now `winml-ort` and `ort-genai` instead of `winml` and `winml-genai`; `winml perf` defaults to automatic runtime selection while retaining explicit choices (#1366, #1308).
- `winml perf` - canonical cache controls are `--use-cache` / `--no-use-cache`; the older `--ignore-cache` / `--no-ignore-cache` forms remain hidden deprecated aliases, and conflicting toggles are rejected (#1271).
- `winml analyze` - removes the obsolete `--htp-metadata` option in favor of pattern-rule-based runtime support checks (#1218).
- `winml serve` - replaces wildcard CORS with same-origin request protection, validates allowed CLI commands, and rejects HTTP attempts to enable remote code, including through nested model/configuration loading (#1321).
- **Memory reports** - classic `memory_measurement.schema_version` is now `3`, with revised baseline/delta boundaries; the outer report remains schema version `2`, and load-only measurements use `load_memory.version = 1` (#1429).
- **Operator tracing** - measured tracing runs default to 10 iterations; explicit iteration overrides remain available (#1406).
- **Evaluation recipe labels** - recipes previously labelled FP16 without an actual quantization/conversion step are labelled FP32; historical result paths and reference labels are preserved rather than rewritten (#1411).

### ✨ Improvements

- **CGIR workflows** - `export`, `config`, `build`, `compile`, `perf`, and `eval` integrate standalone MLIR, Windows ML Runtime with CGC compilation, and ONNX through the WinMLCG EP, using the required preview Runtime/EP packages (#1426).
- `winml eval --runtime pytorch` - evaluates Hugging Face models natively on CPU or CUDA and accepts existing PyTorch models through the evaluation API, preserving checkpoint dtype and reporting the selected runtime (#1282).
- `winml eval` - adds independent `--reference-device` and `--reference-ep` controls for ONNX comparison and integrates the shared device-selection path (#1324, #1412).
- `winml perf --device-luid` - selects the physical adapter reported by `winml sys`, keeps inference and monitoring on that adapter, and distinguishes identical GPUs; the option does not apply to `ort-genai` (#1398).
- **Operator profiling** - adds basic OpenVINO CPU/NPU and TensorRT RTX GPU tracing, excludes warmup samples, preserves raw trace artifacts, and improves QNN schematics and detail-fallback diagnostics (#1377, #1406, #1288, #1289).
- **Model-loading memory** - adds load-only measurements from runtime/device readiness through model compilation, separates sampled peaks from OS lifetime diagnostics, and prepares missing process counters on the explicitly selected GPU while preserving unavailable values and signed deltas (#1429).
- `winml sys` - uses DXCore for GPU/NPU identity and adapter LUIDs, retains WMI/PnP enrichment and fallback, and recognizes NVIDIA ACPI PnP identifiers (#1351, #1290).
- `winml sys` - expands memory metadata, dedicated/shared GPU memory, NPU memory, and Windows build details in system reports (#1312, #1387, #1421, #1425).
- `winml analyze` - evaluates matched subgraphs with pattern-level runtime rules before operator-level fallback, reports pattern coverage, and includes optimization findings in JSON output (#1218, #1285).
- `winml optimize` - adds EP/device targeting, displays custom operator domains, and accelerates capability checks (#1256, #1306, #1298).
- **Graph optimization** - adds routed affine and positive-Exp scale folding while preserving fast QNN grouped-convolution regions (#1301, #1317).
- **Qwen3 GenAI bundles** - shares decoder context/iterator weights through a common EP context and adds VitisAI support for the transformer stages (#1305).
- **Model caches** - unifies cache/rebuild controls across build, eval, and perf; evaluation enables model-cache reuse by default and reports controls that do not apply to prebuilt inputs (#1269, #1270, #1271).
- `winml export --batch-size` - supports validated static batch sizes and carries them through input specifications and export metadata (#1315).
- **Model adapters** - adds the Unlimited-OCR vision tower for feature extraction and audeering Wav2Vec2 dimensional-emotion speech regression (#1018, #1084).
- **Vision recipes** - adds CPU configurations for RT-DETR, ViTPose, and OWLv2 zero-shot detection, plus QNN NPU LayoutLM document-QA configurations (#1190, #1189, #1196, #1296, #1369).
- **Language recipes** - adds mMARCO MiniLM, NLI MiniLM, multilingual E5, and Spanish BERT configurations (#1191, #1192, #1210, #1185).
- **Audio recipes** - adds Wav2Vec2 deepfake classification, MMS-1B-all CPU configurations, and a QNN NPU W8A16 configuration for dimensional-emotion regression (#1194, #1177, #1318).

### 🐛 Fixes

- **Hugging Face export** - repairs Marian, TrOCR/Manga-OCR, LayoutLM QA, BLIP decoder, and SAM wrapper paths, and corrects eager-attention selection during model loading (#1323, #1330, #1392, #1403, #1375).
- **FP16 conversion** - handles initializer-backed graph outputs, prevents quantization naming collisions, and validates captured local-function tensors against stricter ONNX Runtime graph-attribute requirements (#1280, #1417, #1377).
- **Calibration and quantization** - clamps DistilBERT mask constants before calibration and fixes input-dtype, synthetic-calibration, and large-model QDQ serialization regressions found during release evaluation (#1420, #1411).
- **FP16 recipes** - restores missing quantization configuration for RoBERTa-large SQuAD2 and DeBERTa-v3-base NLI recipes (#1245, #1244).
- **ViTPose export** - derives dummy inputs from the model configuration (#1299).
- **Compiled-model caches** - keys EPContext reuse by compile identity and handles multiple QNN EP context partitions in perf (#1295, #1361).
- **GenAI perf** - honors the selected device and EP during automatic builds and aligns performance metric schemas (#1404, #1307).
- **Performance monitoring** - corrects multi-GPU monitoring, preserves memory baselines, and distinguishes unavailable measurements from real samples (#1313, #1414).
- **Runtime initialization** - includes CPU in vendor compatibility, lazily loads session backends and monitors, and removes the manual ONNX Runtime DLL preload path (#1303, #1382, #1380).
- `winml analyze` - repairs schema fallback for runtime-specific operators (#1304).
- **Image-to-text evaluation** - repairs evaluator behavior (#1410).
- `winml sys --format json` - keeps EP installation notices and download progress out of JSON output while retaining them in human-readable modes (#1314).

### 🔧 Internals & CI

- **Release evaluation** - adds an opt-in manifest-driven release sweep with explicit model/task, precision, and machine/EP/device targeting, while preserving non-release selection modes; cross-EP evaluation and structured perf-result handling are improved (#1411, #1281, #1316).
- **Evaluation reliability** - adds targeted timeout exclusions and failure categorization, separates download time from execution budgets, prioritizes model execution, and prevents stage logs from being mistaken for final artifact paths (#1358, #1357, #1400, #1427).
- **LLM evaluation** - adds a schema-normalized evaluation runner (#1277).
- **E2E stability** - isolates native EP CLI invocations in subprocesses, improves shared-RDP GPU coverage, and updates memory assertions to the current RAM label (#1416, #1418, #1430).
- **GitHub releases** - uses the GitHub App service connection and preserves UTF-8 release notes, including BOM-aware output for release tasks (#1292, #1293).
- **CI supply chain** - pins GitHub Actions, updates action dependencies, and introduces a seven-day Dependabot cooldown (#1408, #1409).
- **Dependency maintenance** - updates aiohttp, cryptography, Jupyter dependencies, and the Windows ML ONNX Runtime compatibility range; removes unused Jupyter server packages from the development dependency set (#1278, #1279, #1216, #1386, #1423).
- **Runtime dispatch** - centralizes perf runtime names in shared constants (#1345).
- **Contributor workflows** - adds skills for model-support contributions and correctness-gated auto-optimization (#1415, #1428).
- **Release-line synchronization** - carries the v0.3.0 and v0.3.1 release changes back to main, including the dataset-ID and native-warning-spooling fixes already shipped through earlier release cherry-picks (#1291, #1397, #1262, #1266).

### 📦 Assets

- `winml_cli-0.4.0-py3-none-any.whl`
- `rules-v0.4.0.zip`

## WinML CLI v0.3.1

This hotfix restores compatibility with current Transformers and Windows runtime dependencies, fixes VitisAI cache permissions, and stabilizes Hugging Face model export and evaluation.

### 🐛 Fixes

- **Hugging Face export and evaluation** — stabilized SDPA and attention-mask handling, decoder wrappers, BLIP export, QNN/HTP tracing, and processor resolution for Transformers 4.57 (#1372, #1379, #1384).
- **VitisAI compilation** — moved the compilation cache under the user-writable WinML cache root instead of installation-relative protected paths (#1378).
- **Runtime dependencies** — constrained `plotext` to the compatible 5.x series, required the published Windows ML runtime build, and routed QNN acquisition through the architecture-aware Windows ML EP Catalog so wheel installations preserve both QNN and DirectML without installing a conflicting standard ONNX Runtime distribution (#1372).

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
- `winml perf --runtime ort-genai` — benchmarks prebuilt or automatically cached GenAI bundles with TTFT, token throughput, prompt-template controls, EP overrides, and isolated pre-compilation (#1015, #1042, #1046, #1054, #1109).
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
