# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
