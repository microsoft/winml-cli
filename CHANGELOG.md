# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## WinML CLI v0.0.4

### 🚀 Platform upgrades

- **WinML runtime → 2.0** (#441)
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

If you set `MODELKIT_RULES_DIR` anywhere (shell profile, CI pipeline, user env), rename it to `WINMLCLI_RULES_DIR`. Same `os.pathsep`-separated multi-directory semantics; relative paths still resolve from `src/winml/modelkit/analyze/utils/`.

Related PRs: #411 (Parquet migration), #600 (rules zip in release), #627 (versioned filename), #587 (env var rename as part of ModelKit → WinML CLI Wave 1).

### 📦 Assets

- `winml_cli-0.0.3-py3-none-any.whl`
- `rules-v0.0.3.zip`
