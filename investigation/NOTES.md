# NOTES — deviations from the plan (LoRA on QNN EPContext)

Date: 2026-06-08. Device: Snapdragon (Windows arm64). All three questions
(Q1/Q2/Q3) were **CONFIRMED**, but several mechanics differed from the plan's
assumptions. Per operating rule 1, contradictions are recorded here rather than
silently "fixed".

## 1. ORT version / workspace
- Plan assumed ORT **1.23**; the prepared interpreter is **onnxruntime 1.24.5**
  (x64 emulated Python 3.11.9), `onnx` present, `safetensors` present.
- Workspace is `c:\Users\zhenni\repos\wmk` (plan was drafted for
  `qwen3_0.6b_bundle`). Cosmetic only — no real models were used.

## 2. QNN registration — the big one (Section 0/Step 0/8.2 deviation)
The plan's
`providers=[("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})]` does
**NOT** apply to this build. `QNNExecutionProvider` is **not** in
`ort.get_available_providers()`; QNN ships here as a **WinML plugin EP** (MSIX:
`MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_2.2450.47.0_arm64`).

Registration path actually used (mirrors the repo's
`winml.add_ep_for_device`, marked "NEVER modify"):
```python
from windowsml import EpCatalog
with EpCatalog() as cat:
    for p in cat.find_all_providers():
        if p.name == "QNNExecutionProvider":
            p.ensure_ready()
            ort.register_execution_provider_library(p.name, p.library_path)
devs = {d.device.type: d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"}
so = ort.SessionOptions()
so.add_provider_for_devices([devs[ort.OrtHardwareDeviceType.NPU]], {})
```
EPContext dump still works via the standard session-config entries
(`ep.context_enable` / `ep.context_file_path` / `ep.context_embed_mode`) with
this registration. See `investigation/qnn_common.py`.

## 3. Target device = NPU (QnnHtp), not QnnCpu.dll
Per the user's explicit request, all QNN runs targeted the **NPU** device
class. QNN exposes three EpDevices here: NPU, GPU, CPU. The NPU backend
(QnnHtp) executes the fp32 toy graph at **fp16 precision**, so numeric error is
larger than the plan's `atol=1e-4` guesses.

- Observed max abs error vs CPU reference: **≤ 3.4e-3** across Q2/Q3.
- Recorded tolerance loosened to `ATOL_QNN = 1e-2` (operating rule 2,
  defensible: HTP fp16). CPU EP stays exact (`ATOL_CPU = 1e-5`).
- **Structural** claims (EPContext count, MatMul=0, input lists, absorbed
  initializers) are unchanged and are the load-bearing ones — all held exactly.
- This is effectively the Section 6 "HTP follow-up" promoted to the primary
  run. No separate QnnCpu.dll run was performed (this ORT build has no
  `backend_path` entry point; the QNN CPU EpDevice was available but the user
  asked for NPU).

## 4. OrtLoraAdapter Python API (ORT 1.24.5)
Discovered via `dir(onnxruntime)` / `help(...)`:
```python
fmt = ort.AdapterFormat()
fmt.set_parameters({name: ort.OrtValue.ortvalue_from_numpy(arr)})
fmt.export_adapter(path)                 # writes *.onnx_adapter
ad = ort.LoraAdapter(); ad.Load(path)    # memory-maps the file
ro = ort.RunOptions(); ro.add_active_adapter(ad)
sess.run(None, feeds, run_options=ro)
```

## 5. Q1 nuance — rejection, not silent no-op (recorded contradiction)
Plan §1.4 predicted the adapter override would **succeed silently** and simply
not change the output. Reality on this build: attaching an adapter whose
parameters (`A`/`B`/`W`) do **not** name actual graph inputs of the precompiled
`base_ctx.onnx` (inputs = `["x"]`) raises:
```
InvalidArgument: Invalid input name: A
```
i.e. ORT validates adapter parameter names against the model's input list and
**rejects** unknown ones. The plan's conclusion — *you cannot inject/replace
LoRA inside a precompiled EPContext on QNN* — is **CONFIRMED**, but the
mechanism is an explicit validation failure rather than a silent ignore. The
EPContext node exposes no LoRA-overridable input, so there is nothing for the
adapter to bind to. Verdict recorded as `CONFIRMED no-op` (output unchanged;
only the plain run is valid).

## 6. Q3 used the "more realistic" PEFT path (Section 0 Q3.a, recommended option)
- The verbatim Appendix A loader (`inference/lora_loader.py`) keys off ONNX
  input names ending in `_lora_A` / `_lora_B`. The canonical `switchable.onnx`
  (Section 3.2) names its inputs literally `A`/`B`, which the loader would skip
  — the "PEFT name parsing fails" case the task anticipates.
- Rather than alter the loader, I built `switchable_peft.onnx` with inputs
  `layer0.weight_lora_A` `[8,4]` / `layer0.weight_lora_B` `[4,8]` and PEFT-style
  `adapter.safetensors` / `adapter2.safetensors`
  (`base_model.model.layer0.lora_{A,B}.weight`), plus `adapter_config.json`
  with `lora_alpha=4, r=4` (scaling = 1.0). The verbatim loader then runs
  **unmodified** (regex match + `.T` transpose + scaling fold into B).
- Consequence: `switchable_ctx.onnx` `ctx_inputs` are
  `["x", "layer0.weight_lora_A", "layer0.weight_lora_B"]` rather than the
  illustrative `["x","A","B"]` in the plan's table — structurally identical
  (3 inputs incl. the two live LoRA tensors), just PEFT-named.
- The canonical `switchable.onnx` (inputs `A`/`B`) is retained and validated
  for math-equivalence to `baked.onnx` in `run_cpu.py`.

## 7. Timing notes (operating rule 4)
- `time.perf_counter`, ≥3 trials, median reported. QNN-NPU per-`Run()` medians
  for Q3 swaps are single/double-digit ms (7–47ms) vs a ~500–680ms session
  compile — `no_recompile_on_swap=True`, `compile_dominates=True`. Run timings
  are noisy (first-iteration warmup, x64 emulation) but always an order of
  magnitude below compile.
- Q2.b: every fresh session build (incl. re-running `baked.onnx` after
  `baked2.onnx`) costs ~0.5–0.7s — no free recompile, no in-session swap path
  for initializer-based LoRA. Confirmed.

## Artifacts
- `investigation/`: `env_check.py`, `qnn_common.py`, `build_models.py`,
  `run_cpu.py`, `run_qnn_jit.py`, `run_qnn_aot.py`, `run_user_script.py`,
  `inspect_graph.py`, `write_results.py`, `results.json`, `NOTES.md`.
- `inference/lora_loader.py` (verbatim Appendix A).
- `models/lora_test/`: `base.onnx`, `baked.onnx`, `baked2.onnx`,
  `switchable.onnx`, `switchable_peft.onnx`, `*_ctx.onnx` (base/baked/baked2/
  switchable), `adapter.safetensors`, `adapter2.safetensors`,
  `adapter_config.json`, `weights.npz`, `cpu_refs.npz`, `qnn_jit_outputs.npz`,
  `fake.onnx_adapter`.
