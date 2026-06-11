# LoRA on QNN EPContext — Investigation Summary

**Date:** 2026-06-08 · **Device:** Snapdragon (Windows arm64) · **EP:** QNN on **NPU** (QnnHtp)
**ORT:** 1.24.5 · **Verdict:** all three questions **CONFIRMED**

---

## TL;DR

| Question | What was asked | Verdict |
|---|---|---|
| **Q1** | Can an adapter be attached to an *already-compiled* EPContext model and change its output? | **CONFIRMED no-op** — no |
| **Q2.a** | Does QNN JIT *bake* LoRA-as-initializers into the EPContext correctly? | **CONFIRMED** — yes |
| **Q2.b** | Does switching adapter (different initializers) force a fresh compile? | **CONFIRMED** — yes |
| **Q3** | Does LoRA-as-graph-inputs allow free runtime adapter swap on CPU **and** QNN? | **CONFIRMED** — yes |

**The one rule that explains everything:** *whatever is an initializer at compile
time is frozen into the QNN binary; whatever is a graph input stays live and
feedable at every `Run()`.*

---

## How to ship LoRA on QNN (the practical takeaway)

- **Want a fixed adapter baked into the device binary?** Put `A`/`B` as
  **initializers** and let QNN compile (Q2.a). It folds into the EPContext.
  You cannot change it afterwards without recompiling (Q2.b).
- **Want to swap adapters at runtime with no recompile?** Put `A`/`B` as
  **graph inputs** and feed adapter values from `safetensors` each `Run()`
  (Q3). This is the supported swap path on QNN, identical on CPU.
- **Never** try to attach an `OrtLoraAdapter` to a precompiled `*_ctx.onnx`
  expecting it to inject LoRA — it cannot reach inside the frozen binary (Q1).

---

## Evidence

All checks share the **same toy layer** and **same input** so outputs are directly comparable.

- **Base layer:** `y = x · W`, all tensors `8×8` fp32, deterministic seeds (`W`=seed 0, `A`=1, `B`=2, `A2`=11, `B2`=12).
- **LoRA math:** `y = x·W + (x·A)·B`, with `A:[8,4]`, `B:[4,8]` (rank 4).
- **Adapter #1** = `(A, B)`; **Adapter #2** = `(A2, B2)` — different seeds so it must yield a different output.
- **Fixed input** `x0 = arange(8)/8`:
  ```
  [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]
  ```

**CPU ground-truth outputs (exact, the reference every check compares against):**

| Run | Output `y` (8 values) |
|---|---|
| **no adapter** (`base`, `y=x·W`) | `[ 0.9756,  2.6100, -0.0734, -0.8991, -1.0413, -0.3954,  1.1583,  1.7522]` |
| **adapter #1** (`baked`) | `[ 1.5969,  2.9864, -1.2246,  3.0695, -1.7509, -0.5002,  1.4314,  1.1675]` |
| **adapter #2** (`baked2`) | `[ 2.6761, -0.6875, -1.6200, -2.4637, -1.0137,  3.5913,  3.3981,  1.6432]` |

The three rows are visibly different — that is the signal we use everywhere: *if the adapter took effect, the output moves off the `base` row toward the matching adapter row.*

---

### Q1 — adapter on a precompiled EPContext does nothing

**1. Model / adapter prepared**
- Model: `base_ctx.onnx` — the already-compiled EPContext dump of `base.onnx` (single `EPContext` node, only input `x`, **no** LoRA inside).
- Adapter: a fake `OrtLoraAdapter` built from `A`, `B`, **and a deliberately huge `W+5.0` override** — if it were honored at all, the output would shift dramatically (obvious to spot).

**2. Code**
```python
# build a fake adapter whose params WOULD change the result if applied
fmt = ort.AdapterFormat()
fmt.set_parameters({
    "A": ort.OrtValue.ortvalue_from_numpy(A),
    "B": ort.OrtValue.ortvalue_from_numpy(B),
    "W": ort.OrtValue.ortvalue_from_numpy(W + 5.0),   # large, unmistakable delta
})
fmt.export_adapter("fake.onnx_adapter")

sess = make_qnn_session(MODELS / "base_ctx.onnx", device_type=NPU)
y_plain = sess.run(None, {"x": X0})[0]            # no adapter

ad = ort.LoraAdapter(); ad.Load("fake.onnx_adapter")
ro = ort.RunOptions(); ro.add_active_adapter(ad)
y_adapter = sess.run(None, {"x": X0}, run_options=ro)[0]   # adapter "active"
```

**3. Output without vs with adapter**

| Run | Result |
|---|---|
| **without adapter** (`y_plain`) | `≈ [0.9756, 2.6100, …]` (matches CPU `base` within 6.2e-4) |
| **with adapter** (`y_adapter`) | **never produced** — `sess.run(...)` raised `InvalidArgument: [ONNXRuntimeError] : Invalid input name: A` |

**4. Description.** The compiled EPContext node exposes no LoRA-overridable input, so ORT refuses to bind the adapter parameters *before any inference runs*. There is therefore no "with-adapter" output to compare — the structural rejection is itself the proof that an adapter cannot be injected into a frozen binary. *(Nuance vs plan: the plan expected a **silent** no-op; reality is an **explicit rejection** — same conclusion, stricter mechanism.)*

---

### Q2.a — JIT bake of LoRA-as-initializers

**1. Model / adapter prepared**
- Models: `base.onnx` (no LoRA), `baked.onnx` (`A,B` as **initializers**), `baked2.onnx` (`A2,B2` as initializers).
- "Adapter" here = the initializer values themselves, frozen at compile time.

**2. Code**
```python
cfg = {"ep.context_enable": "1",
       "ep.context_file_path": str(ctx),
       "ep.context_embed_mode": "1"}
sess = make_qnn_session(MODELS / src, device_type=NPU, session_config=cfg)  # JIT compile + dump
y = sess.run(None, {"x": X0})[0]
info = inspect_model(ctx)   # EPContext / MatMul / inputs / initializers
```

**3. Output without vs with adapter** (QNN NPU, vs the CPU reference rows above)

| Run | QNN output | Max error vs CPU |
|---|---|---|
| **no adapter** (`base`) | matches `base` row | 6.2e-4 |
| **adapter #1 baked** (`baked`) | matches `baked` row | 2.7e-3 |
| **adapter #2 baked** (`baked2`) | matches `baked2` row | 3.4e-3 |
| `baked` vs `baked2` | — | **5.54** (the two adapters clearly diverge) |

**4. Description.** Each `*_ctx.onnx` collapses to **1 `EPContext`, 0 `MatMul`, inputs `["x"]`, no `A`/`B` initializers** — the LoRA branch was absorbed into the QNN binary. The output tracks the matching CPU adapter row (within NPU/fp16 tolerance), and the two baked models differ by 5.54, proving each adapter was genuinely baked in.

---

### Q2.b — switching a baked adapter forces a fresh compile

**1. Model / adapter prepared.** Same `baked.onnx` / `baked2.onnx` from Q2.a (LoRA as initializers).

**2. Code**
```python
for src in ["baked.onnx", "baked2.onnx", "baked.onnx"]:   # x3 trials
    t0 = time.perf_counter()
    sess = make_qnn_session(MODELS / src, device_type=NPU)  # no caching options
    build_s = time.perf_counter() - t0
    sess.run(None, {"x": X0}); del sess
```

**3. Output without vs with adapter.** Output values are the same `baked` / `baked2` rows as Q2.a — the focus here is **cost**, not value:

| Step | Fresh build time |
|---|---|
| `baked` → `baked2` → `baked` (median per build) | **0.50 – 0.66 s each** |

**4. Description.** Every adapter change requires building a brand-new QNN session — even returning to a previously-used `baked.onnx` re-pays the full ~0.5–0.7 s compile. There is no in-session swap path when the adapter is an initializer.

---

### Q3 — graph-input LoRA + the Appendix A user script (free runtime swap)

**1. Model / adapter prepared**
- Model: `switchable_peft.onnx` — `A`/`B` are real **graph inputs** named `layer0.weight_lora_A [8,4]` and `layer0.weight_lora_B [4,8]`; only `W` is an initializer.
- Adapters: real **PEFT-style** `adapter.safetensors` (#1) and `adapter2.safetensors` (#2), loaded through the **verbatim Appendix A loader** (`inference/lora_loader.py`): PEFT-name regex + `.T` transpose + `α/r` scaling fold.

**2. Code**
```python
names = [i.name for i in sess.get_inputs()]
lora1 = load_adapter("adapter.safetensors",  names)   # {layer0.weight_lora_A: …, _B: …}
lora2 = load_adapter("adapter2.safetensors", names)

y1 = sess.run(None, {"x": X0, **lora1})[0]   # feed adapter #1
y2 = sess.run(None, {"x": X0, **lora2})[0]   # swap to adapter #2 — same session
y3 = sess.run(None, {"x": X0, **lora1})[0]   # swap back — still no recompile
```

**3. Output without vs with adapter**

| Run | Output | Match |
|---|---|---|
| **adapter #1** (`y1`, CPU) | `[ 1.5969,  2.9864, …]` | = CPU `baked`, err **1.2e-7** |
| **adapter #2** (`y2`, CPU) | `[ 2.6761, -0.6875, …]` | = CPU `baked2`, err **2.4e-7** |
| **adapter #1** (QNN NPU) | tracks `baked` row | err **1.7e-3** |
| **adapter #2** (QNN NPU) | tracks `baked2` row | err **3.4e-3** |

Feeding adapter #1 reproduces the `baked` output; swapping to adapter #2 *in the same session* reproduces the `baked2` output (the two differ by 5.54). Swapping back to #1 returns the `baked` output again.

**4. Description.** Because the LoRA tensors are graph inputs, each `Run()` can feed different adapter weights with **no recompile**: per-run swaps measured **7–47 ms** against a one-time **~600 ms** session build → `no_recompile_on_swap = True`. The AOT dump `switchable_ctx.onnx` is still 1 `EPContext` / 0 `MatMul`, but keeps the LoRA tensors **live as `ctx_inputs`**: `["x", "layer0.weight_lora_A", "layer0.weight_lora_B"]` — i.e. the QNN binary was compiled with those weights left feedable.

---

## Environment deviations (full detail in `NOTES.md`)

1. **QNN is a WinML plugin EP here**, not a `backend_path` provider. Registered via
   `windowsml.EpCatalog` → `register_execution_provider_library` →
   `SessionOptions.add_provider_for_devices` (mirrors the repo's
   `winml.add_ep_for_device`). The plan's
   `providers=[("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})]` does **not** apply.
2. **Ran on NPU/QnnHtp** (per request), which executes fp32 at **fp16** → numeric
   tolerance loosened to `1e-2` (observed max 3.4e-3). Structural claims unchanged.
3. **ORT 1.24.5** (plan assumed 1.23). LoRA API:
   `ort.AdapterFormat().set_parameters({name: OrtValue}).export_adapter(path)` →
   `ort.LoraAdapter().Load(path)` → `RunOptions.add_active_adapter(...)`.
4. **Q3 used the recommended PEFT path**: built `switchable_peft.onnx`
   (`layer0.weight_lora_A/_B` inputs) + PEFT `adapter.safetensors` so the
   Appendix A loader runs **unmodified**.

---

## Deliverables

| Path | What |
|---|---|
| `investigation/results.json` | `env`, `q1__*`, `q2_a__*`, `q2_b__*`, `q3__*`, `verdict` |
| `investigation/NOTES.md` | every deviation from the plan |
| `investigation/*.py` | `env_check`, `qnn_common`, `build_models`, `run_cpu`, `run_qnn_jit`, `run_qnn_aot`, `run_user_script`, `inspect_graph`, `write_results` |
| `inference/lora_loader.py` | verbatim Appendix A loader |
| `models/lora_test/*.onnx` | source models + `base/baked/baked2/switchable` `_ctx.onnx` AOT dumps |
| `models/lora_test/adapter*.safetensors` | fake PEFT adapters |

*Nothing was installed; nothing was committed.*
