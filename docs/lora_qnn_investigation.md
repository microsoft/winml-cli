# LoRA on QNN EPContext — Investigation Plan

> **Audience:** an automated agent (or engineer) running on a Snapdragon /
> QNN‑capable Windows device.
> **Author host:** this plan was drafted on a CPU‑only machine, so none of the
> steps below have been executed. Treat every numeric tolerance as a starting
> guess to be tightened/loosened from real measurements.
> **Scope:** validate, with minimal code, how LoRA adapters interact with the
> ONNX Runtime QNN Execution Provider when the model is compiled into an
> `EPContext` node (either JIT at session init, or AOT into a `_ctx.onnx`).

---

## 0. Things to confirm (read this first)

This plan exists to answer **three specific questions** about how LoRA
adapters interact with ONNX Runtime + QNN. Every script, model, and
assertion below maps to one of them. If a step does not contribute to Q1,
Q2, or Q3, it is out of scope.

> **No real models or weights are required.** Despite the workspace being
> named `qwen3_0.6b_bundle`, this investigation deliberately uses **only a
> toy 8×8 fp32 model built in‑process from numpy via `onnx.helper`** (see
> Section 3). Do **not** download Qwen3 (or any HuggingFace model), do not
> use any `.safetensors` from disk other than the small ones you generate
> yourself in Phase Q3.a from `weights.npz`. The toy model's input is
> literally named `"x"`, not `"input_ids"` — the `input_ids`/`attention_mask`
> snippet in Appendix A is illustrative of the LLM end‑user pattern, not
> what Phase Q3 actually feeds.

### Q1 — Precompiled model + adapter: expected to **NOT** work

> *If I have already compiled the model into an EPContext (`_ctx.onnx`) on
> QNN, can I attach an adapter at inference time and have it actually
> affect the output?*

**Expected answer: no.** The EPContext node is an opaque QNN binary built
before the adapter existed; ORT has no hook to mutate its frozen weights at
`Run()` time, and the `OrtLoraAdapter` API only overrides ONNX‑visible
initializers — of which the EPContext node has none.

**Methodology:** start from a precompiled `base_ctx.onnx` (no LoRA inside).
Attempt to attach a LoRA adapter via `OrtLoraAdapter` at `Run()` time.
Compare the output to a plain run on the same EPContext model. If the two
runs produce identical tensors, the adapter is being ignored — confirming
Q1.

### Q2 — Uncompiled model + adapter, JIT compiled by QNN: behavior unknown

> *If I give QNN an ONNX model that already has the LoRA branch in it (with
> `A`, `B` as initializers) and let the QNN EP JIT‑compile it during
> session creation, does the compiled EPContext correctly include the LoRA
> effect? And if I want to use a different adapter, does that trigger
> another JIT compile?*

**Expected answer:**
 - **Part A — numeric:** JIT QNN should bake the LoRA branch into the
   resulting EPContext exactly like a normal constant subgraph. Output
   should match the CPU EP reference on the same model.
 - **Part B — switching:** because the adapter values are *initializers* of
   the source ONNX, switching to a different adapter requires producing a
   different ONNX, which in turn forces a fresh JIT compile in a new
   session. There is **no in‑session swap path** for this representation.

**Methodology:**
 - Build `baked.onnx` (LoRA as initializers).
 - Run on **CPU EP** → record `y_cpu_baked` and inspect the live graph.
 - Run on **QNN EP** with `ep.context_enable=1` → record `y_qnn_baked`,
   dump `baked_ctx.onnx`, and inspect: it must be a single `EPContext`
   node, no remaining `MatMul`s, no `A`/`B` in `graph.initializer`.
 - Assert `np.allclose(y_qnn_baked, y_cpu_baked)`.
 - Build `baked2.onnx` (same model, different `A2`/`B2` initializer values)
   and instantiate a **new** QNN session — observe and time the compile
   step. Repeat the session creation twice and confirm the compile cost
   recurs (no cross‑session reuse without `ep.context_*` caching).

### Q3 — LoRA promoted to model inputs: does the "user script" work?

> *If I instead rewrite the model so `A` and `B` are real ONNX graph
> inputs, can a normal user feed adapter values from a `safetensors` file
> at inference time, on both CPU and QNN, with the loader code in
> Appendix A?*

**Expected answer: yes.** The EPContext node carries `A` and `B` as inputs;
feeding different values at `Run()` time switches adapters with no
recompile. This is the OpenVINO‑style mechanism applied to ONNX.

**Methodology:**
 - Build `switchable.onnx` (LoRA as graph inputs).
 - Build a fake `adapter.safetensors` whose values, when loaded by the
   Appendix A loader, equal the same `A`/`B` used in `baked.onnx`.
 - Run the **user‑facing call** from Appendix A on CPU EP → output must
   match `y_cpu_baked`.
 - Run the same call on QNN EP (JIT, no `ep.context_*` flags first, then
   with the AOT dump) → output must match `y_cpu_baked`.
 - Swap to a *second* adapter file (different values). The same session
   must accept it with **no recompile** — confirm by timing two
   back‑to‑back `Run()` calls with different adapter overlays.

### What is *not* a question here

- Quantized / HTP execution (mentioned as optional in Section 6).
- Performance comparison between Q2 and Q3.
- Adapter blending, multi‑adapter management, real Qwen3 weights.

---

## 1. Background and mechanism

### 1.1 What `EPContext` is

When an Execution Provider (EP) like **QNN**, **TensorRT**, or
**OpenVINO‑with‑cache** claims a subgraph, ONNX Runtime asks the EP to
**compile** that subgraph into a backend‑specific binary (for QNN: an HTP/CPU
context blob produced by the QNN SDK). The compiled blob is then represented in
the ONNX graph as a single node:

```
op_type = "EPContext"
domain  = "com.microsoft"
attrs   = { ep_cache_context: <bytes or file path>,
            embed_mode: 0|1,
            partition_name: "...",
            source: "QNN", ... }
inputs  = [ <the original graph inputs that fed this subgraph> ]
outputs = [ <the original outputs of this subgraph> ]
```

Two ways to obtain such a model:

| Mode | When compile happens | What you ship |
|---|---|---|
| **JIT** | At `InferenceSession(...)` construction on the target device | Original ONNX; binary is built each session unless you also dump it |
| **AOT** | Offline, by enabling EPContext dump session options and running once | A new ONNX where the compiled subgraph(s) are replaced by `EPContext` nodes |

The dump is triggered with:

```python
so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", "model_ctx.onnx")
so.add_session_config_entry("ep.context_embed_mode", "1")  # binary inline
```

The first time you instantiate a session with these options, the EP compiles
and ORT writes `model_ctx.onnx`. From then on, loading `model_ctx.onnx` just
restores the QNN context — no re‑compile.

### 1.2 What gets baked into the context

At the moment QNN's `Compile` callback runs, the EP can see, for the claimed
subgraph:

- **Initializers** (constants). These are read out and embedded into the QNN
  graph as weights/parameters. After compile they are *gone* — they live only
  inside the binary blob.
- **Graph inputs** that flow into the subgraph. These remain inputs of the
  resulting `EPContext` node and must be fed at every `Run()`.
- **Op structure**. The QNN compiler is free to constant‑fold, fuse, and
  re‑layout aggressively across constants. From the ORT graph after compile you
  will not see the original MatMul/Add ops at all — just one `EPContext` node.

Consequence: **whatever is an initializer at compile time is frozen; whatever
is a graph input stays live.** This single rule explains every LoRA behavior
that follows.

### 1.3 The two LoRA representations in ONNX

Let `W ∈ R^{in×out}` be a base linear weight. A LoRA adapter adds a low‑rank
correction `ΔW = B·A` with `A ∈ R^{in×r}`, `B ∈ R^{r×out}` and a scaling
`s = α/r` typically folded into `A` or `B`. There are two ways to express this
in an ONNX graph:

#### (a) LoRA as **initializers** ("fixed adapter")

```
y = MatMul(x, W) + MatMul(MatMul(x, A_const), B_const)
                       ↑                ↑
                       initializer      initializer
```

QNN at compile time sees three constant matmuls touching one activation. It
will fold them — at the limit, into a single effective `W' = W + B·A` weight
inside the context. Nothing remains to feed at runtime. Cannot be changed
without recompiling.

#### (b) LoRA as **graph inputs** ("switchable adapter")

```
y = MatMul(x, W) + MatMul(MatMul(x, A_input), B_input)
                       ↑                ↑
                       graph input      graph input
```

QNN compiles a graph that expects `A_input` and `B_input` as runtime tensors.
The `EPContext` node carries them as inputs. Different adapter = different
tensor values fed to `session.Run({..., "A_input": ..., "B_input": ...})`.

This is exactly what
[`add_lora_inputs_to_unquantized_model`](../lm_to_onnx/onnx_utils/onnx_surgeries.py#L226)
does in this repo.

### 1.4 Why `OrtLoraAdapter` does not help on QNN

ORT 1.20 introduced `OrtLoraAdapter` + the `lora_parameters` selected‑adapters
run option. Its mechanism is: at `Run()` time, **override the values of named
initializers** of the main model with values from the adapter file.

- On CPU/CUDA the EPs consult initializers on every run, so the override takes
  effect.
- On QNN (and any EP that subgraph‑compiles), the initializers were consumed at
  **compile** time. Overriding them later changes nothing about what the
  pre‑compiled QNN binary executes. The API call succeeds; the output does not
  change. This is true **whether the compile was JIT or AOT** — the
  jit‑vs‑aot distinction only changes *when* the bake happens, not *whether*.

Therefore on QNN the only way to switch adapters is the graph‑input pattern
(1.3b). The only way to "fix" an adapter into a context is the initializer
pattern (1.3a) or a pre‑merge `W' = W + BA` in PyTorch before export.

---

## 2. Hypotheses (restated, one per question in Section 0)

> **H1 ↔ Q1 (irreversibility):** Once a model has been collapsed into an
> `EPContext` node, neither `OrtLoraAdapter` nor any other runtime override
> can inject or replace LoRA weights inside that node. Adapter substitution
> post‑compile is impossible on QNN.
>
> **H2 ↔ Q2 (JIT bake works, but switching is not in‑session):**
> *(a)* If LoRA `A`/`B` are present as **initializers** in an ONNX model, the
> QNN EP at JIT session‑init folds the LoRA branch into the compiled
> `EPContext` node. The resulting binary produces the LoRA‑adjusted output,
> matching the CPU EP reference on the same model.
> *(b)* Switching to a different adapter requires producing a different
> ONNX (different initializer values) and creating a new session, which
> triggers a fresh JIT compile.
>
> **H3 ↔ Q3 (graph‑input LoRA is the supported swap path):** If `A`/`B` are
> instead **graph inputs** at compile time, JIT and AOT QNN both produce an
> `EPContext` node whose input list includes them. The user‑facing loader
> in Appendix A is sufficient to feed values from a `safetensors` adapter,
> on both CPU and QNN, and swapping adapters within one session triggers no
> recompile.

The one‑word verdict of the whole experiment is whichever of
`{H1, H2, H3} ∈ {confirmed, falsified}` we end up writing into
`results.json` (Section 5).

---

## 3. Test artifacts

Create the following layout under the workspace:

```
investigation/
  build_models.py        # builds the 3 source ONNX models
  run_cpu.py             # reference outputs on CPU EP
  run_qnn_jit.py         # JIT compile on QNN + dump EPContext
  run_qnn_aot.py         # load dumped EPContext, attempt adapter overrides
  inspect.py             # asserts node op_types / inputs in produced ONNX
  results.json           # captured numbers (filled by the run scripts)
models/lora_test/
  base.onnx              # y = W·x                          (no LoRA at all)
  baked.onnx             # y = W·x + (x·A)·B  (A,B init.)   (fixed LoRA)
  switchable.onnx        # y = W·x + (x·A)·B  (A,B input)   (graph-input LoRA)
  base_ctx.onnx          # AOT EPContext dump of base.onnx
  baked_ctx.onnx         # AOT EPContext dump of baked.onnx
  switchable_ctx.onnx    # AOT EPContext dump of switchable.onnx
```

### 3.1 Model shapes (tiny, deterministic)

| Symbol | Shape | Source |
|---|---|---|
| `x` | `[1, 8]` | graph input, fp32 |
| `W` | `[8, 8]` | initializer, seeded RNG (e.g. `np.random.default_rng(0).standard_normal(...).astype(np.float32)`) |
| `A` | `[8, 4]` | seeded RNG `(seed=1)` |
| `B` | `[4, 8]` | seeded RNG `(seed=2)` |
| `y` | `[1, 8]` | output |

Rank `r = 4`. Keep everything **fp32** so the QNN CPU backend can run it
without any QDQ work. (HTP requires quantization; we are not doing HTP in this
plan.)

Use `onnx.helper` directly, opset 17 or 20 — no PyTorch dependency needed.

### 3.2 The three source models

**`base.onnx`** — sanity baseline:
```
inputs : x [1,8]
inits  : W [8,8]
nodes  : MatMul(x, W) -> y
outputs: y [1,8]
```

**`baked.onnx`** — LoRA as initializers (H1 target):
```
inputs : x [1,8]
inits  : W [8,8], A [8,4], B [4,8]
nodes  :
  t0 = MatMul(x,  A)        # [1,4]
  t1 = MatMul(t0, B)        # [1,8]
  t2 = MatMul(x,  W)        # [1,8]
  y  = Add(t2, t1)
outputs: y [1,8]
```

**`switchable.onnx`** — LoRA as graph inputs (H3 target):
```
inputs : x [1,8], A [8,4], B [4,8]
inits  : W [8,8]
nodes  : same MatMul/MatMul/Add as above
outputs: y [1,8]
```

All three should produce mathematically identical outputs on CPU when given the
same `W`, `A`, `B`. Use the same RNG seeds so the values agree.

---

## 4. Execution plan (step by step)

The plan is organised so each phase answers one of the three questions from
Section 0. Steps 0–2 set up shared artifacts; **Phase Q2**, **Phase Q1**,
**Phase Q3** then run independently. (Q2 before Q1 because Q1 needs the
AOT‑dumped EPContext that Phase Q2 produces.)

### Step 0 — Environment

Already present in this repo: a `.venv` with Python 3.11 and
`onnxruntime-qnn==1.23`, `onnx==1.19.1`, `numpy`. Sanity‑check:

```powershell
cd C:\repos\qwen3_0.6b_bundle
.venv\Scripts\python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Expected: `"QNNExecutionProvider"` is in the list. Also `pip install safetensors`
if not already present — needed for Phase Q3.

### Step 1 — Build source models (`build_models.py`)

Pure `onnx.helper` + `numpy_helper.from_array`. Save with
`save_as_external_data=False`. Validate with `onnx.checker.check_model`.

Produce all of:
 - `base.onnx` (sanity baseline)
 - `baked.onnx`  (LoRA initializers — for **Q2**)
 - `baked2.onnx` (same shape as `baked.onnx` but `A2`, `B2` from a different
   RNG seed — for the Q2 "different adapter = new compile" experiment)
 - `switchable.onnx` (LoRA as graph inputs — for **Q3**)

Also write the raw numpy arrays of `A`, `B`, `A2`, `B2` to
`models/lora_test/weights.npz` so Phase Q3 can construct a fake
`adapter.safetensors` containing the same values.

### Step 2 — CPU reference (`run_cpu.py`)

For a fixed input `x0 = np.arange(8, dtype=np.float32).reshape(1, 8) / 8`:

1. Run `base.onnx` on CPU EP → `y_cpu_base`.
2. Run `baked.onnx` on CPU EP → `y_cpu_baked`.
3. Run `baked2.onnx` on CPU EP → `y_cpu_baked2`.
4. Run `switchable.onnx` on CPU EP with feeds `{x, A, B}` (same values used
   to build `baked.onnx`) → `y_cpu_switch`.
5. Assertions:
   - `np.allclose(y_cpu_baked, y_cpu_switch, atol=1e-6)` — the two LoRA
     representations are mathematically equivalent.
   - `not np.allclose(y_cpu_base, y_cpu_baked, atol=1e-3)` — LoRA actually
     changes the output.
   - `not np.allclose(y_cpu_baked, y_cpu_baked2, atol=1e-3)` — the second
     adapter produces a *different* output (so Phase Q2 can detect it).
6. Persist all four outputs to `results.json` (or `.npz`).

---

### Phase Q2 — Uncompiled model + adapter via JIT QNN (`run_qnn_jit.py`)

**Goal:** confirm that (a) JIT QNN correctly bakes a LoRA branch present as
initializers into the EPContext, and (b) switching adapter requires a fresh
compile.

**Q2.a — JIT compile, dump, and verify numeric correctness.**

For each of `baked.onnx`, `baked2.onnx`, `base.onnx`:

```python
import onnxruntime as ort, time

so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", out_ctx_path)
so.add_session_config_entry("ep.context_embed_mode", "1")

providers = [("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})]
t0 = time.perf_counter()
sess = ort.InferenceSession(src_path, so, providers=providers)
t_compile = time.perf_counter() - t0
y_qnn = sess.run(None, {"x": x0})[0]
```

> If `QnnCpu.dll` is not on `PATH`, use the absolute path that shipped with
> `onnxruntime-qnn` (typically under
> `.venv\Lib\site-packages\onnxruntime\capi\`). On an HTP device you can
> additionally try `"QnnHtp.dll"` — see Section 6.

Record `y_qnn_base`, `y_qnn_baked`, `y_qnn_baked2`, and `t_compile_*`
for each into `results.json`.

Then run the **inspection step** (see `inspect.py`, Section 4.x below) on
the three dumped `*_ctx.onnx` files. Required structural assertions for Q2:

| File | `EPContext` count | `MatMul` count | EPContext input names | `A`/`B` in initializer |
|---|---|---|---|---|
| `base_ctx.onnx`   | ≥ 1 | 0 | `["x"]` | n/a |
| `baked_ctx.onnx`  | ≥ 1 | 0 | `["x"]` | **absent** (absorbed) |
| `baked2_ctx.onnx` | ≥ 1 | 0 | `["x"]` | **absent** (absorbed) |

Numeric assertions for Q2.a:

| Comparison | Expected | Meaning |
|---|---|---|
| `np.allclose(y_qnn_base,   y_cpu_base,   atol=1e-4)` | True | QNN CPU baseline correct |
| `np.allclose(y_qnn_baked,  y_cpu_baked,  atol=1e-4)` | True | **LoRA effect survived JIT bake** |
| `np.allclose(y_qnn_baked2, y_cpu_baked2, atol=1e-4)` | True | Second adapter also baked correctly |
| `not np.allclose(y_qnn_baked, y_qnn_baked2, atol=1e-3)` | True | The two adapters really do differ end‑to‑end |

**Q2.b — Different adapter = different compile.**

In a single Python process:

```python
providers = [("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})]

# No EPContext caching options here — we want each session to compile fresh.
for src in ["baked.onnx", "baked2.onnx", "baked.onnx"]:
    t0 = time.perf_counter()
    sess = ort.InferenceSession(src, providers=providers)
    dt = time.perf_counter() - t0
    print(f"{src}: session-build = {dt*1000:.1f} ms")
    y = sess.run(None, {"x": x0})[0]
```

Expected observation: session construction time is **non‑trivial for every
session** (tens to hundreds of ms even on the toy model). The QNN EP
compiles `baked.onnx` and `baked2.onnx` independently because their
initializer values differ — there is no in‑process "swap A/B values" path
for initializer‑based LoRA. Record the timings in `results.json` under
`q2_b.compile_times_ms`.

If the third call (re‑running `baked.onnx`) is dramatically faster than the
first, ORT/QNN is doing some in‑process caching; note it but it does not
falsify Q2.b (the point is that *no recompile* is impossible without
graph‑input LoRA).

---

### Phase Q1 — Adapter on a precompiled EPContext is a no‑op (`run_qnn_aot.py`)

**Goal:** confirm that attaching a LoRA adapter to a model that has *already*
been collapsed into an EPContext node does nothing.

Start from `base_ctx.onnx` produced in Phase Q2 (this is an EPContext model
with **no** LoRA inside).

**Q1.a — `OrtLoraAdapter` override on EPContext model — must be a no‑op.**

Build a fake `OrtLoraAdapter` file that, if it worked, would change `W`. The
ORT Python API:

```python
adapter = ort.AdapterFormat()  # exact API spelling may differ; see ORT docs
# ... or use the helper script `onnxruntime.tools.convert_lora_to_onnx_adapter`
ro = ort.RunOptions()
ro.add_active_adapter(adapter)
y = sess.run(None, {"x": x0}, run_options=ro)
```

> Implementation note for the executing agent: in ORT 1.23 the Python entry
> points may be `ort.LoraAdapter.create_from_array(...)` /
> `RunOptions.add_active_adapter(...)`. If the exact names differ, consult
> `import onnxruntime; help(onnxruntime)` on the target machine and adapt.

Compare `y` against `y_qnn_base` from Phase Q2:
- **Expected:** `np.allclose(y, y_qnn_base)` — the override did nothing.
  This confirms **H1 ↔ Q1**.
- **If different:** Q1 is falsified for this configuration; capture the diff
  and stop — that's a significant finding worth a separate writeup.

**Q1.b — Bolting LoRA *outside* the EPContext (informational only).**

For completeness, demonstrate the only workable post‑hoc route: open
`base_ctx.onnx` with `onnx.helper`, insert `MatMul/MatMul/Add` nodes around
the `EPContext` node, save as `base_ctx_plus_lora.onnx`, and run. The extra
ops will fall back to **CPU EP** (they are not inside the QNN binary).

- This is **not** "adding LoRA into the EPContext" — it is adding LoRA
  *around* a black box. Log it explicitly so nobody mistakes the working
  numeric result for a refutation of Q1.

---

### Phase Q3 — Graph‑input LoRA + the Appendix A user script (`run_user_script.py`)

**Goal:** confirm that the end‑user inference code from **Appendix A** works
as advertised on both CPU and QNN, and that swapping the adapter inside one
session triggers no recompile.

**Q3.a — Build a fake `adapter.safetensors`.**

Using the `A`, `B` arrays saved in Step 1's `weights.npz`, write a PEFT‑style
safetensors file whose keys, when stripped by the loader regex in
Appendix A, map to the input names of `switchable.onnx`. Because
`switchable.onnx` was built by hand and its inputs are literally named
`"A"` and `"B"` (not the long `layers.0.self_attn.q_proj.weight_lora_A`
strings the production loader expects), you have two options:

 - **Simplest:** stub `load_adapter` for this test so it just returns
   `{"A": A_np, "B": B_np}` from the npz file. This still exercises the
   *invocation pattern* in the appendix — what `feeds = {..., **lora_feeds}`
   looks like at the call site.
 - **More realistic:** rebuild `switchable.onnx` with input names like
   `"layer0.weight_lora_A"` / `"layer0.weight_lora_B"` and a safetensors
   file using PEFT‑style keys (`base_model.model.layer0.lora_A.weight`,
   etc.). This exercises the full loader, including the name‑mapping regex
   and the `.T` transpose. Recommended if time permits.

Do the same for the second adapter (`A2`, `B2` → `adapter2.safetensors`).

**Q3.b — Run the Appendix A user script on CPU.**

Literally execute the call from Appendix A §"The actual user call" against
`switchable.onnx`. Then repeat with `adapter2.safetensors`. Assert:
 - `np.allclose(y, y_cpu_baked)`  for adapter #1
 - `np.allclose(y, y_cpu_baked2)` for adapter #2

**Q3.c — Run the same script on QNN (JIT, in‑session swap).**

Replace the `providers=` line with the QNN one. Same call, same loader.
Measure:

```python
t0 = time.perf_counter()
sess = ort.InferenceSession("switchable.onnx", providers=qnn_providers)
t_compile = time.perf_counter() - t0

t1 = time.perf_counter(); y1 = sess.run(None, {**base_feeds, **lora1})[0]; dt1 = time.perf_counter() - t1
t2 = time.perf_counter(); y2 = sess.run(None, {**base_feeds, **lora2})[0]; dt2 = time.perf_counter() - t2
t3 = time.perf_counter(); y3 = sess.run(None, {**base_feeds, **lora1})[0]; dt3 = time.perf_counter() - t3
```

Assertions for Q3.c:

| Check | Expected | Meaning |
|---|---|---|
| `np.allclose(y1, y_cpu_baked,  atol=1e-4)` | True | Adapter 1 works on QNN |
| `np.allclose(y2, y_cpu_baked2, atol=1e-4)` | True | Adapter 2 works on QNN |
| `np.allclose(y3, y_cpu_baked,  atol=1e-4)` | True | Round‑trip back to adapter 1 still correct |
| `dt1 ≈ dt2 ≈ dt3` (within an order of magnitude of each other) | True | **No recompile on adapter swap** |
| `t_compile ≫ max(dt1, dt2, dt3)` | True | The one compile happened at session construction, not per Run |

**Q3.d — Optional: AOT EPContext dump still supports the same swap.**

Repeat Q3.c but with `ep.context_enable=1` to dump `switchable_ctx.onnx`,
then open the dumped model in a *fresh* session and rerun the two swap
calls. Assertions identical to Q3.c. Confirms that whether QNN compiles
JIT or AOT does not affect the swap mechanism.

---

### Step 4.x — Shared inspection helper (`inspect.py`)

Used by both Phase Q2 and Phase Q3.d:

```python
import onnx
m = onnx.load(path)
ops = [n.op_type for n in m.graph.node]
ep_ctx_nodes = [n for n in m.graph.node if n.op_type == "EPContext"]
matmul_nodes = [n for n in m.graph.node if n.op_type == "MatMul"]
for n in ep_ctx_nodes:
    print(path, "EPContext inputs:", list(n.input))
print(path, "initializers:", [t.name for t in m.graph.initializer])
```

This is what produces the rows of the tables in Phases Q2 and Q3.

### Step 7 — Write results

Append to `results.json`:

```json
{
  "env": {"ort": "...", "qnn_backend": "QnnCpu.dll", "device": "..."},

  "q2_a__jit_bake_works": {
    "node_counts": {
      "base_ctx":   {"EPContext": 1, "MatMul": 0, "ctx_inputs": ["x"]},
      "baked_ctx":  {"EPContext": 1, "MatMul": 0, "ctx_inputs": ["x"]},
      "baked2_ctx": {"EPContext": 1, "MatMul": 0, "ctx_inputs": ["x"]}
    },
    "max_abs_err": {
      "qnn_baked_vs_cpu_baked":   ...,
      "qnn_baked2_vs_cpu_baked2": ...,
      "qnn_baked_vs_qnn_baked2":  ...
    }
  },

  "q2_b__swap_requires_recompile": {
    "compile_times_ms": [..., ..., ...]
  },

  "q1__precompiled_plus_adapter_is_noop": {
    "max_abs_err__override_vs_qnn_base": ...,
    "adapter_api_used": "ort.LoraAdapter.create_from_array(...) or ..."
  },

  "q3__graph_input_user_script": {
    "node_counts": {
      "switchable_ctx": {"EPContext": 1, "MatMul": 0,
                         "ctx_inputs": ["x", "A", "B"]}
    },
    "max_abs_err": {
      "cpu_adapter1_vs_cpu_baked":  ...,
      "cpu_adapter2_vs_cpu_baked2": ...,
      "qnn_adapter1_vs_cpu_baked":  ...,
      "qnn_adapter2_vs_cpu_baked2": ...
    },
    "timings_ms": {
      "session_build": ...,
      "run_adapter1": ...,
      "run_adapter2": ...,
      "run_adapter1_again": ...
    }
  },

  "verdict": {
    "Q1_precompiled_plus_adapter": "NOT supported (confirmed) | unexpected",
    "Q2_uncompiled_plus_adapter":  "works via JIT bake; swap requires recompile (confirmed) | ...",
    "Q3_graph_input_user_script":  "works on CPU and QNN, no recompile on swap (confirmed) | ..."
  }
}
```

---

## 5. Decision matrix

Read each row as: "if you observe this, this is what it means for the
three questions in Section 0."

| Outcome | Interpretation |
|---|---|
| Q1 ✓, Q2 ✓ (both parts), Q3 ✓ | All three hypotheses confirmed. **Q1**: never attach adapters post‑compile. **Q2**: baking‑via‑initializers works as a fixed‑adapter shipping option but is not switchable. **Q3**: the repo's graph‑input surgery + Appendix A loader is the supported runtime swap path on QNN. |
| Q1 ✗ (override changed output) | `OrtLoraAdapter` *did* affect a precompiled EPContext — unexpected. QNN may expose a live initializer hook. Investigate before relying on Q1. |
| Q2.a numeric mismatch | QNN compiler dropped/altered the LoRA branch silently during JIT bake. Inspect EPContext node I/O, ORT logs, and any compile‑time warnings. |
| Q2.a structural (`MatMul` still in graph after JIT) | QNN refused to claim part of the subgraph and ran it on CPU EP. Reproduce with HTP‑friendly shapes / dtypes (Section 6). The numeric result will still be correct, but the "baked into EPContext" claim is false for this configuration. |
| Q2.b: third re‑compile is ~free | ORT/QNN is in‑process caching compiled contexts. Doesn't falsify Q2.b but is worth documenting. |
| Q3.c: `dt2` ≫ `dt1` (second Run dramatically slower than first) | Adapter swap is *not* free — investigate whether QNN is recompiling per Run for some shape‑related reason. Would falsify the headline Q3 claim. |
| Q3.c numeric mismatch | Either QNN rejects activation×activation MatMul (check graph for CPU fallback), or input plumbing (transpose/scaling) in the loader is wrong. |

---

## 6. Optional HTP follow‑up

If the target machine has Snapdragon HTP, repeat **Steps 3, 5, 6a** with:

```python
providers = [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})]
```

You will likely need to:

1. Quantize `baked.onnx` and `switchable.onnx` to **QDQ fp16 or int8** form
   (the existing repo pipeline under `lm_to_onnx/` does this; or use
   `onnxruntime.quantization` for a tiny calibration set).
2. Loosen numeric tolerances substantially (e.g. `atol=5e-2`).
3. Confirm that the **structural** claims (H1 graph shape, H2 immutability,
   H3 input list) are unchanged. Structural claims are the load‑bearing ones
   for the original question; numeric drift on HTP is expected and not a
   refutation of H1.

If HTP rejects the switchable model because of activation×activation MatMul
not being supported on HTP for the chosen dtype, that is itself a useful data
point — report it but do not treat it as a refutation of H3 (which is about
the ORT/EPContext mechanism, not HTP op coverage).

---

## 7. Out of scope (deliberately)

- Quantization‑aware LoRA training, per‑tensor adapter encodings.
- Multi‑adapter context binaries / weight sharing across contexts.
- Performance benchmarking (latency, memory) between the two LoRA shapes.
- Anything involving real Qwen3 weights — use the toy 8×8 model only.

These come **after** the three structural claims above are nailed down.

---

## 8. Reference snippets

### 8.1 Building `baked.onnx`

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(0)
W = rng.standard_normal((8, 8)).astype(np.float32)
A = np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32)
B = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float32)

x  = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
y  = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])

inits = [numpy_helper.from_array(W, "W"),
         numpy_helper.from_array(A, "A"),
         numpy_helper.from_array(B, "B")]

nodes = [
    helper.make_node("MatMul", ["x", "A"], ["t0"]),
    helper.make_node("MatMul", ["t0", "B"], ["t1"]),
    helper.make_node("MatMul", ["x", "W"], ["t2"]),
    helper.make_node("Add",    ["t2", "t1"], ["y"]),
]

graph = helper.make_graph(nodes, "baked_lora", [x], [y], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "models/lora_test/baked.onnx")
```

### 8.2 JIT compile + dump on QNN

```python
import onnxruntime as ort, numpy as np

x0 = (np.arange(8, dtype=np.float32) / 8.0).reshape(1, 8)

so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path",
                            "models/lora_test/baked_ctx.onnx")
so.add_session_config_entry("ep.context_embed_mode", "1")

sess = ort.InferenceSession(
    "models/lora_test/baked.onnx", so,
    providers=[("QNNExecutionProvider", {"backend_path": "QnnCpu.dll"})],
)
y = sess.run(None, {"x": x0})[0]
print("y =", y)
```

### 8.3 Inspecting the dumped graph

```python
import onnx
m = onnx.load("models/lora_test/baked_ctx.onnx")
print([n.op_type for n in m.graph.node])
ep = [n for n in m.graph.node if n.op_type == "EPContext"][0]
print("EPContext inputs :", list(ep.input))
print("EPContext outputs:", list(ep.output))
print("Initializers     :", [t.name for t in m.graph.initializer])
```

Expected output (illustrative):

```
['EPContext']
EPContext inputs : ['x']
EPContext outputs: ['y']
Initializers     : []
```

---

## 9. Handoff checklist for the executing agent

- [ ] Confirm `onnxruntime-qnn` is importable and `QNNExecutionProvider`
      appears in `ort.get_available_providers()`. `pip install safetensors`
      if missing.
- [ ] Locate `QnnCpu.dll` (and `QnnHtp.dll` if applicable) under the
      `onnxruntime/capi` directory of the venv; record absolute path in
      `results.json`.
- [ ] Implement `build_models.py`, `run_cpu.py`, `run_qnn_jit.py`,
      `run_qnn_aot.py`, `run_user_script.py`, `inspect.py` per Section 4.
- [ ] **Q2:** run Phase Q2 (a + b); write `q2_*` blocks in `results.json`.
- [ ] **Q1:** run Phase Q1 using `base_ctx.onnx` produced in Q2; write
      `q1_*` block. If `OrtLoraAdapter` Python API differs in ORT 1.23,
      adapt and document.
- [ ] **Q3:** run Phase Q3 (a + b + c, and d if time permits); write
      `q3_*` block.
- [ ] (Optional) Repeat Phases Q2 / Q3 with `backend_path="QnnHtp.dll"` —
      see Section 6.
- [ ] Fill in `verdict.Q1`, `verdict.Q2`, `verdict.Q3`.

When done, the deliverables are:

1. `investigation/results.json` with the per‑question verdicts.
2. `models/lora_test/*_ctx.onnx` (the AOT EPContext dumps) and
   `adapter.safetensors` / `adapter2.safetensors`.
3. A short note on any deviations from this plan (especially around the
   `OrtLoraAdapter` Python API spelling, any QNN backend path quirks, and
   any case where Q3.c shows the swap is *not* free).

---

## Appendix A — End‑user inference call (graph‑input LoRA)

This is the *consumer* side of the graph‑input LoRA design (H3): once a model
has been produced where `lora_A` / `lora_B` are real ONNX graph inputs (either
the `switchable.onnx` from this plan, or a production model built with
[`add_lora_inputs_to_unquantized_model`](../lm_to_onnx/onnx_utils/onnx_surgeries.py#L226)),
and the user has an `adapter.safetensors` file, the inference code looks like
the following. This is included for reference — it is **not** part of the
validation steps above.

### Mental model

`session.run(outputs, feeds)` requires every graph input to be supplied. With
LoRA‑as‑inputs that means: the normal inputs (`input_ids`, `attention_mask`,
KV cache, etc.) **plus** an A and B tensor for every adapted linear. The
adapter file just provides the values. The entire "adapter system" reduces to
building one dict:

```
feeds = { ...normal inputs..., **lora_feeds }
```

where `lora_feeds` maps ONNX input name → numpy array loaded from the
safetensors.

### The loader (one‑time helper)

```python
# inference/lora_loader.py
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

# PEFT files store:
#   base_model.model.<orig_layer_path>.lora_A.weight  : [r, in]
#   base_model.model.<orig_layer_path>.lora_B.weight  : [out, r]
_PEFT_KEY_RE = re.compile(
    r"^base_model\.model\.(?P<layer>.+)\.lora_(?P<which>[AB])\.weight$"
)

def load_adapter(
    adapter_path: str | Path,
    onnx_input_names: list[str],
    *,
    alpha: float | None = None,        # if None, read adapter_config.json
    rank:  int   | None = None,
    fold_scaling_into: str = "B",      # "A" or "B"
) -> dict[str, np.ndarray]:
    """
    Build a feeds dict mapping ONNX LoRA input names -> numpy arrays.

    The ONNX surgery in this repo names inputs as:
        "<base_weight_name>_lora_A"
        "<base_weight_name>_lora_B"
    where <base_weight_name> typically ends in something like
        "model.layers.0.self_attn.q_proj.weight"
    which matches the PEFT layer path verbatim (modulo the
    "base_model.model." prefix and the ".lora_X.weight" suffix).
    """
    adapter_path = Path(adapter_path)
    raw = load_file(str(adapter_path))

    # 1. Resolve scaling
    if alpha is None or rank is None:
        cfg = json.loads((adapter_path.parent / "adapter_config.json").read_text())
        alpha = alpha if alpha is not None else cfg["lora_alpha"]
        rank  = rank  if rank  is not None else cfg["r"]
    scaling = float(alpha) / float(rank)

    # 2. Group PEFT entries by layer
    peft_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for k, v in raw.items():
        m = _PEFT_KEY_RE.match(k)
        if not m:
            continue   # ignore optimizer state, embeddings tweaks, etc.
        peft_by_layer.setdefault(m["layer"], {})[m["which"]] = v

    # 3. Build the feeds dict by matching ONNX input names
    feeds: dict[str, np.ndarray] = {}
    for inp in onnx_input_names:
        if not (inp.endswith("_lora_A") or inp.endswith("_lora_B")):
            continue
        which   = inp[-1]                         # 'A' or 'B'
        base    = inp[: -len("_lora_X")]          # strip "_lora_A"/"_lora_B"
        layer   = base[: -len(".weight")] if base.endswith(".weight") else base

        peft_entry = peft_by_layer.get(layer)
        if peft_entry is None or which not in peft_entry:
            # Layer not adapted — feed zeros so this layer behaves as base.
            # (Shape must be inferred from the ONNX input metadata; left as
            #  an exercise — typically use sess.get_inputs() shape.)
            continue

        w = peft_entry[which].astype(np.float32, copy=False)
        # PEFT:  A:[r,in], B:[out,r].
        # ONNX surgery expects A:[in,r], B:[r,out]  (plain MatMul order).
        w = w.T
        if which == fold_scaling_into:
            w = w * scaling
        feeds[inp] = np.ascontiguousarray(w)

    return feeds
```

Two pitfalls baked into that loader:

1. **Transpose.** PEFT and the ONNX MatMul convention disagree by one
   transpose. Verify once against a known‑good CPU output.
2. **Scaling.** PEFT stores A and B unscaled and applies `α/r` at runtime.
   Folding the scalar into one of the matrices at load time is mathematically
   equivalent and avoids an extra graph input.

### The actual user call

```python
import numpy as np
import onnxruntime as ort
from inference.lora_loader import load_adapter

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
onnx_input_names = [i.name for i in sess.get_inputs()]

# Load once; reuse for every Run().
lora_feeds = load_adapter("adapter.safetensors", onnx_input_names)

# Normal inputs
input_ids      = np.array([[1, 2, 3, 4]], dtype=np.int64)
attention_mask = np.ones_like(input_ids)

outputs = sess.run(None, {
    "input_ids":      input_ids,
    "attention_mask": attention_mask,
    **lora_feeds,                   # <-- the only LoRA-aware line
})
```

That is the whole API. From the call site's perspective, an adapter is just
"extra entries in the feeds dict, prepared once."

### Switching adapters at runtime

```python
lora_chinese = load_adapter("adapter_chinese.safetensors", onnx_input_names)
lora_code    = load_adapter("adapter_code.safetensors",    onnx_input_names)

sess.run(None, {..., **lora_chinese})   # request 1
sess.run(None, {..., **lora_code})      # request 2
sess.run(None, {..., **zeros_overlay})  # request 3: pure base model
```

Blending two adapters is `0.5 * A1 + 0.5 * A2` (and the matching for B)
computed on the host side and fed as a single overlay. No ORT API call is
involved — it is plain numpy.

### Performance: `IOBinding`

For decoder LLMs, re‑sending hundreds of small A/B tensors through the feeds
dict every token wastes time. Use `IOBinding` so LoRA tensors live in
pre‑allocated `OrtValue`s and are bound by reference; rebind only when the
adapter changes. Switch the device tag from `"cpu"` to the QNN device tag for
zero‑copy on Snapdragon.

```python
io = sess.io_binding()
lora_ortvalues = {
    name: ort.OrtValue.ortvalue_from_numpy(arr, "cpu", 0)
    for name, arr in lora_feeds.items()
}
for name, v in lora_ortvalues.items():
    io.bind_ortvalue_input(name, v)

for step in range(max_new_tokens):
    io.bind_cpu_input("input_ids",      input_ids)
    io.bind_cpu_input("attention_mask", attention_mask)
    io.bind_output("logits")
    sess.run_with_iobinding(io)
    logits = io.copy_outputs_to_cpu()[0]
    # ... sample next token, update input_ids ...
```

### Integration with the existing repo

To wire this into `inference/`:

1. Add `inference/lora_loader.py` (the loader above).
2. In `session_manager.py`, after building the session, compute
   `lora_input_names = [i.name for i in sess.get_inputs() if "_lora_" in i.name]`.
3. When the user selects an adapter, call `load_adapter(path, lora_input_names)`
   and cache the result on the session manager.
4. At every `Run()` (or `IOBinding`), merge `**lora_feeds` into the feeds dict.

No other code in `qwen_inference.py` needs to change — the model just happens
to have a few hundred extra named inputs that the session manager fills in.
