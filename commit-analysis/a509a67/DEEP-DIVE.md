# Commit a509a67 — Deep-Dive: Implementation vs Design

## How to read this doc

This is the opinionated cross-reference of commit `a509a67`
("feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor"). Three upstream artifacts are assumed in front of the reader: `temp/commit-analysis/a509a67/SUMMARY.md` (integrated narrative of what shipped, ~200 lines), `temp/commit-analysis/a509a67/DESIGN-DOCS-INDEX.md` (catalog of 42 design / plan / review docs under `docs/design/session/` and `docs/plans/`), and the 48 per-file diff analyses under `temp/commit-analysis/a509a67/per-file/`. This doc does *not* recap any of that material; it judges the gap between what was designed and what was shipped, and identifies the load-bearing risks and follow-ups.

## Methodology

The cross-reference reads from three corners simultaneously: (1) the seed spec
`docs/design/session/2026-05-11-ep-device-refactor.md` v1.2 (and its planned v1.3 reconciliation per landing-page §I2) for what was *promised*; (2) `SUMMARY.md` plus eight high-signal per-file analyses (`session__ep_device`, `session__session`, `session__monitor__qnn_monitor`, `commands__perf`, `commands__compile`, `sysinfo__device`, `winml`, `session__ep_registry`, `session__monitor__ep_monitor`, `session____init__`) for what *shipped*; (3) the audits (`2026-05-12-impl-status.md`, `2026-05-13-consolidation-audit.md`, `2026-05-13-post-refactor-taxonomy-audit.md`, `2026-05-13-remaining-issues.md`) for what the team already knows is in drift. Excluded from primary read: the 11 monitor iteration docs (consolidated into PRD v2.4.1 + coreloop v2.4.1), the rejected v1 / v2 taxonomy cleanup plans (superseded by v3 + v3-verify), and the per-file analyses of mechanical migrations (`commands/config.py`, `commands/eval.py`, `compiler/cli.py`, `config/build.py`, `config/precision.py`, etc.). Confidence is highest on the architectural critique sections and on divergences with a paper trail; it is lower on the QNN SDK runtime claims (no QNN SDK on the analysis host) and on the QHAS schema-rename completeness (no fixture corpus reviewed).

## Headline: where impl diverges from design

The divergences below are ranked by **practical blast radius** (how likely a future contributor stumbles into the gap, weighted by how badly they get hurt). Most are improvements over spec — the design doc was a 2026-05-11 snapshot and the shipped code is two weeks of iteration further forward — but a handful are real drift that should be acknowledged.

---

### D1. Catalog `EP_DEVICE_SPECS` replaces both `_ep_defaults` match-statement *and* `_QNN_BACKEND` map; ORT 1.23.5 made the spec's `backend_type` setting toxic

- **Divergence title:** Spec §3.4 `_ep_defaults(ep_device)` returns `{"backend_type": _QNN_BACKEND[ep_device.device]}` for QNN; shipped `_ep_defaults` returns a fresh copy of the catalog entry's `default_provider_options` and the QNN-NPU catalog entry contains `htp_performance_mode="burst"` + `htp_graph_finalization_optimization_mode="3"` — and **deliberately does not** set `backend_type`.
- **What design said:** `docs/design/session/2026-05-11-ep-device-refactor.md` §3.4 specifies the QNN match arm as `return {"backend_type": _QNN_BACKEND[ep_device.device]}` (line 309–315). §7 reaffirms "Strict 4-tuple matching including CPU. Rationale: reproducibility." The spec is silent on `htp_performance_mode`.
- **What impl shipped:** `src/winml/modelkit/session/ep_device.py:166–199` (the `EP_DEVICE_SPECS` tuple) for the QNN-NPU row, with explicit defaults `htp_performance_mode="burst"` and `htp_graph_finalization_optimization_mode="3"`; `src/winml/modelkit/session/session.py:88–104` (`_ep_defaults` free function) for the catalog-lookup-and-copy. Per `per-file/session__session.md` §"Symbol-level changes": the QNN backend_type comment in `_ep_defaults` reads "must not be passed explicitly (crashes ORT 1.23.5 exit 127)" — see also `per-file/session__monitor__qnn_monitor.md`: the QNN monitor's `get_provider_options` does not emit `backend_type` either.
- **Judgment:** **Improvement** (a clear one). Shipping `backend_type` in provider options would crash with the same `exit(127)` mode the symmetric-singleton fix was added to defend against — the spec's recipe would have been a latent landmine. The substitution of `htp_performance_mode="burst"` is verified to deliver ~3× ResNet-50 throughput per the commit-body verification block (5.73 ms → 1.90 ms / 175 → 526 samples/s) and is a measured-not-guessed improvement.
- **Why:** ORT 1.23.5 auto-derives `backend_type` from the resolved `OrtEpDevice` handle via `add_provider_for_devices`. Explicitly setting it produces a native crash; the spec was written against an older ORT contract. The `htp_performance_mode` value is a measured win that the spec did not foresee.
- **Recommendation:** **Amend the spec to v1.3.** The QuantSpec doc (`2026-05-14-quant-spec-design.md`, DRAFT) already implicitly anchors on the catalog-as-truth pattern; v1.3 should retire the match-statement vs catalog-tuple distinction and document the verified burst-mode default as the canonical QNN-NPU recipe.

---

### D2. `WinMLSession.perf()` is a `@contextmanager`, not a regular method

- **Divergence title:** Spec's perf() body implies a regular method that returns `PerfStats`; shipped impl is a `@contextlib.contextmanager` that yields `PerfContext`.
- **What design said:** `docs/design/session/2026-05-11-ep-device-refactor.md` §3.4 lines 239–268 shows `def perf(self, monitor: EPMonitor | None = None, ...)` with a `try/finally` body that rebuilds `self._session` and runs the benchmark inline — i.e. a normal method called by a benchmark driver. The PRD `monitor/1_prd.md` §1.1 says `session.perf(warmup=10, monitor=...)` is a context manager but the spec did not reconcile the two views.
- **What impl shipped:** `src/winml/modelkit/session/session.py:605–774` (`@contextmanager def perf(...)`), yielding a frozen `PerfContext(stats, monitor)` dataclass. See `per-file/session__session.md` "Symbol-level changes" → `WinMLSession.perf`, and SUMMARY.md "Breaking changes for callers" (the 6th bullet on `PerfContext`).
- **Judgment:** **Improvement.** The context-manager form is the only ergonomic way to compose with `HWMonitor` (which is already a context manager per FR-9) and to enforce the C-2 teardown-order invariant (reset the InferenceSession before the monitor's `__exit__` flushes the CSV). A regular method that owns both `__enter__`/`__exit__` of an internal monitor and returns a stats object would have either swallowed teardown errors silently or leaked partial state on exception.
- **Why:** The PRD's `with session.perf(warmup=10, monitor=mon) as ctx:` shape is load-bearing for the standalone-profile idiom (FR-7). The spec's snippet was pseudo-code, not contract.
- **Recommendation:** **Amend spec v1.3.** This is the canonical example of "spec rot" that the landing page §I2 already calls out.

---

### D3. The shipped flow rebuilds the baseline `InferenceSession` on perf-exit *only when monitor contributed differing provider options* — spec rebuilds unconditionally

- **Divergence title:** Spec's `finally` block unconditionally rebuilds the baseline session via a second `_build_session_options(..., None, ...)` call; shipped impl preserves the pre-perf `InferenceSession` identity when monitor and base agree on provider options.
- **What design said:** `2026-05-11-ep-device-refactor.md` §3.4 lines 256–268: the `finally` always calls `self._session = ort.InferenceSession(..., sess_options=_build_session_options(..., None, ...))`. There is no "skip if identical" fast-path in the spec.
- **What impl shipped:** `per-file/session__session.md` "Symbol-level changes" → `WinMLSession.perf`: *"Baseline rebuild on exit only when `_session_rebuilt` was True (preserves pre-perf `InferenceSession` object identity when monitor contributed nothing — tests assert on this)."* Plus the auto-reset is only triggered when `new_prov != self._provider_options`.
- **Judgment:** **Improvement** (and it has tests pinning the behaviour).
- **Why:** ORT InferenceSession construction on a QNN-EPContext model is expensive; recreating it after every plain `with session.perf():` call would punish callers who didn't even attach a monitor. The identity-preservation also lets external code that holds a session reference (e.g. for `session.io_config`) survive a perf window. The spec's unconditional rebuild was protective but wasteful.
- **Recommendation:** **Amend spec v1.3** to document the fast-path. Tests are the contract.

---

### D4. `EPDeviceSpec` catalog as a separate concept (template) layered over `EPDevice` (instance) — the spec proposed neither

- **Divergence title:** Spec defines only `EPDevice`; shipped impl introduces `EPDeviceSpec` as a sibling frozen dataclass and an ordered tuple catalog (`EP_DEVICE_SPECS`, 13 entries). This is the foundation of "order encodes preference" and "single source of truth for provider defaults."
- **What design said:** The seed spec `2026-05-11-ep-device-refactor.md` §3 designs `EPDevice` only. The `_ep_defaults` function in §3.4 uses a `match` statement (one arm per EP) plus a separate `_QNN_BACKEND` dict. There is no notion of a per-(EP, device) catalog row.
- **What impl shipped:** `src/winml/modelkit/session/ep_device.py:151–199` adds `EPDeviceSpec` (`@dataclass(frozen=True, kw_only=True, slots=True)`) and the 13-entry `EP_DEVICE_SPECS` tuple. The follow-up `docs/design/session/2026-05-13-ep-device-spec-design.md` is the post-hoc design document.
- **Judgment:** **Improvement, large.** This is the most important architectural decision the commit makes, and it is missing from the seed spec because it didn't exist when the seed was written.
- **Why:** The spec's match-statement-per-EP shape would have forced every new EP to edit `_ep_defaults` in `session/session.py` (the file with the most blast radius). The catalog moves that knowledge into `ep_device.py` where every other taxonomy lookup lives. `VALID_EPS`, `VALID_DEVICES`, `eps_for_device`, `default_ep_for_device`, `default_device_for_ep`, `_ep_defaults`, the new `_resolve_ep_monitor` dispatch, and the `lookup_device_spec` O(1) probe all derive from the same tuple. The Kubernetes `PodSpec`/`Pod` analogy in `2026-05-13-ep-device-spec-design.md` is the right framing.
- **Recommendation:** **Promote the EPDeviceSpec doc and amend spec v1.3 to mention it.** The doc as it stands reads like a downstream redesign; the unified v1.3 spec should treat catalog-as-truth as a first-class design choice, not a clean-up.

---

### D5. Spec doesn't address how `WinMLSession.compile()` interacts with `add_provider_for_devices`; impl resurrects the method to actually drive `ort.ModelCompiler.compile_to_file` with Bug A + Bug B fixes

- **Divergence title:** The seed spec mentions `compile()` only in passing and does not specify the compile-time session-construction sequence as distinct from `perf()`. Shipped impl resurrects `WinMLSession.compile()` from stub-state by fixing two coupled bugs that previously prevented the compile pipeline from running at all.
- **What design said:** `docs/design/session/2026-05-11-ep-device-refactor.md` §3 mentions compile only in passing — the seed spec's worked example is `perf()`. There is no specification of the compile-time session-construction sequence, no rule for whether/when `InferenceSession` should be constructed eagerly vs deferred when `enable_ep_context=True`, and no design for how the compile path should consume `_build_session_options(...)`.
- **What impl shipped:** `src/winml/modelkit/session/session.py` (`WinMLSession.compile()` body) implements two fixes. **Bug A** — defer `InferenceSession` creation when `enable_ep_context=True`: pre-state, the eager session construction inside `WinMLSession.__init__` triggered ORT to look for an EPContext binary that doesn't exist yet, raising a runtime error before `compile()` ever got control. The compile pipeline could literally never run from `WinMLSession.compile()` until Bug A was fixed. **Bug B** — call the new free `_build_session_options(ep_device, ep_config, monitor=None, base_session_options=...)` (the module-level free function that replaced the deleted `WinMLSession._build_session_options` instance method) and then invoke `ort.ModelCompiler.compile_to_file(...)`. The `ort.ModelCompiler` call is wrapped in `_suppress_native_output(compile_log)` to redirect QNN SDK native stdout to `<onnx_path>.parent/compile.log`. The threading completes via `WinMLCompileConfig.for_ep_device(ep_device, ...)` (new factory in `compiler/configs.py`), `CompileContext` carrying the typed `EPDevice` alongside the dict-form config, and `CompileStage._finalize_output`'s three-way filename search that prefers `{stem}_{device_category}_ctx.onnx` (e.g. `*_npu_ctx.onnx`) over the legacy `_qnn_ctx.onnx` and `_ctx.onnx` patterns.
- **Judgment:** **Improvement (large).** The spec didn't anticipate that `compile()` needed its own session-construction sequence — different from `perf()` — because the spec's worked example was `perf()` and the v1.2 author hadn't yet discovered that `enable_ep_context=True` makes eager session construction fatal. Shipping this PR's `compile()` without Bug A's fix would have left the method dead-on-arrival.
- **Why:** Without Bug A, `ort.InferenceSession` was constructed eagerly with `enable_ep_context=True`, which causes ORT to search for an EPContext binary that doesn't exist yet (the compile pipeline hasn't produced it). That's a runtime error before `compile()` body runs. Bug B is what actually makes the compile call work: the new free `_build_session_options(...)` is the only entry point that knows how to assemble session options *without* needing a `WinMLSession` instance — exactly what `compile()` needs because it's constructing the compiler, not consuming a built session.
- **Recommendation:** **Amend spec v1.3** with the compile-time session-construction sequence as a separate concern from `perf()`. Document the EPContext defer-create rule explicitly: `InferenceSession` construction MUST be deferred when `enable_ep_context=True` until the compile pipeline has emitted the context binary. The free `_build_session_options(...)` function's role as the compile-path session-options assembler should be elevated to a first-class design choice in v1.3, not buried as an implementation detail.

---

### D6. `WinMLQairtSession` constructor has a `None` default that auto-resolves to `resolve_device("qnn", "npu")`; `WinMLSession` has *no* default — asymmetry across subclass

- **Divergence title:** `WinMLSession.__init__(onnx_path, ep_device, *, ...)` requires `ep_device` positional; `WinMLQairtSession.__init__(ep_device: EPDevice | None = None, ...)` has a `None` default that triggers a hidden `resolve_device("qnn", "npu")` call at construction.
- **What design said:** `2026-05-11-ep-device-refactor.md` §2 (non-goals): *"Refactoring WinMLQairtSession... It is fixed in a follow-up PR."* §3.3 prescribes `ep_device: EPDevice  # required, no default` for `WinMLSession`. Spec is silent on QAIRT defaults.
- **What impl shipped:** Per `2026-05-12-code-review/session_qairt_qairt_session.md` and SUMMARY.md "Breaking changes" 9th bullet: QAIRT shipped with the default-None auto-resolution that the spec said was out of scope for this PR.
- **Judgment:** **Acceptable trade-off — but document the asymmetry.** The default is sensible (QAIRT is a QNN-specific subclass; "no device given" can only mean QNN-NPU). The cost is that two near-identical class signatures behave differently on missing args.
- **Why:** Making QAIRT default-None saves the CLI boundary from having to know the QAIRT/non-QAIRT distinction when the user typed `winml run --qairt model.dlc`. But the silent registry side-effect inside `__init__` makes QAIRT's "no-args" construction non-trivial to mock in tests.
- **Recommendation:** **Do nothing structural; add a one-paragraph note to spec v1.3** that explicitly carves QAIRT out of the "ep_device is required positional" rule and explains the rationale.

---

### D7. Two-singleton DLL registration race patched with symmetric defensive guards — design fix (single-singleton) explicitly deferred (I1)

- **Divergence title:** Spec does not anticipate the two-singleton problem at all. The shipped code carries *symmetric defensive guards* in both `WinMLEPRegistry.register_ep` (`session/ep_registry.py`) and `WinML.register_execution_providers` (`winml.py`) to prevent ORT's non-idempotent `register_execution_provider_library` from invoking C++ `exit(127)` on second-call.
- **What design said:** Nothing. The spec assumes one registration path. The diagnostic docs `2026-05-13-gap1-diagnostic.md` and `2026-05-13-t6-analyze-crash-diagnostic.md` are the discovery trail.
- **What impl shipped:** Per `per-file/session__ep_registry.md` "Symbol-level changes → register_ep" and `per-file/winml.md` "Symbol-level changes → register_execution_providers": both registration sites probe `module.get_ep_devices()` for prior loads before calling `register_execution_provider_library`. Probes wrapped in `try/except Exception → already_loaded = False` (conservative fallback for older ORTs).
- **Judgment:** **Acceptable patch; the design fix is the long-term answer.** The landing-page §I1 explicitly defers consolidation; for this PR, the patch is the right call.
- **Why:** Process-killing native crashes with no Python traceback are unacceptable. The symmetric guards work — verified by `2026-05-13-cli-claims-reverify.md` 6/6 PASS. But two singletons with two in-memory caches, two registration codepaths, and asymmetric failure-recording (`register_to_ort` writes to `_registration_failures`; `register_ep` raises typed exceptions instead — see `per-file/session__ep_registry.md` "Risks / subtleties") is technical debt that grows with every new analyze/runtime code path.
- **Recommendation:** **Follow-up PR.** Collapse to one singleton. Architecture test that asserts only one symbol named `register_execution_provider_library` is called anywhere under `src/winml/`. The patch is fine for shipping; consolidate before the third singleton appears.

---

### D8. Spec's `_build_session_options` always rebuilds via a single call; shipped impl auto-resets a *compiled* session inside `perf()` with a WARNING when monitor's provider options differ

- **Divergence title:** Per `per-file/session__session.md` §"perf()": *"if `_session is not None and new_prov != self._provider_options` the compiled session is auto-reset (with a WARNING) so new options take effect."* Spec does not contemplate that the session might be already-compiled at perf-time.
- **What design said:** `2026-05-11-ep-device-refactor.md` §3.4 lines 238–268 assumes `_session` is freshly built inside `perf()` from the current `_build_session_options(...)`. Compile-path interleaving is not addressed.
- **What impl shipped:** `src/winml/modelkit/session/session.py:605–774` does the diff-check-and-auto-reset dance.
- **Judgment:** **Improvement** (handles the real-world case the spec ignored).
- **Why:** Once `WinMLSession.compile()` (the new Bug-A/Bug-B-fixed compile path) builds an EPContext model, the session is "compiled" and holding cached state. Calling `perf(monitor=...)` against that session must either rebuild or warn-and-rebuild. The spec's "always rebuild" was implicit; the shipped code is explicit and emits a WARNING so the user sees the cost.
- **Recommendation:** **Amend spec v1.3** with the auto-reset rule.

---

### D9. `_resolve_ep_monitor` drops OpenVINO; spec's `EPMonitorMismatch` exception assumed a populated monitor-per-EP map

- **Divergence title:** PRD `monitor/1_prd.md` FR-11 *requires* the perf CLI to "fail hard with a descriptive error (no silent fallback)" when op-tracing is requested against an EP with no matching monitor. Shipped `commands/perf.py:_resolve_ep_monitor` only knows `qnn` and `vitisai`; OpenVINO falls into the `RuntimeError("Op-tracing not available for EP 'openvino'...")` branch.
- **What design said:** PRD FR-11 (`monitor/1_prd.md` line 175): *"commands/perf.py MUST resolve the correct EPMonitor class via explicit dispatch... If op-tracing is requested against an EP that has no matching monitor, the command MUST fail hard with a descriptive error."* SUMMARY.md "Op-tracing parity for OpenVINO / VitisAI" notes the commit body labels OpenVINO as a "placeholder for parity."
- **What impl shipped:** `src/winml/modelkit/commands/perf.py:117–187` (`_resolve_ep_monitor`), per `per-file/commands__perf.md` "_resolve_ep_monitor only knows about QNN and VitisAI."
- **Judgment:** **Acceptable trade-off short-term, regression long-term.** The shipped behaviour is *technically* spec-compliant — the spec said "fail hard," and the impl does. The issue is the inconsistency with VitisAI, which *does* have a `VitisAIMonitor` import even though SUMMARY.md flags both VitisAI and OpenVINO as "placeholders for parity." Asymmetric placeholder coverage signals "we forgot" rather than "we deferred."
- **Why:** If VitisAI gets a no-op monitor stub, OpenVINO should too — or neither should. Today's state suggests `_resolve_ep_monitor` was written before the OpenVINO monitor branch was decided.
- **Recommendation:** **Follow-up issue (small).** Either wire `OpenVinoMonitor` parallel to `VitisAIMonitor` (no-op for op-tracing), or remove the VitisAI special case and unify under a single "non-QNN monitors are unsupported for op-tracing" rule.

---

### D10. Spec said `EPMonitor` ABC retains `to_dict()`; v2.4 PRD dropped it; impl complies with v2.4 — but transitional monitors still expose `to_dict()` ad-hoc

- **Divergence title:** Two design docs disagree across versions. Seed spec implies `to_dict()` is the canonical accessor (used by perf JSON output). PRD v2.4 §1.1 + SC-10 drops `to_dict()` from the ABC contract in favour of typed `result` and `proof` accessors. Shipped code complies with v2.4 on the ABC side but `commands/perf.py:_monitor_to_json_dict` still calls `monitor.to_dict()` as a fallback for VitisAI / OpenVINO.
- **What design said:** PRD `monitor/1_prd.md` §1.1: *"v2.4 drops to_dict() from the EPMonitor contract — replaced by typed accessors."* But: *"VitisAIMonitor.proof / OpenVINOMonitor.proof -> ProofOfExecution (typed accessor + new ProofOfExecution class flagged as a follow-up PR — out of scope for this lift)."*
- **What impl shipped:** Per `per-file/session__monitor__ep_monitor.md`: ABC drops `to_dict()`. Per `per-file/commands__perf.md` "_monitor_to_json_dict" (lines 57–96 in `commands/perf.py`): the helper falls through `monitor.result` → `monitor.to_dict()` → `{}`. The middle step exists because VitisAI/OpenVINO monitors still carry their own `to_dict` until `ProofOfExecution` lands.
- **Judgment:** **Acceptable transitional state.** PRD explicitly carves out the `ProofOfExecution` follow-up, so the temporary duck-typed `to_dict()` fallback is documented.
- **Why:** Cleanly typed but staged migration. Worth verifying that the duck-typed `hasattr(monitor, "to_dict")` check survives the OpenVINO/VitisAI integration; if those monitors lose `to_dict()` before `proof` lands, the helper returns `{}` and the JSON output silently degrades.
- **Recommendation:** **Track in a follow-up.** Add a deprecation warning inside `_monitor_to_json_dict` when the `to_dict()` branch fires, so the transitional state is loud.

---

### D11. `models/auto.py` positional `ep_device` argument is a footgun — review summary called it out as Decision M3 (deferred)

- **Divergence title:** `WinMLAutoModel.from_pretrained(model_id_or_path, ep_device, *, ...)` makes `ep_device` the second positional arg. Any caller that previously passed `task=` or a config positionally rebinds those to `ep_device` with no type-error.
- **What design said:** The spec is silent on `models/auto.py` signatures. The 2026-05-12 code-review summary (`2026-05-12-review-summary.md`) flagged it as a BLOCKER. The landing-page §M3 deferred it to follow-up audit.
- **What impl shipped:** Per SUMMARY.md "Breaking changes" lines on `WinMLAutoModel`: `from_pretrained` is positional `ep_device`, `from_onnx` is keyword-only `ep_device`. Inconsistent across the two factory methods.
- **Judgment:** **Regression** (mild). Positional in one factory and keyword-only in the sibling factory is the worst of both worlds — the user must memorize which is which.
- **Why:** Any caller passing config positionally will silently rebind to `ep_device`, which is a frozen dataclass and will produce a `TypeError` deep inside `_load_model` — far from the call site. The fix is to make both factory methods keyword-only past position 1.
- **Recommendation:** **Follow-up PR (small).** Make `from_pretrained` keyword-only-past-position-1 (`ep_device` becomes `*, ep_device` instead of positional). This is a one-line change with broad caller audit cost; the landing page already lists it as M3.

---

### D12. `_finalize_output` input-search prefers device-category filename pattern; output filename remains EP-short-name

- **Divergence title:** Per SUMMARY.md and `per-file/compiler__stages__compile.md`: input-search list prepends `{stem}_{device_category}_ctx.onnx` (e.g. `*_npu_ctx.onnx`) ahead of the legacy `_qnn_ctx.onnx` and `_ctx.onnx` patterns. But the *output* filename written by `WinMLSession.compile()` is still `{stem}_{provider_short}_ctx.onnx` (e.g. `resnet50_qnn_ctx.onnx`). Asymmetry between input and output naming protocols.
- **What design said:** Spec is silent on filename conventions.
- **What impl shipped:** SUMMARY.md "Bug fixes worth calling out" 3rd bullet (`CompileStage._finalize_output` naming-protocol fix).
- **Judgment:** **Acceptable for this commit; flag for follow-up.** The asymmetry is documented in the commit-body verification block as intentional. The risk is naming drift over multiple compile passes — a user who compiles with `--ep qnn`, then re-runs with `--device npu` (no `--ep`), produces two files differing only by suffix.
- **Recommendation:** **Follow-up issue.** Unify on one convention (device-category for both input and output, or EP-short-name for both). Add a `--output-naming {device|ep}` flag if both forms must coexist.

---

### Lower-severity divergences (drift-but-not-broken)

| # | Topic | Spec | Impl | Recommendation |
|---|---|---|---|---|
| D13 | `ensure_initialized()` module-level entry point | Not in spec | Added in `ep_registry.py`; called by `QNNMonitor.is_available()` to break import cycle | Document in v1.3 spec |
| D14 | `available_eps()` `lru_cache(maxsize=1)` with no invalidation API | Not in spec | Shipped; "hardware/EPs don't change during a process lifetime" docstring claim is contradicted by the fact that `register_ep` *does* change the set | Document the caveat; consider `cache_clear()` after registration |
| D15 | `EPDevice.from_dict` raises `KeyError` not typed exception | Not in spec | `KeyError` if `ep`/`device`/`vendor_id`/`device_id` missing | Promote to `ValueError` with structural context |
| D16 | `register_ep` "catalog miss" branch falls back to `ort.get_ep_devices()` for bundled EPs (CPU, DML) | Not in spec | Shipped; SUMMARY.md "Architectural move #5" | Document in v1.3 |
| D17 | `_get_ep_registry()` lazy import shim for `ep_device` ↔ `ep_registry` cycle | Not in spec | Module-level `WinMLEPRegistry: Any = None` sentinel patched lazily | Document, or restructure to break cycle properly |
| D18 | `__init_subclass__` guard on `requires_session_teardown` non-bool | Not in spec | Class-definition-time TypeError | Improvement; document |
| D19 | `EPMonitor.set_onnx_op_types` no-op default with `# noqa: B027` | PRD allows it | Shipped | Improvement; document |
| D20 | QHAS schema-rename (`time_us → inference_us`, etc.) | Not in spec | Shipped in `qnn/_internal.py` | Document in v1.3; external consumers of old keys silently see KeyError or None |

## Verified design-impl alignment

Where the impl genuinely matches the spec. Parsimonious:

- **`EPDevice` dataclass shape.** Frozen, 5 fields (`ep`, `device`, `vendor_id`, `device_id`, `vendor`), `__post_init__` lowercase invariant on `device`, `to_dict`/`from_dict` for JSON round-trip, no `OrtEpDevice` handle stored. Matches spec §3.1 exactly. (Cite: `session/ep_device.py:54–82`; spec §3.1 lines 47–71.)
- **5-exception taxonomy.** `EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch` — names, intent, and message structure match spec §4. Shipped with `# noqa: N818`. (Cite: `session/ep_device.py:31–47`; spec §4 lines 387–399.)
- **Strict 4-tuple matching.** `(ep, device.type.name.lower(), vendor_id, device_id)` everywhere, including CPU. Matches spec §3.4 line 202–207 + §7 line 430. (Cite: `session/session.py:171+` for the filter inside `_build_session_options`.)
- **Three-layer provider-options merge.** `_ep_defaults` → `ep_config.provider_options` → `monitor.get_provider_options()`. Monitor wins last. Matches spec §3.4 lines 274–297. (Cite: `session/session.py:107–131`; per-file analysis `session__session.md` §`_build_provider_options`.)
- **Hard break (Option A).** No `device="auto"` on `WinMLSession.__init__`, no `ep=` short-form kwarg, no policy paths, no `_find_ep_device`, no `set_provider_selection_policy(PREFER_NPU)`. Matches spec §3.3 lines 168–177. (Cite: SUMMARY.md "Breaking changes" first three bullets.)
- **`WinMLEPRegistry.register_ep` is additive.** Bulk `register_to_ort` unchanged; `register_ep` is a new method. Matches spec §3.5 line 322–340. (Cite: `session/ep_registry.py:151–211`.)
- **`EPMonitor` integrates via `perf(monitor=...)`, not the constructor.** Matches spec §3.3 paragraph at line 168 and §3.4 (b). (Cite: `session/session.py:605+`.)
- **PRD FR-14 four-layer fallback chain** (ONNX → EP-authoritative → heuristic → raw) implemented in `QNNMonitor._resolve_op_type`. (Cite: `session/monitor/qnn_monitor.py` per `per-file/session__monitor__qnn_monitor.md`.)
- **PRD FR-17 information-hiding architecture test** at `tests/unit/architecture/test_qnn_imports.py`. (Cite: `tests/unit/architecture/test_qnn_imports.py` — verified present in this branch.)
- **PRD SC-2: `optracing/` deleted.** No grep hits in `src/`. (Cite: SUMMARY.md "Architectural move #3"; 7 files deleted listed in SUMMARY.md table.)

## Spec drift that nobody documented

The landing page (`2026-05-13-remaining-issues.md` §I2) flags v1.3 as queued but unwritten. Per the DESIGN-DOCS-INDEX.md catalog, the following items are *all* shipped behaviours that have no design-doc paper trail beyond audit notes. Each is classified as **spec rot** (impl improved, spec not updated) or **scope creep** (impl exceeded scope without acknowledgment).

| # | Spec says | Code does | Type |
|---|---|---|---|
| SD1 | `perf()` is a regular method | `@contextmanager` yielding `PerfContext` | spec rot |
| SD2 | `_ep_defaults` is a match-statement returning `{"backend_type": ...}` | `_ep_defaults` returns `dict(spec.default_provider_options)` from catalog; QNN-NPU defaults are `htp_performance_mode="burst"` + `htp_graph_finalization_optimization_mode="3"`, **no `backend_type`** | spec rot (ORT 1.23.5 forced this) |
| SD3 | Spec defines `EPDevice` only | Impl ships `EPDeviceSpec` catalog as primary truth source | scope creep (architecturally necessary) |
| SD4 | Spec implies single registration path | Impl has two singletons + symmetric defensive guards | scope creep (forced by `winml.py` legacy singleton, deferred to I1) |
| SD5 | Spec is silent on baseline-session-rebuild fast path | Impl preserves InferenceSession identity when monitor contributes no differing options | scope creep |
| SD6 | Spec is silent on auto-reset on differing provider options | Impl auto-resets with WARNING log | scope creep |
| SD7 | Spec's `_QNN_BACKEND = {"npu": "htp", "gpu": "gpu", "cpu": "cpu"}` constant | Constant does not exist in shipped code; QNN backend is auto-derived by ORT from `OrtEpDevice` | spec rot |
| SD8 | Spec is silent on `WinMLQairtSession.__init__` defaults | Impl ships `ep_device: EPDevice | None = None` with auto-resolution side-effect | scope creep (§2 explicitly out of scope, but shipped anyway) |
| SD9 | Spec is silent on `_resolve_ep_monitor` CLI dispatch | Impl ships in `commands/perf.py:117–187` with QNN + VitisAI; OpenVINO falls to RuntimeError | scope creep (PRD FR-11 anchors it) |
| SD10 | Spec is silent on `ensure_initialized()` | Impl ships module-level cycle-breaker in `ep_registry.py` | scope creep |
| SD11 | Spec's `_SHORT_TO_CANONICAL` table has 7 entries | Impl renamed to `_SHORT_TO_FULL` and has 9 entries (added `cuda`, `tensorrt`); these were latent gap bugs | scope creep + bug fix |
| SD12 | Spec is silent on `available_eps()` aggregator | Impl ships `lru_cache(1)`-d aggregator over WinML + ORT | scope creep |
| SD13 | Spec is silent on QHAS summary-key renames | Impl renames `time_us → inference_us`, etc. (8 keys total) — breaks external consumers reading the old keys | scope creep |
| SD14 | Spec is silent on `_get_ep_registry()` lazy import shim | Impl ships `WinMLEPRegistry: Any = None` sentinel + lazy `importlib.import_module` to break `ep_device ↔ ep_registry` cycle | scope creep (architectural workaround) |
| SD15 | Spec is silent on `_to_int` hardening (`int(...) → round(float(...))`) in qnn_monitor | Bundle A monitor silent-failure fix shipped | scope creep (bug fix) |
| SD16 | Spec is silent on `_require()` × 19 hardening in `qnn/_internal.py` | Bundle A monitor silent-failure fix shipped; raises `KeyError` with context on schema drift | scope creep (bug fix) |
| SD17 | Spec is silent on benchmark JSON write ordering vs op-trace status check | Impl moves `write_json_report` to **after** op-trace status check (Bundle A A3 fix) | scope creep (bug fix) |
| SD18 | Spec is silent on `_finalize_output` naming-protocol three-way search | Impl adds device-category pattern as primary search | scope creep |
| SD19 | Spec is silent on `NvTensorRtRtx` casing fix across 5 files | Impl fixes the latent typo in `analyze/runtime_checker/check_ops.py` and `winml.py` docstring | scope creep (bug fix) |
| SD20 | Spec is silent on `_compile_provider` shadow dict deletion | Impl deletes the dead shadow dict per `2026-05-13-final-taxonomy-cleanup-plan-v3.md` | scope creep |
| SD21 | Spec is silent on `WinMLSession.compile()` resurrection | Impl ships Bug A (defer `InferenceSession` when `enable_ep_context=True`) + Bug B (use free `_build_session_options` + `ort.ModelCompiler.compile_to_file`) + EPDevice-aware `_finalize_output` three-way search | scope creep (compile path was de facto broken pre-this-PR; this is a bug fix + architectural enhancement) |

**Summary:** 7 spec rot items + 14 scope creep items = 21 documented drift instances. None are bugs; all are improvements over the v1.2 spec. The v1.3 bump should treat the spec as needing to *catch up to the code*, not the other way around.

## Architectural critique

Independent of the design docs. The questions in the brief, with judgments:

### Q1: Is `EPDevice` the right level of abstraction? Should it carry the `OrtEpDevice` handle, or is the deferred resolution at session-build time the right call?

**Answer: Deferred resolution is right.** `EPDevice` is a wire-format value object — frozen, JSON-serializable, no runtime dependency on ORT. Storing the `OrtEpDevice` handle on `EPDevice` would couple every CLI argument, every config file, and every test fixture to a live ORT installation. The handle is correctly re-derived inside `_build_session_options(...)` via the registration + filter + dedup pipeline. The one subtle cost is that `_build_session_options` runs `WinMLEPRegistry.get_instance().register_ep(...)` *every time* — which is idempotent post-the-defensive-guard but does walk the catalog. For the volume of session constructions this codebase does, that's negligible. **Verdict: correct call.**

### Q2: Is catalog-tuple ordering as silent preference encoding the right contract?

**Answer: Acceptable trade-off, but the contract should be explicit, not silent.** Today, swapping the order of `EP_DEVICE_SPECS[0]` (QNN-NPU) and `EP_DEVICE_SPECS[1]` (DML-GPU) silently changes the result of `default_device_for_ep("QNNExecutionProvider")` and the auto-resolution priority in `auto_detect_device()`. The docstring at `ep_device.py:166–172` says "Order encodes preference" but does not pin the *names* of the preference axes (per-EP-first-device vs per-device-first-EP). Two future contributors could legitimately disagree on whether moving CPU to position 0 changes "the default device for QNN" or "the default EP for CPU" — both reads are supportable.

The alternative is explicit fields: `EPDeviceSpec(..., is_preferred_for_ep: bool = False, is_preferred_for_device: bool = False)` and `default_device_for_ep` filters by `is_preferred_for_ep`. This is more verbose but eliminates the ordering-as-contract footgun. **Verdict: improvement candidate — add explicit preference fields in a follow-up, keep the tuple's order semantically meaningful for readability only.**

### Q3: Is `EPMonitor.requires_session_teardown` a ClassVar invariant the right ergonomics, or should the session always tear down regardless?

**Answer: ClassVar is correct for the QNN case, but the design over-fits.** The `requires_session_teardown=True` flag exists because QNN flushes its profiling CSV only on `InferenceSession.__del__`. So `WinMLSession.perf().__exit__` must drop the session **before** `monitor.__exit__`. This is genuinely vendor-specific behaviour — there's no universal reason to teardown before monitor exit, and forcing it for all monitors would penalize HWMonitor / VitisAIMonitor for QNN's peculiarity.

That said: an `__init_subclass__` guard that rejects non-bool shadowing at class definition (which the impl has) is a code-smell signal that the class-level invariant is fragile. A cleaner design is to invert: monitors that need teardown-before-exit *declare* it via an explicit method (`def requires_teardown_before_exit(self) -> bool`) and the session calls that method at `__exit__` time. This avoids the class-vs-instance shadowing trap entirely. **Verdict: acceptable for QNN-only present, refactor before second monitor needs the same dance.**

### Q4: Is the two-singleton fix (symmetric defensive guards) acceptable for the long term, or is I1 a real ticking bomb?

**Answer: I1 is a real ticking bomb. The patch is the right move for *this* PR but not for the next.** The probability of process-killing native crashes scales with the number of independent registration call sites. Today: 2 (WinMLEPRegistry, WinML in `winml.py`). Adding a third (e.g. an analyze-pipeline-specific registration) without auditing both existing guards is straightforward to do accidentally. The asymmetric error reporting between `register_to_ort` (writes failures to a dict) and `register_ep` (raises typed exceptions) is the canary — two singletons with two failure-recording schemes is twice the surface area for callers reading `registration_failures` to get confused.

The right fix is one canonical registrar (probably `WinMLEPRegistry`) plus a thin internal-only wrapper for the `winml.py` AppSDK path. The deferral is defensible *if* it ships in the next PR. Past the next PR, it becomes legacy debt. **Verdict: ship the patch, commit I1 to the next PR's title.**

### Q5: Is dropping `_resolve_ep_monitor`'s OpenVINO entry while keeping VitisAI's an acceptable inconsistency?

**Answer: No.** The commit body labels both VitisAI and OpenVINO as "placeholders for parity." Wiring one and not the other reads as "we forgot," not "we deferred." The smallest fix is to either (a) add an OpenVINO branch that returns `NullEPMonitor` for op-tracing requests with a logged-but-not-error warning, or (b) drop the VitisAI special case and let both fall through to the generic `RuntimeError` branch. Either is fine; the current state is the worst of both. **Verdict: tiny follow-up issue, mostly aesthetic but unmistakably drift.**

## Test coverage critique

The commit body claims ~720 passing tests post-squash + 6-command CLI matrix + QDQ-on-NPU verification + new architecture regression tests (`test_qnn_imports.py`, `test_ep_device_import_rule.py`). The per-file analyses surface a number of un-tested code paths. Ranked by load-bearing-ness:

### TC1: Symmetric singleton fix has no integration test that simulates the dual-singleton race under realistic conditions

The commit ships the symmetric defensive guards in *two* files (`winml.py` and `session/ep_registry.py`) and the verification is via 6 CLI commands at `2026-05-13-cli-claims-reverify.md`. None of those commands exercise the actual race window: a synthetic test that calls both `WinMLEPRegistry.register_ep("QNNExecutionProvider")` *and* `WinML().register_execution_providers(ort=ort)` in the same process (in *both* orders) and asserts no `exit(127)` would be the integration test that pins the fix. Without it, a future contributor who refactors either guard has no fail-fast signal. **Severity: high. Effort: S.**

### TC2: `_require()` × 19 rollout in `qnn/_internal.py` has no per-key unit test

Per SUMMARY.md "Three silent-failure paths closed" item 2: the `_require` helper is rolled out 19 times. Each call site raises `KeyError(f"Required QHAS field {key!r} is missing in {context}")` on schema drift. There is no unit test that exercises each of the 19 paths with a synthetic minimal-JSON fixture missing one key. Without it, the schema-drift-surface is not regression-protected. **Severity: medium. Effort: M.**

### TC3: QHAS schema-rename has no fixture verifying both old and new schemas parse

SUMMARY.md "Behavior changes" notes the 8-key rename (`time_us → inference_us`, etc.). External consumers reading the old keys silently see KeyError or None. There is no fixture that simulates an SDK emitting *both* schemas and verifies the parser handles the post-rename one. **Severity: medium. Effort: S.**

### TC4: CUDA / TensorRT runtime is acknowledged untested

The commit body acknowledges the gap: `_SHORT_TO_FULL` was bug-fixed for `cuda`/`tensorrt` but only the 6-command QNN-on-Snapdragon-X-Elite verification ran. The CUDA/TensorRT auto-resolution path is not exercised end-to-end. **Severity: low for this PR, high before a CUDA-using customer hits it. Effort: depends on CI hardware.**

### TC5: OpenVINO monitor is a placeholder with no test

`_resolve_ep_monitor` rejects OpenVINO op-tracing with `RuntimeError`. There is no test that confirms this rejection (positive test of negative behaviour). And there is no integration test that exercises non-op-tracing OpenVINO inference, so the "behaviour change: OpenVINO is now candidate EP for all three device categories" item in SUMMARY.md is unverified. **Severity: medium. Effort: S for the rejection test, L for OpenVINO inference (needs hardware).**

### TC6: `_resolve_op_type` four-layer fallback chain — coverage matrix?

The PRD FR-14 fallback chain has four layers. `per-file/session__monitor__qnn_monitor.md` says `_resolve_op_type` is implemented but does not document per-layer test coverage. A coverage matrix asserting L1 hit, L2 hit (L1 miss), L3 hit (L1+L2 miss), L4 hit (L1+L2+L3 miss) is the load-bearing test for the FR-14 contract. **Severity: medium. Effort: S.**

### TC7: `EPDevice.from_dict` `KeyError` on missing keys

`per-file/session__ep_device.md` "Risks" notes that `EPDevice.from_dict` requires `ep`, `device`, `vendor_id`, `device_id` keys. A round-trip from an older serialized form raises bare `KeyError`. No test for this; tests probably only round-trip a freshly-constructed `EPDevice`. **Severity: low. Effort: S.**

### TC8: `WinMLSession.perf()` auto-reset on differing provider options

`per-file/session__session.md` "Behavior" notes that `perf()` may auto-reset a compiled session with a WARNING. The "tests assert on this" comment in the per-file analysis refers to object-identity preservation, not the auto-reset path. The auto-reset is silent except for a log; no test pins the WARNING emission. **Severity: medium (silent state corruption if the log is suppressed). Effort: S.**

### TC9: `_build_op_type_map` swallows all exceptions and returns `{}`

`per-file/session__session.md` "Risks": `_build_op_type_map` returns `{}` on missing ONNX, corrupt protobuf, missing `onnx` package. Op-tracing then silently falls through L1 to L2+. There is no negative test that pins the silent-degradation behaviour. A future "improvement" that surfaces ONNX-parse errors would break callers that rely on the silent fallback. **Severity: low. Effort: S.**

### TC10: `models/auto.py` positional `ep_device` audit (landing-page M3)

Already on the deferred list. No automated check that callers use `ep_device=` keyword. **Severity: medium. Effort: S (audit + ruff rule or pyflakes check).**

## Risks ranked by likelihood × blast radius

| # | Risk | Likelihood | Blast radius | Trigger | Mitigation |
|---|---|---|---|---|---|
| R2 | Third registration call site added without auditing both symmetric guards | Med | Process-killing `exit(127)` | New analyze pipeline or new tool that calls `ort.register_execution_provider_library` directly | Resolve I1 — collapse to one singleton |
| R3 | `_resolve_ep_monitor` rejects OpenVINO with confusing error referring to `device None` | Low | UX (one CLI command) | `winml perf --ep openvino --op-tracing basic` without `--device` | Fix error message; wire OpenVINO monitor stub |
| R4 | Catalog reorder by future maintainer flips `default_device_for_ep` / `default_ep_for_device` silently | Med | Module (CLI auto-resolution semantics) | Reordering tuple entries for any reason | Add explicit preference fields to `EPDeviceSpec` (Q2); architecture test asserting first entry per-device equals expected |
| R5 | QHAS SDK schema drift breaks parsing; `_require()` raises `KeyError` with context but caller surfaces it as generic "parse_failed" | Med | Local (one op-trace output) | QNN SDK minor-version bump | TC2 + typed `QhasSchemaError` (already on follow-up list) |
| R6 | `available_eps()` `lru_cache(1)` masks dynamic plugin install | Low | Process (one stale lookup) | Test that registers a new EP mid-process | TC + `cache_clear()` after registration |
| R7 | `EPDevice` JSON schema changes in a future PR without `from_dict` version handling | Med | Data-loss (old serialized configs unreadable) | Adding a field to `EPDevice` | Add `version` field + version-aware `from_dict` |
| R8 | `_build_op_type_map` returns `{}` silently; L1 fallback chain silently degraded | Low | Local (one op-trace's name column shows raw paths) | Corrupt ONNX file | Log at WARNING (currently DEBUG); surface in `OpTraceResult.error` |
| R9 | `models/auto.py` positional `ep_device` rebinding | Med | Local (one model load TypeErrors deep in `_load_model`) | Any caller using positional args | Make `ep_device` keyword-only (D11/M3) |
| R10 | `WinMLSession._build_op_type_map` is a `@staticmethod` on the class; future refactor moves session.py and `node.op_type` extraction logic gets duplicated | Low | Module (parser drift) | Refactor that moves staticmethod | Promote to module-level free function (parity with `_build_session_options`) |
| R11 | Architecture regression test (`test_ep_device_import_rule.py`) only catches AST imports, not literal-mapping detection | Med | Module (silent regression) | Future contributor inlines `{"qnn": "QNNExecutionProvider"}` in another file | Extend test per `2026-05-13-final-taxonomy-cleanup-plan-v3.md` Gap 1 |
| R12 | `analyze/runtime_checker/check_ops.py` has hardcoded 5-EP argparse `choices` with no `CARVE-OUT` comment | Low | UX (analyze CLI lags catalog) | Adding a 6th NPU EP | Add CARVE-OUT comment matching `check_patterns.py` |

## Open questions the design-docs explicitly left dangling

From DESIGN-DOCS-INDEX.md, listing every DRAFT or deferred-decision item:

### OQ1 — QuantSpec §10 open questions (DRAFT)

`2026-05-14-quant-spec-design.md` is explicitly DRAFT. §10 lists 5 unanswered design questions:
- Field set finalization (`precision`/`weight_type`/`activation_type`/`symmetric`/`per_channel_weights` — final or evolving?)
- `None` semantics (does `default_quant=None` mean "no opinion" or "use catalog default elsewhere"?)
- Override-vs-merge for user-supplied `quant=` arg
- Validation policy (warn vs error on `QuantSpec` mismatch with model)
- Naming (`QuantSpec` vs `QuantizationSpec` vs `QuantScheme`)

§11 lists 3 unchecked decision boxes (direction approved? landing option? open questions answered?). The commit body and SUMMARY.md both explicitly say "do not implement." Two design hooks exist (`_pre_bench.py` "Surface" sub-block placeholder; `_EP_NAME_ALIASES` migration stub) but no functional code touches QuantSpec.

**Recommendation:** User decision pending. The DRAFT is well-scoped; landing option (a) DEFER is the author's lean and seems right unless a downstream consumer needs per-EP quantization metadata before next quarter.

### OQ2 — I1 two-singleton consolidation

Tracked. See D7 / R2. Recommend: next PR.

### OQ3 — Is the v3 taxonomy cleanup the final state, or is there a v4 implied by code that doesn't match v3?

The v3 plan (`2026-05-13-final-taxonomy-cleanup-plan-v3.md`) says "No v4 expected." But:
- `_EP_NAME_ALIASES` migration stub is still present, marked for removal post-`feat/update-pkg-deps` merge. Until then, every new casing-mismatch must be added by hand.
- `_compile_provider` dead method on `CompileStage` (landing-page M2) — survives at HEAD per audit notes.
- `models/auto.py` positional `ep_device` audit (M3) — not closed.
- `commands/analyze.py` not migrated to typed `EPDevice` — still passes raw strings.
- `commands/build.py` partial migration — auto-resolves but passes strings downstream.

These are all "v4 territory" if you take v3's "post-cleanup state" claim literally. They are not blocking but argue for a v4 doc that closes M1–M3 + the residual analyze/build migrations.

### OQ4 — Iteration 11's decision to delete `QNNProfiler` entirely

Tracked. SC-2 confirms 0 grep hits in `src/`. Not actually dangling.

### OQ5 — `ProofOfExecution` typed accessor follow-up

PRD §1.1 explicitly carves out the `ProofOfExecution` class as a follow-up PR. Shipped code uses transitional `monitor.to_dict()` for VitisAI / OpenVINO. Once `ProofOfExecution` lands, the transitional path in `commands/perf.py:_monitor_to_json_dict` should be retired. Tracked but not blocking.

### OQ6 — `feat/update-pkg-deps` rebase

`_EP_NAME_ALIASES` is a one-line replacement target. Stub is hand-curated; rebase removes the stub.

### OQ7 — Per-EP install hints

`commands/compile.py` hardcodes `EPNotDiscovered`'s install hint to mention `onnxruntime-qnn`. Misleading for OpenVINO/VitisAI/CUDA. Per SUMMARY.md "Open follow-ups" — not closed.

### OQ8 — `winml compile --list` requires `resolve_device` to succeed

UX regression flagged in `per-file/commands__compile.md`. Pure listing fails with `EPNotDiscovered` on hostless boxes. Not closed.

## What I would do next (prioritized)

Nine concrete items. Effort: S < 1 day, M < 3 days, L < 1 week (assuming Claude-pair-programmed; multiply by 2 for solo human).

1. **Write spec v1.3.** Reconcile all 20 spec-drift items in this doc. Anchor on `EPDeviceSpec` catalog as truth, document `perf()` as `@contextmanager`, document the two-singleton patch with I1 follow-up, document `_get_ep_registry()` lazy shim, document `ensure_initialized()`, document QHAS schema rename. **Effort: M.** Site: `docs/design/session/2026-05-11-ep-device-refactor.md` → bump to v1.3, or write new `2026-05-15-ep-device-refactor-v1.3.md`.

2. **Close I1: collapse to one singleton.** Make `WinMLEPRegistry` the canonical registrar. `winml.py:WinML.register_execution_providers` delegates to it. Add an architecture test asserting `register_execution_provider_library` is called from exactly one site under `src/winml/`. **Effort: M.** Sites: `winml.py`, `session/ep_registry.py`, `tests/unit/architecture/`. Also writes the TC1 integration test.

3. **Resolve D11/M3: make `WinMLAutoModel.from_pretrained`'s `ep_device` keyword-only.** Audit all callers. **Effort: S.** Sites: `models/auto.py`, plus any downstream caller using positional args.

4. **Add TC2 + TC3 + TC6 fixture tests for QNN parser.** Per-key `_require()` rollout (19 paths) + post-rename schema fixture + four-layer fallback matrix. **Effort: M.** Sites: `tests/unit/session/monitor/`.

5. **Wire OpenVINO monitor stub or remove VitisAI special case (D9).** Tiny consistency fix. **Effort: S.** Site: `commands/perf.py:_resolve_ep_monitor`.

6. **Migrate `commands/analyze.py` and `commands/build.py` to typed `EPDevice` end-to-end.** SUMMARY.md flags both as cosmetic-only / partial. The CLI surface is the contract; passing raw strings downstream silently undoes the typed-`EPDevice` win for half the CLI commands. **Effort: M.** Sites: `commands/analyze.py`, `commands/build.py`, `analyze/analyzer.py` (signature), downstream `build_hf_model`.

7. **Add explicit preference fields to `EPDeviceSpec` (Q2/R4).** `is_preferred_for_ep: bool = False`, `is_preferred_for_device: bool = False`. Migrate `default_device_for_ep` and `default_ep_for_device` to filter by these flags rather than tuple order. Keep ordering for readability only. **Effort: S.** Site: `session/ep_device.py`.

8. **Parameterize the `EPNotDiscovered` install hint in `commands/compile.py`.** Map EP → package name. Misleading hardcoded `onnxruntime-qnn` for OpenVINO/VitisAI/CUDA. **Effort: S.** Site: `commands/compile.py`, plus a per-EP install-hint table (probably in `session/ep_device.py` as a new optional field on `EPDeviceSpec`).

9. **Promote `_build_op_type_map` to module-level free function** (R10) for parity with `_build_session_options`. One-line move + import update. **Effort: S.** Site: `session/session.py`.

## Confidence statement

**High confidence:**
- The architectural critique sections — `EPDevice` design choices, catalog ordering, the two-singleton patch — these are all visible in code, well-documented in audit trails, and unambiguous in their judgments.
- The spec-drift enumeration. The audits already named most items; this doc puts them in one table with classification.
- The test-coverage gaps. These are direct citations from the per-file analyses.

**Medium confidence:**
- The QuantSpec recommendation to defer. I haven't read the body of `2026-05-14-quant-spec-design.md`, only its summary in DESIGN-DOCS-INDEX.md. A QNN/QDQ specialist might overrule on whether per-EP quantization metadata is needed before next quarter.
- The "ClassVar invariant for `requires_session_teardown`" critique (Q3). The class-vs-instance shadow trap is real but the over-fit concern presupposes a second monitor needs the same dance. If QNN remains the only `requires_session_teardown=True` monitor for the foreseeable future, the current design is fine. Someone with ORT internals expertise might know whether DML / OpenVINO / CUDA have similar teardown-ordering requirements.
- The R7 concern about `EPDevice` schema versioning. I haven't checked whether `EPDevice.from_dict` is called with stored data anywhere (e.g., from a config-file write). If it's only ever a same-process round-trip via `WinMLCompileConfig.to_dict / from_dict`, the versioning concern is academic.

**Lower confidence (where a domain expert might overrule):**
- The QHAS schema-rename impact (SD13/TC3). I don't have a corpus of external consumers of the old keys; the concern is theoretical.
- The `htp_performance_mode="burst"` recommendation as canonical default. SUMMARY.md cites the +3× verification; whether burst-mode is universally safe (vs power/thermal-constrained mode for laptops) is a QNN SDK question. Someone with extended Snapdragon NPU experience would know whether the catalog should encode multiple per-EP-device defaults conditioned on workload (e.g. `htp_performance_mode="balanced"` for long-running workloads).
- The CUDA / TensorRT runtime gap (TC4). Without a CUDA test box, the bug-fix that added `cuda` / `tensorrt` to `_SHORT_TO_FULL` is verified by code review only. A user with a CUDA setup may find that the deduction path still has gaps (e.g. CUDA `compute_capability` mismatch handling).
- The `requires_session_teardown` design (Q3). A QNN SDK specialist might point out that future SDK versions could change the CSV-flush timing, obviating the C-2 invariant entirely — in which case the ClassVar becomes dead.
- The full PRD FR-14 fallback chain correctness. The four layers are implemented per `per-file/session__monitor__qnn_monitor.md`; whether they cover all observed `op_path` shapes from real QNN SDK output requires SDK exposure I don't have.
