# WinML CLI Agent Design

> Status: Draft — 2026-06-11  
> Context: Strategic design for the agent layer of winml-cli

---

## 1. Context: Why Agent Matters for winml-cli

### 1.1 winml-cli vs Olive — The Real Distinction

Microsoft Olive already exists as a pass-based optimization framework supporting QNN, DML, and other Windows EPs. The temptation is to dismiss winml-cli's agent as redundant with Olive. That would be wrong — the distinction is fundamental:

| Dimension | Olive | winml-cli |
| --- | --- | --- |
| Target user | ML engineer who understands ORT internals | WinApp developer who wants their model to work on Windows |
| Workflow | Compose passes manually, specify EP upfront | `config` + `build` — two commands, full pipeline |
| Hardware selection | Manual EP specification | `--device auto` — detects hardware, selects EP |
| Explainability | Silent pipeline output | Designed for transparency |
| Windows-first | Cross-platform, Windows supported | Built exclusively for Windows hardware diversity |
| Operator diagnostics | Not available | `winml analyze` — operator linting, EP compatibility |
| Agent-ready | Not designed for it | First-class design goal |

**Analogy:** Olive is webpack (powerful, expert-configured); winml-cli is Vite (opinionated, works for most cases out of the box).

### 1.2 The Core Gap Agent Should Fill

WinApp developers lack access to a senior ML engineer who:

- Knows why a model fails on QNN NPU for this specific operator pattern
- Can read an error message and immediately know the root cause
- Understands which optimization knob to turn for which problem
- Knows how a config that works on Snapdragon X Elite will behave on Intel Meteor Lake

**The agent's job is to be that person.**

---

## 2. Agent Design Philosophy

### 2.1 The Wrong Design (Current Autoconfig)

The current autoconfig agent runs a **headless search loop**:
Explorer → Optimizer → Reviewer → repeat

**Problems with this approach:**

- A Python script can do benchmark loops faster, cheaper, and more reliably than an LLM agent
- Results (config files) are not auditable — developer cannot verify why a config was chosen
- No explainability — developer doesn't understand what was decided or why
- Treats developer as absent; no collaborative interaction
- The "agentic" overhead (LLM inference cost per loop iteration) adds nondeterminism without intelligence

Autoconfig search is useful as a **sub-tool**, not as the primary value proposition of the agent layer.

### 2.2 The Right Design: Diagnosis + Guidance over Search

Agent excels at **judgment, diagnosis, and explanation** — not computation. The redesign centers on:

> **When a developer encounters a problem, the agent gives explanation + executable next step — not a config file.**

#### Design Principles

1. **Explain, don't just output**  
   Instead of silently picking an EP, say: *"I picked QNN EP because your device has a Qualcomm NPU. Operator coverage is 97% — the remaining 3% fall back to CPU, which is acceptable for these specific ops."*
2. **Fix, don't just diagnose**  
   When an incompatible operator is found, apply the graph transformation — don't just flag it.
3. **Developer talks, agent acts**  
   The agent is interactive and conversational. Developer says "this model is slow on GPU" → agent asks clarifying questions, runs targeted experiments, explains findings.
4. **Progressive trust**  
   Show confidence levels. Be explicit about uncertainty. Let the developer see what the agent is doing. Never give false precision (e.g., "Config A is 3% faster" when standard deviation is 5%).
5. **Windows device diversity as first-class concern**  
   Always reason about what happens on devices the developer doesn't have — not just the machine the agent runs on.

---

## 3. Agent Types

### 3.1 Diagnostic Agent *(highest priority)*

**Trigger:** Model fails to load, crashes at inference, throws EP compatibility error  
**Developer question:** "My model fails on QNN NPU — why? What do I do?"

**Agent responsibilities:**

- Parse error message → identify root cause (unsupported op, shape mismatch, driver version, etc.)
- Analyze model graph → enumerate incompatible operators per EP
- Propose and apply concrete fix (graph transformation, operator substitution, fallback EP)
- Verify fix with `winml eval` accuracy check

**Why this is Olive-incompatible:** Olive doesn't converse, doesn't diagnose, doesn't explain. It fails silently or produces a broken model.

**Example interaction:**

```javascript
Developer: winml build failed. Error: "QNNExecutionProvider: Unsupported op at node /conv/Conv_3"
Agent: Found it. Conv_3 has dynamic padding — QNN NPU requires static shapes.
       I'll apply DynamicToFixedShape transform and re-run the compile.
       [applies fix] → Build succeeded. NPU latency: 12.3ms. Accuracy delta: 0.01%.
```

---

### 3.2 Decision Guidance Agent

**Trigger:** Developer is at a decision point in the pipeline (which EP? which precision? to quantize or not?)  
**Developer question:** "I don't know what options to pick. What's the tradeoff?"

**Agent responsibilities:**

- Run quick comparative benchmarks (not exhaustive search)
- Present tradeoffs with numbers: latency gain vs accuracy delta vs model size
- Make a recommendation with reasoning, not just a number
- Let developer override with understanding of consequences

**Key difference from autoconfig:** This is interactive and decision-oriented, not headless. The developer is in the loop.

---

### 3.3 Cross-Device Confidence Agent *(winml-cli unique)*

**Trigger:** Developer has a working config, asks "will this work on my users' devices?"  
**Developer question:** "My app ships on many Windows hardware configs. Will this be okay?"

**Agent responsibilities:**

- Given a config optimized for Device A, reason about behavior on Device B, C...
- Identify configs that are device-specific (compiled QNN binaries only work on Qualcomm)
- Generate multi-device config with automatic EP fallback chain (QNN → DML → CPU)
- Surface warnings: "This config will fail on Intel Meteor Lake — here's the fallback"

**Why this matters:** WinApp developers ship to millions of devices. No other tool addresses Windows hardware diversity in the deployment sense.

---

### 3.4 Regression Detection Agent *(CI/CD scenario)*

**Trigger:** ORT version bump, driver update, or scheduled CI run  
**Developer question:** "Something changed — my model got slower / broke"

**Agent responsibilities:**

- Compare before/after perf numbers with statistical validity (not point estimates)
- Correlate change with known ORT/EP changelog entries
- Identify which layer / operator regressed using profiler output
- Propose workaround or file structured bug report

---

## 4. Role of Autoconfig (Perf Search) in This Design

Autoconfig (opset × EP × opt\_level search) is **not abandoned** — it becomes a sub-tool invoked by the agents above when appropriate.

**When it's invoked:**

- Diagnostic Agent: "Let me try a few configs to see if this performance issue is config-related"
- Decision Guidance Agent: "Let me run a quick sweep across opt\_level to show you the tradeoff"

**What changes:**

- It's not the primary entry point
- Its output is explained, not just surfaced as a config file
- It runs fewer, targeted experiments (not exhaustive grid search)
- Accuracy validation (`winml eval`) is mandatory before any recommendation

---

### 3.5 Model Recommendation Agent *(developer onboarding scenario)*

**Trigger:** Developer describes a problem they want to solve but doesn't know which model to use  
**Developer question:** "I want to add real-time object detection to my app. What model should I use?"

**Agent responsibilities:**

- Understand the developer's task from natural language (classification, detection, segmentation, NLP, etc.)
- Query `winml catalog` to enumerate available built-in models for that task
- Factor in the developer's current hardware (`winml sys`) — which EPs are available, memory constraints, NPU presence
- Recommend the best-fit model with reasoning: why this model, expected latency on their device, accuracy tier
- Output a ready-to-run command so the developer can immediately try it

**Why this is valuable:** Developers often don't know which model architecture fits their use case, let alone which one performs well on their specific Windows hardware. The catalog has verified, pre-tested models — the agent bridges "I have a problem" to "here is a model you can ship today."

**Example interaction:**

```javascript
Developer: I want to classify product images in my e-commerce app. Should be fast on low-end laptops.
Agent: For image classification on low-end hardware without an NPU, I recommend:

  → microsoft/resnet-50  (catalog built-in)
    - Task: image-classification
    - Latency on CPU: ~18ms p50 (tested on Intel Core i5)
    - Size: 98MB FP32, 25MB w8a16
    - Best EP on your machine: CPUExecutionProvider

  Runner-up: google/vit-base-patch16-224 (better accuracy, ~2x slower on CPU)

  To build and benchmark:
    winml build -c $(winml config -m microsoft/resnet-50 --device auto) -o resnet_out/
    winml perf -m resnet_out/model.onnx --device auto --iterations 100
```

**What makes this different from a search engine:** The recommendation is hardware-aware — the same question asked on a machine with a Qualcomm NPU would surface a different model (or a different EP for the same model) with different expected numbers. It's not a static lookup, it's a contextual match.

---

## 5. Key Concerns to Track

| Concern | Mitigation |
| --- | --- |
| Device heterogeneity: config found on Dev's machine may not generalize | Cross-Device Confidence Agent explicitly addresses this; output includes device scope |
| Trust/auditability: developer can't verify agent recommendation | All recommendations include reasoning + confidence + "how I tested this" |
| Olive overlap at implementation layer | winml-cli uses ORT under the hood like Olive; the differentiation is UX + Windows-first + explainability, not reimplementing optimization passes |
| Accuracy validation | `winml eval` is mandatory in every agent loop that modifies the model |
| Agent hallucinating perf numbers | All perf claims require iteration ≥ 1000 and report p50/p90/p99 with std dev |

---

## 6. Open Questions

1. **Scope**: Should the agent be a CLI mode (`winml agent`) or embedded into existing commands (`winml build --agent`)?
2. **Olive relationship**: Should winml-cli contribute opset search back to Olive, or maintain it independently? Needs alignment with Olive team.
3. **Offline / no-LLM mode**: Should the agent work without LLM (rule-based fallback) for air-gapped CI environments?
4. **Multi-device testing**: Cross-Device Confidence Agent requires access to multiple devices or a device simulation layer — how to implement?
