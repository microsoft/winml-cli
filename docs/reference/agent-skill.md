# Agent Skill

winml-cli ships a **Copilot Skill** (`use-winml-cli`) that lets AI coding agents
drive the entire model-building pipeline on your behalf. When a coding agent has
this skill attached, it can inspect models, generate configs, run builds, and
interpret results — without you having to remember exact flags or stage ordering.

---

## What the skill provides

The skill teaches the agent:

| Capability | What the agent learns |
|---|---|
| **Pipeline shape** | The stage order (`inspect → export → analyze → optimize → quantize → compile → perf`) and when to enter mid-pipeline |
| **Flag discovery** | Always run `winml <command> --help` before quoting a command — never fabricate flags |
| **Output mapping** | Which command's `-o` produces the artifact the user actually needs |
| **Scope awareness** | Which model architectures are supported (classic DL) vs. out-of-scope (LLMs, diffusion) |
| **Hardware detection** | Use `winml sys --list-ep` to confirm what's available before targeting an EP |
| **Two paths** | When to use primitives (debugging, exploring) vs. config + build (production, CI) |

---

## How to use it

### In GitHub Copilot (Chat / Workspace)

The skill is automatically available when working in this repository. Ask
Copilot to build, benchmark, or debug a model and it will follow the skill's
guidance:

```
@workspace Build microsoft/resnet-50 for my NPU and show me the latency
```

### In other agents (Copilot Extensions, custom MCP)

Point the agent at the skill file:

```
skills/use-winml-cli/SKILL.md
```

The skill uses standard Copilot Skill format (YAML front-matter + markdown
body). Any agent that supports skill ingestion can consume it directly.

---

## Skill location

```
winml-cli/
└── skills/
    └── use-winml-cli/
        └── SKILL.md          ← the skill definition
```

---

## Key principles encoded in the skill

1. **Inspect first** — always run `winml inspect` before building to catch
   unsupported architectures early.

2. **Don't fabricate flags** — if a flag isn't in `--help`, it doesn't exist.
   The skill enforces this as a hard rule.

3. **Published outputs only** — each command has an explicit `-o` output; never
   fish artifacts from internal cache.

4. **EP-compiled models are EP-bound** — don't benchmark a QNN-compiled model on
   the CPU EP. Use the pre-compile optimized ONNX for cross-EP comparison.

5. **Scope gate** — the agent will refuse to attempt generative/decoder-only
   models (GPT, LLaMA, Phi, Stable Diffusion) and explain they're out of scope.

---

## Example agent interaction

```
User: Can I run ConvNeXt on my Snapdragon X Elite NPU?

Agent (with skill):
1. Runs `winml sys --list-ep` → confirms QNNExecutionProvider is registered
2. Runs `winml inspect -m microsoft/convnext-tiny-224` → confirms supported
3. Runs `winml config --onnx ... -d npu -o config.json`
4. Runs `winml build -c config.json -m microsoft/convnext-tiny-224 -o output/`
5. Runs `winml perf -m output/model.onnx -d npu --monitor`
6. Reports latency + NPU utilization to user
```

---

## Updating the skill

The skill lives at `skills/use-winml-cli/SKILL.md` in the repository root.
When commands or flags change, update both the docs site and the skill file to
keep agent behavior aligned with the CLI.
