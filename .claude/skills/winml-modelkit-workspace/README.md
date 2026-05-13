# winml-modelkit eval workspace

Dev-only infrastructure for iterating + validating the skill. Three pillars, each with its own runbook the driving agent should follow:

| Pillar | What it measures | How to iterate |
|---|---|---|
| **1. Trigger** | Does the agent decide to load the skill given a user prompt? | [`trigger_eval/RUNBOOK.md`](trigger_eval/RUNBOOK.md) |
| **2. Response** | Given the skill loads, is the advice sound? | [`response_eval/RUNBOOK.md`](response_eval/RUNBOOK.md) |
| **3. E2E** | Does the agent's recipe actually work on real hardware? | [`/run-e2e`](../../commands/run-e2e.md) slash command + [`e2e_eval/RUNBOOK.md`](e2e_eval/RUNBOOK.md) |

For design rationale (why three pillars, what's out of scope), see [`winml-cli-skill-design.md`](../../../winml-cli-skill-design.md) at repo root.

When a user asks the driving agent to iterate any pillar, **the agent should open the relevant RUNBOOK and follow its steps**.
