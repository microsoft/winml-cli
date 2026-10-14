# PR Routing

Every confirmed champion gets one standalone `promotion_handoff.json` with
schema `model-support-promotion-v1`. `promotion.py create` validates the final
bundle and records its manifest, the promotion context, and exactly two routes.

| PR class | Owner | Label |
| --- | --- | --- |
| `optimizer` | auto-optimize | `model-opt-by-skill` |
| `recipe` | adding-model-support | `model-scale-by-skill` |

Auto-optimize creates and reviews only optimizer PRs.
Adding-model-support creates and reviews only recipe PRs.
Neither owner creates the other class.

## Classification

- A required generic capability change selects `optimizer`.
- A durable Hugging Face model ID, revision, and task select `recipe`.
- Recipe-only work starts `ELIGIBLE`.
- Optimizer-only work leaves recipe `BLOCKED_IDENTITY`.
- Mixed work orders optimizer before recipe and keeps recipe
  `BLOCKED_ON_OPTIMIZER`.

## Mixed Results

The optimizer owner records its Draft PR and reviewed commit. After the PR is
merged, fetch current main before running the mixed merge update. The helper
does not fetch. It requires local `refs/remotes/origin/main` to equal the given
current-main commit and verifies:

```text
reviewed commit -> merged commit -> current main
```

Only then does it mark optimizer `MERGED` and recipe `ELIGIBLE`. Squash or
rebase history that does not preserve the reviewed commit stays blocked.

## Commands

```text
python scripts/promotion.py create --bundle <bundle> --context <context.json>
python scripts/promotion.py validate --handoff <promotion_handoff.json>
python scripts/promotion.py update --handoff <promotion_handoff.json> --route <optimizer|recipe> --status <status>
python scripts/promotion.py update --handoff <promotion_handoff.json> --route optimizer --status MERGED --optimizer-repo <repo> --merged-commit <sha> --current-main-commit <sha>
```

The handoff is outside the bundle. Route updates never mutate the validated
model bundle, report, or manifest.
## Bundle lifecycle

Finalize and validate the replayed bundle before promotion.py create. Freeze it once the handoff records its manifest hash. Keep subsequent PR URLs and review state in the standalone handoff. Changed model evidence requires a new versioned bundle and handoff; never rewrite the bundle behind an existing handoff.

Replay requires PowerShell 7.3 or newer. The generated wrapper enables native-command error handling: a nonzero native exit stops replay before later commands can overwrite the failure.

## Creation ownership

Feature Gap Engineer returns implementation and public-path evidence only.
The main agent finalizes and validates the bundle, creates the handoff, then
creates at most one eligible optimizer Draft PR and verifies its label before
Check-in Reviewer. With no generic source change, skip optimizer PR creation
and optimizer label requirements. Missing labels block that route, not an
already validated model bundle.
