# `winml` CLI Quality Documentation

This folder is the home for everything related to auditing and maintaining the quality of the `winml` ModelKit CLI. It contains the rules, the workflow that applies them, the per-command invocation matrix, the latest audit findings, and the captured UX evidence those findings reference.

## What each file is for

| File | Audience | Purpose |
|---|---|---|
| **[`quality-checklist.md`](quality-checklist.md)** | feature owners, reviewers, agents | The 6 rule sections (R1.x Discoverability, R2.x Consistency, R3.x UX, R4.x Reliability & Performance, R5.x Functional Correctness, R6.x Install & Environment). **Single source of truth** for what "Ready" means. |
| **[`quality-check-skill.md`](quality-check-skill.md)** | AI agents, audit drivers | The 5-phase operational workflow (Setup → Static triangulation → Runtime probing → Multi-clause decomposition → Cross-command sweep → Wrap-up) that applies the checklist systematically. Includes failure-mode catalog and CI regression-test template. Ends with a pointer to the human-only Feature-Owner Self-Check (SC-1 / SC-2 / SC-3). |
| **[`CLI_commands.md`](CLI_commands.md)** | feature owners, agents | Per-command invocation matrix — for every subcommand, the representative success scenarios and the failure scenarios that an audit must probe. This is the **input** to Phase 2 of the skill. |
| **[`CLI_quality_check_report.md`](CLI_quality_check_report.md)** | release captain, feature owners | The latest audit findings produced by running the skill against the checklist. 58 findings as of 2026-05-08. Ends with the **Feature-Owner Self-Check appendix** that the release captain signs off. |
| **[`CLI_UX_Capture.md`](CLI_UX_Capture.md)** | reviewers verifying findings | Verbatim captures of `--help` output and one happy-path run per command. Cited as evidence by the report. |

## Reading order

### …if you are a feature owner about to declare a command "Ready"

1. Skim **[`quality-checklist.md`](quality-checklist.md)** — these are the rules your command is graded against.
2. Open **[`CLI_quality_check_report.md`](CLI_quality_check_report.md)**, jump to your command via the Navigation table, and confirm every finding is either fixed or has a tracking issue.
3. Run the three checks in the **Feature-Owner Self-Check Appendix** at the bottom of the report and fill in the SC-1 / SC-2 / SC-3 rows. **Releases require this appendix to be filled in.**

### …if you are an AI agent asked to audit the CLI

1. Read **[`quality-checklist.md`](quality-checklist.md)** end-to-end (the rules).
2. Read **[`quality-check-skill.md`](quality-check-skill.md)** (the 5-phase workflow). Do not skip phases.
3. Use **[`CLI_commands.md`](CLI_commands.md)** as the Phase 2 invocation seed — every "Failure scenarios" row corresponds to a probe you must run.
4. Capture verbatim output in a refreshed copy of **[`CLI_UX_Capture.md`](CLI_UX_Capture.md)**.
5. Emit findings into a refreshed copy of **[`CLI_quality_check_report.md`](CLI_quality_check_report.md)**, including the Feature-Owner Self-Check appendix template (SC-1 / SC-2 / SC-3 rows left as `TODO` — the human owner fills them in).

### …if you are a release captain

1. Open **[`CLI_quality_check_report.md`](CLI_quality_check_report.md)**.
2. Confirm the Summary table P0 count is acceptable for the release.
3. Confirm the **Feature-Owner Self-Check Appendix** has SC-1 / SC-2 / SC-3 all `PASS <date> <owner>` (no `TODO`, no `DEFERRED` without a written plan).
4. Ship.

## Skill registration

The VS Code skill entry-point lives at [`.github/skills/wmk-eval/cli-command-quality-checklist.md`](../../.github/skills/wmk-eval/cli-command-quality-checklist.md). It is now a thin pointer that delegates to the files in this folder.

## Document responsibilities (single-writer rule)

To keep these documents from drifting:

- **`quality-checklist.md`** — owned by CLI maintainers. Changes go through PR review. Add a rule only when a real audit finding had no rule to file under.
- **`quality-check-skill.md`** — owned by whoever drives audits (currently AI agent + reviewer). Update when a failure mode recurs that the workflow did not catch.
- **`CLI_commands.md`** — owned by feature owners + audit drivers. Update when a new command, flag, or scenario is added.
- **`CLI_quality_check_report.md`** — produced by the audit. Replaced wholesale on each audit pass; previous versions live in git history.
- **`CLI_UX_Capture.md`** — produced by the audit; refreshed alongside the report.
