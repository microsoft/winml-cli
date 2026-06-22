# Release Built-in Models & Recipes

This runbook is the single entry point for two recurring release tasks:

1. **Refresh `examples/recipes/README.md`** (Built-in Model count + Models table) on the current working branch and push.
2. **Pick recipe configs** onto a fresh branch forked from `main` and open a PR.

The two non-trivial pieces of logic (eval-result counting and recipe picking) live in
Python scripts; everything else (`git`, `gh`) is invoked directly from the commands below.

| Script | Purpose |
|---|---|
| [scripts/rebuild_recipes_readme.py](../scripts/rebuild_recipes_readme.py) | Regenerate the `## Models` section of `examples/recipes/README.md` from eval results. Prose before `## Models` is preserved verbatim. |
| [scripts/pick_builtin_recipes.py](../scripts/pick_builtin_recipes.py) | Copy qualifying recipe configs into `examples/recipes/<slug>/`. Supports `--dry-run` and `--prune`. Does **not** modify the README. |

---

## Definitions

### The 10 (EP, device) buckets

A *bucket* is one folder under `examples/<ep>/<device>/`. The full set is:

| # | EP | Device | Folder | Result filename pattern |
|---|---|---|---|---|
| 1 | `dml` | gpu | `examples/dml/gpu/<slug>/` | `<task>_eval_result.json` |
| 2 | `mlas` | cpu | `examples/mlas/cpu/<slug>/` | `<task>_eval_result.json` |
| 3 | `migraphx` | gpu | `examples/migraphx/gpu/<slug>/` | `<task>_eval_result.json` |
| 4 | `nv_tensorrt_rtx` | gpu | `examples/nv_tensorrt_rtx/gpu/<slug>/` | `<task>_eval_result.json` |
| 5 | `openvino` | cpu | `examples/openvino/cpu/<slug>/` | `<task>_eval_result.json` |
| 6 | `openvino` | gpu | `examples/openvino/gpu/<slug>/` | `<task>_eval_result.json` |
| 7 | `qnn` | gpu | `examples/qnn/gpu/<slug>/` | `<task>_eval_result.json` |
| 8 | `openvino` | npu | `examples/openvino/npu/<slug>/` | `<task>_<precision>_eval_result.json` |
| 9 | `qnn` | npu | `examples/qnn/npu/<slug>/` | `<task>_<precision>_eval_result.json` |
| 10 | `vitisai` | npu | `examples/vitisai/npu/<slug>/` | `<task>_<precision>_eval_result.json` |

`<slug>` = `<hf_id>` with the first `/` replaced by `_` (e.g. `microsoft/resnet-50` → `microsoft_resnet-50`).

See also [test_config.md](test_config.md) and [generate_config.md](generate_config.md) for the layout these results come from.

### Configs and eval results per bucket

- **CPU/GPU buckets** (7 total): one config per task, `<task>_config.json`, with one matching eval result `<task>_eval_result.json` (EP default precision — treated as fp16-equivalent).
- **NPU buckets** (3 total): up to three configs per task — `<task>_fp16_config.json`, `<task>_w8a8_config.json`, `<task>_w8a16_config.json` — each with a matching `<task>_<precision>_eval_result.json` when its eval passes.

### Built-in Model criterion

A `(model, task)` pair is **Built-in** iff:

- All 7 CPU/GPU buckets contain `<task>_eval_result.json` (CPU/GPU eval passes), **and**
- All 3 NPU buckets contain `<task>_fp16_eval_result.json` (NPU fp16 eval passes).

NPU w8a8 / w8a16 eval results play **no role** in the Built-in criterion.

### Recipe picking criterion

For each Built-in `(model, task)`, recipes are copied from an NPU bucket into `examples/recipes/<slug>/`:

- **fp16** recipe: always picked (Built-in guarantees NPU fp16 passed on every NPU EP, so the source always exists).
- **w8a8** recipe: picked iff at least one NPU EP has `<task>_w8a8_eval_result.json`; sourced from that EP.
- **w8a16** recipe: same, for `<task>_w8a16_eval_result.json`.

Composite tasks (e.g. CLIP zero-shot) match multiple files via `<task>_<precision>_config*.json` and all are copied. CPU/GPU configs are not copied — the recipe set is intentionally NPU-shaped.

---

## Stage 1 — Refresh `recipes/README.md` (this branch)

Regenerate the README in-place (preserving prose above `## Models`), commit on the **current** branch, push.

```powershell
uv run python scripts/rebuild_recipes_readme.py
git add examples/recipes/README.md
git diff --cached --quiet examples/recipes/README.md
if ($LASTEXITCODE -ne 0) {
    git commit -m "examples/recipes: refresh built-in model README"
    git push origin (git rev-parse --abbrev-ref HEAD)
} else {
    Write-Host "README unchanged; nothing to commit."
}
```

---

## Stage 2 — Pick recipe configs onto a branch off `main` and open a PR

Both Python scripts read `<ep>/<device>/.../<task>_*_eval_result.json` from disk,
so they must run on a branch that **has** those eval results (typically your
working branch). All generation happens **before** we touch `main`. We then
snapshot the final `examples/recipes/` to a temp dir, switch to a fresh branch
forked from `main`, and replace `examples/recipes/` wholesale — so additions
**and deletions** relative to `main` both land in the PR, and the fresh branch
never needs the scripts.

```powershell
# 0. Run on the branch that contains the eval results. Tree must be clean.
git status --porcelain  # must be empty

# 1. Pick recipes into examples/recipes/<slug>/.
uv run python scripts/pick_builtin_recipes.py --prune

# 2. Rebuild README so the Models table matches the picked configs.
uv run python scripts/rebuild_recipes_readme.py

# 3. Snapshot the final examples/recipes/ (configs + README) to a temp dir.
$tmp = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("builtin-recipes-" + [guid]::NewGuid().Guid))
Copy-Item -Recurse examples/recipes/* $tmp
Write-Host "snapshot at $tmp"

# 4. Restore working branch (drop the picker/README changes locally).
git checkout -- examples/recipes
git clean -fd examples/recipes

# 5. Fork from origin/main.
git fetch origin main
$branch = "shzhen/update-builtin-recipes-" + (Get-Date -Format yyyyMMdd)
git switch --create $branch origin/main

# 6. Replace examples/recipes/ wholesale with the snapshot (README included).
Get-ChildItem examples/recipes | Remove-Item -Recurse -Force
Copy-Item -Recurse $tmp/* examples/recipes/ -Force
Remove-Item -Recurse -Force $tmp

# 7. Commit, push, open PR. No Python is run on this branch.
git add examples/recipes
git commit -m "examples/recipes: refresh built-in model recipes"
git push --set-upstream origin $branch
gh pr create --base main --head $branch `
    --title "Refresh built-in model recipes" `
    --body "Auto-generated by ``scripts/pick_builtin_recipes.py --prune``."
```

The fresh branch only ever sees `examples/recipes/` — the scripts and this
runbook stay on the working branch and never leak into the PR. If you need
to recover from an interrupted run, the snapshot lives at the `$tmp` path
printed by step 3.
