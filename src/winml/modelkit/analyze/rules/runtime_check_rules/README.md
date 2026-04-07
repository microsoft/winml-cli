# Runtime Check Rules

This directory contains zip files with runtime check rules (negative rules and tables) used by the static analyzer. Each zip corresponds to a specific `{EP}_{Device}_{Domain}_opset{N}` combination.

The zip files are **not tracked by git**. They are hosted in a separate repo.

## Setup

### Option 1: Download script (recommended)

First, configure git credentials for the `gim-home` org (one-time setup):

```bash
gh auth switch --user <your_gim-home_account>
gh auth setup-git
```

Then download:

```bash
uv run python scripts/download_rules.py
```

The script does a sparse checkout (downloads only the zip folder, not the full repo) and copies files here.

Use `--force` to re-download all files even if they already exist locally.

### Option 2: Manual copy

Copy all `*.zip` files from [`gim-home/ModelKitArtifacts/op_check_results/rules/`](https://github.com/gim-home/ModelKitArtifacts/tree/main/op_check_results/rules) into this directory.

## What happens if zips are missing

The analyzer will log a warning and treat affected operators as unknown. Analysis results will be incomplete but will not crash.
