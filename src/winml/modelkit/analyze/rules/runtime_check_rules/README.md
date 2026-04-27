# Runtime Check Rules

This directory contains parquet runtime rule artifacts used by the analyzer.

Files are **not tracked by git** and are expected to come from `ModelKitArtifacts`.

## Setup

### Option 1: Download script (recommended)

Requires [GitHub CLI](https://cli.github.com) (`gh`) with an account that has access to `gim-home`.

```bash
uv run python scripts/download_rules.py --account <your_gim-home_account>
```

The script sparse-checkouts `gim-home/ModelKitArtifacts/rules` and copies all `*.parquet`
files here (preserving subdirectories).

Use `--force` to re-download all files even if they already exist locally.

### Option 2: Manual copy

Copy all runtime rule parquet files from:

`gim-home/ModelKitArtifacts/rules/`

### Option 3: Use external rules directories via environment variable

Set `MODELKIT_RULES_DIR` to one or more directories containing parquet rule artifacts.

Important: relative paths are resolved from `src/winml/modelkit/analyze/utils/` (the
directory of `rule_loader.py`), not from the current terminal working directory.

- Windows (PowerShell, user-level absolute path): `[Environment]::SetEnvironmentVariable("MODELKIT_RULES_DIR", "C:\*path*\rules", "User")`
- Windows (PowerShell, user-level repo-relative path): `[Environment]::SetEnvironmentVariable("MODELKIT_RULES_DIR", "..\..\..\..\..\..\ModelKitArtifacts\rules", "User")`

Multiple directories are supported using `os.pathsep` (`;` on Windows, `:` on Unix-like systems).

## Rule lookup order

The analyzer searches directories in this order:

1. Directories listed in `MODELKIT_RULES_DIR` (left to right)
2. Embedded default directory: `src/winml/modelkit/analyze/rules/runtime_check_rules/`

`MODELKIT_RULES_DIR` takes precedence over the embedded default when the same parquet file
exists in multiple locations.

## What happens if parquet rules are missing

The analyzer logs warnings and treats affected operators as unknown. Analysis results remain
available but may be incomplete.
