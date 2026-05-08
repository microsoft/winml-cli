# Runtime Check Rules

This directory contains parquet runtime rule artifacts used by the analyzer.

Files are **not tracked by git**. They are bundled in wheel installs and can
also be fetched from release assets or `ModelKitArtifacts` for source builds.

## Setup

No manual download is required for normal `pip`/wheel installs. Runtime rule
parquet files are bundled and installed with the package.

If `winml analyze` reports missing parquet files, first reinstall the package.

### Option 1: Download from the latest GitHub release (for source builds)

If you are building from source code (for example, cloning this repo), download
the parquet assets from the latest WinML-ModelKit release.

```bash
gh release download --repo microsoft/WinML-ModelKit --pattern '*.parquet' --dir src/winml/modelkit/analyze/rules/runtime_check_rules
```

`gh release download` defaults to the latest release. Use `--tag <version>`
to pin a specific release if you need a reproducible snapshot.

### Option 2: Download script (Microsoft internal fallback)

Requires [GitHub CLI](https://cli.github.com) (`gh`) with an account that has access to `gim-home`.

```bash
uv run python scripts/download_rules.py --account <your_gim-home_account>
```

The script sparse-checkouts `gim-home/ModelKitArtifacts/rules` and copies all `*.parquet`
files here (preserving subdirectories).

Use `--force` to re-download all files even if they already exist locally.

### Option 3: Manual copy (Microsoft internal fallback)

Copy all runtime rule parquet files from:

`gim-home/ModelKitArtifacts/rules/`

### Option 4: Use external rules directories via environment variable

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

`winml analyze` exits with code 2 and prints an error. Reinstall the package first,
or use one of the fallback methods above to provide parquet rule files.
