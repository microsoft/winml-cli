# Runtime Check Rules

This directory contains parquet runtime rule artifacts used by the analyzer.

Files are **not tracked by git**. They are bundled in wheel installs and can
also be fetched from release assets or `ModelKitArtifacts` for source builds.

## Setup

### Option 1: Download from the latest GitHub release (for source builds)

If you are building from source code (for example, cloning this repo), download
the `rules.zip` asset from the latest WinML-ModelKit release.

```bash
gh release download --repo microsoft/WinML-ModelKit --pattern 'rules.zip' --dir .
```

`gh release download` defaults to the latest release. Use `--tag <version>`
to pin a specific release if you need a reproducible snapshot.

Then extract `rules.zip` into this directory:

```powershell
Expand-Archive -Path .\rules.zip -DestinationPath src\winml\modelkit\analyze\rules\runtime_check_rules -Force
```

```bash
unzip -o rules.zip -d src/winml/modelkit/analyze/rules/runtime_check_rules
```

The zip preserves file paths relative to `runtime_check_rules/`.

### Option 2: Download script (Microsoft internal fallback)

Requires [GitHub CLI](https://cli.github.com) (`gh`) with an account that has access to `gim-home`.

```bash
uv run python scripts/download_rules.py --account <your_gim-home_account>
```

The script sparse-checkouts `gim-home/ModelKitArtifacts/rules` and copies all `*.parquet`
files here (preserving subdirectories).

This script downloads from the internal `ModelKitArtifacts` repo, not from
WinML-ModelKit release assets.

Use `--force` to re-download all files even if they already exist locally.

### Option 3: Manual copy (Microsoft internal fallback)

Copy all runtime rule parquet files from:

`gim-home/ModelKitArtifacts/rules/`

### Option 4: Use external rules directories via environment variable

Set `WINMLCLI_RULES_DIR` to one or more directories containing parquet rule artifacts.

Important: relative paths are resolved from `src/winml/modelkit/analyze/utils/` (the
directory of `rule_loader.py`), not from the current terminal working directory.

- Windows (PowerShell, user-level absolute path): `[Environment]::SetEnvironmentVariable("WINMLCLI_RULES_DIR", "C:\*path*\rules", "User")`
- Windows (PowerShell, user-level repo-relative path): `[Environment]::SetEnvironmentVariable("WINMLCLI_RULES_DIR", "..\..\..\..\..\..\ModelKitArtifacts\rules", "User")`

Multiple directories are supported using `os.pathsep` (`;` on Windows, `:` on Unix-like systems).

## Rule lookup order

The analyzer searches directories in this order:

1. Directories listed in `WINMLCLI_RULES_DIR` (left to right)
2. Embedded default directory: `src/winml/modelkit/analyze/rules/runtime_check_rules/`

`WINMLCLI_RULES_DIR` takes precedence over the embedded default when the same parquet file
exists in multiple locations.

## What happens if parquet rules are missing

`winml analyze` exits with code 2 and prints an error. Reinstall the package first,
or use one of the fallback methods above to provide parquet rule files.
