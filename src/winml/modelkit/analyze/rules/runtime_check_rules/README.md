# Runtime Check Rules

This directory contains parquet runtime rule artifacts used by the analyzer.

Files are **not tracked by git**. They are bundled in wheel installs and can
also be fetched from release assets or `ModelKitArtifacts` for source builds.

## Setup

### Option 1: Download from a GitHub release (for source builds)

If you are building from source code (for example, cloning this repo), download
the `rules-v<version>.zip` asset from a winml-cli release. Replace `<version>`
with the release version you want (e.g. `0.0.3`) and `<tag>` with the matching
release tag (e.g. `v0.0.3`); all current releases are pre-releases, so an
explicit tag is required (`gh release download` without a tag skips
pre-releases).

```bash
gh release download <tag> --repo microsoft/winml-cli --pattern 'rules-v<version>.zip' --dir .
```

Then extract the archive into this directory:

```powershell
Expand-Archive -Path .\rules-v<version>.zip -DestinationPath src\winml\modelkit\analyze\rules\runtime_check_rules -Force
```

```bash
unzip -o rules-v<version>.zip -d src/winml/modelkit/analyze/rules/runtime_check_rules
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
winml-cli release assets.

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

`WINMLCLI_RULES_DIR` overrides — it does not augment — the embedded default:

- If `WINMLCLI_RULES_DIR` is set, only the directories it lists are searched (left to right).
  The embedded default directory is **not** consulted, so those directories must contain every
  parquet rule you need.
- If `WINMLCLI_RULES_DIR` is unset or empty, only the embedded default directory is searched:
  `src/winml/modelkit/analyze/rules/runtime_check_rules/`.

## What happens if parquet rules are missing

`winml analyze` exits with code 2 and prints an error. Reinstall the package first,
or use one of the fallback methods above to provide parquet rule files.
