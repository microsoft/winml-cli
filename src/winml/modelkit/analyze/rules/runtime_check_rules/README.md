# Runtime Check Rules

This directory contains zip files with runtime check rules (negative rules and tables) used by the static analyzer. Each zip corresponds to a specific `{EP}_{Device}_{Domain}_opset{N}` combination.

The zip files are **not tracked by git**. They are hosted in a separate repo.

## Setup

### Option 1: Download from the latest GitHub release (for external contributors)

Rule zips are published as individual assets on the **latest** [WinML-ModelKit release](https://github.com/microsoft/WinML-ModelKit/releases/latest). No special access required — this is the recommended path for external contributors who do not have `gim-home` org membership.

Each asset is named `{EP}_{Device}_{Domain}_opset{N}.zip` (for example, `QNNExecutionProvider_NPU_ai.onnx_opset17.zip`). Download only the combinations you need and place them in this directory.

To download all assets from the latest release with [GitHub CLI](https://cli.github.com):

```bash
gh release download --repo microsoft/WinML-ModelKit --pattern '*.zip' --dir src/winml/modelkit/analyze/rules/runtime_check_rules
```

`gh release download` defaults to the latest release. Pin to a specific tag with `--tag <version>` (for example, `--tag v0.0.1`) if you need a reproducible snapshot.

### Option 2: Download script (Microsoft internal)

For Microsoft developers with access to the `gim-home` org. Requires [GitHub CLI](https://cli.github.com) (`gh`) authenticated with such an account.

```bash
uv run python scripts/download_rules.py --account <your_gim-home_account>
```

The script uses the specified `gh` account's token to authenticate, does a sparse checkout (downloads only the zip folder, not the full repo), and copies files here.

Use `--force` to re-download all files even if they already exist locally.

### Option 3: Manual copy (Microsoft internal)

Copy all `*.zip` files from [`gim-home/ModelKitArtifacts/op_check_results/rules/`](https://github.com/gim-home/ModelKitArtifacts/tree/main/op_check_results/rules) into this directory.

### Option 4: Use external rules directory via environment variable

Set `MODELKIT_RULES_DIR` to one or more directories containing runtime rule zip files.

Important: relative paths are resolved from `src/winml/modelkit/analyze/utils/` (the directory of `rule_loader.py`), not from your current terminal working directory.

- Windows (PowerShell, user-level absolute path): `[Environment]::SetEnvironmentVariable("MODELKIT_RULES_DIR", "C:\*path*\rules_zip", "User")`
- Windows (PowerShell, user-level repo-relative path): `[Environment]::SetEnvironmentVariable("MODELKIT_RULES_DIR", "..\..\..\..\..\..\ModelKitArtifacts\rules_zip", "User")`

Multiple directories are supported using `os.pathsep` (`;` on Windows, `:` on Unix-like systems).

### Option 5: Expand rule zips via CLI command

You can materialize delta snapshots to full payloads in-place with:

```bash
winml expand_rules
```

This command reads all entries from `MODELKIT_RULES_DIR`, resolves each via
`_resolve_env_rules_dir_entry`, and performs in-place rewrite for each existing
directory that contains matching zip files.

After a folder is successfully expanded (and has at least one matching zip),
an empty marker file named `expanded` is created in that folder.

You can also override the path entry:

```bash
winml expand_rules --rules-dir-entry C:\path\to\rules_zip
```

Multiple explicit entries are supported:

```bash
winml expand_rules --rules-dir-entry C:\path\a --rules-dir-entry C:\path\b
```

## Rule zip lookup order

The analyzer searches zip files in this order:

1. Directories listed in `MODELKIT_RULES_DIR` (left to right)
2. This embedded directory: `src/winml/modelkit/analyze/rules/runtime_check_rules/`

This means `MODELKIT_RULES_DIR` takes precedence over the embedded default directory when the same zip filename exists in both locations.

## What happens if zips are missing

The analyzer will log a warning and treat affected operators as unknown. Analysis results will be incomplete but will not crash.
