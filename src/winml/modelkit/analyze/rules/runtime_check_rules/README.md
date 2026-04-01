# Runtime Check Rules

This directory contains zip files with runtime check rules (negative rules and tables) used by the static analyzer. Each zip corresponds to a specific `{EP}_{Device}_{Domain}_opset{N}` combination.

The zip files are **not tracked by git**. They are hosted on a GitHub Release and downloaded on demand.

## Setup (first time)

```bash
uv run python scripts/download_rules.py
```

This reads `rules_manifest.json`, compares sha256 hashes, and downloads only missing or changed files from the `runtime-rules` GitHub Release.

## Check status

```bash
uv run python scripts/download_rules.py --check
```

## Update rules after running the runtime checker

```bash
# 1. Generate new zips (also auto-updates rules_manifest.json)
uv run python -m winml.modelkit.analyze.runtime_checker.result_processor \
    <input_dir> --opset_version 17 --opset_domain ai.onnx --update-zip

# 2. Upload to the existing release (overwrites same-name assets)
gh release upload runtime-rules *.zip --repo microsoft/ModelKit --clobber

# 3. Commit the updated manifest
git add rules_manifest.json
git commit -m "update runtime check rules"
```

## Auto-download on git pull (optional)

Install the post-merge hook to automatically download rules when `rules_manifest.json` changes:

```bash
cp scripts/post-merge-rules-check.sh .git/hooks/post-merge
chmod +x .git/hooks/post-merge
```

The hook only runs when the manifest file is part of the merge diff. Download failures are non-blocking (warning only).

## How it works

- `rules_manifest.json` is the source of truth: it lists every zip file with its sha256 hash and size.
- Download uses `gh` CLI for authentication (works with private repos), falling back to direct URL for public repos.
- The `winml analyze` command warns at startup if any rule files are missing.
- At build time, zip files in this directory are included in the wheel via `pyproject.toml` package-data.
- A CI workflow (`.github/workflows/verify-rules.yml`) checks that manifest hashes match release assets on PRs that modify the manifest.

## Recovering a deleted release

If the `runtime-rules` release is accidentally deleted, recreate it from local zip files:

```bash
# Recreate the release
gh release create runtime-rules \
    src/winml/modelkit/analyze/rules/runtime_check_rules/*.zip \
    --repo microsoft/ModelKit \
    --title "Runtime check rules" \
    --notes "Runtime check rule zip files for static analyzer." \
    --prerelease

# Verify
uv run python scripts/download_rules.py --check
```

Any developer who has the zip files locally can perform this recovery.
