# Runtime Check Rules

This directory contains zip files with runtime check rules (negative rules and tables) used by the static analyzer. Each zip corresponds to a specific `{EP}_{Device}_{Domain}_opset{N}` combination.

The zip files are **not tracked by git**. They are hosted on a GitHub Release and downloaded on demand.

## Setup (first time)

```bash
winml rules download
```

This reads `rules_manifest.json`, compares sha256 hashes, and downloads only missing or changed files from the `runtime-rules` GitHub Release.

## Check status

```bash
winml rules status
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

## How it works

- `rules_manifest.json` is the source of truth: it lists every zip file with its sha256 hash and size.
- `winml rules download` uses `gh` CLI for authentication (works with private repos), falling back to direct URL for public repos.
- The `winml analyze` command warns at startup if any rule files are missing.
- At build time, zip files in this directory are included in the wheel via `pyproject.toml` package-data.

## CI cache key

Use `winml rules cache-key` to get a short hash of the manifest for cache invalidation:

```yaml
- run: echo "key=$(winml rules cache-key)" >> $GITHUB_OUTPUT
  id: rules-key
- uses: actions/cache@v4
  with:
    path: src/winml/modelkit/analyze/rules/runtime_check_rules/*.zip
    key: rules-${{ steps.rules-key.outputs.key }}
```
