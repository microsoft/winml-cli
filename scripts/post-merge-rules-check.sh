#!/usr/bin/env bash
# post-merge hook: download runtime check rules if manifest changed.
# Install: cp scripts/post-merge-rules-check.sh .git/hooks/post-merge && chmod +x .git/hooks/post-merge

MANIFEST="src/winml/modelkit/analyze/rules/runtime_check_rules/rules_manifest.json"

# Check if manifest was part of the merge diff
if git diff-tree -r --name-only ORIG_HEAD HEAD | grep -q "$MANIFEST"; then
    echo "[post-merge] rules_manifest.json changed, checking rule files..."
    if command -v uv >/dev/null 2>&1; then
        uv run python scripts/download_rules.py || echo "[post-merge] WARNING: rule download failed, run 'python scripts/download_rules.py' manually"
    elif command -v python >/dev/null 2>&1; then
        python scripts/download_rules.py || echo "[post-merge] WARNING: rule download failed, run 'python scripts/download_rules.py' manually"
    else
        echo "[post-merge] WARNING: python not found, run 'python scripts/download_rules.py' manually"
    fi
fi
