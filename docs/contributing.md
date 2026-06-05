# Contributing

For the full contributing guide — development setup, coding conventions, testing, PR checklist, and CLA — see [`CONTRIBUTING.md`](https://github.com/microsoft/winml-cli/blob/main/CONTRIBUTING.md) in the repository root.

## Quick Reference

```bash
# Clone and set up
git clone https://github.com/microsoft/winml-cli.git
cd winml-cli
uv sync --extra dev
uv run pre-commit install

# Download runtime check rules (required for `winml analyze`)
gh release download <tag> --repo microsoft/winml-cli --pattern 'rules-v*.zip' --dir .
# Windows:
Expand-Archive -Path .\rules-v*.zip -DestinationPath src\winml\modelkit\analyze\rules\runtime_check_rules -Force
# Linux/macOS:
# unzip -o rules-v*.zip -d src/winml/modelkit/analyze/rules/runtime_check_rules

# Run tests
uv run pytest tests/ -m "not e2e and not npu and not gpu"

# Lint and format
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

# Docs preview
uv run mkdocs serve
```

## See also

- [Installation](getting-started/installation.md) — user-facing setup
- [Commands](commands/overview.md) — CLI reference
