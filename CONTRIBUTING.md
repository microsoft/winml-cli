# Contributing

We're always looking for your help to improve the product (bug fixes, new features, documentation, etc).

## Contribute a code change

* Start by reading the project [README](./README.md) to understand the scope and goals of ModelKit.
* If your change is non-trivial or introduces new public facing APIs, please use the [feature request issue template](https://github.com/microsoft/ModelKit/issues/new) to discuss it with the team first.
* For all other changes, you can directly create a pull request (PR) and we'll be happy to take a look.
* Make sure your PR adheres to the coding conventions and standards below.

## Getting started

See the [README](./README.md#getting-started) for prerequisites and installation instructions. Then set up your development environment:

```bash
uv sync
uv run pre-commit install
```

This installs all dependencies and enables [pre-commit hooks](https://pre-commit.com/) that automatically check license headers, formatting (Ruff), trailing whitespace, and YAML syntax on every commit.

### Runtime check rules (for `analyze` development)

If you are working on the analyzer (`winml analyze`, `src/winml/modelkit/analyze/`), you need to populate the runtime check rule zips locally. Without them, the analyzer logs a warning and treats affected operators as unknown — analysis results are incomplete but commands do not crash.

External contributors (without `gim-home` org access) should download the rule zips from the **latest** [WinML-ModelKit release](https://github.com/microsoft/WinML-ModelKit/releases/latest) into `src/winml/modelkit/analyze/rules/runtime_check_rules/`:

```bash
gh release download --repo microsoft/WinML-ModelKit --pattern '*.zip' --dir src/winml/modelkit/analyze/rules/runtime_check_rules
```

Each asset is named `{EP}_{Device}_{Domain}_opset{N}.zip`; download only the combinations relevant to your work.

For all setup options (including the internal `gim-home` download script and the `MODELKIT_RULES_DIR` override), see [`src/winml/modelkit/analyze/rules/runtime_check_rules/README.md`](./src/winml/modelkit/analyze/rules/runtime_check_rules/README.md).

## Coding conventions and standards

### Python code style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [Google's Python style guide](https://google.github.io/styleguide/pyguide.html). A maximum line length of 100 characters is enforced.

This project uses the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **Mypy** for type checking
- **Pytest** for testing

Before submitting a pull request, please ensure:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pytest tests/
```

### Testing

New code *must* be accompanied by unit tests. Code coverage should aim at maintaining over 80% coverage.

### License header

All Python source files **must** include the MIT license header at the top of the file:

```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
```

The pre-commit hook configured in [Getting started](#getting-started) will automatically insert this header on new `.py` files.

## Licensing guidelines

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of conduct

See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## Report a security issue

See [SECURITY.md](./SECURITY.md).
