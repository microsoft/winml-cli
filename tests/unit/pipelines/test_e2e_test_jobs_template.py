# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_python_e2e_steps_force_utf8_encoding() -> None:
    template = REPO_ROOT / ".pipelines" / "templates" / "e2e-test-jobs.yml"
    text = template.read_text(encoding="utf-8")

    for command in (
        'python", "scripts/e2e_eval/run_eval.py"',
        "python -m pytest tests/e2e/test_${{ target }}_e2e.py",
    ):
        command_index = text.index(command)
        prefix = text[max(0, command_index - 500) : command_index]
        assert '$env:PYTHONUTF8 = "1"' in prefix
        assert '$env:PYTHONIOENCODING = "utf-8"' in prefix


def test_pytest_step_tolerates_only_post_junit_access_violation() -> None:
    template = REPO_ROOT / ".pipelines" / "templates" / "e2e-test-jobs.yml"
    text = template.read_text(encoding="utf-8")
    pytest_index = text.index("python -m pytest tests/e2e/test_${{ target }}_e2e.py")
    pytest_block = text[pytest_index : pytest_index + 1800]

    assert "$accessViolationExitCodes" in pytest_block
    assert "-1073741819" in pytest_block
    assert "3221225477" in pytest_block
    assert 'SelectNodes("//testsuite")' in pytest_block
    assert "$failures -eq 0 -and $errors -eq 0" in pytest_block
