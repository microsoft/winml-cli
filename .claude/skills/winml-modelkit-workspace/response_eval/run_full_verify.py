"""Run verify_commands.py across all with_skill responses in an iteration, write a markdown report.

Usage:
    python run_full_verify.py [iter-name]      # default: latest iter-* directory
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # response_eval/
VERIFY = HERE / "verify_commands.py"
ITER_ROOT = HERE / "iterations"


def latest_iter() -> Path:
    candidates = sorted(ITER_ROOT.glob("iter-*"))
    if not candidates:
        raise SystemExit(f"No iter-* directories found under {ITER_ROOT}")
    return candidates[-1]


def main(argv: list[str]) -> int:
    if argv:
        iter_dir = ITER_ROOT / argv[0]
        if not iter_dir.exists():
            raise SystemExit(f"Iteration not found: {iter_dir}")
    else:
        iter_dir = latest_iter()

    rows = []
    total_commands = 0
    total_passed = 0

    for eval_dir in sorted(iter_dir.glob("eval-*")):
        resp = eval_dir / "with_skill" / "run-1" / "outputs" / "response.md"
        if not resp.exists():
            continue
        r = subprocess.run(
            ["python", str(VERIFY), str(resp), "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        report = json.loads(r.stdout)
        rows.append((eval_dir.name, report))
        total_commands += report["command_count"]
        total_passed += report["passed"]

    out: list[str] = []
    out.append(f"# {iter_dir.name} — CLI command verification report\n")
    out.append("Static check only: parse `winml` commands from each response, verify subcommand exists and every flag appears in `winml <cmd> --help`. No bare positional model arg.\n")
    out.append("## Overall\n")
    out.append(f"- Commands extracted: **{total_commands}**")
    out.append(f"- Passed static check: **{total_passed}/{total_commands}**\n")
    for eval_name, report in rows:
        out.append(f"## {eval_name}\n")
        out.append(f"Commands: {report['command_count']} | Passed: {report['passed']}/{report['command_count']}\n")
        for r in report["commands"]:
            cmd = r["command"]
            tag = "OK" if r["passed"] else "FAIL"
            out.append(f"### `{cmd}`")
            out.append(f"- **{tag}**")
            if r["failures"]:
                for f in r["failures"]:
                    out.append(f"  - {f}")
            out.append("")

    report_path = iter_dir / "cli_verification.md"
    report_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Static: {total_passed}/{total_commands}")
    return 0 if total_passed == total_commands else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
