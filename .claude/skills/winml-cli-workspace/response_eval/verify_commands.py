"""Static check: do the winml commands in a response match the CLI contract?

For each `winml ...` command extracted from a response:
  1. The subcommand must exist (`winml <sub> --help` returns a non-empty flag set).
  2. Every flag used in the command must appear in that --help output.
  3. No bare positional model arg (winml is click-based, model IDs go via `-m`).

The CLI's `--help` output is the source of truth — we read its contract,
we don't test its behavior. CLI behavior is the project's CI's job.

Usage:
    python verify_commands.py <response.md> [--json]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(r"C:/repo/WinML-ModelKit")

# Subcommands that legitimately take a positional model arg (none today).
POSITIONAL_OK: set[str] = set()


def extract_commands(text: str) -> list[str]:
    """Pull out winml command invocations from the response.

    Handles inline-backticked commands and code-block lines.
    Skips snippets containing `<placeholder>` tokens, and bare
    `winml <sub>` references with no flags/args (those are prose).
    """
    cmds: list[str] = []
    seen: set[str] = set()
    placeholder_re = re.compile(r"<[^>]+>")

    def _accept(snippet: str) -> None:
        snippet = re.sub(r"^(?:\$\s+|&\s+|uv\s+run\s+|python\s+-m\s+)", "", snippet).strip()
        if placeholder_re.search(snippet):
            return
        if not snippet.startswith("winml"):
            return
        toks = snippet.split()
        if len(toks) < 2:
            return
        if toks[1].startswith("--") and toks[1] not in ("--help", "--version"):
            return
        # `winml <sub>` with no flags/args is a prose reference — skip.
        if len(toks) == 2 and not toks[1].startswith("-"):
            return
        if snippet not in seen:
            seen.add(snippet)
            cmds.append(snippet)

    for m in re.finditer(r"`([^`\n]+)`", text):
        _accept(m.group(1).strip())
    for line in text.splitlines():
        m = re.match(r"^\s*[`$&]?\s*(?:uv\s+run\s+|python\s+-m\s+)?(winml\b[^`#]*)", line)
        if m:
            _accept(m.group(1).strip().rstrip("`").strip())
    return cmds


def help_flags_for(subcmd: str) -> set[str]:
    """Return the set of flag tokens (e.g. '-m', '--model') from `winml <subcmd> --help`."""
    try:
        out = subprocess.run(
            ["uv", "run", "winml", subcmd, "--help"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return set()
    return set(re.findall(r"(?<![A-Za-z0-9])(-{1,2}[a-zA-Z][\w-]*)", out.stdout))


def parse_command(cmd: str) -> tuple[str, list[str], list[str]]:
    """Return (subcommand, flag_tokens_used, positional_tokens)."""
    toks = cmd.split()
    assert toks[0] == "winml"
    if len(toks) < 2:
        return "", [], []
    sub = toks[1]
    if sub.startswith("-"):
        return "", [], []
    rest = toks[2:]
    flags: list[str] = []
    positional: list[str] = []
    i = 0
    while i < len(rest):
        t = rest[i]
        if t.startswith("-"):
            flags.append(t.split("=", 1)[0])
            # Consume the next token as a value if it isn't itself a flag.
            if "=" not in t and i + 1 < len(rest) and not rest[i + 1].startswith("-"):
                i += 2
                continue
        else:
            positional.append(t)
        i += 1
    return sub, flags, positional


def verify(response_path: str) -> dict:
    text = Path(response_path).read_text(encoding="utf-8")
    commands = extract_commands(text)
    results: list[dict] = []
    for cmd in commands:
        sub, flags, positional = parse_command(cmd)
        if not sub:
            continue
        failures: list[str] = []
        help_flags = help_flags_for(sub)
        if not help_flags:
            failures.append(f"unknown subcommand `{sub}` (or `--help` failed)")
        else:
            for f in flags:
                if f not in help_flags:
                    failures.append(f"flag `{f}` not in `winml {sub} --help`")
            if positional and sub not in POSITIONAL_OK:
                failures.append(f"positional arg(s) {positional} — `winml {sub}` is flag-only")
        results.append({"command": cmd, "subcommand": sub, "failures": failures, "passed": not failures})
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "response_path": str(response_path),
        "command_count": total,
        "passed": passed,
        "failed": total - passed,
        "commands": results,
    }


def render_human(report: dict) -> str:
    lines: list[str] = []
    n = report["command_count"]
    if n == 0:
        return "No winml commands found in response."
    lines.append(f"Found {n} winml command(s):\n")
    for r in report["commands"]:
        lines.append(f"  $ {r['command']}")
        if r["failures"]:
            for f in r["failures"]:
                lines.append(f"    FAIL: {f}")
        else:
            lines.append("    OK (flags + shape match --help)")
        lines.append("")
    lines.append("=" * 60)
    lines.append(f"Passed: {report['passed']}/{n}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = list(argv)
    as_json = "--json" in args
    if as_json:
        args.remove("--json")
    if len(args) != 1:
        print(__doc__)
        return 2
    report = verify(args[0])
    print(json.dumps(report, indent=2) if as_json else render_human(report))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
