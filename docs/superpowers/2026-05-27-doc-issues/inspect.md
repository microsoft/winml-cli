# Issues: docs/commands/inspect.md

Source verified against: `src/winml/modelkit/commands/inspect.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--model` is listed as "required" in the flag table** (doc line 22: "Required unless `--help` is used") but the source marks it `required=False` (line 63). The command accepts `--model-type` or `--model-class` as alternatives; source line 165 raises `UsageError` only when all three (`model_id`, `model_type`, `model_class`) are None. Users who read the doc and omit `-m` expecting a usage error will instead succeed with `--model-type`.

- **`--list-tasks` flag is not documented at all.** Source lines 98–103 define `@click.option("--list-tasks", "list_tasks", is_flag=True, ...)`. Omitting it from the flags table means users cannot discover this flag. Running `winml inspect --list-tasks` exits early printing all known tasks (lines 157–161) — a useful shortcut completely hidden from the doc.

- **`--model-type` and `--model-class` flags are not documented.** Source lines 104–116 define `--model-type` (can replace `-m`) and `--model-class` (can replace `-m`). The doc synopsis says `-m <model_id>` is the only input path. Users have no way to discover the type-only or class-only inspection paths shown in the source docstring examples.

## Important (misleading or stale)

- **`-v` / `--verbose` flag is absent from the flag table.** Source lines 78–83 define `@click.option("-v", "--verbose", is_flag=True, ...)`. Verbose mode changes JSON/table output to include full configuration details (passed as `verbose=verbose` to `output_json` and `output_table` at lines 229–231).

- **"How it works" says `--hierarchy` uses `AutoModel.from_config()` and records a "forward-pass trace"** — source lines 449–458 show `extract_hierarchy(model_id)` is called, but this is `from ..inspect.hierarchy import extract_hierarchy` which is a separate module. The source comment at line 451 says "requires model_id" (line 452: `if include_hierarchy and model_id:`), not just a config fetch. The claim that "no real weights are downloaded" should be verified against `extract_hierarchy`.

- **`--format` choices are documented as `table | json`** — source line 74 confirms `click.Choice(["table", "json"])`, so this is correct. However the doc uses backtick-escaped `table` and `json` which is fine.

## Minor (polish)

- The `--help / -h` row in the flag table is auto-added by Click and does not need to be listed explicitly.
- The synopsis shows `$ winml inspect -m <model_id> [options]` but since `-m` is not required, the synopsis should read `$ winml inspect [options]` or include alternates.
- The example `winml inspect -m facebook/convnext-tiny-224 -v -H` uses `-v` which is a real and functional flag, but since `-v` is not in the flag table the user has no context for it. Consistent with the missing `--verbose` entry.

## Verified correct (key claims checked)

- `-m` / `--model` short form exists → source line 62.
- `-f` / `--format` with `Choice(["table", "json"])`, default `"table"` → source lines 70–76.
- `-t` / `--task` with no required constraint, default `None` → source lines 85–90.
- `-H` / `--hierarchy` as `is_flag=True, default=False` → source lines 91–97.
- Command does not accept `--device`, `--ep`, `--precision`, `--output` → confirmed absent.
- `--format json` output goes to stdout, banners go to stderr → source lines 33–35.
- `--list-tasks` requires no model and lists `KNOWN_TASKS` → source lines 157–161.
