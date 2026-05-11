## Functional Correctness: ask of feature owners

**Goal**: every command owner personally validates their command is functionally correct before the **202605 Release**.

Two issues per command. **Both are about functional correctness** — does the command do what it claims, on every input it advertises.

### Issue type 1 — *Run feature-owner E2E acceptance tests*

Identify the functional scenarios for your command (every model format, every supported EP, every flag combination advertised in `--help`) and **implement them as E2E tests in CI**.

### Issue type 2 — *Feature-owner self-check*

Walk your command through a short checklist to confirm it is functionally correct. Each item is answered with linked evidence, or marked **N/A** with a reason.

1. Behavior matches the **PRD / design spec** — both good path and bad path, across all valid option combinations.
2. **No flag is silently ignored or silently changed** during execution.
3. **Functionally correct across all supported EPs.**
4. Runs successfully in a **fresh venv** with the package newly installed (attach log).

## Auto scanned bugs: 46 bugs already filed

**Goal**:  zero open audit bugs per command. That is the bar for "ready to ship" for a user facing CLI product.

Ran every command through the 6-category quality checklist ([docs/cli-command-quality-checklist.md](docs/cli-command-quality-checklist.md)) and filed one issue per concrete defect. Owners pick them up via the milestone query — **no need to open each one in this meeting**.

> Triage query: [`is:issue state:open label:quality milestone:"202605 Release"`](https://github.com/microsoft/WinML-ModelKit/issues?q=is%3Aissue+state%3Aopen+label%3Aquality+milestone%3A%22202605+Release%22)

### Issue categories

- **Discoverability** — can a new user figure the command out from `--help`, `--version`, and error messages alone?
- **Consistency** — does every command use the same flag names, defaults, and short forms as its siblings?
- **UX** — is one run pleasant: actionable errors, bounded log volume, clean rendering on every terminal?
- **Reliability & Performance** — is the command's contract trustworthy: bad input rejected up front, conflicts not silently overridden, fast startup, documented exit codes?
- **Functional Correctness** — does the command actually do what its `--help` claims, on every supported model format and EP?
- **Install & Environment** — does it work on a fresh `pip install`, honor env vars, and stay out of the user's working directory?

### Issues — picked for impact and ease of understanding

**Discoverability**
- [#511](https://github.com/microsoft/WinML-ModelKit/issues/511) Subcommand descriptions in `winml --help` are truncated mid-word.
- [#546](https://github.com/microsoft/WinML-ModelKit/issues/546) `inspect --task bogus-task` leaks `TasksManager` jargon and points to Optimum docs.

**Consistency**
- [#541](https://github.com/microsoft/WinML-ModelKit/issues/541) `-t` means `--model-type` in `hub` but `--task` in `inspect` / `export` / `config`.
- [#514](https://github.com/microsoft/WinML-ModelKit/issues/514) `analyze` lacks `-o` short alias; only `--output` works (every other command has both).
- [#565](https://github.com/microsoft/WinML-ModelKit/issues/565) `-d` short flag is used only by `compile --device`; every other command uses `--device` with no short.

**UX**
- [#542](https://github.com/microsoft/WinML-ModelKit/issues/542) **P0** — `winml inspect` reports both a bogus HF id and a missing local file as "Network error".
- [#535](https://github.com/microsoft/WinML-ModelKit/issues/535) `winml export` crashes with `UnicodeEncodeError` on rocket emoji `\U0001f680` under cp1252 (cmd.exe).
- [#562](https://github.com/microsoft/WinML-ModelKit/issues/562) Missing-required-option errors lack a runnable example (just `Missing option '--model'`).

**Reliability & Performance**
- [#521](https://github.com/microsoft/WinML-ModelKit/issues/521) **P0** — `compile --device cpu --ep qnn` silently overrides device to NPU and exits 0.
- [#552](https://github.com/microsoft/WinML-ModelKit/issues/552) `perf --iterations 0` succeeds with garbage stats and exit 0.
- [#558](https://github.com/microsoft/WinML-ModelKit/issues/558) `winml sys` takes 11.5 s warm for what should be a sub-second local-state command.

**Functional Correctness**
- [#525](https://github.com/microsoft/WinML-ModelKit/issues/525) **P0** — `config --precision int8` silently produces `uint8/uint8`.
- [#534](https://github.com/microsoft/WinML-ModelKit/issues/534) **P0** — `export --dynamo` and `--torch-module` advertised in `--help`; both fail on the documented example model.
- [#550](https://github.com/microsoft/WinML-ModelKit/issues/550) **P0** — `perf --compare-devices` advertised but unimplemented.
- [#555](https://github.com/microsoft/WinML-ModelKit/issues/555) **P0** — `quantize --precision banana` silently falls back to defaults, prints "Success!", exits 1.