# Issues: docs/commands/build.md

Source verified against: `src/winml/modelkit/commands/build.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--random-init` flag does not exist.** The flag table lists `--random-init` as "Skip weight download; build with random weights". A full search of `build.py` finds no `--random-init` or `random_init` option definition. The behavior (random-weight build) is supported by omitting `-m` (see `build.py:247`: "Omit for random-weight build"), but there is no `--random-init` flag. Users who pass `--random-init` will get "No such option".
- **`--config` / `-c` listed as *(required)* but source marks it `required=False`.** `build.py:237` sets `required=False` with `default=None`. When `-c` is omitted, config is auto-generated from `-m`. The doc makes it sound mandatory.

## Important (misleading or stale)

- **`--qnn-sdk-root` should not appear in this page.** The flag does not exist in `build.py` (confirmed: zero hits for `qnn_sdk_root` or `qnn-sdk-root` in the option definitions). It is a `winml compile` flag only. Its appearance in the flag table is a copy-paste error.
- **`--no-compile` is documented as a simple flag but source defines a `--no-compile/--compile` toggle pair.** `build.py:275-282` shows `--no-compile/--compile` as a boolean toggle with `default=None`. The doc only shows `--no-compile`, omitting `--compile` (which forces compilation on when the config has a compile section). The `--compile` positive form is useful and undocumented.
- **Flag table omits `--trust-remote-code`.** `build.py:312-314` defines this via `cli_utils.trust_remote_code_option(...)`. Users building custom architecture models (e.g., Mu2) need it.
- **`--max-optim-iterations` table shows default `3` but source default is `None`.** `build.py:309` sets `default=None`. The actual default of `3` is enforced inside the pipeline helpers (`build.py:1112, 1234`), not at the CLI layer. If the user does not pass the flag, Click resolves it as `None`, not `3`.

## Minor (polish)

- **Flag table omits `--verbose` / `-v`.** Defined at `build.py:315-320`.
- **"How it works" says pipeline is "export → optimize → quantize → compile" in the intro, but the synopsis shows the full correct form.** The command map table in overview.md correctly shows "export → optimize → quantize → compile". The build.md intro paragraph at line 44 says only "export → quantize → compile" (missing optimize). Minor omission but inconsistent.

## Verified correct (key claims checked)

- `--config` / `-c` path, optional → `build.py:233-241`
- `--model` / `-m` string default None → `build.py:242-248`
- `--output-dir` / `-o` path default None → `build.py:249-256`
- `--use-cache` flag default false → `build.py:257-262`
- `--rebuild` flag default false → `build.py:263-268`
- `--no-quant` flag default false → `build.py:269-274`
- `--no-optimize` flag default false → `build.py:299-304`
- `--no-analyze` flag default false → `build.py:293-298`
- `--ep` defined via `cli_utils.ep_option` → `build.py:283-286`
- `--device` defined via `cli_utils.device_option` default `auto` → `build.py:287-292`
- Mutual exclusion: `--output-dir` and `--use-cache` → `build.py:376-379`
- `--use-cache` not supported in module mode → `build.py:491-495`
- ONNX input skips export stage → `build.py:691-711` (`_build_onnx_pipeline`)
- No `wmk` or `ModelKit` strings in user-facing prose → confirmed
