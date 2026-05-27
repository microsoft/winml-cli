# Issues: docs/commands/perf.md

Source verified against: `src/winml/modelkit/commands/perf.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- (none)

## Important (misleading or stale)

- **`--compare-devices` flag does not exist in source.** The flag table lists `--compare-devices | TEXT | — | Not yet implemented`. A full search of `perf.py` shows zero occurrences of `compare_devices` or `compare-devices` as a defined click option. The flag is documented but never registered; passing it will produce a "No such option" error. The note "Not yet implemented" is insufficient — the flag should either be removed from the table entirely or marked explicitly as "not defined, will error if passed".
- **`--op-tracing` is hidden in source.** `perf.py:1184`: `hidden=True` — the flag is intentionally hidden from `--help` output. The doc exposes it in the flag table without any note that it does not appear in `--help`. Consider adding "(hidden from --help output; not ready for general use)" to the description.
- **Default output path documented as "`{model_slug}_perf.json`" is wrong.** Source `perf.py:871` generates the path as `~/.cache/winml/perf/<slug>[/<module_class>]/<timestamp>.json`, not a file in the current working directory. Users expecting a local file will be confused.

## Minor (polish)

- **Flag table omits `--verbose` / `-v`.** Defined at `perf.py:1183-1191`.
- **Flag table omits `--build-config` / `-c` (the shared build config option).** `perf.py:1192` registers `@cli_utils.build_config_option`.
- **`--shape-config` description says "Ignored for pre-exported ONNX files and in `--module` mode"** — correct; both branches issue warnings at `perf.py:1280-1284` and `perf.py:1351-1356`. The doc accurately describes this behavior.

## Verified correct (key claims checked)

- `--model` / `-m` required (enforced in body, not `required=True`) → `perf.py:1092`, `perf.py:1243`
- `--task` string default None → `perf.py:1093-1098`
- `--iterations` IntRange min=1 default 100 → `perf.py:1099-1105`
- `--warmup` IntRange min=0 default 10 → `perf.py:1106-1111`
- `--device` choice `auto|cpu|gpu|npu` default `auto` → `perf.py:1113` via `cli_utils.device_option`
- `--precision` string default `auto` → `perf.py:1114-1120`
- `--ep` via `cli_utils.ep_option` → `perf.py:1121-1124`
- `--batch-size` int default 1 → `perf.py:1129-1135`
- `--shape-config` path default None → `perf.py:1136-1142`
- `--no-quantize` flag default false → `perf.py:1143-1148`
- `--rebuild` flag default false → `perf.py:1149-1153`
- `--ignore-cache` flag default false → `perf.py:1154-1159`
- `--module` string default None → `perf.py:1160-1170`
- `--monitor` flag default false → `perf.py:1171-1176`
- `--op-tracing` choice `basic|detail` default None → `perf.py:1177-1184`
- `--compare-devices` marked "not yet implemented" → confirmed not implemented (flag absent from source)
- Statistics include mean, min, max, P50, P90, P95, P99, std → `perf.py:104-109` (BenchmarkResult fields)
- `--monitor` includes hw metrics in JSON → `perf.py:127`, `perf.py:167-168`
- No `wmk` or `ModelKit` strings in user-facing prose → confirmed
