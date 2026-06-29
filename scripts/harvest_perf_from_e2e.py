"""Harvest perf_result.json files for the 29 newly-added (model, task) tuples.

Source: C:\\Users\\shzhen\\Downloads\\e2e_model_coverage_result 2\\e2e_model_coverage_result
        Each EP dir contains date subdirs, each with a models/ subdir holding
        <owner>__<name>__<task>/eval_result.json.

Behavior:
- For every (EP_dir, new_model, task) we pick the latest date subdir that has
  a matching model dir with perf.passed == True.
- We parse iterations/warmup from perf.command, the latency table + Throughput
  from perf.stdout_output, and Inputs/Outputs from perf.stderr_output.
- raw_samples_ms and latency_ms.warmup_mean stay empty/null (not in source).
- We never overwrite an existing *_perf_result.json file.
- Output filename uses {task}_perf_result.json for cpu/gpu and
  {task}_fp16_perf_result.json for npu (matching the existing fp16 configs).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SRC_ROOT = Path(r"C:\Users\shzhen\Downloads\e2e_model_coverage_result 2\e2e_model_coverage_result")

# Source EP dir -> (examples ep folder, hardware, EP name string, NPU precision suffix or None)
SRC_EP_MAP = {
    "DmlExecutionProvider_GPU":      ("dml",             "gpu", "DmlExecutionProvider",       None),
    "MLASExecutionProvider_CPU":     ("mlas",            "cpu", "CPUExecutionProvider",       None),
    "OpenVINOExecutionProvider_CPU": ("openvino",        "cpu", "OpenVINOExecutionProvider",  None),
    "OpenVINOExecutionProvider_GPU": ("openvino",        "gpu", "OpenVINOExecutionProvider",  None),
    "OpenVINOExecutionProvider_NPU": ("openvino",        "npu", "OpenVINOExecutionProvider",  "fp16"),
    "QNNExecutionProvider_GPU":      ("qnn",             "gpu", "QNNExecutionProvider",       None),
    "QNNExecutionProvider_NPU":      ("qnn",             "npu", "QNNExecutionProvider",       "fp16"),
    "TRTRTXExecutionProvider_GPU":   ("nv_tensorrt_rtx", "gpu", "NvTensorRTRTXExecutionProvider", None),
    "VitisAIExecutionProvider_NPU":  ("vitisai",         "npu", "VitisAIExecutionProvider",   "fp16"),
}

# The 29 newly-added (hf_id, task) tuples (must match generate_example_configs.py).
NEW_MODELS = [
    ("AdamCodd/vit-base-nsfw-detector", "image-classification"),
    ("ahotrod/electra_large_discriminator_squad2_512", "question-answering"),
    ("amunchet/rorshark-vit-base", "image-classification"),
    ("apple/mobilevit-small", "image-classification"),
    ("cross-encoder/ms-marco-MiniLM-L4-v2", "text-classification"),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", "text-classification"),
    ("dima806/fairface_age_image_detection", "image-classification"),
    ("distilbert/distilbert-base-cased-distilled-squad", "question-answering"),
    ("distilbert/distilbert-base-uncased", "fill-mask"),
    ("distilbert/distilbert-base-uncased-distilled-squad", "question-answering"),
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "text-classification"),
    ("Falconsai/nsfw_image_detection", "image-classification"),
    ("google-bert/bert-base-multilingual-cased", "fill-mask"),
    ("google-bert/bert-base-multilingual-cased", "masked-lm"),
    ("hustvl/yolos-small", "object-detection"),
    ("Intel/dpt-hybrid-midas", "depth-estimation"),
    ("Isotonic/distilbert_finetuned_ai4privacy_v2", "token-classification"),
    ("Jean-Baptiste/camembert-ner-with-dates", "token-classification"),
    ("kredor/punctuate-all", "token-classification"),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "zero-shot-classification"),
    ("microsoft/resnet-18", "image-classification"),
    ("monologg/koelectra-small-v2-distilled-korquad-384", "question-answering"),
    ("sentence-transformers/all-mpnet-base-v2", "feature-extraction"),
    ("sentence-transformers/all-mpnet-base-v2", "fill-mask"),
    ("sentence-transformers/all-mpnet-base-v2", "sentence-similarity"),
    ("sentence-transformers/multi-qa-mpnet-base-dot-v1", "feature-extraction"),
    ("sentence-transformers/multi-qa-mpnet-base-dot-v1", "fill-mask"),
    ("sentence-transformers/multi-qa-mpnet-base-dot-v1", "sentence-similarity"),
    ("valentinafeve/yolos-fashionpedia", "object-detection"),
]


def _to_slug(hf_id: str) -> str:
    return hf_id.replace("/", "_")


def _to_src_dir_name(hf_id: str, task: str) -> str:
    return f"{hf_id.replace('/', '__')}__{task}"


def _list_date_dirs(ep_dir: Path) -> list[Path]:
    """Date subdirs sorted newest-first (string sort works for MMDD)."""
    return sorted((d for d in ep_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)


def _find_source_entry(ep_dir: Path, hf_id: str, task: str) -> Path | None:
    """Return the eval_result.json for the latest date that has a matching model dir."""
    name = _to_src_dir_name(hf_id, task)
    for date_dir in _list_date_dirs(ep_dir):
        candidate = date_dir / "models" / name / "eval_result.json"
        if candidate.exists():
            return candidate
    return None


# ---- Parsers -----------------------------------------------------------------

_RE_ITER = re.compile(r"--iterations\s+(\d+)")
_RE_WARM = re.compile(r"--warmup\s+(\d+)")
_RE_THROUGHPUT = re.compile(r"Throughput:\s*([0-9.]+)\s*samples/sec")


def _parse_iter_warm(command: str) -> tuple[int | None, int | None]:
    it = _RE_ITER.search(command or "")
    wm = _RE_WARM.search(command or "")
    return (int(it.group(1)) if it else None, int(wm.group(1)) if wm else None)


def _parse_latency_and_throughput(stdout: str) -> tuple[dict | None, float | None]:
    """Extract the single latency data row (Avg/P50/P90/P95/P99/Min/Max/Std)
    and Throughput samples/sec from the perf stdout."""
    if not stdout:
        return None, None

    throughput = None
    m = _RE_THROUGHPUT.search(stdout)
    if m:
        throughput = float(m.group(1))

    # Find the data row inside the box-drawn table: a line whose alnum-stripped
    # content is exactly 8 floats separated by the box vertical "|" or "│".
    latency = None
    for raw in stdout.splitlines():
        # Replace any non-ASCII char with a single space, then collapse to fields.
        cleaned = re.sub(r"[^\x20-\x7e]", " ", raw)
        # Split on '|' too if present
        cleaned = cleaned.replace("|", " ")
        parts = cleaned.split()
        if len(parts) != 8:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        # Header row is text only and won't pass float(); separator rows are
        # whitespace-only and have len(parts)!=8. So the first numeric 8-tuple is
        # the latency data row.
        avg, p50, p90, p95, p99, mn, mx, std = nums
        latency = {
            "mean": avg,
            "min": mn,
            "max": mx,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "std": std,
            "warmup_mean": None,
        }
        break

    return latency, throughput


def _parse_inputs_outputs(stderr: str) -> dict | None:
    """Parse the 'Inputs:' / 'Outputs:' block that perf prints at the end of stderr.

    Format example:
        Inputs:      pixel_values         [1, 3, 384, 384]       float32
                     extra_input          [1, 16]                int32
        Outputs:     start_logits         [1, 512]
                     end_logits           [1, 512]
    """
    if not stderr:
        return None

    lines = stderr.splitlines()
    # Find the Inputs: line. Walk from end to find the *last* occurrence (since
    # repeated runs may print multiple).
    in_idx = None
    out_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("Inputs:"):
            in_idx = i
        elif ln.lstrip().startswith("Outputs:"):
            out_idx = i
    if in_idx is None or out_idx is None or out_idx <= in_idx:
        return None

    def _strip_label(s: str, label: str) -> str:
        s = s.lstrip()
        if s.startswith(label):
            s = s[len(label):]
        return s

    def _parse_row(row: str) -> tuple[str, list[int], str | None] | None:
        # Expect: name  [shape...]  dtype?
        m = re.match(r"\s*(\S+)\s+\[([^\]]*)\](?:\s+(\S+))?\s*$", row)
        if not m:
            return None
        name = m.group(1)
        shape_str = m.group(2)
        dtype = m.group(3)
        shape: list[int] = []
        for tok in shape_str.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                shape.append(int(tok))
            except ValueError:
                # Symbolic dim - keep as 0 placeholder is fabricating; instead skip the model_info entirely.
                return None
        return name, shape, dtype

    # Input rows: from in_idx (with label stripped) until out_idx
    input_rows: list[str] = [_strip_label(lines[in_idx], "Inputs:")]
    for i in range(in_idx + 1, out_idx):
        if lines[i].strip():
            input_rows.append(lines[i])

    output_rows: list[str] = [_strip_label(lines[out_idx], "Outputs:")]
    for i in range(out_idx + 1, len(lines)):
        if not lines[i].strip():
            break
        # Bail if we hit a non-tensor row (e.g. another log line)
        if re.match(r"\s*\S+\s+\[", lines[i]) is None:
            break
        output_rows.append(lines[i])

    inputs_parsed = [_parse_row(r) for r in input_rows]
    outputs_parsed = [_parse_row(r) for r in output_rows]
    if any(p is None for p in inputs_parsed) or any(p is None for p in outputs_parsed):
        return None

    input_names = [p[0] for p in inputs_parsed]
    input_shapes = [p[1] for p in inputs_parsed]
    input_types = [p[2] for p in inputs_parsed if p[2] is not None]
    output_names = [p[0] for p in outputs_parsed]
    output_shapes = [p[1] for p in outputs_parsed]

    info: dict = {
        "input_names": input_names,
        "input_shapes": input_shapes,
    }
    if len(input_types) == len(input_names):
        info["input_types"] = input_types
    info["output_names"] = output_names
    info["output_shapes"] = output_shapes
    return info


# ---- Main --------------------------------------------------------------------

def _normalize_timestamp(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        # Source format: '2026-05-22T13:02:25.123456' (no tz) or with tz.
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return ts


def build_perf_record(eval_json_path: Path, ep_name: str, hardware: str,
                      hf_id: str, task: str, precision: str) -> dict | None:
    data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    perf = data.get("perf") or {}
    if not perf.get("passed"):
        return None

    iterations, warmup = _parse_iter_warm(perf.get("command", "") or "")
    latency, throughput = _parse_latency_and_throughput(perf.get("stdout_output", "") or "")
    if latency is None and throughput is None:
        return None  # nothing usable to pull

    model_info = _parse_inputs_outputs(perf.get("stderr_output", "") or "") or {}
    timestamp = _normalize_timestamp(data.get("run_timestamp"))

    benchmark_info: dict = {
        "model_id": hf_id,
        "task": task,
        "device": hardware,
        "ep": ep_name,
        "precision": precision,
        "iterations": iterations,
        "warmup": warmup,
        "batch_size": 1,
        "timestamp": timestamp,
    }

    return {
        "benchmark_info": benchmark_info,
        "model_info": model_info,
        "latency_ms": latency if latency is not None else {
            "mean": None, "min": None, "max": None,
            "p50": None, "p90": None, "p95": None, "p99": None,
            "std": None, "warmup_mean": None,
        },
        "throughput": {
            "samples_per_sec": throughput,
            "batches_per_sec": throughput,
        },
        "raw_samples_ms": [],
    }


_PRECISION_SUFFIXES = ("fp16", "w8a16", "w8a8")


def _split_stem(stem: str) -> tuple[str, str]:
    """Split a config stem (without the trailing ``_config``) into (task, precision)."""
    for p in _PRECISION_SUFFIXES:
        if stem.endswith(f"_{p}"):
            return stem[: -(len(p) + 1)], p
    return stem, ""


def main() -> int:
    new_keys = {(hf, task) for hf, task in NEW_MODELS}

    written = 0
    skipped_existing = 0
    no_source = 0
    not_passed = 0
    parse_failed = 0

    for src_dir_name, (ep_folder, hardware, ep_name, _npu_default_precision) in SRC_EP_MAP.items():
        src_ep_dir = SRC_ROOT / src_dir_name
        if not src_ep_dir.is_dir():
            print(f"[skip] source EP dir missing: {src_ep_dir}")
            continue

        ep_root = EXAMPLES / ep_folder / hardware
        if not ep_root.is_dir():
            continue

        # Iterate the existing config files belonging to one of the 29 new (hf_id, task) tuples.
        slug_to_hfid = {hf.replace("/", "_"): hf for hf, _ in NEW_MODELS}

        for model_dir in ep_root.iterdir():
            if not model_dir.is_dir():
                continue
            hf_id = slug_to_hfid.get(model_dir.name)
            if hf_id is None:
                continue
            for cfg in model_dir.glob("*_config.json"):
                stem = cfg.name[: -len("_config.json")]
                task, precision = _split_stem(stem)
                if (hf_id, task) not in new_keys:
                    continue

                out_file = model_dir / f"{stem}_perf_result.json"
                if out_file.exists():
                    skipped_existing += 1
                    continue

                src_eval = _find_source_entry(src_ep_dir, hf_id, task)
                if src_eval is None:
                    no_source += 1
                    continue

                record = build_perf_record(src_eval, ep_name, hardware, hf_id, task, precision)
                if record is None:
                    try:
                        data = json.loads(src_eval.read_text(encoding="utf-8"))
                        if not (data.get("perf") or {}).get("passed"):
                            not_passed += 1
                        else:
                            parse_failed += 1
                    except Exception:
                        parse_failed += 1
                    continue

                out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                written += 1

    print()
    print(f"written           : {written}")
    print(f"skipped (existing): {skipped_existing}")
    print(f"no source entry  : {no_source}")
    print(f"perf not passed   : {not_passed}")
    print(f"parse failed      : {parse_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
