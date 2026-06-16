"""E2E test for the transformer-only Qwen3 export path.

Produces two transformer-only ONNX files whose I/O matches
``qwen3_gqa_fp16_ctx.onnx`` / ``qwen3_gqa_fp16_iter.onnx``:

  decoder_prefill: input_hidden_states [1, 64, 1024]  → output_hidden_states + KV
  decoder_gen    : input_hidden_states [1,  1, 1024]  → output_hidden_states + KV

with FP16 past/present KV named ``past_keys_{i}`` / ``past_values_{i}``,
``com.microsoft::GroupQueryAttention``, ``LpNormalization``, and 1x1 Conv
projections.

Important: ``install()`` MUST be called before importing the composite model
machinery so the registry hot-patches take effect.

Generation (``model.generate(...)``) is NOT supported by this build path —
the inference feeds in ``WinMLDecoderOnlyModel`` still target the eager
I/O signature. Use the eager ``WinMLQwen3Model`` build path for end-to-end
generation.

Run::

    python test_qwen_transformer_only.py

This builds each transformer sub-model and then runs the w8a16
quantization on the exported transformer ONNX files (no surgery needed —
files are already transformer-only).
"""

import os
import sys
import pathlib
import subprocess

# Put the in-repo `src/` ahead of site-packages so `import winml` always
# resolves to the editable source tree — no manual copy-to-venv needed.
_repo_root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

model_id = "Qwen/Qwen3-0.6B"
MAX_CACHE = 256

# component name -> (HF task, seq_len, artifact prefix). Order matters
# (prefill first). The prefix is how the built npu_ctx file is named so the
# parent can verify success by artifact appearance (the build segfaults on
# native QNN/ORT teardown AFTER writing the file, so exit codes are unreliable).
SUB_MODELS = {
    "decoder_prefill": ("feature-extraction", 64, "feat_"),
    "decoder_gen": ("text2text-generation", 1, "txt2txt_"),
}

ARTIFACTS_DIR = (
    pathlib.Path.home() / ".cache" / "winml" / "artifacts" / model_id.replace("/", "_")
)


def _latest_ctx_mtime(prefix: str) -> float:
    """Newest mtime of a ``{prefix}*_optimized_npu_ctx.onnx`` artifact, or 0."""
    files = list(ARTIFACTS_DIR.glob(f"{prefix}*_optimized_npu_ctx.onnx"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _build_one(task: str, seq_len: int) -> None:
    """Build a SINGLE transformer sub-model in this (fresh) process.

    Invoked as a subprocess by ``main()`` so each sub-model exports in a
    clean interpreter — building both in one process leaves PyTorch/ORT
    state from the first build that corrupts/kills the second.
    """
    from winml.modelkit.models.hf.qwen_transformer_only import install as install_qwen_transformer_only

    install_qwen_transformer_only()

    from winml.modelkit.config import WinMLBuildConfig
    from winml.modelkit.models.auto import WinMLAutoModel

    WinMLAutoModel.from_pretrained(
        model_id,
        task=task,
        config=WinMLBuildConfig(quant=None, compile=None),
        precision="fp16",
        device="npu",
        ep="qnn",
        force_rebuild=True,
        shape_config={"max_cache_len": MAX_CACHE, "seq_len": seq_len},
    )
    # The QNN/ORT teardown segfaults (0xC0000005) on interpreter shutdown
    # AFTER the artifact is fully written. Skip the buggy cleanup with a hard
    # exit so the parent sees a clean exit code 0.
    print(f"BUILD COMPLETE: task={task} seq_len={seq_len}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _find_optimized(prefix: str) -> pathlib.Path:
    """Locate the cached transformer-only ``{prefix}*_optimized.onnx`` file."""
    cands = [
        p for p in ARTIFACTS_DIR.glob(f"{prefix}*_optimized.onnx")
        if not p.name.endswith("_optimized_npu_ctx.onnx")
    ]
    if not cands:
        raise FileNotFoundError(
            f"No {prefix}*_optimized.onnx in {ARTIFACTS_DIR} — build the sub-model first."
        )
    return max(cands, key=lambda p: p.stat().st_mtime)


class _SubShim:
    """Minimal stand-in exposing the ``_onnx_path`` quant needs."""

    def __init__(self, onnx_path: pathlib.Path):
        self._onnx_path = str(onnx_path)


class _ModelShim:
    """Minimal stand-in exposing ``sub_models`` for ``quantize_built_model``."""

    def __init__(self, sub_models: dict):
        self.sub_models = sub_models


def _run_quant() -> None:
    """Quantize the cached transformer ONNX files (no composite/QNN load).

    Runs as its own subprocess so any ORT teardown crash can't poison the
    parent. Builds a shim ``model`` whose ``sub_models[name]._onnx_path``
    point straight at the cached ``*_optimized.onnx`` files.
    """
    # Dump a native C-stack if the calibration InferenceSession segfaults
    # (otherwise the crash is silent — no Python traceback).
    import faulthandler
    faulthandler.enable()

    from qwen3_transformer_only_quantize import quantize_built_model

    sub_models = {
        name: _SubShim(_find_optimized(prefix))
        for name, (_task, _seq, prefix) in SUB_MODELS.items()
    }
    model = _ModelShim(sub_models)
    print("=== Running transformer w8a16 quantization ===", flush=True)
    for name, sub in sub_models.items():
        print(f"  {name}: {sub._onnx_path}", flush=True)

    try:
        quantize_built_model(
            model,
            model_id=model_id,
            max_cache_len=MAX_CACHE,
            prefill_seq=64,
        )
    except BaseException:
        import traceback
        print("QUANT FAILED with exception:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    print("QUANT COMPLETE", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    # 1) Build each sub-model in its OWN subprocess (fresh state each time).
    #    Judge success by whether a FRESH npu_ctx artifact appeared, NOT by the
    #    subprocess exit code: the native QNN/ORT layer segfaults (0xC0000005)
    #    on teardown AFTER the artifact is fully written to disk.
    import time as _time

    for name, (task, seq_len, prefix) in SUB_MODELS.items():
        print(f"\n########## BUILD {name} (task={task}, seq_len={seq_len}) ##########", flush=True)
        before = _latest_ctx_mtime(prefix)
        start = _time.time()
        rc = subprocess.run(
            [sys.executable, "-u", str(pathlib.Path(__file__).resolve()),
             "--build-sub", task, str(seq_len)],
            cwd=str(_repo_root),
        ).returncode

        after = _latest_ctx_mtime(prefix)
        if after > before and after >= start - 1:
            status = "OK" if rc == 0 else f"OK (ignored teardown exit {rc})"
            print(f"########## {name} {status}: fresh {prefix}*_optimized_npu_ctx.onnx ##########", flush=True)
        else:
            raise SystemExit(
                f"Sub-model build failed for {name} (exit {rc}) — "
                f"no fresh {prefix}*_optimized_npu_ctx.onnx in {ARTIFACTS_DIR}"
            )

    # 2) Report the built transformer-only ONNX files (no composite/QNN load —
    #    that creates QNN EP sessions that segfault the parent on teardown).
    for name, (_task, _seq, prefix) in SUB_MODELS.items():
        print(f"\n=== {name} ===")
        print(f"  optimized : {_find_optimized(prefix).name}")
        ctx = sorted(ARTIFACTS_DIR.glob(f"{prefix}*_optimized_npu_ctx.onnx"))
        if ctx:
            print(f"  npu_ctx   : {ctx[-1].name}")

    # 3) Quantization — run in its OWN subprocess for the same teardown-crash
    #    isolation. Judge by whether quant files appeared.
    print("\n########## QUANTIZE ##########", flush=True)
    before = max(
        (p.stat().st_mtime for p in ARTIFACTS_DIR.glob("*quant.onnx")),
        default=0.0,
    )
    qstart = _time.time()
    rc = subprocess.run(
        [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), "--quant"],
        cwd=str(_repo_root),
    ).returncode
    after_files = list(ARTIFACTS_DIR.glob("*quant.onnx"))
    after = max((p.stat().st_mtime for p in after_files), default=0.0)
    if after > before and after >= qstart - 1:
        status = "OK" if rc == 0 else f"OK (ignored teardown exit {rc})"
        print(f"########## QUANTIZE {status} ##########", flush=True)
        for p in sorted(after_files, key=lambda x: x.stat().st_mtime)[-len(SUB_MODELS):]:
            print(f"  {p.name}", flush=True)
    else:
        raise SystemExit(
            f"Quantization failed (exit {rc}) — no fresh *quant.onnx in {ARTIFACTS_DIR}"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--build-sub":
        _build_one(sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--quant":
        _run_quant()
    else:
        main()

