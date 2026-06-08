"""E2E test for Qwen3 decoder-only pipeline.

Uses sub_model_kwargs to set per-component shape_config:
  - decoder_prefill: max_cache_len=256, seq_len=64
  - decoder_gen:     max_cache_len=256, seq_len=1

Set env var ``QUANTIZE=1`` to also run the MOPS-style Step 3:
transformer-only surgery + winml quantize on both sub-models
(embeddings and lm_head are stripped and not quantized).
"""

import os

from transformers import AutoTokenizer

from winml.modelkit.config import WinMLBuildConfig
from winml.modelkit.models.winml.composite_model import WinMLCompositeModel

model_id = "Qwen/Qwen3-0.6B"

model = WinMLCompositeModel.from_pretrained(
    model_id,
    task="text-generation",
    # config=WinMLBuildConfig(quant=None, compile=None),
    config=WinMLBuildConfig(quant=None),
    precision="fp16",
    device="npu",
    ep="qnn",
    force_rebuild=False,
    sub_model_kwargs={
        "decoder_prefill": {"shape_config": {"max_cache_len": 256, "seq_len": 64}},
        "decoder_gen": {"shape_config": {"max_cache_len": 256, "seq_len": 1}},
    },
)

# Verify ONNX I/O shapes
for name, sub in model.sub_models.items():
    io = sub.io_config
    shapes = dict(zip(io["input_names"], io["input_shapes"]))
    print(f"\n=== {name} ===")
    for k, v in shapes.items():
        print(f"  {k}: {v}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "8 * 7 = ?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print("\nAnswer:", content)

if os.environ.get("QUANTIZE") == "1":
    # Reuse the already-built decoder_prefill/decoder_gen ONNX files:
    # surgery (strip embed + lm_head) + transformer-only quantize.
    print("\n=== QUANTIZE=1 — running transformer-only quantization ===")
    from qwen3_quantize import quantize_built_model

    quantize_built_model(
        model,
        model_id=model_id,
        max_cache_len=256,
        prefill_seq=64,
    )
