# Validation Scripts

## ViLT L2 named-input parity (portable)

The command below reproduces the ViLT L2 parity check for both fp32 and fp16 ONNX artifacts.
It computes and prints/captures:

- cosine similarity (PyTorch logits vs ONNX logits)
- max absolute delta
- top-answer agreement

PowerShell example (clean clone, portable OUT path):

```powershell
$OUT = Join-Path (Get-Location) "temp/vilt-l2"
uv run winml build `
  -c examples/recipes/dandelin_vilt-b32-finetuned-vqa/cpu/cpu/visual-question-answering_fp32_config.json `
  -m dandelin/vilt-b32-finetuned-vqa `
  -o "$OUT/artifacts/cpu-fp32" `
  --no-analyze --no-color --rebuild

uv run winml build `
  -c examples/recipes/dandelin_vilt-b32-finetuned-vqa/cpu/cpu/visual-question-answering_fp16_config.json `
  -m dandelin/vilt-b32-finetuned-vqa `
  -o "$OUT/artifacts/cpu-fp16" `
  --no-analyze --no-color --rebuild

uv run python scripts/validation/vilt_l2_named_parity.py `
  --model dandelin/vilt-b32-finetuned-vqa `
  --revision d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08 `
  --fp32 "$OUT/artifacts/cpu-fp32/model.onnx" `
  --fp16 "$OUT/artifacts/cpu-fp16/model.onnx" `
  --json-out "$OUT/l2-named-parity.json"
```