# Swin-Large `winml optimize` failure on QNN — how to unblock

Good news: Swin is in scope (it's one of the explicitly-listed vision-transformer families ModelKit is designed to handle), and the fact that `inspect` and `export` both passed means the architecture is recognized and you have a usable exported ONNX in hand. Optimize/quantize is exactly the stage where most operator-pattern failures land — even when export was clean — so this is a known shape of problem, not a dead end.

Here's the order I'd work through it.

## 1. Read the error literally, then run analyze against the exported ONNX

ModelKit errors are usually specific — they name the op pattern, the EP, and the stage. Before doing anything else, look at the actual op pattern the optimize step is complaining about and write it down. Then point the analyzer at the **exported** ONNX (not the HF ID) so it lints the same graph optimize is choking on:

```bash
winml analyze -m <path-to-your-exported-swin.onnx> --ep QNN
```

(Confirm the exact flag spelling with `winml analyze --help` — the shape is stable but the spelling per command isn't something to memorize.)

The linter classifies every operator as supported / partial / unsupported per EP. It will almost always either confirm what the optimize error already named, or surface a related pattern nearby in the graph. That's your ground truth for "is this an EP fit problem or an optim-config problem."

## 2. Try a different optimize configuration before touching the graph

The skill's explicit guidance for this exact situation: don't hand-edit the ONNX graph. Try a different optim or quantization configuration to dodge the unsupported pattern. Concretely:

- Run `winml optimize --help` and look at what config / level / pattern-toggle flags exist. Pick a less aggressive setting (or a different optim config preset, if one is exposed) and re-run.
- If you're already using a config JSON via `-c`, remember **CLI flags override the config file**, so you can try one-off overrides without editing the JSON.
- If you generated the pipeline with `winml config`, open the JSON and look for the optimize section — overriding the setting there gives you a reproducible knob rather than ad-hoc CLI flags.

Swin's window-attention + roll/shift patterns are the usual suspects for op-pattern friction on NPU EPs, and they often clear up under a different optim configuration without any graph surgery.

## 3. If analyze flags the op as unsupported on QNN (not partial), pivot

If the analyzer comes back with the offending op classified as **unsupported** on QNN (rather than partial / supported-with-a-pattern-variant), no amount of optim-config tweaking will save it — that's the "this model isn't a fit for this EP" escalation the skill calls out. Two reasonable moves:

- Try the same exported ONNX through `optimize` and then `compile --ep Dml` or run it on CPU. Snapdragon X Elite has working DML and CPU EPs, so you still get a deployable artifact, just not NPU-accelerated. Confirm what's actually registered with `winml sys --list-ep` first.
- Swap to a smaller / less exotic Swin variant (e.g. a base or tiny Swin, or a different ViT family entirely) that the analyzer reports as clean on QNN. Swin-Large in particular is the largest of the family and has the most chances to hit a pattern QNN doesn't accelerate.

## 4. Sanity checks while you're in there

- `winml sys --list-ep` — confirm QNN is actually in the registered list on this box. If the optimize stage is somehow being asked to target an EP that isn't registered, the error message can be misleading.
- `winml optimize --help` — confirm every flag you're passing is real. Fabricated flags (a `--preset`, `--mode`, etc.) are the most common silent way to waste a debug cycle here.
- Make sure you're feeding optimize the path to the **exported** ONNX (the `-o` artifact from `winml export`), not the HF ID and not anything from cache.

## What I'd do in your shoes, in order

1. Re-run `winml analyze -m <exported.onnx> --ep QNN` and read the op classification for whatever optimize named.
2. If the op is **partial** or the failure looks pattern-shaped: re-run `winml optimize` with a different/less aggressive optim configuration (check `winml optimize --help` for what's available), feeding it the same exported ONNX.
3. If the op is **unsupported on QNN**: stop fighting it — either fall back to DML/CPU on the same Snapdragon box, or pick a Swin variant the analyzer reports clean.

If you can paste the exact op name / pattern from the optimize error and the analyze output, I can be a lot more specific about which optim knob to turn.
