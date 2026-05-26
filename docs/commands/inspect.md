# winml inspect

> Inspect a model's tasks, classes, and hierarchy before committing to an export.

## When to use this

Use `winml inspect` to understand how winml-cli will treat a HuggingFace model before
running `winml export` or `winml build`. It answers questions like "which task will be
auto-detected?", "which HF model class will be loaded?", and "does this model have a
supported exporter?" without downloading weights or writing any files.

## Synopsis

```bash
$ winml inspect -m <model_id> [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | string | **required** | HuggingFace model ID (e.g. `openai/clip-vit-base-patch32`). Required unless `--help` is used. |
| `--format` | `-f` | `table` \| `json` | `table` | Output format. `table` renders rich panels; `json` emits a machine-readable object. |
| `--task` | `-t` | string | `null` | Override the auto-detected task (e.g. `image-classification`, `feature-extraction`). |
| `--hierarchy` | `-H` | flag | `false` | Print the PyTorch module tree. Instantiates the model with random weights — no weight download required. |
| `--help` | `-h` | flag | — | Show help and exit. |

> `winml inspect` does not accept `--device`, `--ep`, `--precision`, or `--output`.
> It is a read-only discovery command that does not produce any artifacts.

## How it works

`winml inspect` calls into the winml-cli registry to resolve the model ID against the
known loader and exporter configurations. It fetches only the model's `config.json`
from HuggingFace Hub (no weights), uses the architecture field to look up the matching
HF model class and WinML inference class, and then renders the result. When
`--hierarchy` is supplied, the model is instantiated locally with random weights using
`AutoModel.from_config()`, and a forward-pass trace records the full PyTorch module
tree. Because no real weights are downloaded, hierarchy inspection is fast even for
large models.

## Examples

```bash
# Basic inspection — check task detection and loader/exporter classes
$ winml inspect -m microsoft/resnet-50
```

```text
╭─────────────────────────── microsoft/resnet-50 ───────────────────────────╮
│ Task          image-classification                                         │
│ Model Class   ResNetForImageClassification                                 │
│ Exporter      OptimumExporter                                              │
│ WinML Class   WinMLImageClassificationModel                                │
│ Status        Supported                                                    │
╰────────────────────────────────────────────────────────────────────────────╯
```

```bash
# JSON output — useful for scripting or CI pre-flight checks
$ winml inspect -m bert-base-uncased --format json
```

```bash
# Override task when auto-detection picks the wrong one
$ winml inspect -m bert-base-uncased --task feature-extraction
```

```bash
# Print the full PyTorch module hierarchy (no weight download)
$ winml inspect -m openai/clip-vit-base-patch32 --hierarchy
```

```bash
# Combine verbose logging with hierarchy for deep diagnostics
$ winml inspect -m facebook/convnext-tiny-224 -v -H
```

## Common pitfalls

- **`--model` is always required.** Unlike some other commands, `winml inspect` has
  no mode that omits `-m`. The flag is marked required; omitting it returns an error.
- **Hierarchy requires a locally installable model config.** If the model config
  references a custom architecture not in the local `transformers` installation,
  `--hierarchy` will fail with an import error. Update `transformers` or omit the flag.
- **Task override affects all output.** Passing `--task` changes which exporter and
  WinML class are reported, not just the task field. If the override is incompatible
  with the model architecture, the status will show as unsupported.
- **`--format json` is silent on unsupported models.** When the model is not found in
  the winml-cli registry, the command raises a `ClickException`. Wrap the call in
  `winml inspect ... && ...` or check the exit code when scripting.
- **No weight download does not mean no network access.** The `config.json` is always
  fetched from HuggingFace Hub. Set `HF_HUB_OFFLINE=1` if you need fully offline
  inspection of a locally cached model.

## See also

- [hub.md](hub.md) — browse the curated catalog and check accuracy verdicts before
  inspecting
- [Load and export concept](../concepts/load-and-export.md) — how `winml.hierarchy.tag`
  metadata is written and what you can do with the module tree
- [How winml-cli Works](../concepts/how-it-works.md) — pipeline overview showing where
  inspect fits before export
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — background on loaders,
  exporters, and EP-specific configurations
