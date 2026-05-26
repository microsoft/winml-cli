# Quickstart

This page proves your winml-cli install works end-to-end. You will export a
Hugging Face image classifier to ONNX and then inspect the resulting artifact.
No quantization, no execution-provider selection — just the two commands you
need to confirm everything is wired up correctly. Estimated time: 5 minutes.

## Verify the install

Run the following command to enumerate available devices and execution providers
on your machine:

```bash
uv run winml sys --list-device --list-ep
```

`--list-device` and `--list-ep` print only the hardware and EP inventory,
skipping SDK versions and Python environment details that plain `winml sys`
would include. If the command exits without error, your winml-cli install is
ready. See [`winml sys`](../commands/sys.md) for the full flag reference.

## Export your first model

```bash
uv run winml export -m microsoft/resnet-50 -o resnet50.onnx
```

!!! note "What just happened"
    winml-cli downloaded the `microsoft/resnet-50` weights from Hugging Face,
    ran the eight-step Hierarchy-preserving Tags Protocol (HTP) to trace the
    PyTorch module tree, and wrote an ONNX file to `resnet50.onnx`. Each ONNX
    node carries a `hierarchy_tag` metadata property recording its full PyTorch
    ancestry, which downstream quantization and compilation steps use to reason
    about the graph. See [`winml export`](../commands/export.md) for the full
    flag reference.

## Inspect the artifact

```bash
uv run winml inspect -m resnet50.onnx
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

When you pass a local `.onnx` file, `winml inspect` reads the embedded model
metadata directly. When you pass a Hugging Face model ID instead, it reads
the model's `config.json` from the Hub without downloading weights. In both
cases it resolves the loader, exporter, and WinML inference class that
winml-cli will use for this architecture. See
[`winml inspect`](../commands/inspect.md) for output-format and hierarchy
options.

## What's next

- **[End-to-End walkthrough](end-to-end.md)** — full pipeline from Hugging Face to NPU.
- **[How winml-cli Works](../concepts/how-it-works.md)** — understand what each command does under the hood.
- **[ConvNeXt primitives sample](../samples/convnext-primitives.md)** — see every pipeline stage in detail with a representative model.

## See also

- [`winml export`](../commands/export.md)
- [`winml inspect`](../commands/inspect.md)
- [Load and export](../concepts/load-and-export.md)
