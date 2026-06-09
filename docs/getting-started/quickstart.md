# Quickstart

This page proves your winml-cli install works end-to-end. You will inspect a
Hugging Face image classifier, then export it to ONNX. No quantization, no
execution-provider selection — just the commands you need to confirm everything
is wired up correctly. Estimated time: 5 minutes.

## Verify the install

Run the following command to enumerate available devices and execution providers
on your machine:

```bash
uv run winml sys --list-device --list-ep
```

`--list-device` and `--list-ep` print only the hardware and EP inventory,
skipping runtime-version and Python environment details that plain `winml sys`
would include. If the command exits without error, your winml-cli install is
ready. See [`winml sys`](../commands/sys.md) for the full flag reference.

## Inspect the model

Before downloading any weights, confirm that winml-cli recognises the model:

```bash
uv run winml inspect -m microsoft/resnet-50
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

!!! note "What just happened"
    `winml inspect` read only the model's `config.json` from Hugging Face Hub —
    no weights downloaded — and confirmed that `microsoft/resnet-50` maps to a
    supported task, a known model class, and a compatible ONNX exporter. Always
    inspect before export to catch unsupported architectures early. See
    [`winml inspect`](../commands/inspect.md) for output-format and hierarchy
    options.

## Export the model

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

## What's next

- **[End-to-End walkthrough](end-to-end.md)** — full pipeline from Hugging Face to NPU.
- **[How winml-cli Works](../concepts/how-it-works.md)** — understand what each command does under the hood.
- **[ConvNeXt primitives sample](../samples/convnext-primitives.md)** — see every pipeline stage in detail with a representative model.

## See also

- [`winml export`](../commands/export.md)
- [`winml inspect`](../commands/inspect.md)
- [Load and export](../concepts/load-and-export.md)
