# Quickstart

## Verify the install

Run the following command to enumerate available devices and execution providers
on your machine:

```bash
uv run winml sys --list-device --list-ep
```

`--list-device` and `--list-ep` print only the hardware and EP inventory. If the command exits without error, your winml-cli install is
ready. See [`winml sys`](../commands/sys.md) for the full flag reference.

## Inspect the model

Before downloading any models, confirm that winml-cli recognises the model:

```bash
uv run winml inspect -m microsoft/resnet-50
```

```text
+--------------------------- microsoft/resnet-50 ---------------------------+
| Task          image-classification                                         |
| Model Class   ResNetForImageClassification                                 |
| Exporter      OptimumExporter                                              |
| WinML Class   WinMLImageClassificationModel                                |
| Status        Supported                                                    |
+---------------------------------------------------------------------------+
```

!!! tip
    Always inspect before build to catch unsupported architectures early.

## Build the model

```bash
uv run winml build -m microsoft/resnet-50 -o resnet_out/ --no-quant
```

`winml build` runs all pipeline steps in sequence — export, optimize, quantize. You can start a model build without a config file, or provide one to configure each step in the sequence (see [`winml config`](../commands/config.md) to customize).
All intermediate artifacts land in `resnet_out/`, so you can reuse any stage independently.

After a successful build, you will find the following outputs in `resnet_out/`:

- **A standard ONNX file for each completed stage** — load, inspect, or pass any of these to a downstream tool independently.
- **`analyze_result.json`** — detailed model compatibility insights for each Windows ML EP, including supported, partially supported, and unsupported operators, detected optimization patterns, and recommended optimization workflows.
- **A declarative `winml_build_config` file** — automatically generated after the build step to capture the full workflow end-to-end.

## Benchmark the model

```bash
uv run winml perf -m resnet_out/model.onnx --device auto --iterations 50 --monitor
```

`--device auto` lets the CLI resolve the best available device on your machine — NPU first, then GPU, then CPU.

## What's next

- **[How winml-cli Works](../concepts/how-it-works.md)** — understand what each command does under the hood.
- **[BERT sample](../samples/bert-config-build.md)** — see the config + build + perf workflow in detail with a representative model.

## See also

- [`winml build`](../commands/build.md)
- [`winml inspect`](../commands/inspect.md)
- [`winml perf`](../commands/perf.md)
- [`winml sys`](../commands/sys.md)
