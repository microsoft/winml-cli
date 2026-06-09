# winml-cli

winml-cli is a CLI toolkit to build portable, performant, and high-quality models for [Windows ML](https://learn.microsoft.com/en-us/windows/ai/windows-ml/).

## What you can do

- **Build once, run anywhere.** Compose your own workflow from primitive commands (`export`, `analyze`, `optimize`, `quantize`, `compile`), or hand a config to the built-in pipeline. Same portable ONNX, two complementary paths.
- **Drill into the details.** Inspect operators, pinpoint compatibility errors, and trace performance bottlenecks at any stage of the pipeline.
- **AI-ready.** Built-in agent skills work with mainstream coding agents — let the agent drive the pipeline for you.

## What you get out of the box

- **One toolkit, every EP.** All [supported execution providers](concepts/eps-and-devices.md#eps-winml-cli-supports) live behind the same commands.
- **Repeatable and traceable.** Configs are deterministic; every pipeline run records inputs, outputs, and decisions at each stage.
- **Quality gates built in.** The analyzer catches operator-compatibility issues before deployment and suggests fixes automatically.

## Where to start

- **[Installation](getting-started/installation.md)** — get the `winml` CLI running locally.
- **[Quickstart](getting-started/quickstart.md)** — export a Hugging Face model in five minutes.
- **[End-to-End Tour](getting-started/end-to-end.md)** — full pipeline targeting whatever hardware you have (NPU / GPU / CPU).

## Learn the model

- **[How winml-cli Works](concepts/how-it-works.md)** — the pipeline from a PyTorch model to an EP-compiled artifact.
- **[Commands](commands/overview.md)** — reference for all 12 `winml` subcommands.
- **[Samples](samples/convnext-primitives.md)** — end-to-end walkthroughs for ConvNeXt, BERT, and CLIP.

## License

MIT. See [LICENSE](https://github.com/microsoft/winml-cli/blob/main/LICENSE.txt).
