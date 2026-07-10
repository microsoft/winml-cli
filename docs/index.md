# winml-cli

WinML CLI is a command line tool for building portable, performant, and high-quality AI models for Windows ML. It takes you from a source model — whether from Hugging Face or your own pipeline — to a hardware-optimized artifact in a reproducible workflow.

Purpose-built for Windows hardware diversity, the CLI handles conversion, graph optimization, and compilation across AMD, Intel, NVIDIA, and Qualcomm targets. The CLI fits naturally into CI/CD pipelines so teams can validate and ship models easily.

## What you can do

- **Build once, run across hardwares.** Compose your own workflow from primitive commands (`export`, `analyze`, `optimize`, `quantize`, `compile`), or use an auto-generated config with `winml build` — both produce portable models that run across hardware.
- **Drill into the details.** Deep insights into operator compatibility, shape mismatches, graph optimizations, and EP-aware tuning at any stage of the pipeline.
- **AI-ready.** CLI-driven tools with built-in skills, friendly to work with mainstream agents.

## What you get out of the box

- **All Windows ML EPs supported.** Every [supported execution provider](concepts/eps-and-devices.md#eps-winml-cli-supports) is available behind the same commands.
- **Curated model catalog.** A [verified set of models](reference/supported-models.md) that run across all Windows ML EPs — a reliable starting point.
- **Bring your own ONNX.** Not only for converting from PyTorch — bring an [existing ONNX model](tutorials/build-from-onnx.md) to get operator-compatibility insights and optimize it based on the analysis.

## Where to start

- **[Installation](getting-started/installation.md)** — get the `winml` CLI running locally.
- **[Quickstart](getting-started/quickstart.md)** — export a Hugging Face model in five minutes.

## Learn the model

- **[How winml-cli Works](concepts/how-it-works.md)** — the pipeline from a PyTorch model to an EP-compiled artifact.
- **[Commands](commands/overview.md)** — reference for all 12 `winml` subcommands.
- **[Samples](samples/bert-config-build.md)** — walkthroughs for BERT and CLIP.

## Repository access

To request access to the WinML CLI repository, visit [aka.ms/winml-cli](https://aka.ms/winml-cli).

## License

MIT. See [LICENSE](https://github.com/microsoft/winml-cli/blob/main/LICENSE.txt).
