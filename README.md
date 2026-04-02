# ModelKit

Accelerate Model Deployment on WinML.

ModelKit is a Python toolkit for converting and optimizing PyTorch models to ONNX format, targeting deployment on the [Windows ML](https://learn.microsoft.com/en-us/windows/ai/windows-ml/) runtime. It supports multiple hardware backends including QNN (Qualcomm Neural Processing SDK) and OpenVINO.

## Features

- **Universal ONNX Export** — Convert PyTorch and Hugging Face models to ONNX with hierarchy preservation
- **Model Analysis** — Validate ONNX models for operator support, shape inference, and backend compatibility
- **Quantization** — INT8/INT16 quantization with calibration dataset support
- **Optimization** — Graph optimizations tailored for target execution providers
- **Performance Profiling** — Operation-level tracing and hardware monitoring
- **Multi-Backend Support** — QNN, OpenVINO, DirectML, and ONNX Runtime CPU/GPU

## Getting Started

### Prerequisites

- Windows 10/11
- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/microsoft/ModelKit.git
cd ModelKit
uv python install 3.10
uv sync
```

### Usage

ModelKit provides a CLI tool `winml`:

```bash
# Export a Hugging Face model to ONNX
uv run winml export --model microsoft/resnet-50 --output ./output

# Analyze an ONNX model
uv run winml analyze --model ./output/model.onnx

# Quantize an ONNX model
uv run winml quantize --model ./output/model.onnx
```

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/microsoft/ModelKit/issues).


## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE.txt).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft
sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
