# Graph and IR

A `.onnx` file is, at rest, a binary-serialized Protocol Buffer. Open it in any hex editor and you will find the familiar `ONNX` magic bytes followed by a dense encoding of every number the model has ever learned, plus the structural description of how those numbers are combined to produce a prediction. The file is self-contained: weights and computation recipe live together, making the artifact portable without any accompanying framework installation.

That computation recipe is a **graph** — a directed acyclic structure of operators wired together by named data edges. The graph is what the ONNX Intermediate Representation (IR) actually defines. When winml-cli loads or transforms a model, every operation works against this graph structure, not against framework-specific objects.

## What is in a .onnx file

An ONNX `ModelProto` wraps a single `GraphProto`. Inside the graph you will find:

- **Inputs** — typed, named entry points that accept runtime tensors (e.g., `pixel_values: float32[1, 3, 224, 224]`).
- **Outputs** — typed, named exit points that carry the model's predictions back to the caller.
- **Nodes** — individual operators (Conv, MatMul, Softmax, …) that transform tensors. Each node names its inputs and outputs using the same string identifiers used throughout the graph.
- **Initializers** — constant tensors embedded in the file. Learned weights, biases, and lookup tables are stored here; they are treated as graph inputs that are always pre-supplied.
- **Metadata** — key–value string properties attached at the model level. winml-cli uses this area to store information such as `winml.io.inputs` (serialized tensor specs) and `winml.hierarchy.tag` attributes on individual nodes.

## Graphs as IR

ONNX functions as an Intermediate Representation: a portable, framework-neutral description of a computation that can be loaded by any conforming runtime. Unlike a Python object graph or a compiled binary, the ONNX IR makes data flow completely explicit. Every node declares the exact names of its input and output edges; those names form a namespace shared across the whole graph, so any consumer can trace a tensor from the model inputs through every transformation to the final output.

This explicit wiring unlocks two capabilities that winml-cli relies on heavily. First, **shape inference** can propagate concrete or symbolic dimensions through the graph without running it — a prerequisite for correct quantization and for generating input specs automatically. Second, **EP-targeted compilation** can partition the graph by examining which nodes an Execution Provider supports, fuse eligible sub-graphs into accelerated kernels, and serialize the result back into a valid ONNX file using the `EPContext` convention. Neither of these would be tractable on an opaque binary or a dynamic execution trace.

Because the IR is static — describing the full computation at load time rather than at call time — winml-cli can inspect, validate, and transform a model without a GPU, a framework, or sample data.

## Opsets and versioning

Every operator in ONNX belongs to a **domain**, and every domain advances through numbered **opset versions**. An opset is a snapshot of the operator catalog: it defines which operators exist, what their inputs and outputs mean, and how edge cases are handled. When a model declares `opset_import { domain: "" version: 17 }`, it is saying "all unnamed-domain operators in this file must be interpreted according to the rules published in opset 17."

winml-cli defaults to **opset 17** when exporting a PyTorch model to ONNX. This is the value of `opset_version: int = 17` in `WinMLExportConfig` (`src/winml/modelkit/export/config.py`, line 75). Opset 17 introduced layer-normalisation and group-normalisation operators in native form, eliminating the multi-node decompositions required by earlier opsets, which is why it is the recommended baseline for modern transformer and vision architectures.

Higher opsets unlock additional operators and fix known edge-case behavior, but not every Execution Provider supports the latest opset. QNN, for instance, may lag behind the ONNX standard by one or two versions. If you need to target an older EP, pass a custom export configuration:

```bash
# Write a config override
echo '{"opset_version": 16}' > export_cfg.json

# Export with the override
winml export -m prajjwal1/bert-tiny -o bert.onnx --export-config export_cfg.json
```

You can also check the opset a saved model declares:

```bash
winml inspect -m bert.onnx
```

```text
Opset: ai.onnx == 17
```

When winml-cli's optimization and quantization pipelines transform a model, they preserve the declared opset unless explicitly instructed otherwise, so the model you receive after `winml quantize` will carry the same opset version as the model you supplied.

## See also

- [EP and Device](eps-and-devices.md)
- [Weight and Activation](weight-and-activation.md)
- [Datatype and Quantization](quantization.md)
- [winml inspect command](../commands/inspect.md)
- [winml export command](../commands/export.md)
