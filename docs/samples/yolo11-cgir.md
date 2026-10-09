# YOLO11 - ONNX to CGIR (experimental)

This sample starts with an Ultralytics YOLO11 object-detection checkpoint,
exports an ONNX model, converts that model to a reusable CGIR artifact, and
measures it on the GPU. It also shows the ONNX + WinMLCG Execution Provider
path for testing CGC without keeping a CGIR file.

## Before you start

Complete [CGC environment setup](../getting-started/cgc-onboarding.md#environment-setup)
and activate that virtual environment. Inspect the available EPs and devices:

```powershell
winml sys --list-device --list-ep
```

## Step 1: Export YOLO11 to ONNX

Install the exporter into the same environment:

```powershell
uv pip install "ultralytics==8.4.146" --index-url https://packagefeedproxy.microsoft.io/pypi/simple
```

Place the Ultralytics `yolo11n.pt` checkpoint in the current directory, then
export a static, batch-one FP32 model without ONNX simplification:

```powershell
yolo export model=yolo11n.pt format=onnx imgsz=640 batch=1 dynamic=False simplify=False opset=17 device=cpu
```

This produces `yolo11n.onnx` with input shape `[1, 3, 640, 640]` and raw
detection output shape `[1, 84, 8400]`.

## Step 2: Benchmark ONNX

Test ONNX through the WinMLCG EP:

```powershell
winml perf -m .\yolo11n.onnx --runtime winml-ort --ep winmlcg -o .\model-winmlcg-perf.json
```

!!! warning
	With `windowsml==2.7.25.dev0` and `onnxruntime-windowsml==1.30.0.202609102321`,
	two full benchmark runs produced reports but exited with code 1 on the
	validation machine. The cause is unresolved; a saved report alone does
	not indicate a successful command.

To test online CGIR conversion through the Windows ML Runtime API instead:

```powershell
winml perf -m .\yolo11n.onnx --runtime winml-runtime -o .\model-runtime-perf.json
```

!!! warning
	With `windowsml==2.7.25.dev0`, the tested CLI's online conversion path
	fails with `Runtime.load_model() got an unexpected keyword argument 'io_counts'`.
	Use the offline build and MLIR Runtime path below with this combination.

## Step 3: Build CGIR

Generate the CGC build configuration:

```powershell
winml config -m .\yolo11n.onnx --backend cgc -o .\cgc-config.json
```

Build the reusable `.mlir` model:

```powershell
winml build -m .\yolo11n.onnx -c .\cgc-config.json -o .\model-cgir
```

The default configuration applies CGC compatibility rewrites and FP16
conversion before exporting MLIR. FP16 can change accuracy; add `--no-quant`
to the build command to skip quantization.

## Step 4: Benchmark CGIR

Run the converted model through the Windows ML Runtime API:

```powershell
winml perf -m .\model-cgir\model.mlir --runtime winml-runtime -o .\model-cgir\model-perf.json
```

!!! important
	The `.mlir` artifact cannot be used directly by ONNX Runtime. Load it
	through the Windows ML Runtime API for CGC inference.

## Step 5: Compare output tensors

Compare the CGIR model against the original FP32 ONNX model on ten generated
inputs, using ONNX Runtime on CPU as the reference:

```powershell
winml eval --mode compare -m .\model-cgir\model.mlir --runtime winml-runtime --reference .\yolo11n.onnx --reference-device cpu --samples 10 -o .\model-cgir\tensor-similarity.json
```

This checks raw output similarity, not object-detection accuracy. Inspect
absolute errors as well as cosine similarity. Detection mAP still requires
a labeled dataset with matching preprocessing and postprocessing and has not
been validated by this sample.

## See also

- [CGC Onboarding](../getting-started/cgc-onboarding.md)
