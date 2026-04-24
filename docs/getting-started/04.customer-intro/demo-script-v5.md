# ModelKit Demo Script

**Model**: `facebook/convnext-base-224` (all three demos)

---

## Demo 1: Build ConvNeXT with Primitive Commands

Let's start with ConvNeXT. First, `inspect`

ConvNeXt is a family of CNN model inspired by Vision Transformers, introduced by Facebook in 2022.

It adopts several design choices from Transformers, and offers high accuracy while retaining the efficiency of CNNs, therefore it is widely adopted for tasks such as image classification, detection, and segmentation.

this tells us everything about the model. Task, model class, I/O shapes. No weights loaded, just metadata.

`wmk inspect -m facebook/convnext-base-224`

Now we export from PyTorch to ONNX.

`wmk export -m facebook/convnext-base-224 -o convnext/model.onnx -v`

Let's run the analyzer right away. It checks every operator against EPs — tells you what's supported, what's partial, what needs fixing. And it generates an optimization config automatically.

`wmk analyze -m convnext/model.onnx  --optim-config optim.json`

We apply the optimizer with that config. The analyzer told us what to fix, the optimizer fixes it.

`wmk optimize -m convnext/model.onnx -c optim_config.json -o convnext/model_opt.onnx`

Now quantize — compress the optimized model to INT8. At this point, we have a portable model. It can run on any ONNX Runtime backend.

`wmk quantize -m convnext/model_opt.onnx -o convnext/model_opt_int8.onnx`

Now let's compile for QNN — this generates device-specific binaries for the NPU.

`wmk compile -m convnext/model_opt_int8.onnx --ep qnn -o convnext/model_compiled.onnx`

And benchmark on NPU. Look at the latency — let's keep this number in mind.

`wmk perf -m convnext/model_compiled.onnx --ep qnn --iterations 100`

Now the same optimized model on CPU for comparison. See the difference? That's roughly a 25x speedup — the quantized model on NPU versus the original on CPU. Same model, same accuracy, completely different performance.

`wmk perf -m convnext/model_opt.onnx --ep cpu --iterations 100`

---

## Demo 2: Build ConvNeXT with Config + Build

Same model, different approach. Instead of running each command manually, let's use `config` and `build`.

`wmk config` generates a JSON config. Let me show you what's inside. This is the config — it contains all settings for each pipeline step. Task, I/O shapes, optimization flags, quantization parameters, all auto-detected. You can review it, revise it, or pass it directly to the build command.

`wmk config -m facebook/convnext-base-224 -o convnext_config.json`

`wmk build` takes that config and runs the full pipeline. Export, analyze, optimize, quantize, compile — all in one go.

`wmk build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/`

And let's benchmark the result. Same model, same quality — but two commands instead of eight.

`wmk perf -m convnext_build/model.onnx --ep qnn --iterations 100`

---

## Demo 3: Benchmark ConvNeXT in One Command

And the simplest way — one command. `wmk perf` with a model ID. It handles everything: load, export, optimize, benchmark. Live hardware monitoring included.

`wmk perf -m facebook/convnext-base-224 --ep qnn --iterations 100 --monitor`

Same ConvNeXT, three different approaches. Full control, automated pipeline, or one command. Pick what fits your workflow.
