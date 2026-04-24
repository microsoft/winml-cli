# MVP

## Opening

In the next 20 minutes, I'll introduce ModelKit — a new product we've built over the past few months.

## What is ModelKit?

So, what is ModelKit? *ModelKit is a CLI toolkit to build portable, performant and high-quality models for Windows ML.*

**Goals**

With ModelKit, you can build a model once and run anywhere, using either built-in pipelines or composing your own from primitive commands.

You can also drill down into model details with ModelKit, pinpoint errors, or performance bottlenecks.

And ModelKit is AI-ready — we provide built-in skills that work with all mainstream coding agents.

**Promises**

ModelKit promises you an out-of-box user experience — one toolkit covers all the EPs, as well as full repeatability and traceability throughout both commands and pipelines.

And we build quality gates into ModelKit to catch compatibility issues and suggest fixes automatically.

## ModelKit — Command List

Let's quickly go through the commands. Basically, we bucketize the commands into four categories.

- The primitive commands — you can either use them individually or compose them into workflows.
- The pipeline commands that help you build and benchmark models end-to-end.
- And insight commands that enable model analysis and debugging.
- And a few utilities that support daily usage.

Now let's see how to use ModelKit in practice

## Background — BYOM Workflow

Some background info, Before we jump into the demos, let me explain the workflow behind ModelKit.

Here is a typical pipeline your model goes through. It is quite straightforward. Given a source model, the workflow will export, analyze, optimize, quantize if needed, and evaluate before shipping.

Here, I want to highlight are three commands that serve as quality gates.

- Analyze for portability, checking whether your model runs on the target EP
- Optimize, help improve the graph performance
- Quantize which affects the model accuracy

These three steps define the quality of your output. And ModelKit gives you full control over each one.

## Three Ways to Build ConvNeXt with ModelKit

I'm going to show you three ways to build models with ModelKit. And for easy comparison, three demos will all use ConvNeXT.

- First, I'll go with primitive commands. You'll see how to craft a model step by step with ModelKit.
- Then, I'll build ConvNeXT again with the config-driven pipeline, with only two commands.
- And last, I'll show you how to quick-bench a model with ModelKit — in one command

## Build ConvNeXt with Primitive Commands

In this demo, I'll walk you through these primitive commands. Each one handles a single stage of the pipeline.

#### Demo 1: Build ConvNeXT with Primitive Commands

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

OK, let me recap. In demo one, we used primitive commands to bring ConvNeXT to Windows ML — step by step, in three phases.

- Inspect the model.
- Build a portable ONNX through export, analyze, optimize, quantize.
- Then benchmark on device with compile, perf, and eval.

This gives you full control — you can jump into any stage, try different settings to fix errors or tweak performance.

## Analyze ConvNeXt for EP Compatibility

Let me go deeper on the analyzer, because it is the key to building portable ONNX models.

**The analyzer is made of two parts — Linter and AutoConf.**

The linter is like ESLint, but for ONNX. As you saw, it checks operators compatibility and classifies them — green for supported, gray for partial, red for unsupported.

AutoConf detects suboptimal patterns and generates the config for the optimizer.

Together they form the analyze-optimize loop

## Build ConvNeXt with Config-Driven Pipeline

#### Demo2

Same model, different approach. Instead of running each command manually, let's use `config` and `build`.

`wmk config` generates a JSON config. Let me show you what's inside. This is the config — it contains all settings for each pipeline step. Task, I/O shapes, optimization flags, quantization parameters, all auto-detected. You can review it, revise it, or pass it directly to the build command.

`wmk config -m facebook/convnext-base-224 -o convnext_config.json`

`wmk build` takes that config and runs the full pipeline. Export, analyze, optimize, quantize, compile — all in one go.

`wmk build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/`

And let's benchmark the result. Same model, same quality — but two commands instead of eight.

`wmk perf -m convnext_build/model.onnx --ep qnn --iterations 100`

In demo two, we used config and build. Two commands instead of five.

- Config command generates the build config — auto-detects everything.
- Build command orchestrates the full pipeline.

Same result, repeatable and tweakable. Think CMake for models.

## **Primitives Commands vs. Config-Driven Pipeline**

So, when do you use which?

Primitive commands are for flexible workflows — you can start from any stage, try different settings, fix errors, do experiments. Great for exploring and debugging, just like the coding phase in a development lifecycle

Config-driven pipeline is for delivery — repeatable, scriptable, easy to share with your team. Same quality, but reproducible.

Both approaches produce the same portable ONNX. It's about where you are and how much control you need.

## Benchmark ConvNeXt in One Command

>
>
>
> And the simplest way — one command. `wmk perf` with a model ID. It handles everything: load, export, optimize, benchmark. Live hardware monitoring included.
>
> `wmk perf -m facebook/convnext-base-224 --ep qnn --iterations 100 --monitor`
>
> Same ConvNeXT, three different approaches. Full control, automated pipeline, or one command. Pick what fits your workflow.
>

And the third way — the easiest. Say someone hands you a production-ready model. You just want a quick smoke test — does it run, how fast is it? One command. `wmk perf` with a model ID.

Load, export, optimize, benchmark — all behind the scenes. Think of it as a sanity check in the QA process.

## Why ModelKit?

OK, that's all for the demos.

Same ConvNeXT, three different approaches. Full control, automated pipeline, or one command. Pick what fits your best.

> Three ways to build model with ModelKit
>
> - Primitive commands for development — iterate, debug, experiment.
> - Config-driven pipeline for polish and hand over.
> - And one command for QA — validate, benchmark, deliver.

Now — if you want any of these

- build models for Windows ML,
- quick-bench a model
- catch compatibility issues ahead of time
- troubleshoot errors or performance bottlenecks
- or you just want AI to do the heavy lifting

**Please reach out to us for early access. Your feedback is the most valuable thing to us.**

**Roadmap**

And regarding the roadmap

- ModelKit is ready for early access by the end of this month
- We'll release the public beta in Q2, with coding agent skills and AITK integration.
- After that, we'll continue bringing more into ModelKit — LLM support, MLIR, and broader device coverage.
