Unfortunately, Phi-3-mini-4k-instruct isn't something the WinML ModelKit (`winml` CLI) can handle today. It's a decoder-only generative LLM, and the current ModelKit pipeline is built for classic deep learning models — CNNs, encoders, vision transformers, and NLP classifiers (think ResNet, ViT, BERT, SegFormer, etc.). Generative architectures like Phi, LLaMA, Mistral, Qwen, GPT, and Stable Diffusion are explicitly out of scope. If you try to push Phi-3 through, it'll fail somewhere mid-pipeline (export, optimize, or quantize) and the error won't necessarily make the root cause obvious.

LLM support (with LoRA) is on the public ModelKit roadmap for late 2026, but it isn't there yet — I don't want to point you down a path that's just going to waste your time.

A few options that might actually work for you on Intel NPU today:

- **Intel's OpenVINO GenAI** has first-class support for Phi-3-mini and is designed specifically for running LLMs on Intel NPUs/iGPUs. That's almost certainly the shortest path to what you want.
- **ONNX Runtime GenAI** also has a Phi-3 recipe and can target the OpenVINO execution provider on Intel NPU.
- **Hugging Face Optimum-Intel** wraps OpenVINO and gives you a familiar `transformers`-style API.

If you have a classic (non-generative) model you also want to run on your Intel NPU — say a vision or embedding model — `winml` is a great fit for that and I'd be happy to walk you through it. Just let me know.
