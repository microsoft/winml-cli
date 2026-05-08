# `winml` CLI — Console Capture for UX Review

**Captured**: 2026-05-08 · Snapdragon X Elite ARM64 · Windows 11 · PowerShell 7 with `[Console]::OutputEncoding = UTF8` and `PYTHONIOENCODING=utf-8` · onnxruntime 1.23.4 (windowsml) · QNN NPU.

Each section shows two captures:

1. **Help** — exact `--help` text (what users see when they run `winml <cmd> --help`).
2. **Run** — a representative real invocation, including stderr and final exit code / wall time.

Captures are raw — emoji, box-drawing, library warnings, and timing all preserved as a real terminal would render them. Output may include leading library warnings (e.g. `huggingface_hub`, `onnxruntime`); these are part of the user experience.

---

## Table of contents

- [Top-level (`winml`)](#top-level-winml)
- [`winml --version`](#winml---version)
- [`winml analyze`](#winml-analyze)
- [`winml build`](#winml-build)
- [`winml compile`](#winml-compile)
- [`winml config`](#winml-config)
- [`winml eval`](#winml-eval)
- [`winml expand_rules`](#winml-expand_rules)
- [`winml export`](#winml-export)
- [`winml hub`](#winml-hub)
- [`winml inspect`](#winml-inspect)
- [`winml optimize`](#winml-optimize)
- [`winml perf`](#winml-perf)
- [`winml quantize`](#winml-quantize)
- [`winml sys`](#winml-sys)

---

## Top-level (`winml`)

### Help

```
===> winml --help
Usage: winml [OPTIONS] COMMAND [ARGS]...

  WML ModelKit - Accelerate Model Deployment on WinML.

  Universal ONNX export with QNN and OpenVINO backend support.

Options:
  --version      Show the version and exit.
  -v, --verbose  Increase verbosity (-v=INFO, -vv=DEBUG)
  -q, --quiet    Quiet mode - errors only
  --help         Show this message and exit.

Commands:
  analyze       Analyze ONNX model for runtime support with live progress.
  build         Build a WinML-optimized ONNX model from a HuggingFace model
  compile       Compile ONNX model to EP-specific format.
  config        Generate WinMLBuildConfig for a HuggingFace model or .onnx f
  eval          Evaluate model accuracy on a dataset.
  expand_rules  Expand runtime rules zip files in-place when directories and
  export        Export HuggingFace model to ONNX format with HTP.
  hub           Browse ModelKit's curated built-in model catalog.
  inspect       Inspect input model's ModelKit configuration.
  optimize      Optimize ONNX model with capability-driven optimizer.
  perf          Benchmark model inference performance.
  quantize      Quantize ONNX model by inserting QDQ nodes.
  sys           Display system information for ModelKit export.
```

---

## `winml --version`

### Help

```
===> winml --version
winml, version 0.0.2
```

---

## `winml analyze`

### Help

```
===> winml analyze --help
Usage: winml analyze [OPTIONS]

  Analyze ONNX model for runtime support with live progress.

  Performs static analysis to detect patterns and check operator
  compatibility, showing real-time per-operator results.

  Exit Codes:

      0: Model fully supported

      1: Partial support — some unsupported operators

      2: Error — invalid input or analysis failure

  Examples: \b     winml analyze --model model.onnx --ep qnn     winml analyze
  --model model.onnx --ep ov --device GPU     winml analyze --model model.onnx
  --output results.json

Options:
  -m, --model PATH                Path to ONNX model file to analyze
                                  [required]
  --ep [nvtensorrtrtxexecutionprovider|migraphxexecutionprovider|vitisaiexecutionprovider|qnnexecutionprovider|dmlexecutionprovider|openvinoexecutionprovider|cpuexecutionprovider|qnn|openvino|ov|vitisai|vitis|cpu|dml|nv_tensorrt_rtx|migraphx]
                                  Target execution provider. Full names:
                                  QNNExecutionProvider,
                                  OpenVINOExecutionProvider,
                                  VitisAIExecutionProvider. Aliases: qnn,
                                  ov/openvino, vitis/vitisai. If not
                                  specified, analyzes all supported EPs
  --device [cpu|gpu|npu]          Target device type (CPU, GPU, NPU). If not
                                  specified, uses NPU as default  [default:
                                  NPU]
  -v, --verbose                   Increase verbosity (-v=INFO, -vv=DEBUG)
  -q, --quiet                     Quiet mode - errors only to stderr
  -c, --config PATH               WinMLBuildConfig JSON file (from winml
                                  config). Provides defaults; explicit CLI
                                  options take precedence.
  --output PATH                   Save JSON output to file
  --information / --no-information
                                  Include detailed recommendations (default:
                                  enabled)
  --htp-metadata PATH             Path to HTP metadata JSON file for enhanced
                                  pattern extraction
  --run-unknown-op / --no-run-unknown-op
                                  Run unknown operators on local machine if
                                  possible (default: enabled)
  --save-node [partial|unsupported]
                                  Save specific node types for further
                                  analysis. Can be specified multiple times
                                  (e.g., --save-node partial --save-node
                                  unsupported).
  --optim-config PATH             Save auto-discovered optimization config to
                                  JSON file
  --help                          Show this message and exit.
```

### Run

```
===> winml analyze -m temp\cli-audit\resnet.onnx --ep qnn --output temp\ux\out_analyze.json

--- stderr ---

═══════════════════════════════════════════════════════════════════════════════
═
📊 OP CHECK
═══════════════════════════════════════════════════════════════════════════════
═
   📦 Model: resnet.onnx
   🔧 Opset: 17  Producer: pytorch v2.11.0
   📋 Operators: 122 total, 7 unique types
   🎯 Target: QNNExecutionProvider on NPU

───────────────────────────────────────────────────────────────────────────────
─
💻 EP 1: QNNExecutionProvider
───────────────────────────────────────────────────────────────────────────────
─
                📊 OP CHECK — QNNExecutionProvider  ✅ Complete                
 Op Type                    S/P/U                                              
 🟢 Conv (53)               53/0/0       ████████████████████████████████████… 
 🟢 Relu (49)               49/0/0       █████████████████████████████████████ 
 🟢 Add (16)                16/0/0       ████████████                          
 🟢 MaxPool (1)             1/0/0        █                                     
 🟢 GlobalAveragePool (1)   1/0/0        █                                     
 🟢 Flatten (1)             1/0/0        █                                     
 🟢 Gemm (1)                1/0/0        █                                     
 TOTAL (122)                122/0/0      ████████████████████████████████████… 
═══════════════════════════════════════════════════════════════════════════════
═
📈 ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════════════════════
═
   🟢 QNNExecutionProvider: 122/0/0
      Ready to deploy

  S/P/U = Supported/Partial/Unsupported  ██ supported  ██ partial  ██ 
unsupported  ██ unknown


===> exit=0  duration=61.61s
```

---

## `winml build`

### Help

```
===> winml build --help
Usage: winml build [OPTIONS]

  Build a WinML-optimized ONNX model from a HuggingFace model or .onnx file.

  Requires a config file generated by 'winml config'. The config file already
  contains device/precision settings (applied during 'winml config'
  generation). Specify either --output-dir or --use-cache for artifact
  destination.

  If -m points to an existing .onnx file, the build skips export and runs
  optimize -> quantize -> compile directly (ONNX build path).

  \b Examples:     # Full pipeline with pretrained weights     winml build -c
  config.json -m microsoft/resnet-50 -o output/

      # Build from pre-exported ONNX file     winml build -c config.json -m
      model.onnx -o output/

      # Export + optimize only     winml build -c config.json -m bert-base-
      uncased -o output/ --no-quant --no-compile

      # Random-weight build (no download)     winml build -c config.json -o
      output/

      # Use global cache     winml build -c config.json -m microsoft/resnet-50
      --use-cache

      # Force rebuild     winml build -c config.json -m microsoft/resnet-50 -o
      output/ --rebuild

Options:
  -c, --config PATH               WinMLBuildConfig JSON file (from winml
                                  config)  [required]
  -m, --model TEXT                HuggingFace model ID or path to .onnx file.
                                  Omit for random-weight build.
  -o, --output-dir PATH           Output directory for all build artifacts
  --use-cache                     Use ModelKit global cache (~/.cache/winml/).
                                  Mutually exclusive with -o.
  --rebuild                       Overwrite existing artifacts and rebuild
  --no-quant                      Skip quantization (overrides config)
  --no-compile / --compile        Skip compilation (overrides config).
                                  Default: skip.
  --ep TEXT                       Target execution provider for analyzer
                                  (e.g., 'qnn'). Falls back to compile config
                                  EP if not set.
  --device TEXT                   Target device for analyzer (e.g., 'NPU',
                                  'GPU'). Default: NPU.
  --no-analyze                    Skip analyzer loop during build
  --no-optimize                   Skip optimization (for pre-quantized ONNX
                                  models)
  --max-optim-iterations INTEGER  Maximum autoconf re-optimization rounds
                                  (default: 3). --no-analyze sets this to 0.
  --trust-remote-code             Trust remote code for custom model
                                  architectures (e.g., Mu2).
  -v, --verbose                   Enable verbose logging
  --help                          Show this message and exit.
```

### Run

```
===> winml build -c temp\ux\build_cfg.json -m microsoft/resnet-50 -o temp\ux\build_out

--- stderr ---

════════════════════════════════════════════════════════════
🔧 Setup — HuggingFace
════════════════════════════════════════════════════════════
   📦 Model:     microsoft/resnet-50  (pretrained)
   📁 Config:    build_cfg.json
   📂 Output:    temp\ux\build_out

════════════════════════════════════════════════════════════
🎯 Stages
════════════════════════════════════════════════════════════
✅ Export                                          7.6s
   Model class:  AutoModelForImageClassification
   Task:         image-classification
   Input:        pixel_values       [1, 3, 224, 224] float32
   Output:       logits
   📦 Artifact:   temp\ux\build_out\export.onnx  (97.4 MB)
✅ Optimize                                        126.8s
   Analyzing 122 nodes  (iter 1/3)
   - QNNExecutionProvider        122/0/0  ████████████████████████████████████
   Patterns
     No optimization patterns found
   Autoconf converged after 1 iteration(s)
   📦 Artifact:   temp\ux\build_out\optimized.onnx  (97.5 MB)
✅ Quantize                                        16.9s
   Dataset:      default  (image-classification)
   Calibration:  10 samples  (minmax)
   Precision:    uint8/uint16  (weight/activation)
   📦 Artifact:   temp\ux\build_out\quantized.onnx  (24.6 MB)

════════════════════════════════════════════════════════════
📊 Summary
════════════════════════════════════════════════════════════
✅ Build complete in 165.3s
   Export       7.6s
   Optimize     126.8s
   Quantize     16.9s
📦 Final artifact: temp\ux\build_out\model.onnx


===> exit=0  duration=168.66s
```

---

## `winml compile`

### Help

```
===> winml compile --help
Usage: winml compile [OPTIONS]

  Compile ONNX model to EP-specific format.

  This command compiles an ONNX model to an EP-specific format (e.g., QNN
  EPContext) with optional quantization. For pre-quantized models (containing
  QDQ nodes), use --no-quantize.

  \b Examples:     # Compile for NPU (default, uses QNN/VitisAI)     winml
  compile -m model.onnx

      # Compile for NPU with explicit VitisAI EP     winml compile -m
      model.onnx --ep vitisai

      # Compile for GPU with MIGraphX     winml compile -m model.onnx --device
      gpu --ep migraphx

      # Compile pre-quantized model     winml compile -m model_qdq.onnx --no-
      quantize

      # Compile using QAIRT SDK     winml compile -m model.onnx --compiler
      qairt --qnn-sdk-root /path/to/sdk

Options:
  -m, --model PATH                Input ONNX model file (required unless
                                  --list)
  -o, --output PATH               Output file path (e.g., model_compiled.onnx)
  --output-dir PATH               Output directory (default: same as input
                                  model)
  -d, --device [auto|npu|gpu|cpu]
                                  Target device  [default: npu]
  --ep [cpu|cuda|dml|migraphx|nv_tensorrt_rtx|openvino|qnn|vitisai]
                                  Force specific EP. Overrides device-to-
                                  provider mapping.
  --quantize / --no-quantize      Enable/disable quantization (default:
                                  enabled)
  --validate / --no-validate      Validate compiled model (default: enabled)
  -v, --verbose                   Enable verbose output
  --compiler [ort|qairt]          Compiler backend (default: ort)
  --qnn-sdk-root PATH             Path to QAIRT SDK root
  --embed                         Embed EP context in ONNX file (default:
                                  external .bin file)
  --list                          List available compilers for the selected
                                  device and exit
  -c, --config PATH               WinMLBuildConfig JSON file (from winml
                                  config). Provides defaults; explicit CLI
                                  options take precedence.
  --help                          Show this message and exit.
```

### Run

```
===> winml compile -m temp\ux\out_quantize.onnx --device npu --ep qnn -o temp\ux\out_compile.onnx
Input: temp\ux\out_quantize.onnx
Device: npu
EP: qnn
Provider: qnn
Compiler: ort
Output: temp\ux\out_compile.onnx

Compiling model...

Success! Model compiled
Output: temp\ux\out_compile.onnx
Compile time: 1.72s
Total time: 1.76s

--- stderr ---

===> exit=0  duration=4.06s
```

---

## `winml config`

### Help

```
===> winml config --help
Usage: winml config [OPTIONS]

  Generate WinMLBuildConfig for a HuggingFace model or .onnx file.

  This command auto-detects the task, model class, and I/O specifications from
  a HuggingFace model and generates a complete build configuration. When -m
  points to an existing .onnx file, generates a config with export=None for
  the ONNX build path.

  Requires at least one of -m/--model, --model-type, or --model-class.

  \b Examples:     # Basic usage - auto-detect everything     winml config -m
  microsoft/resnet-50

      # Override task     winml config -m bert-base-uncased --task text-
      classification

      # Target NPU with int8 quantization     winml config -m
      microsoft/resnet-50 --device npu --precision int8

      # Target GPU with fp16 (no quantization)     winml config -m bert-base-
      uncased --device gpu --precision fp16

      # Model type only (uses default HF config, auto-detects task)     winml
      config --model-type bert

      # Model type + task     winml config --model-type bert --task fill-mask

      # Override with JSON config file     winml config -m bert-base-uncased
      -c overrides.json

      # Vision model with shape overrides ({"height": 224, "width": 224})
      winml config --model-type resnet -t image-classification --shape-config
      shapes.json

      # Save to file     winml config -m bert-base-uncased -o config.json

      # Generate configs for submodules     winml config -m
      microsoft/resnet-50 --module ResNetConvLayer

Options:
  -m, --model TEXT                HuggingFace model ID (e.g.,
                                  microsoft/resnet-50) or path to .onnx file.
                                  Optional when --model-type is provided.
  -t, --task TEXT                 Override auto-detected task (e.g., image-
                                  classification, text-classification)
  --model-class TEXT              Override auto-detected model class (e.g.,
                                  CLIPTextModelWithProjection)
  --model-type TEXT               Override auto-detected model type (e.g.,
                                  bert, resnet). Can be used without -m to
                                  generate config from default HF settings.
                                  When used without --task, the first
                                  supported task is auto-selected.
  --module TEXT                   Generate configs for submodules matching
                                  this class name (e.g., ResNetConvLayer)
  -c, --config PATH               JSON config file with overrides
                                  (WinMLBuildConfig format)
  --shape-config PATH             JSON file with shape overrides passed to
                                  dummy input generation. Valid keys — text:
                                  sequence_length; vision: height, width,
                                  num_channels; audio: feature_size,
                                  nb_max_frames, audio_sequence_length.
  -d, --device [auto|npu|gpu|cpu]
                                  Target device (affects quant/compile
                                  config). Default: auto (no changes to
                                  config).
  --ep TEXT                       Force specific execution provider (qnn, dml,
                                  migraphx, nv_tensorrt_rtx, vitisai,
                                  openvino, cpu). Overrides device-to-provider
                                  mapping. When used without --device, device
                                  is inferred from EP.
  -p, --precision TEXT            Precision: auto, fp32, fp16, int8, int16, or
                                  w{x}a{y} (e.g., w8a16). Default: auto (based
                                  on device when device is specified).
  -o, --output PATH               Output JSON file path (default: stdout)
  --library TEXT                  Source library for TasksManager (default:
                                  transformers)
  -v, --verbose                   Enable verbose logging
  --no-quant                      Exclude quantization from generated config
                                  (sets quant=None)
  --no-compile / --compile        Exclude compilation from generated config
                                  (sets compile=None). Default: exclude.
  --trust-remote-code             Allow running custom code from model
                                  repository
  --help                          Show this message and exit.
```

### Run

```
===> winml config -m microsoft/resnet-50 --device npu -o temp\ux\out_config.json

--- stderr ---

════════════════════════════════════════════════════════════
📋 CONFIG GENERATION
════════════════════════════════════════════════════════════
   📦 Model:         microsoft/resnet-50
   🧩 Model class:   AutoModelForImageClassification  (auto-detected)
   🏷️ Task:          image-classification  (auto-detected)

   Input:        pixel_values       [1, 3, 224, 224] float32
   Output:       logits

   ⚙️  Resolution:
      Device:     NPU
      Quant:      uint8/uint16  (weight/activation)

   ✅ Config saved to: temp\ux\out_config.json


===> exit=0  duration=19.2s
```

---

## `winml eval`

### Help

```
===> winml eval --help
Usage: winml eval [OPTIONS]

  Evaluate model accuracy on a dataset.

  If --dataset is not provided, a default dataset is used based on the task.

  \b Examples:     # Use default dataset (auto-detected from task)     winml
  eval -m microsoft/resnet-50     winml eval -m model.onnx --model-id
  dslim/bert-base-NER

      # Specify dataset explicitly     winml eval -m microsoft/resnet-50
      --dataset imagenet-1k     winml eval -m model.onnx --model-id
      microsoft/resnet-50 --dataset imagenet-1k

      # Multi-config dataset with column overrides     winml eval -m
      model.onnx --model-id Intel/bert-base-uncased-mrpc \\         --dataset
      glue --dataset-name mrpc \\         --column input_column=sentence1

Options:
  -m, --model TEXT                Model to evaluate. Accepts three forms: (1)
                                  HuggingFace model ID, e.g. `-m
                                  <hf_model_id>`. (2) ONNX file path, e.g. `-m
                                  model.onnx` (requires --model-id). (3)
                                  Composite / split-encoder model as repeated
                                  role=path pairs, e.g. `-m image-
                                  encoder=vision.onnx -m text-
                                  encoder=text.onnx`.
  --model-id TEXT                 HuggingFace model ID when .onnx model file
                                  is provided in --model.
  --dataset TEXT                  HF dataset path (e.g. 'imagenet-1k',
                                  'glue'). If omitted, uses a default dataset
                                  for the task.
  --dataset-name TEXT             Dataset config name for multi-config
                                  datasets (e.g. 'mrpc').
  --task TEXT                     Task (e.g. 'image-classification'). Auto-
                                  detected from --model-id.
  --device [auto|cpu|gpu|npu]     Device to run on. 'auto' detects the best
                                  available device.  [default: auto]
  --ep [nvtensorrtrtxexecutionprovider|migraphxexecutionprovider|vitisaiexecutionprovider|qnnexecutionprovider|dmlexecutionprovider|openvinoexecutionprovider|cpuexecutionprovider|qnn|openvino|ov|vitisai|vitis|cpu|dml|nv_tensorrt_rtx|migraphx]
                                  Target execution provider. Full names:
                                  QNNExecutionProvider,
                                  OpenVINOExecutionProvider,
                                  VitisAIExecutionProvider. Aliases: qnn,
                                  ov/openvino, vitis/vitisai
  --samples INTEGER               Number of dataset samples.  [default: 100]
  --split TEXT                    Dataset split.  [default: validation]
  --shuffle / --no-shuffle        Shuffle dataset before sampling.  [default:
                                  shuffle]
  --streaming                     Stream dataset instead of downloading fully.
  --column TEXT                   Column mapping as key=value (e.g. --column
                                  input_column=image).
  --label-mapping PATH            Path to a JSON file with label mapping:
                                  {"label_name": id}.
  -o, --output PATH               Output JSON file path.
  -v, --verbose                   Enable verbose output.
  --schema                        Print expected dataset schema for the given
                                  --task and exit.
  -c, --config PATH               WinMLBuildConfig JSON file (from winml
                                  config). Provides defaults; explicit CLI
                                  options take precedence.
  --help                          Show this message and exit.
```

### Run

```
===> winml eval -m microsoft/resnet-50 --samples 5

┌─────────────────────────────────┐
│ Evaluation: microsoft/resnet-50 │
└─────────────────────────────────┘

Task:       image-classification
Device:     npu
Dataset:    timm/mini-imagenet
Samples:    100

┌───────────────────────┬─────────┐
│ Metric                │   Value │
├───────────────────────┼─────────┤
│ accuracy              │  0.7600 │
│ total_time_in_seconds │  1.8992 │
│ samples_per_second    │ 52.6533 │
│ latency_in_seconds    │  0.0190 │
└───────────────────────┴─────────┘


--- stderr ---
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
[0;93m2026-05-08 14:21:11.6110754 [W:onnxruntime:, session_state.cc:1316 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
[0;93m2026-05-08 14:21:11.9151960 [W:onnxruntime:, session_state.cc:1318 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
C:\Users\zhenni\repos\wmk\.venv\lib\site-packages\sklearn\metrics\_classification.py:98: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
  type_true = type_of_target(y_true, input_name="y_true")
C:\Users\zhenni\repos\wmk\.venv\lib\site-packages\sklearn\metrics\_classification.py:99: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
  type_pred = type_of_target(y_pred, input_name="y_pred")

===> exit=0  duration=31.3s
```

---

## `winml expand_rules`

### Help

```
===> winml expand_rules --help
Usage: winml expand_rules [OPTIONS]

  Expand runtime rules zip files in-place when directories and zips exist.

Options:
  --rules-dir-entry TEXT  Optional rule directory entry. May be repeated. If
                          omitted, uses all entries from MODELKIT_RULES_DIR.
                          Each entry is resolved by
                          rule_loader._resolve_env_rules_dir_entry.
  --glob TEXT             Zip filename glob to process.  [default: *.zip]
  --help                  Show this message and exit.
```

---

## `winml export`

### Help

```
===> winml export --help
Usage: winml export [OPTIONS]

  Export HuggingFace model to ONNX format with HTP.

  This command converts a HuggingFace model to ONNX format using the
  Hierarchy-preserving Tags Protocol (HTP) with optional full reporting.

  The export process (8 steps): 1. Model Preparation - Load and configure
  model 2. Input Generation - Generate example inputs 3. Hierarchy Building -
  Trace module execution 4. ONNX Export - Convert to ONNX format (TorchScript
  by default) 5. Node Tagger Creation - Create tagger from hierarchy 6. Node
  Tagging - Apply hierarchy tags to nodes 7. Tag Injection - Embed tags in
  ONNX node metadata_props 8. Metadata Generation - Generate reports (if
  --with-report)

  \b Examples:     # Basic export     winml export --model prajjwal1/bert-tiny
  --output model.onnx

      # Short form     winml export -m prajjwal1/bert-tiny -o model.onnx

      # With verbose output and full reporting     winml export -m
      facebook/convnext-tiny-224 -o convnext.onnx -v --with-report

      # Clean ONNX output (no hierarchy metadata, for optimization)     winml
      export -m prajjwal1/bert-tiny -o model.onnx --clean-onnx

      # Use PyTorch dynamo export (for rich node metadata)     winml export -m
      prajjwal1/bert-tiny -o model.onnx --dynamo

      # Include torch.nn modules in hierarchy     winml export -m
      prajjwal1/bert-tiny -o model.onnx --torch-module LayerNorm,Embedding

      # Custom input specifications from JSON file     winml export -m bert-
      base-uncased -o bert.onnx --input-specs inputs.json

      # Custom ONNX export configuration     winml export -m bert-base-uncased
      -o bert.onnx --export-config config.json

Options:
  -m, --model TEXT              HuggingFace model name or local path (e.g.,
                                prajjwal1/bert-tiny)  [required]
  -o, --output PATH             Output ONNX file path (e.g., model.onnx)
                                [required]
  -v, --verbose                 Enable verbose console output (8-step format)
  --with-report                 Generate full export reports (markdown, JSON,
                                console tree)
  --clean-onnx, --no-hierarchy  Skip embedding hierarchy_tag metadata in ONNX
                                (clean ONNX output)
  --dynamo                      Enable PyTorch 2.9+ dynamo export for rich
                                node metadata
  --torch-module TEXT           Include torch.nn modules in hierarchy (comma-
                                separated, e.g., LayerNorm,Embedding)
  --input-specs PATH            JSON file with input specifications (auto-
                                generates if not provided)
  -t, --task TEXT               Override auto-detected task (e.g., image-
                                feature-extraction, feature-extraction)
  --export-config PATH          ONNX export configuration JSON (opset_version,
                                do_constant_folding, etc.)
  --shape-config PATH           JSON with shape overrides (e.g.,
                                {"sequence_length": 2048, "height": 640}).
  -c, --config PATH             WinMLBuildConfig JSON file (from winml
                                config). Provides defaults; explicit CLI
                                options take precedence.
  --help                        Show this message and exit.
```

### Run

```
===> winml export -m microsoft/resnet-50 -o temp\ux\out_export.onnx
Model: microsoft/resnet-50
Output: temp\ux\out_export.onnx
Auto-resolved input specs: ['pixel_values']
Auto-resolved output specs: ['logits']

Starting HTP export...
Detected task: image-classification

Success! Model exported to: temp\ux\out_export.onnx

--- stderr ---
C:\Users\zhenni\repos\wmk\.venv\lib\site-packages\transformers\models\resnet\modeling_resnet.py:72: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if num_channels != self.num_channels:

===> exit=0  duration=24.27s
```

---

## `winml hub`

### Help

```
===> winml hub --help
Usage: winml hub [OPTIONS]

  Browse ModelKit's curated built-in model catalog.

  Lists HuggingFace models that have been validated end-to-end (export ->
  quantise -> run on device) with confirmed accuracy results. Use ``--output``
  to save results to a JSON file.

  \b Accuracy legend:   [+] PASS       -- drop within tolerance   [~] AT_RISK
  -- borderline drop, use with care   [!] REGRESSION -- accuracy degraded
  beyond threshold   drop %         -- relative change vs FP32 baseline

  \b Use ``winml hub --model <model_id>`` for per-model perf and accuracy. Use
  ``winml inspect -m <model_id>`` for architecture details.

  \b Examples:     winml hub     winml hub --model-type bert     winml hub
  --task text-classification     winml hub --model ProsusAI/finbert     winml
  hub --output results/catalog.json

Options:
  -t, --model-type TYPE  Filter by model architecture (e.g. bert, roberta,
                         vit).
  -k, --task TASK        Filter by HuggingFace task (e.g. text-classification,
                         image-segmentation).
  -m, --model MODEL_ID   Show perf and accuracy details for a specific model.
  -o, --output PATH      Save results to a JSON file.
  --help                 Show this message and exit.
```

### Run

```
===> winml hub --task image-classification
┌───────────────── ModelKit Catalog  |  5 validated model(s) ─────────────────┐
│   Model                                 │  Task                │  Model T   │
│ ────────────────────────────────────────┼──────────────────────┼─────────── │
│   google/vit-base-patch16-224           │  image-classificati  │  vit       │
│   rizvandwiki/gender-classification     │  image-classificati  │  vit       │
│   microsoft/swin-large-patch4-window7-  │  image-classificati  │  swin      │
│   microsoft/resnet-50                   │  image-classificati  │  resnet    │
│   facebook/convnext-tiny-224            │  image-classificati  │  convnex   │
└─────────────────────────────────────────────────────────────────────────────┘
Use  winml hub --model <id>  to see perf and accuracy details.

--- stderr ---

===> exit=0  duration=8.08s
```

---

## `winml inspect`

### Help

```
===> winml inspect --help
Usage: winml inspect [OPTIONS]

  Inspect input model's ModelKit configuration.

  Shows the loader, exporter, WinML inference class, I/O specs, and build
  resolution that the pipeline will use for the given model.

  Supports inspection without a model ID via --model-type or --model-class.

  \b Examples:     # Basic inspection     winml inspect -m microsoft/resnet-50

      # Inspect by model type only (no weight download)     winml inspect
      --model-type bert --task fill-mask

      # Override model class     winml inspect -m custom-model --model-class
      BertForCTC

      # JSON output     winml inspect -m google-bert/bert-base-uncased
      --format json

      # List all known tasks     winml inspect --list-tasks

Options:
  -m, --model TEXT           HuggingFace model ID (e.g., microsoft/resnet-50)
  -f, --format [table|json]  Output format (default: table)
  -v, --verbose              Show full configuration details
  -t, --task TEXT            Override auto-detected task (e.g., image-
                             classification, feature-extraction)
  -H, --hierarchy            Show HF module hierarchy (uses random weights, no
                             weight download)
  --list-tasks               List all known tasks and exit
  --model-type TEXT          Override model type (e.g., bert, resnet) — can be
                             used without --model
  --model-class TEXT         Override model class (e.g., BertForMaskedLM) —
                             can be used without --model
  --help                     Show this message and exit.
```

### Run

```
===> winml inspect -m microsoft/resnet-50
┌─────────────────────────────────── Model ───────────────────────────────────┐
│ microsoft/resnet-50                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── Model Information ─────────────────────────────┐
│   Model Type         resnet                                                 │
│   Task               image-classification (via TasksManager)                │
│   Architectures      ResNetForImageClassification                           │
│   Overall Support    Default                                                │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── Loader Configuration ────────────────────────────┐
│   HF Model Class    AutoModelForImageClassification                         │
│   Source            TasksManager                                            │
│   Status            Default                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────── Exporter Configuration ───────────────────────────┐
│   ONNX Config       ResNetOnnxConfig                                        │
│   Source            TasksManager                                            │
│   Status            Default                                                 │
│   OPSET Version     17                                                      │
│                                                                             │
│   Input Tensors                                                             │
│     pixel_values    float32  [B, 3, 224, 224]  range (0, 1)                 │
│                                                                             │
│   Output Tensors                                                            │
│     logits          [B]                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── WinML Inference Class ───────────────────────────┐
│   Class     WinMLModelForImageClassification                                │
│   Source    TASK_TO_WINML_CLASS                                             │
│   Status    Default                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Data Processing ──────────────────────────────┐
│   Processor            ConvNextImageProcessorFast (via auto_class)          │
│   Image Processor      ConvNextImageProcessor (via hf_registry)             │
│   Feature Extractor    ConvNextFeatureExtractor (via hub_config)            │
└─────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── IO Configuration ──────────────────────────────┐
│   Image Size      224 x 224                                                 │
│   Channels        3                                                         │
│   Hidden Sizes    256 → 512 → 1024 → 2048                                   │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────── Cache Status ────────────────────────────────┐
│   Status             4/4 stages cached                                      │
│   Total Size         219.49 MB                                              │
│                                                                             │
│   Pipeline Stages                                                           │
│     + export         97.44 MB                                               │
│     + optimize       97.45 MB                                               │
│     + quantize       24.6 MB                                                │
│     + compile        0.0 MB                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────── Notes ───────────────────────────────────┐
│ • Loader: Using TasksManager defaults                                       │
│ • Exporter: Using TasksManager defaults                                     │
│ • WinML: Using task-based class                                             │
└─────────────────────────────────────────────────────────────────────────────┘

--- stderr ---

===> exit=0  duration=24.28s
```

---

## `winml optimize`

### Help

```
===> winml optimize --help
Usage: winml optimize [OPTIONS]

  Optimize ONNX model with capability-driven optimizer.

  This command applies graph optimizations and operator fusions to an ONNX
  model using the capability-driven optimizer from winml.modelkit.optim.

  CLI options are auto-generated from registered capabilities.

  Configuration precedence (highest to lowest):     1. CLI options
  (--enable-X, --disable-X)     2. Config file options (-c/--config)     3.
  Preset defaults (-p/--preset)     4. Capability defaults

  \b Examples:     # List available capabilities     winml optimize --list-
  capabilities

      # List available rewrite pattern families     winml optimize --list-
      rewrites

      # Pattern rewrite flags follow: --enable-{source-slug}-{target-slug}
      # Run --list-rewrites to discover all available flag names.     #
      Example (all GELU variants -> single Gelu node):     winml optimize -m
      model.onnx -o out.onnx --enable-gelu-singlegelu     # Example (only
      Gelu1 variant -> single Gelu node):     winml optimize -m model.onnx -o
      out.onnx --enable-gelu1-singlegelu

      # Basic optimization with GELU fusion     winml optimize -m model.onnx
      -o model_opt.onnx --enable-gelu-fusion

      # Use transformer preset     winml optimize -m bert.onnx --preset
      transformer-optimized

      # Use config file     winml optimize -m model.onnx -c config.toml

Options:
  -l, --list-capabilities         List all registered optimization
                                  capabilities and exit
  --list-rewrites                 List available pattern rewrite families and
                                  exit
  -m, --model PATH                Input ONNX model file
  -o, --output PATH               Output path (default: {input}_opt.onnx)
  -p, --preset [qnn-compatible|transformer-optimized|full|minimal]
                                  Use optimization preset
  -c, --config PATH               Configuration file (YAML/JSON)
  -v, --verbose                   Enable verbose output
  --enable-gelu-fusion / --disable-gelu-fusion
                                  Fuse multi-operation GELU approximation
                                  patterns into single GELU op
  --enable-fast-gelu-fusion / --disable-fast-gelu-fusion
                                  Fuse fast GELU approximation patterns into
                                  optimized operation
  --enable-bias-gelu-fusion / --disable-bias-gelu-fusion
                                  Fuse bias addition and GELU activation into
                                  single operation
  --enable-quick-gelu-fusion / --disable-quick-gelu-fusion
                                  Fuse QuickGelu variant patterns (x *
                                  sigmoid(1.702 * x))
  --enable-gelu-approximation / --disable-gelu-approximation
                                  Convert exact Gelu/BiasGelu to FastGelu for
                                  improved inference speed
  --enable-bias-softmax-fusion / --disable-bias-softmax-fusion
                                  Fuse Bias+Softmax into single operation
  --enable-bias-dropout-fusion / --disable-bias-dropout-fusion
                                  Fuse Bias+Dropout patterns
  --enable-conv-add-fusion / --disable-conv-add-fusion
                                  Fuse Conv+Add (bias) patterns
  --enable-conv-bn-fusion / --disable-conv-bn-fusion
                                  Fuse Conv+BatchNormalization into modified
                                  Conv weights
  --enable-conv-mul-fusion / --disable-conv-mul-fusion
                                  Fuse Conv+Multiply patterns
  --enable-conv-activation-fusion / --disable-conv-activation-fusion
                                  Fuse Conv+activation (ReLU, LeakyReLU,
                                  Sigmoid, Tanh, Clip)
  --enable-slice-elimination / --disable-slice-elimination
                                  Eliminate redundant Slice operations
  --enable-expand-elimination / --disable-expand-elimination
                                  Eliminate Expand when output shape equals
                                  input shape
  --enable-unsqueeze-elimination / --disable-unsqueeze-elimination
                                  Eliminate Unsqueeze of initializers (fold
                                  into weights)
  --enable-gemm-activation-fusion / --disable-gemm-activation-fusion
                                  Fuse GEMM+activation functions
  --enable-gemm-sum-fusion / --disable-gemm-sum-fusion
                                  Fuse GEMM+Sum patterns
  --enable-gemm-transpose-fusion / --disable-gemm-transpose-fusion
                                  Fuse GEMM+Transpose patterns
  --enable-concat-slice-elimination / --disable-concat-slice-elimination
                                  Eliminate Concat followed by Slice that
                                  extracts original tensors
  --enable-double-qdq-pairs-remover / --disable-double-qdq-pairs-remover
                                  Remove consecutive
                                  QuantizeLinear->DequantizeLinear pairs
  --enable-constant-folding / --disable-constant-folding
                                  Pre-compute constant expressions (may
                                  increase model size)
  --enable-layer-norm-fusion / --disable-layer-norm-fusion
                                  Fuse LayerNorm computation
                                  (ReduceMean->Sub->Pow->Sqrt->Div->Mul->Add)
  --enable-skip-layer-norm-fusion / --disable-skip-layer-norm-fusion
                                  Fuse Add(residual)+LayerNorm into
                                  SkipLayerNormalization
  --enable-simplified-layer-norm-fusion / --disable-simplified-layer-norm-fusion
                                  Fuse simplified LayerNorm (without mean-
                                  centering)
  --enable-transpose-optimizer / --disable-transpose-optimizer
                                  Optimize and eliminate redundant transpose
                                  operations
  --enable-nhwc-transformer / --disable-nhwc-transformer
                                  Transform NCHW to NHWC layout (GPU memory
                                  access optimized)
  --enable-nchwc-transformer / --disable-nchwc-transformer
                                  Transform NCHW to NCHWc layout (CPU SIMD
                                  optimized)
  --enable-conv-add-activation-fusion / --disable-conv-add-activation-fusion
                                  Fuse Conv+Add+Activation chain into single
                                  FusedConv
  --enable-matmul-add-fusion / --disable-matmul-add-fusion
                                  Fuse MatMul+Add operations into single
                                  kernel
  --enable-matmul-activation-fusion / --disable-matmul-activation-fusion
                                  Fuse MatMul+activation functions (ReLU,
                                  Sigmoid, Tanh)
  --enable-matmul-transpose-fusion / --disable-matmul-transpose-fusion
                                  Fuse MatMul+Transpose operations
  --enable-matmul-scale-fusion / --disable-matmul-scale-fusion
                                  Fuse MatMul+Scale (multiply by constant)
  --enable-matmul-bn-fusion / --disable-matmul-bn-fusion
                                  Fuse MatMul+BatchNormalization
  --enable-dynamic-quantize-matmul-fusion / --disable-dynamic-quantize-matmul-fusion
                                  Dynamic quantization for MatMul operations
  --enable-gather-slice-to-split-fusion / --disable-gather-slice-to-split-fusion
                                  Fuse Gather+Slice patterns to Split
                                  operation
  --enable-gather-to-slice-fusion / --disable-gather-to-slice-fusion
                                  Convert Gather to Slice where index is
                                  contiguous
  --enable-pad-fusion / --disable-pad-fusion
                                  Fuse Pad with subsequent Conv/Pool
                                  operations
  --enable-not-where-fusion / --disable-not-where-fusion
                                  Fuse Not+Where patterns
  --enable-gelu-singlegelu / --disable-gelu-singlegelu
                                  Rewrite [Gelu1Pattern, Gelu2Pattern,
                                  Gelu3Pattern, Gelu4Pattern] ->
                                  SingleGeluPattern
  --enable-gelu1-singlegelu / --disable-gelu1-singlegelu
                                  Rewrite Gelu1Pattern -> SingleGeluPattern
  --enable-gelu2-singlegelu / --disable-gelu2-singlegelu
                                  Rewrite Gelu2Pattern -> SingleGeluPattern
  --enable-gelu3-singlegelu / --disable-gelu3-singlegelu
                                  Rewrite Gelu3Pattern -> SingleGeluPattern
  --enable-gelu4-singlegelu / --disable-gelu4-singlegelu
                                  Rewrite Gelu4Pattern -> SingleGeluPattern
  --enable-matmuladd-reshapegemm / --disable-matmuladd-reshapegemm
                                  Rewrite [MatMulAddPattern] ->
                                  ReshapeGemmReshapePattern
  --enable-matmuladd-conv2d4d / --disable-matmuladd-conv2d4d
                                  Rewrite [MatMulAddPattern] ->
                                  Conv2DInplaceLinear4DPattern
  --enable-matmuladd-conv2d3d / --disable-matmuladd-conv2d3d
                                  Rewrite [MatMulAddPattern] ->
                                  Conv2DInplaceLinear3DPattern
  --enable-matmuladd-conv2d2d / --disable-matmuladd-conv2d2d
                                  Rewrite [MatMulAddPattern] ->
                                  Conv2DInplaceLinear2DPattern
  --enable-layernormalization-singlelayernorm / --disable-layernormalization-singlelayernorm
                                  Rewrite [LayerNormalizationPowPattern,
                                  LayerNormalizationMulPattern] ->
                                  TransposedSingleLayerNormalizationPattern
  --enable-layernormpow-singlelayernorm / --disable-layernormpow-singlelayernorm
                                  Rewrite LayerNormalizationPowPattern ->
                                  TransposedSingleLayerNormalizationPattern
  --enable-layernormmul-singlelayernorm / --disable-layernormmul-singlelayernorm
                                  Rewrite LayerNormalizationMulPattern ->
                                  TransposedSingleLayerNormalizationPattern
  --enable-highdimRTR-lowdimRTR / --disable-highdimRTR-lowdimRTR
                                  Rewrite [ReshapeTransposeReshapeOverlyHighDi
                                  mPattern] ->
                                  ReshapeTransposeReshapeLowDimPattern
  --enable-attention-expandedattention / --disable-attention-expandedattention
                                  Rewrite [TransposeAttentionPattern] ->
                                  ExpandedAttentionPattern
  --enable-attention-fusion / --disable-attention-fusion
                                  Fuse attention computation patterns into
                                  optimized operations
  --enable-fuse-rmsnorm / --disable-fuse-rmsnorm
                                  Fuse RMSNorm
                                  (Pow->ReduceMean->Add->Sqrt->Div->Mul) into
                                  LpNormalization(p=2)
  --enable-embed-layer-norm-fusion / --disable-embed-layer-norm-fusion
                                  Fuse embedding+position+token
                                  embeddings+LayerNorm
  --enable-bias-skip-layer-norm-fusion / --disable-bias-skip-layer-norm-fusion
                                  Fuse Bias+Add(residual)+LayerNorm into
                                  BiasSkipLayerNorm (FusionPipe only)
  --enable-clamp-constant-values / --disable-clamp-constant-values
                                  Clamp extreme float constants (e.g., -inf ->
                                  -1e3) to prevent quantization issues
  --enable-remove-isnan-in-attention-mask / --disable-remove-isnan-in-attention-mask
                                  Remove Softmax->IsNaN->Where NaN guard
                                  patterns in attention
  --help                          Show this message and exit.
```

### Run

```
===> winml optimize -m temp\cli-audit\resnet.onnx -o temp\ux\out_optimize.onnx
Input: temp\cli-audit\resnet.onnx
Output: temp\ux\out_optimize.onnx

Loading model...
Running optimizer...
Saving optimized model...

Success! Model optimized: temp\ux\out_optimize.onnx
Nodes: 122 -> 122 (0.0% reduction)

--- stderr ---

===> exit=0  duration=12.11s
```

---

## `winml perf`

### Help

```
===> winml perf --help
Usage: winml perf [OPTIONS]

  Benchmark model inference performance.

  Measures latency and throughput using random input data generated from the
  model's I/O configuration.

  Accepts both HuggingFace model IDs and local .onnx files. HF models go
  through PerfBenchmark; .onnx files use _run_onnx_benchmark.

  \b Examples:     # Basic benchmark (HuggingFace model)     winml perf -m
  microsoft/resnet-50

      # Benchmark a pre-exported ONNX file directly     winml perf -m
      model.onnx --device cpu

      # With custom iterations on NPU     winml perf -m microsoft/resnet-50
      --iterations 500 --device npu

      # Text model with explicit task     winml perf -m bert-base-uncased
      --task text-classification

      # Per-module benchmarking     winml perf -m bert-base-uncased --module
      BertAttention

      # Operator-level profiling (QNN NPU)     winml perf -m model.onnx --op-
      tracing basic

Options:
  -m, --model TEXT             Model identifier: HuggingFace model ID or local
                               .onnx file.
  --task TEXT                  Explicit task (e.g., 'image-classification').
                               Auto-detected if not specified.
  --iterations INTEGER         Number of benchmark iterations  [default: 100]
  --warmup INTEGER             Number of warmup iterations (excluded from
                               statistics)  [default: 10]
  --device [auto|cpu|gpu|npu]  Device to run benchmark on  [default: auto]
  --precision TEXT             Precision mode: auto, fp32, fp16, int8, int16,
                               or w{x}a{y} (e.g., w8a16).  [default: auto]
  --ep TEXT                    Force specific execution provider (qnn, dml,
                               migraphx, nv_tensorrt_rtx, vitisai, openvino,
                               cpu). Overrides device-to-provider mapping.
  -o, --output PATH            Output JSON file path. Defaults to
                               '{model_slug}_perf.json'
  --batch-size INTEGER         Batch size for input generation  [default: 1]
  --shape-config PATH          JSON file with shape overrides (e.g.,
                               {"height": 480, "width": 480}).
  --no-quantize                Skip quantization during model build
  --rebuild                    Force rebuild even if cached artifacts exist
  --ignore-cache               Build from scratch in a temp folder (discard
                               after benchmarking)
  --module TEXT                HF module class name for per-module
                               benchmarking (e.g., 'BertAttention'). Builds
                               and benchmarks each instance separately.
  --monitor                    Show live NPU utilization chart during
                               benchmark
  --op-tracing [basic|detail]  Enable operator-level profiling (requires
                               onnxruntime-qnn)
  --compare-devices TEXT       Compare benchmark across devices (e.g.,
                               'cpu,npu'). Not yet implemented.
  -v, --verbose                Enable verbose output
  -c, --config PATH            WinMLBuildConfig JSON file (from winml config).
                               Provides defaults; explicit CLI options take
                               precedence.
  --help                       Show this message and exit.
```

### Run

```
===> winml perf -m temp\cli-audit\resnet.onnx --iterations 5 --warmup 1 -o temp\ux\out_perf.json
Benchmarking ONNX: temp\cli-audit\resnet.onnx

Device:      auto (npu)
Task:        n/a (direct ONNX)

Latency (ms)
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  Avg │  P50 │  P90 │  P95 │  P99 │  Min │  Max │  Std │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 5.88 │ 5.91 │ 6.13 │ 6.13 │ 6.13 │ 5.67 │ 6.13 │ 0.15 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

Throughput: 169.92 samples/sec

Results saved to: temp\ux\out_perf.json

--- stderr ---

Device:      npu
Inputs:      pixel_values         [1, 3, 224, 224]       float32
Outputs:     logits               [1, 1000]


===> exit=0  duration=11.14s
```

---

## `winml quantize`

### Help

```
===> winml quantize --help
Usage: winml quantize [OPTIONS]

  Quantize ONNX model by inserting QDQ nodes.

  This command applies static quantization to an ONNX model using calibration
  data to determine quantization parameters. The output model contains
  QuantizeLinear and DequantizeLinear nodes for quantization-aware inference.

  \b Examples:     # Basic quantization with defaults (10 samples, uint8)
  winml quantize -m model.onnx

      # Use precision shorthand (same as --weight-type uint8 --activation-type
      uint8)     winml quantize -m model.onnx --precision int8

      # Int16 quantization     winml quantize -m model.onnx --precision int16

      # Custom output path and more samples     winml quantize -m model.onnx
      -o quantized.onnx --samples 100

      # Explicit types with entropy calibration     winml quantize -m
      model.onnx --weight-type int8 --method entropy

Options:
  -m, --model PATH                Input ONNX model file  [required]
  -o, --output PATH               Output path (default: {input}_qdq.onnx)
  -p, --precision TEXT            Quantization precision: int8, int16, or
                                  w{x}a{y} (e.g., w8a16). Overridden by
                                  explicit --weight-type/--activation-type.
  --samples INTEGER               Number of calibration samples (default: 10)
  --method [minmax|entropy|percentile]
                                  Calibration method (default: minmax)
  --weight-type [uint8|int8|uint16|int16]
                                  Weight quantization type. Overrides
                                  --precision.
  --activation-type [uint8|int8|uint16|int16]
                                  Activation quantization type. Overrides
                                  --precision.
  --per-channel                   Use per-channel quantization
  --symmetric                     Use symmetric quantization
  --task TEXT                     Task for calibration dataset selection
                                  (e.g., 'image-classification').
  -v, --verbose                   Enable verbose output
  -c, --config PATH               WinMLBuildConfig JSON file (from winml
                                  config). Provides defaults; explicit CLI
                                  options take precedence.
  --help                          Show this message and exit.
```

### Run

```
===> winml quantize -m temp\cli-audit\resnet.onnx --samples 3 -o temp\ux\out_quantize.onnx
Input: temp\cli-audit\resnet.onnx
Output: temp\ux\out_quantize.onnx
Weight type: uint8
Activation type: uint8
Samples: 3
Method: minmax
Dataset: Random data (synthetic from ONNX I/O specs)

Running quantization...

Success! Model quantized
Output: temp\ux\out_quantize.onnx
QDQ nodes inserted: 256
Total time: 18.22s

--- stderr ---
[2026-05-08T14:19:48] WARNING: Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
[2026-05-08T14:19:54] WARNING: Please consider pre-processing before quantization. See https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 

===> exit=0  duration=24.26s
```

---

## `winml sys`

### Help

```
===> winml sys --help
Usage: winml sys [OPTIONS]

  Display system information for ModelKit export.

  This command gathers and displays information relevant to ONNX model export,
  including Python version, library versions, hardware capabilities, and
  backend SDK availability.

  Use this to diagnose issues with model export or verify your environment is
  properly configured.

  \b Examples:     # Display system info (human-readable format)     winml sys

      # Get output as JSON for scripting     winml sys --format json

      # Show detailed info     winml sys --verbose

      # Compact format for quick overview     winml sys --format compact

      # List available devices     winml sys --list-device

      # List execution providers as JSON     winml sys --list-ep --format json

Options:
  -f, --format [text|json|compact]
                                  Output format: text (human-readable), json,
                                  or compact
  -v, --verbose                   Include additional diagnostic information
  --list-device                   List available devices in priority order
  --list-ep                       List available execution providers
  --help                          Show this message and exit.
```

### Run

```
===> winml sys --format compact
Python: 3.10.19 (Windows)
torch: 2.11.0 | transformers: 4.57.6 | onnx: 1.18.0
QNN: OK | OpenVINO: N/A
Export Ready: ONNX OK

--- stderr ---

===> exit=0  duration=5.09s
```

---


