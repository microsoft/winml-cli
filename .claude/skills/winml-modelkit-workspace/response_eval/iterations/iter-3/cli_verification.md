# iter-3 — CLI command verification report

Static check only: parse `winml` commands from each response, verify subcommand exists and every flag appears in `winml <cmd> --help`. No bare positional model arg.

## Overall

- Commands extracted: **23**
- Passed static check: **23/23**

## eval-is-model-supported

Commands: 4 | Passed: 4/4

### `winml sys --list-ep`
- **OK**

### `winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512`
- **OK**

### `winml export   -m nvidia/segformer-b1-finetuned-ade-512-512 -o segformer.onnx`
- **OK**

### `winml analyze  -m segformer.onnx --ep ov --device npu`
- **OK**

## eval-llm-out-of-scope

Commands: 2 | Passed: 2/2

### `winml inspect -m microsoft/Phi-3-mini-4k-instruct`
- **OK**

### `winml sys --list-device --list-ep`
- **OK**

## eval-npu-vs-cpu-comparison

Commands: 4 | Passed: 4/4

### `winml inspect -m google/vit-base-patch16-224`
- **OK**

### `winml sys --list-ep`
- **OK**

### `winml config -m google/vit-base-patch16-224 --device npu --precision int8 --compile -o vit_npu.json`
- **OK**

### `winml build -c vit_npu.json -m google/vit-base-patch16-224 -o build/vit/ --compile`
- **OK**

## eval-optimize-failure-recovery

Commands: 4 | Passed: 4/4

### `winml optimize --help`
- **OK**

### `winml optimize --list-rewrites`
- **OK**

### `winml sys --list-ep`
- **OK**

### `winml inspect -m microsoft/swin-large-patch4-window7-224`
- **OK**

## eval-ryzen-ai-quick-benchmark

Commands: 3 | Passed: 3/3

### `winml sys --list-ep`
- **OK**

### `winml sys --list-device --list-ep`
- **OK**

### `winml perf -m facebook/convnext-tiny-224 --device npu`
- **OK**

## eval-snapdragon-resnet-build

Commands: 6 | Passed: 6/6

### `winml sys --list-device --list-ep`
- **OK**

### `winml inspect -m microsoft/resnet-50`
- **OK**

### `winml config -m microsoft/resnet-50 --device npu --precision int8 --compile -o resnet50-qnn.json`
- **OK**

### `winml build -c resnet50-qnn.json -m microsoft/resnet-50 -o resnet50-qnn-build/ --compile`
- **OK**

### `winml perf -m microsoft/resnet-50 -c resnet50-qnn.json --device npu --iterations 500 --warmup 50 --monitor`
- **OK**

### `winml perf -m microsoft/resnet-50 --device cpu --iterations 500 --warmup 50 -o resnet50-cpu-perf.json`
- **OK**
