#import "@preview/fletcher:0.5.7": diagram, node, edge

#set page(width: auto, height: auto, margin: 12pt, fill: rgb("#EEEEEE"))
#set text(fill: rgb("#1a2332"))

#text(size: 0.75em)[
  #diagram(
    spacing: (22pt, 25pt),
    node-stroke: 0.8pt + luma(120),
    node-fill: white,
    node-corner-radius: 4pt,
    edge-stroke: 0.8pt + luma(100),
    mark-scale: 60%,

    // Artifacts (circles)
    node((0, 0), text(fill: rgb("#1a2332"))[Source\ Model], fill: rgb("#f5e6d3"), stroke: 1pt + rgb("#c9a87c"), shape: circle, name: <hf>),
    node((7, 0), text(fill: rgb("#1a2332"))[Portable\ ONNX], fill: rgb("#f5e6d3"), stroke: 1pt + rgb("#c9a87c"), shape: circle, name: <onnx>),

    // Export
    node((1.5, 0), text(fill: rgb("#1a2332"))[Load &\ Export], fill: rgb("#c8e6c9"), stroke: 1pt + rgb("#66bb6a"), name: <export>),

    // Analyzer box with Lint and Conf
    node((3, 0), text(fill: rgb("#1a2332"))[Lint], fill: rgb("#e3f2fd"), name: <lint>),
    node((4.2, 0), text(fill: rgb("#1a2332"))[Conf], fill: rgb("#f3e5f5"), name: <conf>),
    node(
      enclose: (<lint>, <conf>),
      fill: rgb("#c8e6c9"),
      stroke: 1pt + luma(180),
      inset: 12pt,
      snap: -1,
      name: <analyzer-box>,
    ),
    node((3.6, 0.7), [Analyzer], stroke: none, fill: none),

    // Optimizer (above analyzer)
    node((3.6, -2.3), text(fill: rgb("#1a2332"))[Optimize], fill: rgb("#c8e6c9"), stroke: 1pt + rgb("#66bb6a"), name: <optimizer>),

    // Quantize (below main line - optional path)
    node((5.5, 1), text(fill: rgb("#1a2332"))[Quantize], fill: rgb("#c8e6c9"), stroke: 1pt + rgb("#66bb6a"), name: <quantize>),

    // Evaluate, Profile, and Deploy
    node((9.5, 0), text(fill: rgb("#1a2332"))[Evaluate], fill: rgb("#c8e6c9"), stroke: 1pt + rgb("#66bb6a"), name: <eval>),
    node((9.5, 1), text(fill: rgb("#1a2332"))[Profile], fill: rgb("#fff9c4"), stroke: 1pt + rgb("#fbc02d"), name: <profiler>),
    node((11, 0), text(fill: rgb("#1a2332"))[Deploy], fill: rgb("#e1bee7"), stroke: (dash: "dashed", paint: rgb("#7b1fa2")), shape: circle, name: <ship>),

    // Main flow
    edge(<hf>, <export>, "-|>"),
    edge(<export>, <lint>, "-|>"),
    edge(<lint>, <conf>, "-|>"),

    // Optimizer feedback loop
    edge(<conf>, <optimizer>, "-|>", bend: -25deg, label: [_config_], label-side: right),
    edge(<optimizer>, <lint>, "-|>", bend: -25deg, label: [_optimized_], label-side: right),

    // Main path (fp16 - skip quantize)
    edge(<analyzer-box>, <onnx>, "-|>", label: [_fp16_], label-side: left),

    // Optional quantize path (n-bit)
    edge(<analyzer-box>, <quantize>, "-|>", bend: -25deg, label: [_n-bit_], label-side: left),
    edge(<quantize>, <onnx>, "-|>", bend: -25deg, label: [_qdq_], label-side: left),

    // Continue to eval and output
    edge(<onnx>, <eval>, "-|>", label: [_compile_]),
    edge(<eval>, <profiler>, "--|>", stroke: (dash: "dashed")),
    edge(<eval>, <ship>, "-|>"),
  )
]
