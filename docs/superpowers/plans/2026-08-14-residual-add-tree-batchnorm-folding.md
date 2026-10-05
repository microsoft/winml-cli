# Residual Add-Tree BatchNormalization Folding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing opt-in `--enable-conv-add-batch-normalization-folding` capability so it safely removes inference BatchNormalization after arbitrary-depth binary residual Add trees with at least two Conv leaves.

**Architecture:** Keep the generated RewritePipe implementation for the fixed one-Conv topology and register its canonical capability object with AlgebraicRewritePipe as a second owner. AlgebraicRewritePipe performs bounded recursive candidate discovery, validates and precomputes every replacement array without mutation, then atomically applies copy-on-write Conv and constant replacements before existing channel-affine folding.

**Tech Stack:** Python 3.11, ONNX 1.18, NumPy 2.2, ONNX Runtime CPU, pytest 8.4, Ruff 0.15, mypy 1.18, Click test runner, Git.

## Global Constraints

- Keep the existing public CLI flag and default-off behavior.
- Preserve the fixed RewritePipe behavior.
- Fold inference BatchNormalization after a proper binary Add tree containing at least two eligible Conv leaves and zero or more immutable constant leaves.
- Support arbitrary tree shape and Add operand order up to a conservative depth limit.
- Preserve graph semantics, public outputs, and unaffected users of shared initializers through copy-on-write replacement.
- Use the standard ONNX domain with exactly one default-domain opset import at version 7 or later.
- Traversal depth may not exceed 64 Add edges.
- No original initializer is modified in place.
- No model mutation occurs during candidate discovery or constant computation.
- Validation uses explicit guards and narrow tensor-conversion failures.
- No new CLI flag, config key, capability alias, or default enablement.
- No replacement or removal of the existing fixed RewritePipe patterns.
- No finite expansion of recursive Add trees into generated pattern skeletons.
- No new optimization pipe.
- No support for Sub, Mul, Div, Concat, activation, projection, normalization, or other operator leaves.
- No dynamic BatchNormalization parameters, training BatchNormalization, or custom-domain operators.
- No folding across shared or publicly observed computed intermediates.
- No architecture, model-name, tensor-name, layer-name, or execution-provider special cases.
- No NPU performance requirement.
- No unrelated algebraic, pattern, or pipeline refactor.
- All tests use pytest with code-generated graphs and expected results; do not add a standalone validation script.
- Use `uv run` for Python commands, keep temporary validation files under `temp\`, and use Windows path separators in commands.
- Source imports remain relative. Tests import public symbols from package-level exports and may import private `_`-prefixed helpers only for focused implementation-detail tests.
- Every Python revision is followed by the affected pytest scope and `uv run ruff check --fix`.
- Every task ends in its own non-amended commit with:

```text
Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>
```

---

## File Structure Map

| File | Responsibility | Planned change |
| --- | --- | --- |
| `src/winml/modelkit/optim/pipes/algebraic.py` | Existing large-file home for conservative recursive algebraic rewrites, graph indexing, constant loading, naming, cleanup, and pass ordering | Register the shared capability; add residual candidate/replacement types, traversal, validation, precomputation, atomic apply, and invoke it before `_fold_channel_affine` |
| `src/winml/modelkit/optim/pipes/rewrite_rules.py` | Generates the canonical fixed-pattern `BoolCapability` from JSON | Read-only; AlgebraicRewritePipe imports `REWRITE_CAPABILITIES["conv-add-batch-normalization-folding"]` rather than creating another definition |
| `src/winml/modelkit/optim/pipes/__init__.py` | Defines `PIPES = [ORTGraphPipe, AlgebraicRewritePipe, RewritePipe, ORTFusionPipe, SurgeryPipe]` | Read-only; tests lock the existing ordering |
| `src/winml/modelkit/optim/analysis.py` | Shared-owner full-pipeline capability probing added by PR #1301 | Read-only unless a test exposes a real defect; exercise the existing shared-owner path |
| `tests/unit/optim/pipes/test_pipe_algebraic.py` | Existing large generated-graph suite for AlgebraicRewritePipe | Add builders and all registration, positive, algebra, copy-on-write, atomicity, idempotence, and fail-closed tests |
| `tests/unit/optim/test_analysis.py` | Optimizer applicability and shared-owner probing | Add a generated residual model and one combined-owner finding test |
| `tests/unit/pattern/test_conv_batchnorm_patterns.py` | Existing fixed RewritePipe Conv/Add/BatchNormalization coverage | Keep unchanged; run it at coexistence gates |
| `docs/superpowers/evidence/2026-08-14-residual-add-tree-batchnorm-folding.md` | Persisted acceptance evidence for the approved real model | Create from a temporary pytest validation using measured counts, signatures, and errors |
| `CHANGELOG.md` | Release-level snapshots rather than per-commit notes | Do not modify: the public flag already exists and no new user-facing command/config surface is introduced |

The two large Python files remain large by repository convention. This change adds cohesive sections beside their related graph-rewrite and generated-graph coverage instead of introducing a parallel pipe or test module.
The recursive topology belongs in AlgebraicRewritePipe because eligibility
depends on bounded whole-tree traversal, graph ownership, broadcast reasoning,
and atomic multi-node mutation already provided there. Finite generated-pattern
expansion cannot cover arbitrary depth, while a new pipe would duplicate the
same index, capability, cleanup, and ordering machinery.

---

### Task 1: Share the Existing Capability and Lock Pass Ordering

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:17-24,90-245`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:17-43,1708-1760`
- Read-only verification: `src/winml/modelkit/optim/pipes/rewrite_rules.py:265-316`
- Read-only verification: `src/winml/modelkit/optim/pipes/__init__.py:26-42`

**Interfaces:**
- Consumes: `REWRITE_CAPABILITIES: dict[str, Any]`, `caps_dict(*capabilities: CapabilityDef) -> dict[str, CapabilityDef]`.
- Produces: `AlgebraicRewritePipeConfig.conv_add_batch_normalization_folding: bool`; no-op scaffold `_fold_residual_add_batch_norms(model: onnx.ModelProto, allocator: _NameAllocator) -> None`; the canonical capability object registered in `ALGEBRAIC_CAPABILITIES`.

- [ ] **Step 1: Add failing registration, identity, CLI, and ordering tests**

Add the import:

```python
from winml.modelkit.optim.pipes.rewrite_rules import REWRITE_CAPABILITIES
```

Extend `TestAlgebraicRegistration` with:

```python
    def test_conv_add_batch_norm_capability_is_shared_with_rewrite_pipe(self) -> None:
        name = "conv-add-batch-normalization-folding"
        capabilities = get_all_capabilities()

        assert AlgebraicRewritePipe.capabilities[name] is REWRITE_CAPABILITIES[name]
        assert capabilities[name] is REWRITE_CAPABILITIES[name]
        assert capabilities[name].default is False
        assert capabilities[name].cli_flags() == (
            "--enable-conv-add-batch-normalization-folding",
            "--disable-conv-add-batch-normalization-folding",
        )

        config = AlgebraicRewritePipe.build_config(
            conv_add_batch_normalization_folding=True,
        )
        assert config.conv_add_batch_normalization_folding is True
        assert AlgebraicRewritePipe.should_process(config)

    def test_cli_lists_shared_conv_add_batch_norm_flag_once(self) -> None:
        result = CliRunner().invoke(optimize, ["--list-capabilities"])

        assert result.exit_code == 0
        assert result.output.count("--enable-conv-add-batch-normalization-folding") == 1

    def test_residual_owner_stays_between_ort_graph_and_rewrite_pipes(self) -> None:
        assert [pipe.name for pipe in PIPES] == [
            "ort_graph",
            "algebraic_rewrite",
            "rewrite",
            "ort_fusion",
            "surgery",
        ]

    def test_residual_batch_norm_runs_before_channel_affine(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            algebraic_pipe,
            "_fold_residual_add_batch_norms",
            lambda model, allocator: calls.append("residual_batch_norm"),
        )
        monkeypatch.setattr(
            algebraic_pipe,
            "_fold_channel_affine",
            lambda model, allocator: calls.append("channel_affine"),
        )
        model = _model([], [], [], [])

        AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
                conv_channel_affine_folding=True,
            ),
        )

        assert calls == ["residual_batch_norm", "channel_affine"]
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestAlgebraicRegistration -v
```

Expected: FAIL because AlgebraicRewritePipe does not own the capability, its config has no `conv_add_batch_normalization_folding` field, and `_fold_residual_add_batch_norms` is undefined.

- [ ] **Step 3: Register the canonical object and add the ordered no-op scaffold**

Change the imports and capability collection:

```python
from ..capabilities import algebraic, misc
from .base import BasePipe, PipeConfig, caps_dict
from .rewrite_rules import REWRITE_CAPABILITIES


CONV_ADD_BATCH_NORMALIZATION_FOLDING = REWRITE_CAPABILITIES[
    "conv-add-batch-normalization-folding"
]

ALGEBRAIC_CAPABILITIES: dict[str, Any] = caps_dict(
    algebraic.STATIC_SPLIT_TO_SLICE,
    misc.GATHER_SLICE_TO_SPLIT_FUSION,
    CONV_ADD_BATCH_NORMALIZATION_FOLDING,
    algebraic.CONV_CHANNEL_AFFINE_FOLDING,
    algebraic.EXP_POSITIVE_SCALE_FOLDING,
)
```

Extend the config:

```python
@dataclass
class AlgebraicRewritePipeConfig(PipeConfig):
    """Configuration for exact algebraic rewrites."""

    static_split_to_slice: bool = False
    sibling_slice_to_split: bool = False
    conv_add_batch_normalization_folding: bool = False
    conv_channel_affine_folding: bool = False
    exp_positive_scale_folding: bool = False
```

Add the scaffold immediately before `_fold_channel_affine`:

```python
def _fold_residual_add_batch_norms(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
) -> None:
    """Fold eligible inference BatchNormalization nodes over residual Add trees."""
```

Wire configuration and ordering:

```python
        return AlgebraicRewritePipeConfig(
            static_split_to_slice=kwargs.get("static_split_to_slice", False),
            sibling_slice_to_split=kwargs.get("gather_slice_to_split_fusion", False),
            conv_add_batch_normalization_folding=kwargs.get(
                "conv_add_batch_normalization_folding",
                False,
            ),
            conv_channel_affine_folding=kwargs.get("conv_channel_affine_folding", False),
            exp_positive_scale_folding=kwargs.get("exp_positive_scale_folding", False),
        )
```

```python
        return (
            config.static_split_to_slice
            or config.sibling_slice_to_split
            or config.conv_add_batch_normalization_folding
            or config.conv_channel_affine_folding
            or config.exp_positive_scale_folding
        )
```

```python
        if (
            config.conv_add_batch_normalization_folding
            and standard_opset is not None
            and standard_opset >= 7
        ):
            _fold_residual_add_batch_norms(result, allocator)
        if (
            config.conv_channel_affine_folding
            and standard_opset is not None
            and standard_opset >= 7
        ):
            _fold_channel_affine(result, allocator)
```

- [ ] **Step 4: Run targeted tests and formatting**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestAlgebraicRegistration tests\unit\pattern\test_conv_batchnorm_patterns.py::test_rewrite_capability_is_registered -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: all selected tests PASS; Ruff reports no remaining errors and formats no unrelated files.

- [ ] **Step 5: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "feat: register residual batch norm folding" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Confirm object identity with the generated RewritePipe capability, exactly one CLI listing, unchanged top-level `PIPES` order, and residual-before-channel-affine call order.

---

### Task 2: Index and Discover Complete Residual Add Trees

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:32-88, before TestConvChannelAffineFolding at current line 727`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:31-184,240-279, before _copy_conv_parameters at current line 1495`

**Interfaces:**
- Consumes: `_GraphIndex.build(model: onnx.ModelProto) -> _GraphIndex`, `_constant_array(index: _GraphIndex, name: str) -> np.ndarray | None`, `_static_shape(index: _GraphIndex, name: str) -> tuple[int, ...] | None`.
- Produces: `MAX_RESIDUAL_ADD_DEPTH = 64`; `_ResidualConstantInput`; `_ResidualAddTree`; `_residual_add_tree(index: _GraphIndex, batch_norm: onnx.NodeProto) -> _ResidualAddTree | None`.

- [ ] **Step 1: Add a reusable generated residual model builder**

Add below `_assert_byte_identical`:

```python
def _residual_batch_norm_model(
    *,
    nested: bool = False,
    reverse_adds: frozenset[int] = frozenset(),
    with_bias: tuple[bool, ...] = (),
    constant_values: tuple[np.ndarray, ...] = (),
    dtype: np.dtype = np.dtype(np.float32),
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    rng = np.random.default_rng(71)
    channels = 2
    shape = [1, channels, 2, 2]
    conv_count = 3 if nested else 2
    tensor_type = onnx.helper.np_dtype_to_tensor_dtype(dtype)

    def info(name: str) -> onnx.ValueInfoProto:
        return onnx.helper.make_tensor_value_info(name, tensor_type, shape)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    inputs: list[onnx.ValueInfoProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    feeds: dict[str, np.ndarray] = {}
    conv_outputs: list[str] = []

    for index in range(conv_count):
        input_name = f"x_{index}"
        weight_name = f"weight_{index}"
        output_name = f"conv_{index}"
        conv_inputs = [input_name, weight_name]
        initializers.append(
            _tensor(
                weight_name,
                rng.normal(size=(channels, channels, 1, 1)).astype(dtype),
            )
        )
        if index < len(with_bias) and with_bias[index]:
            bias_name = f"bias_{index}"
            conv_inputs.append(bias_name)
            initializers.append(
                _tensor(bias_name, rng.normal(size=channels).astype(dtype))
            )
        nodes.append(
            onnx.helper.make_node(
                "Conv",
                conv_inputs,
                [output_name],
                name=f"conv_node_{index}",
            )
        )
        inputs.append(info(input_name))
        value_info.append(info(output_name))
        feeds[input_name] = rng.normal(size=shape).astype(dtype)
        conv_outputs.append(output_name)

    add_index = 0

    def append_add(left: str, right: str) -> str:
        nonlocal add_index
        inputs_for_add = [left, right]
        if add_index in reverse_adds:
            inputs_for_add.reverse()
        output_name = f"sum_{add_index}"
        nodes.append(
            onnx.helper.make_node(
                "Add",
                inputs_for_add,
                [output_name],
                name=f"add_node_{add_index}",
            )
        )
        value_info.append(info(output_name))
        add_index += 1
        return output_name

    root = append_add(conv_outputs[0], conv_outputs[1])
    if nested:
        root = append_add(root, conv_outputs[2])
    for constant_index, values in enumerate(constant_values):
        constant_name = f"constant_{constant_index}"
        initializers.append(_tensor(constant_name, np.asarray(values, dtype=dtype)))
        root = append_add(root, constant_name)

    scale = rng.uniform(0.5, 1.5, size=channels).astype(dtype)
    beta = rng.normal(size=channels).astype(dtype)
    mean = rng.normal(size=channels).astype(dtype)
    variance = rng.uniform(0.5, 1.5, size=channels).astype(dtype)
    initializers.extend(
        [
            _tensor("bn_scale", scale),
            _tensor("bn_beta", beta),
            _tensor("bn_mean", mean),
            _tensor("bn_variance", variance),
        ]
    )
    nodes.append(
        onnx.helper.make_node(
            "BatchNormalization",
            [root, "bn_scale", "bn_beta", "bn_mean", "bn_variance"],
            ["y"],
            name="batch_norm",
            epsilon=0.01,
        )
    )
    return (
        _model(
            nodes,
            inputs,
            [info("y")],
            initializers,
            value_info=value_info,
        ),
        feeds,
    )
```

- [ ] **Step 2: Add failing topology-discovery tests**

Add a new class before `TestConvChannelAffineFolding`:

```python
class TestResidualAddTreeBatchNormFolding:
    """Test generic residual Add-tree BatchNormalization folding."""

    def test_discovers_two_leaf_and_nested_operand_permutations(self) -> None:
        for nested, reverse_adds, expected_adds, expected_convs in (
            (False, frozenset(), 1, 2),
            (False, frozenset({0}), 1, 2),
            (True, frozenset(), 2, 3),
            (True, frozenset({0, 1}), 2, 3),
        ):
            model, _ = _residual_batch_norm_model(
                nested=nested,
                reverse_adds=reverse_adds,
            )
            index = algebraic_pipe._GraphIndex.build(model)
            batch_norm = model.graph.node[-1]

            tree = algebraic_pipe._residual_add_tree(index, batch_norm)

            assert tree is not None
            assert tree.batch_norm is batch_norm
            assert len(tree.adds) == expected_adds
            assert len(tree.convs) == expected_convs
            assert tree.root_add.output == [batch_norm.input[0]]

    def test_topology_discovery_accepts_immutable_constant_leaves(self) -> None:
        model, _ = _residual_batch_norm_model(
            constant_values=(np.asarray(0.25, dtype=np.float32),),
        )
        index = algebraic_pipe._GraphIndex.build(model)

        tree = algebraic_pipe._residual_add_tree(index, model.graph.node[-1])

        assert tree is not None
        assert len(tree.convs) == 2
        assert len(tree.constants) == 1
        assert tree.constants[0].name == "constant_0"

    def test_topology_discovery_rejects_one_conv_and_excessive_depth(self) -> None:
        one_conv, _ = _residual_batch_norm_model()
        one_conv.graph.node[1].op_type = "Identity"
        one_conv.graph.node[1].input[:] = ["x_1"]
        one_conv_index = algebraic_pipe._GraphIndex.build(one_conv)
        assert algebraic_pipe._residual_add_tree(
            one_conv_index,
            one_conv.graph.node[-1],
        ) is None

        deep, _ = _residual_batch_norm_model()
        batch_norm = deep.graph.node[-1]
        previous = batch_norm.input[0]
        for depth in range(65):
            output = f"deep_sum_{depth}"
            deep.graph.node.insert(
                len(deep.graph.node) - 1,
                onnx.helper.make_node("Add", [previous, "constant_depth"], [output]),
            )
            deep.graph.value_info.append(_info(output, [1, 2, 2, 2]))
            previous = output
        deep.graph.initializer.append(
            _tensor("constant_depth", np.asarray(0.0, dtype=np.float32))
        )
        batch_norm.input[0] = previous
        deep_index = algebraic_pipe._GraphIndex.build(deep)
        assert algebraic_pipe._residual_add_tree(deep_index, batch_norm) is None
```

- [ ] **Step 3: Run the new tests and confirm the private interface is absent**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "discovers or topology" -v
```

Expected: FAIL with `AttributeError` for `_residual_add_tree`.

- [ ] **Step 4: Add graph order metadata and immutable topology types**

Add the limit:

```python
MAX_RESIDUAL_ADD_DEPTH = 64
```

Extend `_GraphIndex` and its constructor:

```python
    node_indexes: dict[int, int]
```

```python
            node_indexes=node_indexes,
```

Add beside the existing candidate dataclasses:

```python
@dataclass(frozen=True)
class _ResidualConstantInput:
    """One immutable constant connected to a specific Add input slot."""

    add: onnx.NodeProto
    input_index: int
    name: str
    values: np.ndarray


@dataclass(frozen=True)
class _ResidualAddTree:
    """A complete private Add tree rooted at one inference BatchNormalization."""

    batch_norm: onnx.NodeProto
    root_add: onnx.NodeProto
    adds: tuple[onnx.NodeProto, ...]
    convs: tuple[onnx.NodeProto, ...]
    constants: tuple[_ResidualConstantInput, ...]
    output_shape: tuple[int, ...]
```

- [ ] **Step 5: Implement bounded recursive discovery without mutation**

Insert before `_copy_conv_parameters`:

```python
def _collect_residual_add_tree(
    index: _GraphIndex,
    tensor_name: str,
    output_shape: tuple[int, ...],
    adds: list[onnx.NodeProto],
    convs: list[onnx.NodeProto],
    constants: list[_ResidualConstantInput],
    seen_tensors: set[str],
    depth: int,
) -> bool:
    if depth > MAX_RESIDUAL_ADD_DEPTH or tensor_name in seen_tensors:
        return False
    seen_tensors.add(tensor_name)
    producer = index.producers.get(tensor_name)
    if producer is not None and _is_standard_onnx_node(producer) and producer.op_type == "Conv":
        if (
            len(producer.input) not in (2, 3)
            or len(producer.output) != 1
            or producer.output[0] != tensor_name
            or _static_shape(index, tensor_name) != output_shape
            or tensor_name in index.graph_outputs
            or len(index.consumers.get(tensor_name, [])) != 1
        ):
            return False
        convs.append(producer)
        return True
    if producer is not None and _is_standard_onnx_node(producer) and producer.op_type == "Add":
        if (
            len(producer.input) != 2
            or any(not name for name in producer.input)
            or len(producer.output) != 1
            or producer.output[0] != tensor_name
            or _static_shape(index, tensor_name) != output_shape
            or tensor_name in index.graph_outputs
            or len(index.consumers.get(tensor_name, [])) != 1
        ):
            return False
        adds.append(producer)
        for input_index, input_name in enumerate(producer.input):
            input_producer = index.producers.get(input_name)
            if input_producer is not None and input_producer.op_type in {"Add", "Conv"}:
                if not _collect_residual_add_tree(
                    index,
                    input_name,
                    output_shape,
                    adds,
                    convs,
                    constants,
                    seen_tensors,
                    depth + 1,
                ):
                    return False
                continue
            values = _constant_array(index, input_name)
            if values is None:
                return False
            constants.append(
                _ResidualConstantInput(
                    add=producer,
                    input_index=input_index,
                    name=input_name,
                    values=values,
                )
            )
        return True
    return False


def _residual_add_tree(
    index: _GraphIndex,
    batch_norm: onnx.NodeProto,
) -> _ResidualAddTree | None:
    if (
        not _is_standard_onnx_node(batch_norm)
        or batch_norm.op_type != "BatchNormalization"
        or len(batch_norm.input) != 5
        or any(not name for name in batch_norm.input)
        or len(batch_norm.output) != 1
        or not batch_norm.output[0]
    ):
        return None
    output_shape = _static_shape(index, batch_norm.input[0])
    if output_shape is None or len(output_shape) < 2 or output_shape[1] <= 0:
        return None
    adds: list[onnx.NodeProto] = []
    convs: list[onnx.NodeProto] = []
    constants: list[_ResidualConstantInput] = []
    if not _collect_residual_add_tree(
        index,
        batch_norm.input[0],
        output_shape,
        adds,
        convs,
        constants,
        set(),
        0,
    ):
        return None
    if len(convs) < 2 or len({id(conv) for conv in convs}) != len(convs):
        return None
    root_add = index.producers.get(batch_norm.input[0])
    if root_add is None or root_add.op_type != "Add":
        return None
    return _ResidualAddTree(
        batch_norm=batch_norm,
        root_add=root_add,
        adds=tuple(adds),
        convs=tuple(convs),
        constants=tuple(constants),
        output_shape=output_shape,
    )
```

The consumer map already includes nested-graph captures, so `len(consumers) != 1` rejects captured or externally shared computed intermediates.

- [ ] **Step 6: Run the topology scope and Ruff**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "discovers or topology" -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: all topology tests PASS.

- [ ] **Step 7: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "feat: discover residual Add trees" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Confirm discovery is read-only, recognizes every Add operand order, accepts immutable constant leaves, rejects non-tree aliases, counts distinct Conv nodes, and enforces depth 64.

---

### Task 3: Validate BatchNormalization and Conv Leaves and Precompute Pure Replacements

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:TestResidualAddTreeBatchNormFolding`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:_GraphIndex, candidate dataclasses, residual helper section`

**Interfaces:**
- Consumes: `_ResidualAddTree`, `_GraphIndex.node_indexes`, `_constant_array`, `_initializer_array`.
- Produces: `_ResidualConvReplacement`, `_ResidualConstantReplacement`, `_ResidualBatchNormRewrite`; `_precompute_residual_batch_norm(index: _GraphIndex, tree: _ResidualAddTree) -> _ResidualBatchNormRewrite | None`.

- [ ] **Step 1: Add failing pure-algebra tests**

Add:

```python
    def test_precomputation_scales_every_conv_and_assigns_delta_once(self) -> None:
        model, _ = _residual_batch_norm_model(
            nested=True,
            reverse_adds=frozenset({0, 1}),
            with_bias=(True, False, True),
        )
        index = algebraic_pipe._GraphIndex.build(model)
        tree = algebraic_pipe._residual_add_tree(index, model.graph.node[-1])
        assert tree is not None
        original = model.SerializeToString()

        rewrite = algebraic_pipe._precompute_residual_batch_norm(index, tree)

        assert rewrite is not None
        assert model.SerializeToString() == original
        scale = onnx.numpy_helper.to_array(index.initializers["bn_scale"])
        beta = onnx.numpy_helper.to_array(index.initializers["bn_beta"])
        mean = onnx.numpy_helper.to_array(index.initializers["bn_mean"])
        variance = onnx.numpy_helper.to_array(index.initializers["bn_variance"])
        alpha = scale / np.sqrt(variance + 0.01)
        delta = beta - alpha * mean
        replacements = {replacement.conv.name: replacement for replacement in rewrite.convs}

        for conv in tree.convs:
            replacement = replacements[conv.name]
            weights = onnx.numpy_helper.to_array(index.initializers[conv.input[1]])
            np.testing.assert_allclose(
                replacement.weights,
                weights * alpha.reshape(2, 1, 1, 1),
                rtol=1e-6,
                atol=1e-6,
            )

        anchor = min(tree.convs, key=lambda node: index.node_indexes[id(node)])
        anchor_replacement = replacements[anchor.name]
        anchor_bias = (
            onnx.numpy_helper.to_array(index.initializers[anchor.input[2]])
            if len(anchor.input) == 3
            else np.zeros(2, dtype=np.float32)
        )
        np.testing.assert_allclose(
            anchor_replacement.bias,
            anchor_bias * alpha + delta,
            rtol=1e-6,
            atol=1e-6,
        )
        for conv in tree.convs:
            if conv is anchor:
                continue
            replacement = replacements[conv.name]
            bias = (
                onnx.numpy_helper.to_array(index.initializers[conv.input[2]])
                if len(conv.input) == 3
                else np.zeros(2, dtype=np.float32)
            )
            np.testing.assert_allclose(
                replacement.bias,
                bias * alpha,
                rtol=1e-6,
                atol=1e-6,
            )

    @pytest.mark.parametrize(
        ("attribute_name", "attribute_value"),
        [
            ("training_mode", 1),
            ("epsilon", -0.01),
        ],
    )
    def test_precomputation_rejects_non_inference_batch_norm(
        self,
        attribute_name: str,
        attribute_value: int | float,
    ) -> None:
        model, _ = _residual_batch_norm_model()
        batch_norm = model.graph.node[-1]
        batch_norm.attribute.append(
            onnx.helper.make_attribute(attribute_name, attribute_value)
        )
        index = algebraic_pipe._GraphIndex.build(model)
        tree = algebraic_pipe._residual_add_tree(index, batch_norm)
        assert tree is not None

        assert algebraic_pipe._precompute_residual_batch_norm(index, tree) is None
```

- [ ] **Step 2: Run and confirm the precompute interface is absent**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "precomputation" -v
```

Expected: FAIL with `AttributeError` for `_precompute_residual_batch_norm`.

- [ ] **Step 3: Add dtype indexing and immutable replacement types**

Add `_value_info_dtype` beside `_value_info_shape`:

```python
def _value_info_dtype(value_info: onnx.ValueInfoProto) -> np.dtype[Any] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.UNDEFINED:
        return None
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))
    except TypeError:
        return None
```

Extend `_GraphIndex`:

```python
    dtypes: dict[str, np.dtype[Any]]
```

Build it with:

```python
        dtypes: dict[str, np.dtype[Any]] = {}
        for value_info in (*graph.input, *graph.value_info, *graph.output):
            dtype = _value_info_dtype(value_info)
            if dtype is not None:
                dtypes[value_info.name] = dtype
        for name, initializer in initializers.items():
            try:
                dtypes.setdefault(
                    name,
                    np.dtype(onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)),
                )
            except TypeError:
                continue
```

Pass `dtypes=dtypes` into the `_GraphIndex` constructor.

Add replacement types:

```python
@dataclass(frozen=True)
class _ResidualConvReplacement:
    conv: onnx.NodeProto
    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class _ResidualConstantReplacement:
    add: onnx.NodeProto
    input_index: int
    values: np.ndarray


@dataclass(frozen=True)
class _ResidualBatchNormRewrite:
    batch_norm: onnx.NodeProto
    root_add: onnx.NodeProto
    convs: tuple[_ResidualConvReplacement, ...]
    constants: tuple[_ResidualConstantReplacement, ...]
```

- [ ] **Step 4: Implement finite checked casting and pure Conv/BN precomputation**

Add:

```python
def _finite_cast(
    values: np.ndarray,
    dtype: np.dtype[Any],
) -> np.ndarray | None:
    with np.errstate(over="ignore", invalid="ignore"):
        cast_values = np.asarray(values, dtype=dtype)
    if cast_values.shape != values.shape or not np.isfinite(cast_values).all():
        return None
    return cast_values


def _precompute_residual_batch_norm(
    index: _GraphIndex,
    tree: _ResidualAddTree,
) -> _ResidualBatchNormRewrite | None:
    batch_norm = tree.batch_norm
    try:
        training_mode = int(_attribute(batch_norm, "training_mode", 0))
        epsilon = float(_attribute(batch_norm, "epsilon", 1e-5))
    except (TypeError, ValueError):
        return None
    if training_mode != 0 or not math.isfinite(epsilon) or epsilon < 0:
        return None

    dtype = index.dtypes.get(batch_norm.input[0])
    channels = tree.output_shape[1]
    if dtype is None or not np.issubdtype(dtype, np.floating):
        return None
    parameter_values = [_constant_array(index, name) for name in batch_norm.input[1:]]
    if any(values is None for values in parameter_values):
        return None
    parameters = cast("list[np.ndarray]", parameter_values)
    if any(
        values.ndim != 1
        or len(values) != channels
        or values.dtype != dtype
        or not np.isfinite(values).all()
        for values in parameters
    ):
        return None
    scale, beta, mean, variance = parameters
    calculation_dtype = np.dtype(np.result_type(dtype, np.float32))
    scale_calc = scale.astype(calculation_dtype)
    beta_calc = beta.astype(calculation_dtype)
    mean_calc = mean.astype(calculation_dtype)
    variance_calc = variance.astype(calculation_dtype)
    denominator = variance_calc + epsilon
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        alpha = scale_calc / np.sqrt(denominator)
        delta = beta_calc - alpha * mean_calc
    if not np.isfinite(alpha).all() or not np.isfinite(delta).all():
        return None

    anchor = min(tree.convs, key=lambda node: index.node_indexes[id(node)])
    conv_replacements: list[_ResidualConvReplacement] = []
    for conv in tree.convs:
        weights = _initializer_array(index, conv.input[1]) if len(conv.input) >= 2 else None
        if (
            weights is None
            or weights.dtype != dtype
            or weights.ndim < 1
            or weights.shape[0] != channels
            or not np.isfinite(weights).all()
        ):
            return None
        if len(conv.input) == 3 and conv.input[2]:
            bias = _initializer_array(index, conv.input[2])
            if (
                bias is None
                or bias.dtype != dtype
                or bias.shape != (channels,)
                or not np.isfinite(bias).all()
            ):
                return None
        else:
            bias = np.zeros(channels, dtype=dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_weights = weights.astype(calculation_dtype) * alpha.reshape(
                (channels,) + (1,) * (weights.ndim - 1)
            )
            scaled_bias = bias.astype(calculation_dtype) * alpha
            if conv is anchor:
                scaled_bias = scaled_bias + delta
        replacement_weights = _finite_cast(scaled_weights, dtype)
        replacement_bias = _finite_cast(scaled_bias, dtype)
        if replacement_weights is None or replacement_bias is None:
            return None
        conv_replacements.append(
            _ResidualConvReplacement(
                conv=conv,
                weights=replacement_weights,
                bias=replacement_bias,
            )
        )

    if tree.constants:
        return None
    return _ResidualBatchNormRewrite(
        batch_norm=batch_norm,
        root_add=tree.root_add,
        convs=tuple(conv_replacements),
        constants=(),
    )
```

Rejecting constants here is intentional and temporary: the task delivers reviewable BN/Conv algebra first; Task 5 begins with failing constant tests and replaces this branch.

- [ ] **Step 5: Run focused tests and quality checks**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "precomputation" -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: pure precomputation tests PASS and the source model remains byte-identical.

- [ ] **Step 6: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "feat: precompute residual batch norm folds" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Recalculate `alpha`, `delta`, every Conv weight, and every bias independently; verify the earliest serialized Conv receives `delta` exactly once and no protobuf mutation occurs.

---

### Task 4: Atomically Apply Conv-Only Rewrites and Preserve the BatchNormalization Output

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:TestResidualAddTreeBatchNormFolding`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:residual helper section and _fold_residual_add_batch_norms scaffold`

**Interfaces:**
- Consumes: `_precompute_residual_batch_norm`, `_NameAllocator.new(prefix: str) -> str`, `_new_initializer`, `_remove_nodes`.
- Produces: `_apply_residual_batch_norm_rewrite(model: onnx.ModelProto, allocator: _NameAllocator, rewrite: _ResidualBatchNormRewrite) -> None`; functional `_fold_residual_add_batch_norms`.

- [ ] **Step 1: Add failing public transformation and numerical tests**

Add:

```python
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize(
        "reverse_adds",
        [frozenset(), frozenset({0}), frozenset({0, 1})],
    )
    @pytest.mark.parametrize(
        "with_bias",
        [(False, False, False), (True, False, True), (True, True, True)],
    )
    def test_conv_only_trees_fold_and_preserve_output(
        self,
        nested: bool,
        reverse_adds: frozenset[int],
        with_bias: tuple[bool, ...],
    ) -> None:
        model, feeds = _residual_batch_norm_model(
            nested=nested,
            reverse_adds=reverse_adds,
            with_bias=with_bias,
        )
        expected = _run(model, feeds)
        output_signature = [
            output.SerializeToString() for output in model.graph.output
        ]
        batch_norm_input = model.graph.node[-1].input[0]
        batch_norm_parameters = set(model.graph.node[-1].input[1:])

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert not any(
            node.op_type == "BatchNormalization" for node in transformed.graph.node
        )
        assert transformed.graph.node[-1].op_type == "Add"
        assert transformed.graph.node[-1].output == ["y"]
        assert [
            output.SerializeToString() for output in transformed.graph.output
        ] == output_signature
        assert all(
            len(node.input) == 3
            for node in transformed.graph.node
            if node.op_type == "Conv"
        )
        assert batch_norm_parameters.isdisjoint(
            initializer.name for initializer in transformed.graph.initializer
        )
        assert batch_norm_input not in {
            value.name for value in transformed.graph.value_info
        }
        _assert_valid_with_inferred_shapes(transformed)
        np.testing.assert_allclose(
            _run(transformed, feeds),
            expected,
            rtol=3e-5,
            atol=3e-5,
        )

    def test_multiple_candidates_fold_and_second_run_is_idempotent(self) -> None:
        left, left_feeds = _residual_batch_norm_model()
        right, right_feeds = _residual_batch_norm_model(nested=True)
        for value in (*right.graph.input, *right.graph.output, *right.graph.value_info):
            value.name = f"right_{value.name}"
        for initializer in right.graph.initializer:
            initializer.name = f"right_{initializer.name}"
        for node in right.graph.node:
            node.name = f"right_{node.name}"
            node.input[:] = [f"right_{name}" for name in node.input]
            node.output[:] = [f"right_{name}" for name in node.output]
        left.graph.node.extend(right.graph.node)
        left.graph.input.extend(right.graph.input)
        left.graph.output.extend(right.graph.output)
        left.graph.initializer.extend(right.graph.initializer)
        left.graph.value_info.extend(right.graph.value_info)
        feeds = {**left_feeds, **{f"right_{k}": v for k, v in right_feeds.items()}}
        config = AlgebraicRewritePipeConfig(
            conv_add_batch_normalization_folding=True,
        )

        once = AlgebraicRewritePipe().process(left, config)
        twice = AlgebraicRewritePipe().process(once, config)

        assert not any(node.op_type == "BatchNormalization" for node in once.graph.node)
        assert twice.SerializeToString() == once.SerializeToString()
        np.testing.assert_allclose(
            _run(left, feeds),
            _run(once, feeds),
            rtol=3e-5,
            atol=3e-5,
        )
```

- [ ] **Step 2: Run and confirm BatchNormalization remains**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "conv_only or multiple_candidates" -v
```

Expected: FAIL because the scaffold does not mutate the graph and BatchNormalization remains.

- [ ] **Step 3: Implement mutation-only apply after all arrays and names exist**

Add:

```python
def _apply_residual_batch_norm_rewrite(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    rewrite: _ResidualBatchNormRewrite,
) -> None:
    conv_tensors = [
        (
            onnx.numpy_helper.from_array(
                replacement.weights,
                allocator.new("algebraic_residual_conv_weight"),
            ),
            onnx.numpy_helper.from_array(
                replacement.bias,
                allocator.new("algebraic_residual_conv_bias"),
            ),
        )
        for replacement in rewrite.convs
    ]
    for replacement, (weight_tensor, bias_tensor) in zip(
        rewrite.convs,
        conv_tensors,
        strict=True,
    ):
        model.graph.initializer.extend([weight_tensor, bias_tensor])
        replacement.conv.input[1] = weight_tensor.name
        if len(replacement.conv.input) == 3:
            replacement.conv.input[2] = bias_tensor.name
        else:
            replacement.conv.input.append(bias_tensor.name)
    rewrite.root_add.output[0] = rewrite.batch_norm.output[0]
    _remove_nodes(model, {id(rewrite.batch_norm)})
```

Replace the scaffold:

```python
def _fold_residual_add_batch_norms(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
) -> None:
    """Fold eligible inference BatchNormalization nodes over residual Add trees."""
    index = _GraphIndex.build(model)
    for batch_norm in list(model.graph.node):
        tree = _residual_add_tree(index, batch_norm)
        if tree is None:
            continue
        rewrite = _precompute_residual_batch_norm(index, tree)
        if rewrite is None:
            continue
        _apply_residual_batch_norm_rewrite(model, allocator, rewrite)
        index = _GraphIndex.build(model)
```

All conversion and shape work remains in precomputation. Once apply starts, it only appends already-built tensors, rewires known slots, preserves the BN output name on the root Add, and removes one known node.

- [ ] **Step 4: Run positive tests, fixed-pattern regression, and Ruff**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "conv_only or multiple_candidates" tests\unit\pattern\test_conv_batchnorm_patterns.py -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: all selected tests PASS for float32 Conv-only trees and the fixed pattern remains green.

- [ ] **Step 5: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "feat: apply residual batch norm folds atomically" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Inspect the diff for any mutation before `_precompute_residual_batch_norm` succeeds; confirm the root Add, not another node, receives the exact former BN output name.

---

### Task 5: Scale Constant Leaves and Prove Copy-on-Write Ownership

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:TestResidualAddTreeBatchNormFolding`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:14-18, residual precompute/apply helpers`

**Interfaces:**
- Consumes: `EXTERNAL_DATA_THRESHOLD` from `..onnx`, `_ResidualConstantInput`, `_ResidualConstantReplacement`.
- Produces: `_scaled_residual_constant(values: np.ndarray, alpha: np.ndarray, output_shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray | None`; apply support for constant input-slot rewiring.

- [ ] **Step 1: Add failing constant, shared-initializer, anchor, and dtype tests**

Add:

```python
    @pytest.mark.parametrize(
        "constant",
        [
            np.asarray(0.25, dtype=np.float32),
            np.asarray([[[[0.25]], [[-0.5]]]], dtype=np.float32),
            np.asarray(
                [[[[0.25, -0.5], [0.75, 1.0]], [[-0.25, 0.5], [1.25, -1.0]]]],
                dtype=np.float32,
            ),
        ],
    )
    def test_constant_leaves_fold_with_minimal_broadcast(
        self,
        constant: np.ndarray,
    ) -> None:
        model, feeds = _residual_batch_norm_model(constant_values=(constant,))
        expected = _run(model, feeds)

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert not any(
            node.op_type == "BatchNormalization" for node in transformed.graph.node
        )
        replacement_name = transformed.graph.node[-1].input[1]
        replacement = next(
            initializer
            for initializer in transformed.graph.initializer
            if initializer.name == replacement_name
        )
        channels = onnx.numpy_helper.to_array(
            next(
                initializer
                for initializer in model.graph.initializer
                if initializer.name == "bn_scale"
            )
        ).size
        assert tuple(replacement.dims) == np.broadcast_shapes(
            constant.shape,
            (1, channels, 1, 1),
        )
        np.testing.assert_allclose(
            _run(transformed, feeds),
            expected,
            rtol=3e-5,
            atol=3e-5,
        )

    def test_standard_constant_node_leaf_is_folded_and_pruned(self) -> None:
        model, feeds = _residual_batch_norm_model(
            constant_values=(np.asarray(0.25, dtype=np.float32),),
        )
        payload = next(
            initializer
            for initializer in model.graph.initializer
            if initializer.name == "constant_0"
        )
        model.graph.initializer.remove(payload)
        model.graph.node.insert(
            len(model.graph.node) - 2,
            onnx.helper.make_node(
                "Constant",
                [],
                ["constant_0"],
                name="constant_leaf",
                value=payload,
            ),
        )
        expected = _run(model, feeds)

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert not any(node.name == "constant_leaf" for node in transformed.graph.node)
        assert not any(
            node.op_type == "BatchNormalization" for node in transformed.graph.node
        )
        np.testing.assert_allclose(
            _run(transformed, feeds),
            expected,
            rtol=3e-5,
            atol=3e-5,
        )

    def test_shared_parameters_and_constants_are_copied_not_modified(self) -> None:
        constant = np.asarray([[[[0.25]], [[-0.5]]]], dtype=np.float32)
        model, _ = _residual_batch_norm_model(
            with_bias=(True, True),
            constant_values=(constant,),
        )
        first_conv = next(node for node in model.graph.node if node.name == "conv_node_0")
        second_conv = next(node for node in model.graph.node if node.name == "conv_node_1")
        second_conv.input[1] = first_conv.input[1]
        second_conv.input[2] = first_conv.input[2]
        model.graph.node.insert(
            len(model.graph.node) - 1,
            onnx.helper.make_node(
                "Identity",
                ["constant_0"],
                ["constant_observer"],
                name="constant_observer",
            ),
        )
        model.graph.output.append(_info("constant_observer", [1, 2, 1, 1]))
        originals = {
            initializer.name: initializer.SerializeToString()
            for initializer in model.graph.initializer
        }

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        for name in (first_conv.input[1], first_conv.input[2], "constant_0"):
            original_bytes = originals[name]
            transformed_initializer = next(
                initializer
                for initializer in transformed.graph.initializer
                if initializer.name == name
            )
            assert transformed_initializer.SerializeToString() == original_bytes
        transformed_convs = [
            node for node in transformed.graph.node if node.op_type == "Conv"
        ]
        assert transformed_convs[0].input[1] != transformed_convs[1].input[1]
        assert transformed_convs[0].input[2] != transformed_convs[1].input[2]

    def test_anchor_is_earliest_conv_not_add_operand_order(self) -> None:
        replacements_by_order: list[dict[str, np.ndarray]] = []
        for reverse_adds in (frozenset(), frozenset({0, 1})):
            model, _ = _residual_batch_norm_model(
                nested=True,
                reverse_adds=reverse_adds,
                with_bias=(True, True, True),
            )
            index = algebraic_pipe._GraphIndex.build(model)
            tree = algebraic_pipe._residual_add_tree(index, model.graph.node[-1])
            assert tree is not None
            rewrite = algebraic_pipe._precompute_residual_batch_norm(index, tree)
            assert rewrite is not None
            replacements_by_order.append(
                {replacement.conv.name: replacement.bias for replacement in rewrite.convs}
            )

        assert replacements_by_order[0].keys() == replacements_by_order[1].keys()
        for name in replacements_by_order[0]:
            np.testing.assert_array_equal(
                replacements_by_order[0][name],
                replacements_by_order[1][name],
            )

    @pytest.mark.parametrize(
        ("dtype", "rtol", "atol"),
        [
            (np.dtype(np.float16), 3e-3, 3e-3),
            (np.dtype(np.float32), 3e-5, 3e-5),
            (np.dtype(np.float64), 1e-10, 1e-10),
        ],
    )
    def test_supported_dtypes_meet_required_tolerance(
        self,
        dtype: np.dtype,
        rtol: float,
        atol: float,
    ) -> None:
        model, feeds = _residual_batch_norm_model(
            nested=True,
            with_bias=(True, False, True),
            constant_values=(np.asarray(0.25, dtype=dtype),),
            dtype=dtype,
        )
        expected = _run(model, feeds)

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        np.testing.assert_allclose(_run(transformed, feeds), expected, rtol=rtol, atol=atol)
```

- [ ] **Step 2: Run and confirm constant candidates remain unchanged**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "constant or shared_parameters or anchor or supported_dtypes" -v
```

Expected: constant and dtype cases FAIL because precomputation rejects `tree.constants`; copy-on-write assertions fail until constant slots receive copied tensors.

- [ ] **Step 3: Import the repository threshold and implement bounded constant scaling**

Add:

```python
from ...onnx import EXTERNAL_DATA_THRESHOLD
```

Add:

```python
def _scaled_residual_constant(
    values: np.ndarray,
    alpha: np.ndarray,
    output_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray | None:
    if (
        values.dtype != dtype
        or not np.isfinite(values).all()
        or not _shape_broadcasts_to(tuple(values.shape), output_shape)
    ):
        return None
    alpha_shape = (1, len(alpha)) + (1,) * (len(output_shape) - 2)
    try:
        result_shape = np.broadcast_shapes(values.shape, alpha_shape)
    except ValueError:
        return None
    element_count = _shape_element_count(cast("tuple[int, ...]", result_shape))
    if (
        element_count < 0
        or element_count * dtype.itemsize >= EXTERNAL_DATA_THRESHOLD
    ):
        return None
    calculation_dtype = np.dtype(np.result_type(dtype, np.float32))
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.asarray(values, dtype=calculation_dtype) * alpha.reshape(alpha_shape)
    return _finite_cast(scaled, dtype)
```

Replace the temporary constant rejection in `_precompute_residual_batch_norm`:

```python
    constant_replacements: list[_ResidualConstantReplacement] = []
    for constant in tree.constants:
        replacement_values = _scaled_residual_constant(
            constant.values,
            alpha,
            tree.output_shape,
            dtype,
        )
        if replacement_values is None:
            return None
        constant_replacements.append(
            _ResidualConstantReplacement(
                add=constant.add,
                input_index=constant.input_index,
                values=replacement_values,
            )
        )
    return _ResidualBatchNormRewrite(
        batch_norm=batch_norm,
        root_add=tree.root_add,
        convs=tuple(conv_replacements),
        constants=tuple(constant_replacements),
    )
```

- [ ] **Step 4: Extend atomic apply with prebuilt constant tensors**

At the start of `_apply_residual_batch_norm_rewrite`, build every new protobuf
before the first graph edit:

```python
    constant_tensors = [
        onnx.numpy_helper.from_array(
            replacement.values,
            allocator.new("algebraic_residual_add_constant"),
        )
        for replacement in rewrite.constants
    ]
```

Before renaming the root Add, append and rewire:

```python
    for replacement, constant_tensor in zip(
        rewrite.constants,
        constant_tensors,
        strict=True,
    ):
        model.graph.initializer.append(constant_tensor)
        replacement.add.input[replacement.input_index] = constant_tensor.name
```

- [ ] **Step 5: Run positive coverage and quality checks**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "constant or shared_parameters or anchor or supported_dtypes" -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: all selected tests PASS at float16, float32, and float64 tolerances; original shared tensors remain byte-identical.

- [ ] **Step 6: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "feat: fold residual constant leaves" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Confirm constant scaling uses right-aligned broadcasting, materializes the smallest broadcast result, checks both element and byte bounds, and never overwrites shared payloads.

---

### Task 6: Harden Every Fail-Closed Boundary and Atomicity Case

**Files:**
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:TestResidualAddTreeBatchNormFolding`
- Modify: `src/winml/modelkit/optim/pipes/algebraic.py:residual traversal and precompute helpers`

**Interfaces:**
- Consumes: all residual interfaces from Tasks 2-5.
- Produces: complete fail-closed behavior; `_private_residual_output(index: _GraphIndex, tensor_name: str) -> bool`; no public API.

- [ ] **Step 1: Add failing BN, dtype, Conv, and constant validation matrices**

Add explicit mutation helpers:

```python
def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(value for value in model.graph.initializer if value.name == name)


def _mark_external_unloaded(tensor: onnx.TensorProto) -> None:
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    location = tensor.external_data.add()
    location.key = "location"
    location.value = f"missing-{tensor.name}.bin"
```

Add parameter and tensor cases:

```python
    @pytest.mark.parametrize(
        "case",
        [
            "training",
            "noninteger_training_mode",
            "extra_output",
            "bad_arity",
            "negative_epsilon",
            "nonfinite_epsilon",
            "nonfloat_epsilon",
            "duplicate_epsilon",
            "nonpositive_variance",
            "dynamic_parameter",
            "overridable_parameter",
            "external_parameter",
            "mixed_parameter_dtype",
            "nonfinite_parameter",
            "bad_parameter_shape",
            "integer_data_dtype",
            "unsupported_bfloat16",
        ],
    )
    def test_invalid_batch_norm_candidates_are_byte_identical(self, case: str) -> None:
        model, _ = _residual_batch_norm_model()
        batch_norm = model.graph.node[-1]
        if case == "training":
            batch_norm.attribute.append(onnx.helper.make_attribute("training_mode", 1))
        elif case == "noninteger_training_mode":
            batch_norm.attribute.append(onnx.helper.make_attribute("training_mode", 0.0))
        elif case == "extra_output":
            batch_norm.output.append("saved_mean")
        elif case == "bad_arity":
            batch_norm.input.append("bn_variance")
        elif case == "negative_epsilon":
            batch_norm.attribute.append(onnx.helper.make_attribute("epsilon", -0.01))
        elif case == "nonfinite_epsilon":
            batch_norm.attribute.append(onnx.helper.make_attribute("epsilon", np.inf))
        elif case == "nonfloat_epsilon":
            batch_norm.attribute.append(onnx.helper.make_attribute("epsilon", 0))
        elif case == "duplicate_epsilon":
            batch_norm.attribute.extend(
                [
                    onnx.helper.make_attribute("epsilon", 1e-5),
                    onnx.helper.make_attribute("epsilon", 1e-5),
                ]
            )
        elif case == "nonpositive_variance":
            _initializer(model, "bn_variance").CopyFrom(
                _tensor("bn_variance", np.asarray([-0.02, 1.0], dtype=np.float32))
            )
        elif case == "dynamic_parameter":
            model.graph.initializer.remove(_initializer(model, "bn_scale"))
            model.graph.input.append(_info("bn_scale", [2]))
        elif case == "overridable_parameter":
            model.graph.input.append(_info("bn_scale", [2]))
        elif case == "external_parameter":
            _mark_external_unloaded(_initializer(model, "bn_scale"))
        elif case == "mixed_parameter_dtype":
            _initializer(model, "bn_beta").CopyFrom(
                _tensor("bn_beta", np.zeros(2, dtype=np.float64))
            )
        elif case == "nonfinite_parameter":
            _initializer(model, "bn_mean").CopyFrom(
                _tensor("bn_mean", np.asarray([np.nan, 0.0], dtype=np.float32))
            )
        elif case == "bad_parameter_shape":
            _initializer(model, "bn_scale").CopyFrom(
                _tensor("bn_scale", np.ones((1, 2), dtype=np.float32))
            )
        elif case in {"integer_data_dtype", "unsupported_bfloat16"}:
            elem_type = (
                onnx.TensorProto.INT32
                if case == "integer_data_dtype"
                else onnx.TensorProto.BFLOAT16
            )
            for value_info in (
                *model.graph.input,
                *model.graph.output,
                *model.graph.value_info,
            ):
                value_info.type.tensor_type.elem_type = elem_type
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        _assert_byte_identical(model, transformed)
        assert model.SerializeToString() == original

    @pytest.mark.parametrize(
        "case",
        [
            "dynamic_weight",
            "overridable_weight",
            "overridable_bias",
            "external_weight",
            "external_bias",
            "bad_weight_channels",
            "bad_bias_shape",
            "bad_conv_arity",
            "extra_conv_output",
            "mixed_weight_dtype",
            "nonfinite_weight",
            "nonfinite_bias",
            "overflow_weight",
            "overflow_bias",
        ],
    )
    def test_invalid_conv_parameters_are_byte_identical(self, case: str) -> None:
        model, _ = _residual_batch_norm_model(with_bias=(True, True))
        conv = model.graph.node[0]
        if case == "dynamic_weight":
            model.graph.initializer.remove(_initializer(model, conv.input[1]))
            model.graph.input.append(_info(conv.input[1], [2, 2, 1, 1]))
        elif case == "overridable_weight":
            model.graph.input.append(_info(conv.input[1], [2, 2, 1, 1]))
        elif case == "overridable_bias":
            model.graph.input.append(_info(conv.input[2], [2]))
        elif case == "external_weight":
            _mark_external_unloaded(_initializer(model, conv.input[1]))
        elif case == "external_bias":
            _mark_external_unloaded(_initializer(model, conv.input[2]))
        elif case == "bad_weight_channels":
            _initializer(model, conv.input[1]).CopyFrom(
                _tensor(conv.input[1], np.ones((3, 2, 1, 1), dtype=np.float32))
            )
        elif case == "bad_bias_shape":
            _initializer(model, conv.input[2]).CopyFrom(
                _tensor(conv.input[2], np.ones((1, 2), dtype=np.float32))
            )
        elif case == "bad_conv_arity":
            conv.input.append(conv.input[1])
        elif case == "extra_conv_output":
            conv.output.append("extra_conv_output")
        elif case == "mixed_weight_dtype":
            _initializer(model, conv.input[1]).CopyFrom(
                _tensor(conv.input[1], np.ones((2, 2, 1, 1), dtype=np.float64))
            )
        elif case == "nonfinite_weight":
            _initializer(model, conv.input[1]).CopyFrom(
                _tensor(conv.input[1], np.full((2, 2, 1, 1), np.inf, dtype=np.float32))
            )
        elif case == "nonfinite_bias":
            _initializer(model, conv.input[2]).CopyFrom(
                _tensor(conv.input[2], np.asarray([np.nan, 0.0], dtype=np.float32))
            )
        elif case == "overflow_weight":
            _initializer(model, conv.input[1]).CopyFrom(
                _tensor(conv.input[1], np.full((2, 2, 1, 1), 3e38, dtype=np.float32))
            )
            _initializer(model, "bn_scale").CopyFrom(
                _tensor("bn_scale", np.full(2, 2.0, dtype=np.float32))
            )
        elif case == "overflow_bias":
            _initializer(model, conv.input[2]).CopyFrom(
                _tensor(conv.input[2], np.full(2, 3e38, dtype=np.float32))
            )
            _initializer(model, "bn_scale").CopyFrom(
                _tensor("bn_scale", np.full(2, 2.0, dtype=np.float32))
            )
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert transformed.SerializeToString() == original
```

Add constant cases:

```python
    @pytest.mark.parametrize(
        "case",
        [
            "dynamic",
            "overridable",
            "custom_domain_constant",
            "external_initializer",
            "external_constant",
            "malformed_constant",
            "mixed_dtype",
            "nonfinite",
            "bad_broadcast",
            "unsafe_size",
        ],
    )
    def test_invalid_constant_leaves_are_byte_identical(
        self,
        case: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model, _ = _residual_batch_norm_model(
            constant_values=(np.asarray(0.25, dtype=np.float32),),
        )
        constant_add = model.graph.node[-2]
        if case == "dynamic":
            model.graph.initializer.remove(_initializer(model, "constant_0"))
            model.graph.input.append(_info("constant_0", [1]))
        elif case == "overridable":
            model.graph.input.append(_info("constant_0", [1]))
        elif case == "custom_domain_constant":
            model.graph.initializer.remove(_initializer(model, "constant_0"))
            model.graph.node.insert(
                len(model.graph.node) - 2,
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["constant_0"],
                    domain="custom",
                    value=_tensor("payload", np.asarray(0.25, dtype=np.float32)),
                ),
            )
        elif case == "external_initializer":
            _mark_external_unloaded(_initializer(model, "constant_0"))
        elif case == "external_constant":
            model.graph.initializer.remove(_initializer(model, "constant_0"))
            payload = _tensor("payload", np.asarray(0.25, dtype=np.float32))
            _mark_external_unloaded(payload)
            model.graph.node.insert(
                len(model.graph.node) - 2,
                onnx.helper.make_node("Constant", [], ["constant_0"], value=payload),
            )
        elif case == "malformed_constant":
            model.graph.initializer.remove(_initializer(model, "constant_0"))
            model.graph.node.insert(
                len(model.graph.node) - 2,
                onnx.helper.make_node(
                    "Constant",
                    [],
                    ["constant_0"],
                    value_string=b"not-a-tensor",
                ),
            )
        elif case == "mixed_dtype":
            _initializer(model, "constant_0").CopyFrom(
                _tensor("constant_0", np.asarray(0.25, dtype=np.float64))
            )
        elif case == "nonfinite":
            _initializer(model, "constant_0").CopyFrom(
                _tensor("constant_0", np.asarray(np.inf, dtype=np.float32))
            )
        elif case == "bad_broadcast":
            _initializer(model, "constant_0").CopyFrom(
                _tensor("constant_0", np.ones((3, 3), dtype=np.float32))
            )
        elif case == "unsafe_size":
            monkeypatch.setattr(algebraic_pipe, "EXTERNAL_DATA_THRESHOLD", 1)
        assert "constant_0" in constant_add.input
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert transformed.SerializeToString() == original
```

- [ ] **Step 2: Add failing topology, observation, domain, cycle, and definition tests**

Add:

```python
    @pytest.mark.parametrize(
        "case",
        [
            "zero_conv",
            "one_conv",
            "unsupported_leaf",
            "empty_add_input",
            "extra_add_output",
            "custom_batch_norm",
            "custom_add",
            "custom_conv",
            "public_conv",
            "public_add",
            "shared_conv",
            "shared_add",
            "captured_conv",
            "captured_add",
            "repeated_leaf",
            "duplicate_definition",
            "duplicate_graph_input",
            "duplicate_initializer",
            "cross_kind_collision",
            "cycle",
            "dynamic_conv_shape",
            "dynamic_add_shape",
            "ambiguous_opset",
            "legacy_opset",
        ],
    )
    def test_unsafe_graph_shapes_are_byte_identical(self, case: str) -> None:
        model, _ = _residual_batch_norm_model()
        first_conv, second_conv, root_add, batch_norm = model.graph.node
        if case == "zero_conv":
            first_conv.op_type = "Relu"
            second_conv.op_type = "Relu"
        elif case == "one_conv":
            second_conv.op_type = "Relu"
        elif case == "unsupported_leaf":
            second_conv.op_type = "Relu"
        elif case == "empty_add_input":
            root_add.input[1] = ""
        elif case == "extra_add_output":
            root_add.output.append("extra_add_output")
        elif case == "custom_batch_norm":
            batch_norm.domain = "custom"
        elif case == "custom_add":
            root_add.domain = "custom"
        elif case == "custom_conv":
            first_conv.domain = "custom"
        elif case == "public_conv":
            model.graph.output.append(_info(first_conv.output[0], [1, 2, 2, 2]))
        elif case == "public_add":
            model.graph.output.append(_info(root_add.output[0], [1, 2, 2, 2]))
        elif case == "shared_conv":
            model.graph.node.insert(
                len(model.graph.node) - 1,
                onnx.helper.make_node("Identity", [first_conv.output[0]], ["observed"]),
            )
            model.graph.output.append(_info("observed", [1, 2, 2, 2]))
        elif case == "shared_add":
            model.graph.node.insert(
                len(model.graph.node) - 1,
                onnx.helper.make_node("Identity", [root_add.output[0]], ["observed"]),
            )
            model.graph.output.append(_info("observed", [1, 2, 2, 2]))
        elif case in {"captured_conv", "captured_add"}:
            captured = first_conv.output[0] if case == "captured_conv" else root_add.output[0]
            body = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", [captured], ["captured_out"])],
                "capturing_branch",
                [],
                [_info("captured_out", [1, 2, 2, 2])],
            )
            model.graph.node.insert(
                len(model.graph.node) - 1,
                onnx.helper.make_node(
                    "If",
                    ["condition"],
                    ["capture_result"],
                    then_branch=body,
                    else_branch=body,
                ),
            )
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(
                    "condition",
                    onnx.TensorProto.BOOL,
                    [],
                )
            )
        elif case == "repeated_leaf":
            root_add.input[:] = [first_conv.output[0], first_conv.output[0]]
        elif case == "duplicate_definition":
            second_conv.output[0] = first_conv.output[0]
        elif case == "duplicate_graph_input":
            model.graph.input.append(_info("x_0", [1, 2, 2, 2]))
        elif case == "duplicate_initializer":
            model.graph.initializer.append(
                _tensor("weight_0", np.ones((2, 2, 1, 1), dtype=np.float32))
            )
        elif case == "cross_kind_collision":
            model.graph.initializer.append(
                _tensor(first_conv.output[0], np.asarray(0.0, dtype=np.float32))
            )
        elif case == "cycle":
            first_conv.input[0] = root_add.output[0]
        elif case == "dynamic_conv_shape":
            conv_info = next(
                value for value in model.graph.value_info if value.name == "conv_0"
            )
            conv_info.type.tensor_type.shape.dim[0].ClearField("dim_value")
        elif case == "dynamic_add_shape":
            add_info = next(
                value for value in model.graph.value_info if value.name == "add_0"
            )
            add_info.type.tensor_type.shape.dim[0].ClearField("dim_value")
        elif case == "ambiguous_opset":
            model.opset_import.append(onnx.helper.make_opsetid("", 17))
        elif case == "legacy_opset":
            model.opset_import[0].version = 6
        original = model.SerializeToString()

        transformed = AlgebraicRewritePipe().process(
            model,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        assert transformed.SerializeToString() == original
```

Add an exact boundary test:

```python
    def test_depth_64_folds_and_depth_65_is_unchanged(self) -> None:
        results: list[bool] = []
        for depth in (64, 65):
            model, _ = _residual_batch_norm_model()
            batch_norm = model.graph.node[-1]
            previous = batch_norm.input[0]
            model.graph.initializer.append(
                _tensor("depth_zero", np.asarray(0.0, dtype=np.float32))
            )
            for index in range(depth):
                output = f"depth_sum_{index}"
                model.graph.node.insert(
                    len(model.graph.node) - 1,
                    onnx.helper.make_node("Add", [previous, "depth_zero"], [output]),
                )
                model.graph.value_info.append(_info(output, [1, 2, 2, 2]))
                previous = output
            batch_norm.input[0] = previous
            transformed = AlgebraicRewritePipe().process(
                model,
                AlgebraicRewritePipeConfig(
                    conv_add_batch_normalization_folding=True,
                ),
            )
            results.append(
                any(
                    node.op_type == "BatchNormalization"
                    for node in transformed.graph.node
                )
            )

        assert results == [False, True]

    def test_ineligible_later_candidate_does_not_rollback_earlier_fold(self) -> None:
        first, _ = _residual_batch_norm_model()
        second, _ = _residual_batch_norm_model()
        second_batch_norm = second.graph.node[-1]
        second_batch_norm.attribute.append(
            onnx.helper.make_attribute("training_mode", 1)
        )
        for value in (*second.graph.input, *second.graph.output, *second.graph.value_info):
            value.name = f"later_{value.name}"
        for initializer in second.graph.initializer:
            initializer.name = f"later_{initializer.name}"
        for node in second.graph.node:
            node.name = f"later_{node.name}"
            node.input[:] = [f"later_{name}" for name in node.input]
            node.output[:] = [f"later_{name}" for name in node.output]
        expected_later_bn = second.graph.node[-1].SerializeToString()
        first.graph.node.extend(second.graph.node)
        first.graph.input.extend(second.graph.input)
        first.graph.output.extend(second.graph.output)
        first.graph.initializer.extend(second.graph.initializer)
        first.graph.value_info.extend(second.graph.value_info)

        transformed = AlgebraicRewritePipe().process(
            first,
            AlgebraicRewritePipeConfig(
                conv_add_batch_normalization_folding=True,
            ),
        )

        remaining = [
            node for node in transformed.graph.node if node.op_type == "BatchNormalization"
        ]
        assert [node.name for node in remaining] == ["later_batch_norm"]
        assert remaining[0].SerializeToString() == expected_later_bn
        assert any(
            node.op_type == "Add" and node.output == ["y"]
            for node in transformed.graph.node
        )
```

- [ ] **Step 3: Run the fail-closed matrix and inspect every failure**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding -k "invalid or unsafe or depth_64" -v
```

Expected before hardening: at least the custom-domain Constant, overridable Conv parameter, extra BN output, ambiguous opset, and exact depth-boundary cases FAIL. Do not weaken assertions; add the missing proof guards.

- [ ] **Step 4: Add explicit private-output and immutable-parameter guards**

Add:

```python
def _typed_attribute(
    node: onnx.NodeProto,
    name: str,
    attribute_type: int,
    default: int | float,
) -> int | float | None:
    attributes = [attribute for attribute in node.attribute if attribute.name == name]
    if not attributes:
        return default
    if len(attributes) != 1 or attributes[0].type != attribute_type:
        return None
    return cast("int | float", onnx.helper.get_attribute_value(attributes[0]))


def _private_residual_output(
    index: _GraphIndex,
    tensor_name: str,
) -> bool:
    return (
        tensor_name not in index.graph_outputs
        and len(index.consumers.get(tensor_name, [])) == 1
    )
```

    Replace the permissive attribute conversions at the start of
    `_precompute_residual_batch_norm`:

    ```python
        training_mode = _typed_attribute(
            batch_norm,
            "training_mode",
            onnx.AttributeProto.INT,
            0,
        )
        epsilon = _typed_attribute(
            batch_norm,
            "epsilon",
            onnx.AttributeProto.FLOAT,
            1e-5,
        )
        if (
            training_mode is None
            or epsilon is None
            or training_mode != 0
            or not math.isfinite(epsilon)
            or epsilon < 0
        ):
            return None
    ```

    Use it for every Conv and Add output in `_collect_residual_add_tree`.

Before reading Conv parameters in `_precompute_residual_batch_norm`, add:

```python
        if (
            len(conv.input) not in (2, 3)
            or conv.input[1] in index.graph_inputs
            or (
                len(conv.input) == 3
                and conv.input[2]
                and conv.input[2] in index.graph_inputs
            )
        ):
            return None
```

Before reading BN parameters:

```python
    if any(name in index.graph_inputs for name in batch_norm.input[1:]):
        return None
```

Require the traversal producer domain before dispatch:

```python
    if producer is not None and not _is_standard_onnx_node(producer):
        return False
```

Require a standard-domain Constant when `_constant_array` succeeds through a producer:

```python
            input_producer = index.producers.get(input_name)
            if (
                input_producer is not None
                and (
                    not _is_standard_onnx_node(input_producer)
                    or input_producer.op_type != "Constant"
                )
            ):
                return False
```

The existing top-level `process` guard remains authoritative:

```python
        index = _GraphIndex.build(result)
        if index.definition_collisions or index.has_cycle:
            return result
```

Do not add a broad exception handler. Keep `_constant_array`'s existing narrow conversion exceptions.

- [ ] **Step 5: Resolve exact boundary and arity failures**

Use Add-edge depth, not leaf count:

```python
    if depth > MAX_RESIDUAL_ADD_DEPTH:
        return False
```

Call child recursion with `depth + 1` only for an Add child and keep Conv leaves at the current depth. Validate BN and Conv arity before indexing any input. Require `len(batch_norm.output) == 1` and `len(conv.output) == 1`.

- [ ] **Step 6: Run all residual tests, affected fixed tests, and Ruff**

Run:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding tests\unit\pattern\test_conv_batchnorm_patterns.py -v
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: every residual positive and fail-closed test PASS; every fixed-pattern test PASS; no skipped or expected-failure marker is added.

- [ ] **Step 7: Commit**

```powershell
git add src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "fix: harden residual batch norm folding" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Compare the parameterized cases line-by-line with the spec fail-closed list. For every case, confirm byte equality proves no appended initializer, rewire, rename, or node removal leaked.

---

### Task 7: Verify Shared Analysis, Fixed-Pattern Coexistence, and Public CLI Composition

**Files:**
- Modify: `tests/unit/optim/test_analysis.py:45-118, TestIterOptimizationOutputs around line 688`
- Modify: `tests/unit/optim/pipes/test_pipe_algebraic.py:TestAlgebraicRegistration and TestResidualAddTreeBatchNormFolding`
- Read-only unless a real defect appears: `src/winml/modelkit/optim/analysis.py:360-530`
- Read-only unless a real defect appears: `src/winml/modelkit/optim/pipes/rewrite.py:105-180`

**Interfaces:**
- Consumes: `iter_optimization_outputs(model, get_all_capabilities())`, `AlgebraicRewritePipe.build_config`, `RewritePipe.build_config`, CLI `optimize`.
- Produces: no new production interface; tests prove the existing shared-owner machinery reports `algebraic_rewrite+rewrite`.

- [ ] **Step 1: Add a generated residual analysis model**

Add after `_sibling_slice_model` in `test_analysis.py`:

```python
def _residual_batch_norm_model() -> ModelProto:
    shape = [1, 2, 2, 2]
    inputs = [
        helper.make_tensor_value_info("x0", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("x1", TensorProto.FLOAT, shape),
    ]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    nodes = [
        helper.make_node("Conv", ["x0", "w0"], ["c0"], name="c0"),
        helper.make_node("Conv", ["x1", "w1"], ["c1"], name="c1"),
        helper.make_node("Add", ["c0", "c1"], ["sum"], name="sum"),
        helper.make_node(
            "BatchNormalization",
            ["sum", "scale", "beta", "mean", "variance"],
            ["y"],
            name="bn",
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.ones((2, 2, 1, 1), dtype=np.float32), "w0"),
        numpy_helper.from_array(np.ones((2, 2, 1, 1), dtype=np.float32), "w1"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.zeros(2, dtype=np.float32), "beta"),
        numpy_helper.from_array(np.zeros(2, dtype=np.float32), "mean"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "variance"),
    ]
    value_info = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name in ("c0", "c1", "sum")
    ]
    return _finalize(
        helper.make_graph(
            nodes,
            "residual_batch_norm",
            inputs,
            [output],
            initializer=initializers,
            value_info=value_info,
        )
    )
```

- [ ] **Step 2: Add failing combined-owner analysis assertions**

Add to `TestIterOptimizationOutputs`:

```python
    def test_reports_residual_batch_norm_once_across_both_owners(self) -> None:
        pairs = list(
            iter_optimization_outputs(
                _residual_batch_norm_model(),
                get_all_capabilities(),
            )
        )
        matches = [
            (finding, produced)
            for finding, produced in pairs
            if finding.name == "conv-add-batch-normalization-folding"
        ]

        assert len(matches) == 1
        finding, produced = matches[0]
        assert finding.enable_flag == "--enable-conv-add-batch-normalization-folding"
        assert finding.pipe_name == "algebraic_rewrite+rewrite"
        assert any(
            reference.op_type == "BatchNormalization"
            for reference in finding.removed_nodes
        )
        assert not any(
            node.op_type == "BatchNormalization" for node in produced.graph.node
        )
```

Run:

```powershell
uv run pytest tests\unit\optim\test_analysis.py::TestIterOptimizationOutputs::test_reports_residual_batch_norm_once_across_both_owners -v
```

Expected: PASS without changing `analysis.py`. This is a postimplementation
characterization of PR #1301's shared-owner machinery rather than a new
production behavior. If it fails, preserve the failure as the red test,
diagnose the concrete shared-owner defect, and make only the smallest analysis
correction needed to turn it green.

- [ ] **Step 3: Add fixed and recursive coexistence under one kwargs key**

Import:

```python
from winml.modelkit.optim.pipes import RewritePipe
```

Add:

```python
    def test_fixed_and_residual_candidates_coexist_under_one_flag(self) -> None:
        residual, residual_feeds = _residual_batch_norm_model()
        fixed, fixed_feeds = _residual_batch_norm_model()
        fixed_conv = next(node for node in fixed.graph.node if node.name == "conv_node_1")
        fixed.graph.node.remove(fixed_conv)
        retained_inputs = [
            value for value in fixed.graph.input if value.name != "x_1"
        ]
        del fixed.graph.input[:]
        fixed.graph.input.extend(retained_inputs)
        retained_initializers = [
            value
            for value in fixed.graph.initializer
            if value.name != "weight_1"
        ]
        del fixed.graph.initializer[:]
        fixed.graph.initializer.extend(retained_initializers)
        retained_value_info = [
            value for value in fixed.graph.value_info if value.name != "conv_1"
        ]
        del fixed.graph.value_info[:]
        fixed.graph.value_info.extend(retained_value_info)
        fixed.graph.initializer.append(
            _tensor(
                "fixed_static",
                np.asarray([[[[0.25]], [[-0.5]]]], dtype=np.float32),
            )
        )
        fixed_add = next(node for node in fixed.graph.node if node.name == "add_node_0")
        fixed_add.input[:] = ["conv_0", "fixed_static"]
        for value in (*fixed.graph.input, *fixed.graph.output, *fixed.graph.value_info):
            value.name = f"fixed_{value.name}"
        for initializer in fixed.graph.initializer:
            initializer.name = f"fixed_{initializer.name}"
        for node in fixed.graph.node:
            node.name = f"fixed_{node.name}"
            node.input[:] = [f"fixed_{name}" for name in node.input]
            node.output[:] = [f"fixed_{name}" for name in node.output]
        residual.graph.node.extend(fixed.graph.node)
        residual.graph.input.extend(fixed.graph.input)
        residual.graph.output.extend(fixed.graph.output)
        residual.graph.initializer.extend(fixed.graph.initializer)
        residual.graph.value_info.extend(fixed.graph.value_info)
        feeds = {
            **residual_feeds,
            "fixed_x_0": fixed_feeds["x_0"],
        }
        expected = _run(residual, feeds)
        kwargs = {"conv_add_batch_normalization_folding": True}

        after_algebraic = AlgebraicRewritePipe().process(
            residual,
            AlgebraicRewritePipe.build_config(**kwargs),
        )
        transformed = RewritePipe().process(
            after_algebraic,
            RewritePipe.build_config(**kwargs),
        )

        assert not any(
            node.op_type == "BatchNormalization" for node in transformed.graph.node
        )
        np.testing.assert_allclose(
            _run(transformed, feeds),
            expected,
            rtol=3e-5,
            atol=3e-5,
        )
```

- [ ] **Step 4: Add an exact public CLI composition test**

Add to `TestAlgebraicRegistration`:

```python
    def test_cli_folds_residual_tree_with_existing_flag(
        self,
        tmp_path: Path,
    ) -> None:
        model, feeds = _residual_batch_norm_model(nested=True)
        expected = _run(model, feeds)
        input_path = tmp_path / "residual_input.onnx"
        output_path = tmp_path / "residual_output.onnx"
        onnx.save_model(model, input_path)

        result = CliRunner().invoke(
            optimize,
            [
                "-m",
                str(input_path),
                "-o",
                str(output_path),
                "--enable-conv-add-batch-normalization-folding",
                "--enable-conv-channel-affine-folding",
                "--no-color",
            ],
        )

        assert result.exit_code == 0, result.output
        transformed = onnx.load_model(output_path)
        assert not any(
            node.op_type == "BatchNormalization" for node in transformed.graph.node
        )
        assert transformed.graph.output[0].name == "y"
        np.testing.assert_allclose(
            _run(transformed, feeds),
            expected,
            rtol=3e-5,
            atol=3e-5,
        )
```

Do not create a second CLI option or capability alias.

- [ ] **Step 5: Run analysis, coexistence, CLI, and fixed-pattern tests**

Run:

```powershell
uv run pytest tests\unit\optim\test_analysis.py::TestIterOptimizationOutputs::test_reports_residual_batch_norm_once_across_both_owners tests\unit\optim\pipes\test_pipe_algebraic.py::TestResidualAddTreeBatchNormFolding::test_fixed_and_residual_candidates_coexist_under_one_flag tests\unit\optim\pipes\test_pipe_algebraic.py::TestAlgebraicRegistration tests\unit\pattern\test_conv_batchnorm_patterns.py -v
uv run ruff check --fix tests\unit\optim\test_analysis.py tests\unit\optim\pipes\test_pipe_algebraic.py
uv run ruff format tests\unit\optim\test_analysis.py tests\unit\optim\pipes\test_pipe_algebraic.py
```

Expected: one shared finding, both owners in pipeline order, both topology families removed under one key, CLI composition PASS, and fixed tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add tests\unit\optim\test_analysis.py tests\unit\optim\pipes\test_pipe_algebraic.py
git commit -m "test: cover shared residual batch norm capability" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Confirm `analysis.py` remained unchanged unless a reproduced bug required it, exactly one finding is emitted, and the fixed one-Conv case is still owned by RewritePipe.

---

### Task 8: Record Real-Model Acceptance Evidence

**Files:**
- Create temporarily, then delete: `temp/test_residual_add_tree_batchnorm_real_model.py`
- Create: `docs/superpowers/evidence/2026-08-14-residual-add-tree-batchnorm-folding.md`
- Do not modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `optimize_onnx(model: str | Path | onnx.ModelProto, output: str | Path | None = None, *, config: str | Path | dict[str, Any] | None = None, **capabilities: Any) -> onnx.ModelProto`.
- Produces: persisted measured evidence; no production API.

- [ ] **Step 1: Create a temporary pytest that validates and writes measured evidence**

Create `temp\test_residual_add_tree_batchnorm_real_model.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from winml.modelkit.optim import optimize_onnx
from winml.modelkit.optim.pipes import algebraic as algebraic_pipe


MODEL_DIR = Path(r"D:\AI\isv_models\keen_hominy\0805")
MODEL_PATH = MODEL_DIR / "model_opset17.onnx"
INPUTS_PATH = MODEL_DIR / "model_example_inputs.npz"
PYTORCH_OUTPUTS_PATH = MODEL_DIR / "model_example_outputs_pytorch.npz"
EVIDENCE_PATH = Path(
    r"docs\superpowers\evidence\2026-08-14-residual-add-tree-batchnorm-folding.md"
)


def _signature(model: onnx.ModelProto) -> list[tuple[str, int, tuple[int | str | None, ...]]]:
    signature: list[tuple[str, int, tuple[int | str | None, ...]]] = []
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        dimensions: list[int | str | None] = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
        signature.append((output.name, tensor_type.elem_type, tuple(dimensions)))
    return signature


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _run(path: Path, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path),
        options,
        providers=["CPUExecutionProvider"],
    )
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(names, inputs), strict=True))


def _errors(
    actual: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float]:
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    difference = np.abs(actual64 - expected64)
    denominator = np.maximum(
        np.abs(expected64),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(difference)), float(np.max(difference / denominator))


def _tolerance(dtype: np.dtype) -> tuple[float, float]:
    if dtype == np.dtype(np.float16):
        return 3e-3, 3e-3
    if dtype == np.dtype(np.float32):
        return 3e-5, 3e-5
    if dtype == np.dtype(np.float64):
        return 1e-10, 1e-10
    raise AssertionError(f"unsupported output dtype {dtype}")


def test_real_model_acceptance(tmp_path: Path) -> None:
    optimized_path = tmp_path / "model_residual_bn_folded.onnx"
    before = onnx.load_model(MODEL_PATH, load_external_data=True)
    assert len(before.graph.node) == 232
    assert sum(node.op_type == "BatchNormalization" for node in before.graph.node) == 4
    source_index = algebraic_pipe._GraphIndex.build(before)
    source_trees = [
        algebraic_pipe._residual_add_tree(source_index, node)
        for node in before.graph.node
        if node.op_type == "BatchNormalization"
    ]
    assert all(tree is not None for tree in source_trees)
    eligible_shapes = sorted(
        (len(tree.adds), len(tree.convs))
        for tree in source_trees
        if tree is not None
    )
    assert eligible_shapes == [(1, 2), (1, 2), (2, 3), (2, 3)]

    optimize_onnx(
        MODEL_PATH,
        optimized_path,
        conv_add_batch_normalization_folding=True,
    )
    after = onnx.load_model(optimized_path, load_external_data=True)
    onnx.checker.check_model(after)
    assert sum(node.op_type == "BatchNormalization" for node in after.graph.node) == 0
    assert _signature(after) == _signature(before)

    inputs = _load_npz(INPUTS_PATH)
    pytorch_outputs = _load_npz(PYTORCH_OUTPUTS_PATH)
    baseline_outputs = _run(MODEL_PATH, inputs)
    optimized_outputs = _run(optimized_path, inputs)
    assert baseline_outputs.keys() == optimized_outputs.keys() == pytorch_outputs.keys()

    rows: list[str] = []
    for name in baseline_outputs:
        baseline = baseline_outputs[name]
        optimized = optimized_outputs[name]
        pytorch = pytorch_outputs[name]
        rtol, atol = _tolerance(baseline.dtype)
        np.testing.assert_allclose(optimized, baseline, rtol=rtol, atol=atol)
        optimized_baseline = _errors(optimized, baseline)
        baseline_pytorch = _errors(baseline, pytorch)
        optimized_pytorch = _errors(optimized, pytorch)
        assert optimized_pytorch[0] <= baseline_pytorch[0] + atol
        assert optimized_pytorch[1] <= baseline_pytorch[1] + rtol
        rows.append(
            f"| `{name}` | {optimized_baseline[0]:.9g} | "
            f"{optimized_baseline[1]:.9g} | {baseline_pytorch[0]:.9g} | "
            f"{baseline_pytorch[1]:.9g} | {optimized_pytorch[0]:.9g} | "
            f"{optimized_pytorch[1]:.9g} |"
        )

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    signature_rows = "\n".join(
        f"- `{name}`: elem_type={elem_type}, shape={shape}"
        for name, elem_type, shape in _signature(before)
    )
    EVIDENCE_PATH.write_text(
        "\n".join(
            [
                "# Residual Add-Tree BatchNormalization Folding Evidence",
                "",
                f"- Source model: `{MODEL_PATH}`",
                f"- Total nodes: `{len(before.graph.node)} -> {len(after.graph.node)}`",
                "- BatchNormalization nodes: `4 -> 0`",
                "- Eligible source trees: `2 x (1 Add, 2 Conv); 2 x (2 Add, 3 Conv)`",
                "- ONNX checker: passed",
                "- Output signature: preserved exactly",
                "- NPU performance: not evaluated and not an acceptance gate",
                "",
                "## Output Signature",
                "",
                signature_rows,
                "",
                "## Replay Error",
                "",
                "| Output | opt/base max abs | opt/base max rel | "
                "base/PyTorch max abs | base/PyTorch max rel | "
                "opt/PyTorch max abs | opt/PyTorch max rel |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 2: Run the real-model pytest**

Run:

```powershell
uv run pytest temp\test_residual_add_tree_batchnorm_real_model.py -v -s
```

Expected: PASS with source counts `232` and `4`, exactly two eligible one-Add/two-Conv trees and two eligible two-Add/three-Conv trees, optimized BatchNormalization count `0`, exact output signature preservation, ONNX checker success, optimized-versus-baseline allclose, and non-regressed PyTorch-reference errors. The measured final total node count is written but is not asserted.

- [ ] **Step 3: Inspect generated evidence and remove only the temporary test**

Run:

```powershell
Get-Content docs\superpowers\evidence\2026-08-14-residual-add-tree-batchnorm-folding.md
Remove-Item temp\test_residual_add_tree_batchnorm_real_model.py
git status --short
```

Expected: the evidence contains measured rows for `xyz`, `rgb`, `opacity`, `quaternion`, `scale`, `sh`, `position_offset`, and `landmarks_2d`; only the evidence document is untracked before staging.

- [ ] **Step 4: Apply the documentation/release-note decision**

Inspect `CHANGELOG.md` and leave it unchanged. It is organized as release snapshots, while this work extends an existing flag without adding a command, config key, or default behavior. The approved design and measured evidence are the directly related documentation.

- [ ] **Step 5: Commit the evidence**

```powershell
git add docs\superpowers\evidence\2026-08-14-residual-add-tree-batchnorm-folding.md
git commit -m "docs: record residual batch norm evidence" -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

**Reviewer gate:** Verify `232 / 4 -> 0`, exact signature equality, all eight replay rows, no asserted final total node count, and no NPU gate.

---

## Final Verification and Review Gate

After Task 8's commit, run the exact affected scope:

```powershell
uv run pytest tests\unit\optim\pipes\test_pipe_algebraic.py tests\unit\optim\test_analysis.py tests\unit\pattern\test_conv_batchnorm_patterns.py -v
```

Then run the broader optimizer scope required for this cross-pipe capability:

```powershell
uv run pytest tests\unit\optim\ tests\unit\pattern\test_conv_batchnorm_patterns.py
```

Run source quality gates:

```powershell
uv run ruff check --fix src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py tests\unit\optim\test_analysis.py tests\unit\pattern\test_conv_batchnorm_patterns.py
uv run ruff format src\winml\modelkit\optim\pipes\algebraic.py tests\unit\optim\pipes\test_pipe_algebraic.py tests\unit\optim\test_analysis.py tests\unit\pattern\test_conv_batchnorm_patterns.py
uv run mypy src\
git diff --check
git status --short
```

Expected: all tests PASS with no newly skipped or expected-failure tests; Ruff, mypy, and `git diff --check` succeed; the worktree is clean.

Invoke `superpowers:requesting-code-review` for the complete branch diff from the implementation base through the evidence commit. The reviewer must check:

1. One canonical capability object and one CLI flag.
2. ORTGraphPipe before AlgebraicRewritePipe; residual fold before channel-affine; RewritePipe afterward.
3. Bounded complete-tree discovery and all privacy/capture/domain guards.
4. Exact `alpha`/`delta` algebra, deterministic anchor, minimal constant broadcast, and dtype tolerances.
5. No mutation before complete precomputation and collision-free name reservation.
6. Copy-on-write for every Conv weight, Conv bias, and constant leaf.
7. Root Add output-name preservation, cleanup, multiple candidates, and idempotence.
8. One combined shared-owner analysis finding and fixed-pattern coexistence.
9. Every fail-closed category from the approved spec.
10. Real-model `232 nodes / 4 BatchNormalization -> 0 BatchNormalization`, exact signature, checker, and replay evidence.
11. No architecture-specific logic, unrelated refactor, release-note churn, or NPU acceptance dependency.

If review identifies a defect, reproduce it with a focused failing pytest, run that test to observe the failure, apply the smallest correction, rerun the targeted and broader optimizer scopes, and commit the correction separately with the required trailer. Repeat the final review until it reports no high-confidence correctness findings.
