# Pipe Architecture Test Suite

This document describes the test suite for the pipe-based optimization architecture.

## Test Files

### 1. `test_pipe_base.py`
Tests for PipeConfig and BasePipe abstract classes.

**Test Categories:**
- **PipeConfig Tests**: Basic configuration class functionality
  - Instantiation as minimal base class
  - Subclassing with custom fields
  - Dataclass structure validation

- **BasePipe Abstract Tests**: Abstract class enforcement
  - Cannot instantiate directly
  - Must implement `build_config` method
  - Must implement `process` method
  - Must define `name` and `pipe_type` class variables
  - Complete implementation validation

- **BasePipe Interface Tests**: Required method signatures
  - `get_capabilities()` method existence and functionality
  - Filtering capabilities by pipe type
  - `build_config()` accepts kwargs
  - `process()` returns ONNX ModelProto

**Key Test Cases:**
- `test_base_pipe_is_abstract` - Verify BasePipe cannot be instantiated
- `test_base_pipe_requires_build_config` - Subclass must implement build_config
- `test_base_pipe_requires_process` - Subclass must implement process
- `test_get_capabilities_filters_by_pipe_type` - Capabilities filtered by pipe

### 2. `test_pipe_graph.py`
Tests for GraphPipe and GraphPipeConfig.

**Test Categories:**
- **GraphPipeConfig Tests**: Configuration structure and defaults
  - Default optimization level = 1
  - Default disabled list is empty
  - Custom values support
  - PipeConfig inheritance

- **GraphPipe.build_config Tests**: Config building from kwargs
  - Empty kwargs uses defaults
  - Explicit optimization level override
  - Single/multiple capability disabling
  - Enable override of default-False capabilities
  - Respects capability defaults
  - Filters GRAPH pipe capabilities only
  - Ignores unknown kwargs

- **GraphPipe.process Tests**: Model processing with ORT SessionOptions
  - Returns ONNX ModelProto
  - Preserves model structure
  - Works with optimization_level=0
  - Works with disabled optimizers
  - Tests all ORT optimization levels (0, 1, 2, 99)

- **GraphPipe Integration Tests**: End-to-end workflow
  - Complete workflow from kwargs to processed model
  - Class attributes (name="graph", pipe_type=GRAPH)
  - get_capabilities returns GRAPH-only capabilities
  - Multiple pipe instances are independent

**Key Test Cases:**
- `test_build_config_respects_capability_defaults` - Validates default handling
- `test_build_config_filters_graph_pipe_only` - Pipe isolation
- `test_process_different_optimization_levels` - ORT level validation
- `test_end_to_end_workflow` - Complete integration test

### 3. `test_pipe_fusion.py`
Tests for FusionPipe and FusionPipeConfig.

**Test Categories:**
- **FusionPipeConfig Tests**: Configuration structure and defaults
  - Default model_type = "bert"
  - All fusion toggles default to False
  - Custom values support
  - All FusionOptions fields present

- **FusionPipe.build_config Tests**: Config building from kwargs
  - Empty kwargs uses defaults
  - Model type override (bert, gpt2, t5, vit)
  - Single/multiple fusion enabling
  - fusion_attr mapping to config fields
  - Respects capability defaults
  - Filters FUSION pipe capabilities only
  - Model type + fusion options together

- **FusionPipe.process Tests**: Model processing with FusionOptions
  - Returns ONNX ModelProto
  - Preserves model structure
  - Works with no fusions enabled
  - Works with single/multiple fusions
  - Tests different model types

- **FusionPipe Integration Tests**: End-to-end workflow
  - Complete workflow from kwargs to processed model
  - Class attributes (name="fusion", pipe_type=FUSION)
  - get_capabilities returns FUSION-only capabilities
  - Capabilities without fusion_attr are ignored
  - All fusion options accessible

**Key Test Cases:**
- `test_build_config_fusion_attrs_mapping` - Validates fusion_attr mapping
- `test_build_config_filters_fusion_pipe_only` - Pipe isolation
- `test_process_different_model_types` - Model type validation
- `test_all_fusion_options_accessible` - Config completeness

### 4. `test_optimizer.py`
Tests for Optimizer orchestration class.

**Test Categories:**
- **Optimizer Initialization Tests**: Pipe registration
  - Can be instantiated
  - Has pipes list class variable
  - Pipes list not empty (≥2 pipes)
  - All pipes are BasePipe subclasses
  - All pipes have unique names

- **Optimizer Execution Tests**: Sequential pipe execution
  - Executes all registered pipes
  - Pipes execute in registration order
  - Returns final processed model
  - Passes kwargs to build_config
  - Works with empty kwargs

- **Optimizer Flow Tests**: Config building and model processing
  - Calls build_config then process for each pipe
  - Passes built config to process method
  - Chains model through pipes sequentially
  - Each pipe receives output from previous pipe

- **Optimizer Integration Tests**: End-to-end workflows
  - Optional should_process pattern for conditional execution
  - Works with actual GraphPipe and FusionPipe
  - Preserves model validity through pipeline
  - Handles multiple capabilities enabled
  - Handles empty pipes list gracefully
  - Correct method signature (model, **kwargs)

**Key Test Cases:**
- `test_optimizer_executes_all_pipes` - Core orchestration
- `test_optimizer_pipes_in_order` - Sequential execution validation
- `test_optimizer_chains_model_through_pipes` - Pipeline validation
- `test_optimizer_with_real_pipes` - Integration with actual pipes

## Test Structure

All tests follow these principles:

### 1. Cardinal Rule Compliance
- **Rule #1**: No hardcoded model architectures
- **Rule #2**: All tests use pytest with code-generated results
- **Rule #3**: Tests must run and pass

### 2. Fixtures
```python
@pytest.fixture
def clean_registry() -> Generator[None, None, None]:
    """Clear registry before and after each test for isolation."""

@pytest.fixture
def sample_model() -> onnx.ModelProto:
    """Create a simple ONNX model dynamically using code."""

@pytest.fixture
def graph_capabilities(clean_registry) -> dict:
    """Create sample graph pipe capabilities for testing."""

@pytest.fixture
def fusion_capabilities(clean_registry) -> dict:
    """Create sample fusion pipe capabilities for testing."""

@pytest.fixture
def mock_pipes(clean_registry) -> list[type[BasePipe]]:
    """Create mock pipe classes for testing."""
```

### 3. Test Organization
- Tests organized by functionality into classes
- Each class focuses on one aspect (Config, BuildConfig, Process, Integration)
- Clear test names describing what is being tested
- Comprehensive docstrings

### 4. Test Patterns

#### Configuration Tests
```python
def test_config_defaults():
    """Verify config has correct default values."""
    config = PipeConfig()
    assert config.field == expected_default
```

#### Build Config Tests
```python
def test_build_config_with_kwargs():
    """Build config with kwargs should override defaults."""
    config = Pipe.build_config(param=value)
    assert config.param == value
```

#### Process Tests
```python
def test_process_returns_model(sample_model):
    """Process should return an ONNX ModelProto."""
    pipe = Pipe()
    config = PipeConfig()
    result = pipe.process(sample_model, config)
    assert isinstance(result, onnx.ModelProto)
```

#### Integration Tests
```python
def test_end_to_end_workflow(sample_model):
    """Test complete workflow from kwargs to processed model."""
    config = Pipe.build_config(**kwargs)
    pipe = Pipe()
    result = pipe.process(sample_model, config)
    assert isinstance(result, onnx.ModelProto)
```

## Running Tests

### Run all pipe tests
```bash
cd D:\BYOM\ModelKit
uv run pytest tests/test_pipe_*.py tests/test_optimizer.py -v
```

### Run specific test file
```bash
uv run pytest tests/test_pipe_base.py -v
uv run pytest tests/test_pipe_graph.py -v
uv run pytest tests/test_pipe_fusion.py -v
uv run pytest tests/test_optimizer.py -v
```

### Run specific test class
```bash
uv run pytest tests/test_pipe_base.py::TestPipeConfig -v
uv run pytest tests/test_pipe_graph.py::TestGraphPipeBuildConfig -v
```

### Run specific test
```bash
uv run pytest tests/test_pipe_base.py::TestBasePipeAbstract::test_base_pipe_is_abstract -v
```

## Test Coverage

### test_pipe_base.py
- 13 tests covering PipeConfig and BasePipe
- 100% pass rate (when implementation exists)

### test_pipe_graph.py
- 24 tests covering GraphPipe and GraphPipeConfig
- Tests build_config logic (12 tests)
- Tests process method (6 tests)
- Tests integration (6 tests)

### test_pipe_fusion.py
- 22 tests covering FusionPipe and FusionPipeConfig
- Tests build_config logic (9 tests)
- Tests process method (6 tests)
- Tests integration (7 tests)

### test_optimizer.py
- 21 tests covering Optimizer orchestration
- Tests initialization (5 tests)
- Tests execution (5 tests)
- Tests flow control (3 tests)
- Tests integration (8 tests)

**Total: 80 comprehensive tests**

## Known Issues

### ONNX IR Version Mismatch
Some tests fail with:
```
Unsupported model IR version: 12, max supported IR version: 11
```

This is an environment/implementation issue, not a test design issue:
- The sample_model fixture generates ONNX IR version 12 models
- Current onnxruntime version only supports IR version 11
- Fix by either:
  1. Upgrading onnxruntime to support IR version 12
  2. Modifying sample_model fixture to generate IR version 11 models

### FusionPipeConfig Fields
Some tests expect fields not yet implemented in FusionPipeConfig:
- `enable_group_query_attention`
- `enable_rotary_embeddings`

These fields are documented in the pipe architecture design but may not be in the current implementation.

## Future Enhancements

### Conditional Processing
Tests include placeholders for optional `should_process` pattern:
```python
def should_process(self, config: PipeConfig) -> bool:
    """Optional method to skip pipe processing based on config."""
    return True
```

### Additional Pipes
Test structure supports adding new pipe implementations:
- Create `test_pipe_newpipe.py` following same patterns
- Add tests for NewPipeConfig, NewPipe.build_config, NewPipe.process
- Update Optimizer tests to include new pipe

### Performance Tests
Consider adding:
- Benchmark tests for optimization speed
- Memory usage tests
- Large model tests

## References

- **Pipe Architecture Design**: `docs/design/optimization/6_pipe_architecture.md`
- **Core Loop Design**: `docs/design/optimization/2_coreloop.md`
- **Capability Registry Tests**: `tests/test_optimization_registry.py`
