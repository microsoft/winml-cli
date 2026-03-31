# Complete Pytest Best Practices Guide (2025)

A comprehensive guide covering all aspects of pytest, from basic usage to advanced patterns and project organization.

## Table of Contents

1. [Project Structure & Organization](#project-structure--organization)
2. [Test Discovery & Naming Conventions](#test-discovery--naming-conventions)
3. [Fixtures: The Heart of Pytest](#fixtures-the-heart-of-pytest)
4. [Markers & Test Categorization](#markers--test-categorization)
5. [Parametrization: Data-Driven Testing](#parametrization-data-driven-testing)
6. [Assertions & Error Handling](#assertions--error-handling)
7. [Configuration & Settings](#configuration--settings)
8. [Conftest.py: Shared Test Logic](#conftest-py-shared-test-logic)
9. [Mocking & Monkeypatching](#mocking--monkeypatching)
10. [Database Testing Patterns](#database-testing-patterns)
11. [Performance & Optimization](#performance--optimization)
12. [CI/CD Integration](#cicd-integration)
13. [Plugin Ecosystem](#plugin-ecosystem)
14. [Snapshot & Regression Testing](#snapshot--regression-testing)
15. [Property-Based Testing with Hypothesis](#property-based-testing-with-hypothesis)
16. [Test Asset Generation & Management](#test-asset-generation--management)
17. [Common Patterns & Anti-Patterns](#common-patterns--anti-patterns)
18. [Debugging & Troubleshooting](#debugging--troubleshooting)
19. [Best Practices Checklist](#best-practices-checklist)

---

## Project Structure & Organization

### Recommended Layout

```
project/
├── src/                        # Source code
│   └── myproject/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── api/
│           ├── __init__.py
│           └── endpoints.py
├── tests/                      # Test directory
│   ├── __init__.py            # Makes tests a package (optional - see note below)
│   ├── conftest.py            # Shared fixtures and configuration
│   ├── unit/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── test_engine.py
│   │   └── test_helpers.py
│   ├── integration/           # Integration tests
│   │   ├── __init__.py
│   │   └── test_api.py
│   ├── e2e/                   # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_workflows.py
│   └── fixtures/              # Shared test data/utilities
│       ├── __init__.py
│       └── test_data.py
├── pyproject.toml            # Modern Python project config (preferred)
├── pytest.ini                 # Legacy pytest configuration (avoid)
├── .coveragerc               # Coverage configuration
└── tox.ini                   # Multiple environment testing
```

### Key Principles

1. **Mirror Source Structure**: Test directory structure should mirror your source code
2. **Separate Test Types**: Keep unit, integration, and e2e tests in separate directories
3. **`__init__.py` in Tests**: Optional - use only when you need to import between test modules (see detailed explanation below)
4. **Centralize Fixtures**: Use `conftest.py` for shared fixtures

### Should You Use `__init__.py` in Test Directories?

The use of `__init__.py` in test directories is **optional** and depends on your specific needs:

#### When to USE `__init__.py` in tests ✅

1. **Cross-test imports**: When you need to import helper functions or classes between test modules
   ```python
   # tests/unit/test_user.py
   from tests.helpers.factories import UserFactory  # Requires __init__.py
   ```

2. **Test utilities as a package**: When you have reusable test utilities that need to be imported
   ```
   tests/
   ├── __init__.py
   ├── helpers/
   │   ├── __init__.py
   │   ├── factories.py
   │   └── assertions.py
   ```

3. **Namespace packages**: When you need to avoid naming conflicts with application modules
   ```python
   # Disambiguates tests.models from myapp.models
   from tests.models import TestUser
   from myapp.models import User
   ```

#### When NOT to use `__init__.py` in tests ❌

1. **Simple test structures**: Most projects don't need it - pytest discovers tests without it
2. **Import mode conflicts**: Can cause issues with pytest's import mechanisms
3. **Accidental test collection**: May cause pytest to collect non-test files

#### Best Practice Recommendation

**Default approach**: Start WITHOUT `__init__.py` in test directories. Only add it when you have a specific need for cross-test imports or test utilities.

```
# Recommended minimal structure
tests/
├── conftest.py          # Shared fixtures (no __init__.py needed)
├── unit/
│   └── test_models.py   # Tests work without __init__.py
└── integration/
    └── test_api.py
```

#### pytest.ini Configuration for Import Issues

If you encounter import issues, configure pytest's import mode instead of adding `__init__.py`:

```ini
# pytest.ini
[pytest]
# Use importlib mode for better import handling
import_mode = importlib

# Or use prepend mode (default)
import_mode = prepend
```

### Alternative Layouts

#### Tests Outside Application Code (Recommended)
```
project/
├── src/myproject/
└── tests/
```

#### Tests as Part of Application (Less Common)
```
project/
└── myproject/
    ├── core/
    │   ├── engine.py
    │   └── tests/
    │       └── test_engine.py
    └── utils/
        ├── helpers.py
        └── tests/
            └── test_helpers.py
```

---

## Test Discovery & Naming Conventions

### Default Discovery Rules

Pytest automatically discovers tests following these patterns:

- **Test files**: `test_*.py` or `*_test.py`
- **Test classes**: `Test*` (must not have an `__init__` method)
- **Test functions**: `test_*`
- **Test methods**: `test_*` inside `Test*` classes

### Naming Best Practices

```python
# ❌ Bad: Unclear test names
def test_1():
    pass

def test_user():
    pass

def test_function():
    pass

# ✅ Good: Descriptive test names
def test_user_creation_with_valid_email():
    """Test that a user can be created with a valid email address."""
    pass

def test_user_creation_fails_with_duplicate_email():
    """Test that creating a user with an existing email raises an error."""
    pass

def test_password_reset_sends_email_to_registered_user():
    """Test that password reset email is sent to registered users."""
    pass
```

### Test Class Organization

```python
class TestUserAuthentication:
    """Test cases for user authentication functionality."""

    def test_login_with_valid_credentials_returns_token(self):
        """Test successful login returns authentication token."""
        pass

    def test_login_with_invalid_password_returns_401(self):
        """Test login with wrong password returns 401 status."""
        pass

    def test_login_with_nonexistent_user_returns_404(self):
        """Test login with non-existent user returns 404 status."""
        pass
```

### Custom Discovery Configuration

```ini
# pytest.ini
[pytest]
# Custom patterns for test discovery
python_files = test_*.py check_*.py
python_classes = Test* Check*
python_functions = test_* check_*

# Ignore specific directories
norecursedirs = .git .tox build dist *.egg
```

---

## Fixtures: The Heart of Pytest

### Basic Fixture Concepts

```python
import pytest

# Simple fixture
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"name": "John", "age": 30}

# Fixture with teardown
@pytest.fixture
def database_connection():
    """Create database connection and clean up after test."""
    conn = create_connection()
    yield conn  # This is where the test runs
    conn.close()  # Teardown happens after test

# Using fixtures in tests
def test_user_data(sample_data):
    assert sample_data["name"] == "John"
```

### Fixture Scopes

```python
# Function scope (default) - run once per test function
@pytest.fixture(scope="function")
def function_resource():
    return expensive_setup()

# Class scope - run once per test class
@pytest.fixture(scope="class")
def class_resource():
    return expensive_setup()

# Module scope - run once per module
@pytest.fixture(scope="module")
def module_resource():
    return expensive_setup()

# Session scope - run once per test session
@pytest.fixture(scope="session")
def session_resource():
    return expensive_setup()

# Package scope - run once per package
@pytest.fixture(scope="package")
def package_resource():
    return expensive_setup()
```

### Advanced Fixture Patterns

#### Factory Fixtures
```python
@pytest.fixture
def make_user():
    """Factory fixture for creating users."""
    created_users = []

    def _make_user(name, email=None):
        user = User(name=name, email=email or f"{name}@example.com")
        created_users.append(user)
        return user

    yield _make_user

    # Cleanup all created users
    for user in created_users:
        user.delete()

def test_user_interactions(make_user):
    alice = make_user("alice")
    bob = make_user("bob", "bob@company.com")
    assert alice.can_message(bob)
```

#### Parametrized Fixtures
```python
@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database(request):
    """Test with multiple database backends."""
    return setup_database(request.param)

def test_query_performance(database):
    # This test runs three times, once for each database
    result = database.execute("SELECT * FROM users")
    assert result.execution_time < 100  # ms
```

#### Dynamic Fixture Scope
```python
def determine_scope(fixture_name, config):
    """Dynamically determine fixture scope based on config."""
    if config.getoption("--quick", None):
        return "session"  # Reuse fixtures for speed
    return "function"    # Fresh fixtures for isolation

@pytest.fixture(scope=determine_scope)
def api_client():
    return APIClient()
```

#### Fixture Dependencies
```python
@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def database(config):
    return Database(config["db_url"])

@pytest.fixture
def api_client(config, database):
    # Fixtures can depend on other fixtures
    return APIClient(config["api_url"], database)
```

### Auto-use Fixtures

```python
@pytest.fixture(autouse=True)
def reset_global_state():
    """Automatically run before each test without explicit request."""
    clear_caches()
    reset_singletons()
    yield
    # Cleanup happens after test

@pytest.fixture(autouse=True, scope="session")
def configure_test_environment():
    """Set up test environment once for entire session."""
    os.environ["TESTING"] = "true"
    configure_logging("debug")
```

### Fixture Finalization

```python
@pytest.fixture
def resource_with_finalizer(request):
    """Using request.addfinalizer for cleanup."""
    resource = acquire_resource()

    def cleanup():
        release_resource(resource)

    request.addfinalizer(cleanup)
    return resource

# Equivalent using yield
@pytest.fixture
def resource_with_yield():
    """Using yield for cleanup (preferred)."""
    resource = acquire_resource()
    yield resource
    release_resource(resource)
```

---

## Markers & Test Categorization

### Built-in Markers

```python
import pytest
import sys

# Skip marker
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Conditional skip
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_pattern_matching():
    match value:
        case 1: return "one"
        case _: return "other"

# Expected failure
@pytest.mark.xfail(reason="Known bug #123")
def test_known_issue():
    assert buggy_function() == expected_value

# Strict xfail - fails if test passes
@pytest.mark.xfail(strict=True, reason="Should be fixed in v2.0")
def test_upcoming_fix():
    assert new_feature() == expected

# Platform-specific tests
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only test")
def test_linux_specific():
    pass

# Import skip
def test_optional_dependency():
    numpy = pytest.importorskip("numpy", minversion="1.20.0")
    # Test only runs if numpy >= 1.20.0 is available
```

### Custom Markers

```ini
# pytest.ini - Register custom markers
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    smoke: core functionality that must always work
    integration: requires external services
    unit: fast isolated unit tests
    flaky: tests that occasionally fail
    requires_db: tests that need database access
    requires_network: tests that need network access
```

```python
# Using custom markers
@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow():
    """Test complete user workflow with external services."""
    pass

@pytest.mark.smoke
def test_critical_functionality():
    """Test that must always pass."""
    pass

# Multiple markers
@pytest.mark.unit
@pytest.mark.smoke
def test_core_logic():
    """Fast unit test for critical functionality."""
    pass
```

### Marker Expressions

```bash
# Run only smoke tests
pytest -m smoke

# Run all tests except slow ones
pytest -m "not slow"

# Complex expressions
pytest -m "smoke and not slow"
pytest -m "(unit or integration) and not flaky"

# List all markers
pytest --markers
```

### Applying Markers Dynamically

```python
# In conftest.py
def pytest_collection_modifyitems(items):
    """Dynamically add markers during collection."""
    for item in items:
        # Add marker based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add marker based on test name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
```

---

## Parametrization: Data-Driven Testing

### Basic Parametrization

```python
import pytest

# Single parameter
@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_square(number):
    assert number ** 2 == number * number

# Multiple parameters
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (-2, 4),
])
def test_square_with_expected(input, expected):
    assert input ** 2 == expected

# Using test IDs for better output
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (-2, 4),
], ids=["positive_2", "positive_3", "negative_2"])
def test_square_with_ids(input, expected):
    assert input ** 2 == expected

# ID function
def idfn(val):
    return f"num_{val}"

@pytest.mark.parametrize("number", [1, 2, 3], ids=idfn)
def test_with_id_function(number):
    assert number > 0
```

### Advanced Parametrization

```python
# Nested parametrization
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiplication(x, y):
    # Runs 4 times: (1,10), (1,20), (2,10), (2,20)
    assert x * y == y * x

# Parametrize with marks
@pytest.mark.parametrize("test_input,expected", [
    ("3+5", 8),
    ("2+4", 6),
    pytest.param("6*9", 42, marks=pytest.mark.xfail(reason="Hitchhiker's joke")),
    pytest.param("1/0", 0, marks=pytest.mark.skip(reason="Division by zero")),
])
def test_eval(test_input, expected):
    assert eval(test_input) == expected

# Indirect parametrization (parametrize fixtures)
@pytest.mark.parametrize("db_name", ["sqlite", "postgres"], indirect=True)
def test_database_operations(db_name):
    # db_name fixture receives the parameter value
    assert db_name.connect()
```

### Parametrization Patterns

```python
# Test class parametrization
@pytest.mark.parametrize("browser", ["chrome", "firefox", "safari"])
class TestWebApplication:
    def test_login(self, browser):
        # Each test method runs with each browser
        pass

    def test_search(self, browser):
        pass

# Dynamic parametrization
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests."""
    if "dynamic_value" in metafunc.fixturenames:
        values = load_test_values_from_file()
        metafunc.parametrize("dynamic_value", values)

# Parametrization from fixtures
@pytest.fixture(params=["admin", "user", "guest"])
def user_role(request):
    return create_user_with_role(request.param)

def test_permissions(user_role):
    # Test runs for each user role
    assert user_role.can_access("/dashboard") == user_role.is_admin
```

---

## Assertions & Error Handling

### Enhanced Assertions

```python
# Pytest rewrites assert statements for better output
def test_assertion_introspection():
    data = {"name": "Alice", "items": [1, 2, 3]}
    # Pytest shows detailed diff on failure
    assert data == {"name": "Bob", "items": [1, 2, 3]}

# Custom assertion messages
def test_with_message():
    result = complex_calculation()
    assert result > 0, f"Expected positive result, got {result}"
```

### Exception Testing

```python
import pytest

# Basic exception testing
def test_raises_exception():
    with pytest.raises(ValueError):
        raise ValueError("Invalid value")

# Check exception message
def test_exception_message():
    with pytest.raises(ValueError, match="Invalid.*value"):
        raise ValueError("Invalid value provided")

# Access exception info
def test_exception_info():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("test error")

    assert str(exc_info.value) == "test error"
    assert exc_info.type == ValueError

# Test multiple exceptions (ExceptionGroup)
def test_exception_group():
    with pytest.raises(ExceptionGroup) as exc_info:
        raise ExceptionGroup("errors", [
            ValueError("error 1"),
            TypeError("error 2")
        ])

    assert len(exc_info.value.exceptions) == 2
```

### Warning Testing

```python
import warnings
import pytest

def test_warns():
    with pytest.warns(UserWarning):
        warnings.warn("This is a warning", UserWarning)

def test_warns_with_match():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        warnings.warn("This function is deprecated", DeprecationWarning)

def test_no_warnings():
    # Ensure no warnings are raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean_function()  # Should not raise any warnings
```

### Approximate Comparisons

```python
import pytest

def test_float_comparison():
    assert 0.1 + 0.2 == pytest.approx(0.3)

def test_list_approximate():
    assert [0.1 + 0.2, 0.2 + 0.4] == pytest.approx([0.3, 0.6])

def test_dict_approximate():
    assert {"a": 0.1 + 0.2} == pytest.approx({"a": 0.3})

# Custom tolerance
def test_custom_tolerance():
    assert 1.0001 == pytest.approx(1.0, rel=1e-3)
    assert 1.0001 == pytest.approx(1.0, abs=1e-3)
```

---

## Configuration & Settings

### Configuration File Priority (Critical Knowledge)

Understanding configuration file priority is essential for debugging pytest configuration issues.

**Priority Order** (first match wins - configurations are NEVER merged):

| Priority | File | Notes |
|----------|------|-------|
| 1 (Highest) | `pytest.toml` / `.pytest.toml` | New in pytest 9.0, native TOML |
| 2 | `pytest.ini` / `.pytest.ini` | Classic pytest config |
| 3 | `pyproject.toml` | Modern Python project standard |
| 4 | `tox.ini` | Tox integration |
| 5 (Lowest) | `setup.cfg` | Legacy, not recommended |

> ⚠️ **Critical Gotcha**: If an empty `pytest.ini` file exists in your project, ALL settings in `pyproject.toml` will be ignored! This is a common source of confusion. Delete any empty `pytest.ini` files.

**Configuration Sections by File Type**:

| File Type | Section Name |
|-----------|--------------|
| pytest.ini | `[pytest]` |
| pyproject.toml (pytest 6.0-8.x) | `[tool.pytest.ini_options]` |
| pyproject.toml (pytest 9.0+) | `[tool.pytest]` |
| tox.ini | `[pytest]` |
| setup.cfg | `[tool:pytest]` |

**Best Practice**: Use `pyproject.toml` as your single source of truth for all Python tooling configuration (pytest, ruff, mypy, etc.).

### pyproject.toml Configuration (Recommended)

Using `pyproject.toml` is the modern, preferred approach for Python project configuration. It consolidates all project metadata and tool configurations in one place.

```toml
# pyproject.toml
[tool.pytest.ini_options]
# Minimum pytest version
minversion = "7.0"

# Default command line options
addopts = [
    "--strict-markers",      # Fail on unknown markers
    "--strict-config",       # Fail on config errors
    "--import-mode=importlib",  # Use standard import system (recommended)
    "--verbose",             # Verbose output
    "-ra",                   # Show all test outcomes
    "--cov=myproject",       # Coverage for your project
    "--cov-report=html",     # HTML coverage report
    "--cov-report=term-missing",  # Terminal report with missing lines
]

> 💡 **Recommended**: Always include `--import-mode=importlib` in your `addopts`. This uses Python's standard import system instead of modifying `sys.path`, avoiding common import issues. This has been the default since pytest 6.0 but explicitly setting it ensures consistent behavior.

# Test discovery
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*", "*Tests"]
python_functions = ["test_*"]

# Python path configuration
pythonpath = ["src"]

# Import mode (importlib is recommended for most projects)
import_mode = "importlib"

# Custom markers registration
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: requires external services",
    "unit: fast isolated unit tests",
    "smoke: core functionality that must always work",
    "flaky: tests that occasionally fail",
    "requires_network: tests that need network access",
]

# Output configuration
console_output_style = "progress"

# Directories to ignore
norecursedirs = [".git", ".tox", "dist", "build", "*.egg", "__pycache__"]

# Logging configuration
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

# Warning filters
filterwarnings = [
    "error",                          # Turn warnings into errors
    "ignore::UserWarning",            # Ignore user warnings
    "ignore::DeprecationWarning",     # Ignore deprecation warnings
    "default:.*deprecated.*:DeprecationWarning",  # Show deprecation warnings with "deprecated" in message
]

# Required plugins
required_plugins = [
    "pytest-cov>=4.0",
]

# Test timeout (requires pytest-timeout)
timeout = 300
timeout_method = "thread"

# Strict xfail
xfail_strict = true

# Asyncio configuration (requires pytest-asyncio)
asyncio_mode = "auto"

# Coverage configuration (can also be in [tool.coverage])
[tool.coverage.run]
source = ["myproject"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/.venv/*",
    "*/migrations/*",
    "*/__pycache__/*",
    "*/.pytest_cache/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
]

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
```

### Complete pyproject.toml Example

Here's a complete `pyproject.toml` that includes project metadata along with pytest configuration:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "1.0.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-asyncio>=0.21.0",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/username/myproject"
Documentation = "https://myproject.readthedocs.io"
Repository = "https://github.com/username/myproject.git"
Issues = "https://github.com/username/myproject/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
# ... (configuration from above)

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]
include = '\.pyi?$'

[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Migration from pytest.ini to pyproject.toml

If you have an existing `pytest.ini`, here's how to migrate:

```ini
# OLD: pytest.ini
[pytest]
markers =
    slow: slow tests
testpaths = tests
```

Becomes:

```toml
# NEW: pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: slow tests",
]
testpaths = ["tests"]
```

### pytest 9.0+ Native TOML Configuration

Starting with pytest 9.0, you can use the native `[tool.pytest]` table which provides cleaner TOML syntax:

```toml
# pytest 9.0+ (native TOML arrays - cleaner syntax)
[tool.pytest]
minversion = "9.0"

# Test discovery
testpaths = ["tests"]
pythonpath = ["."]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
norecursedirs = [".git", ".tox", "dist", "build", ".venv", "__pycache__"]

# Command line options (native TOML arrays)
addopts = [
    "--strict-markers",
    "--strict-config",
    "--import-mode=importlib",
    "-ra",
    "--tb=short",
]

# Markers
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
]

# Warning filters
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]

# Required plugins
required_plugins = [
    "pytest-cov>=4.0",
]
```

**Benefits over `[tool.pytest.ini_options]`**:
- Native TOML array syntax (clearer than space-separated strings in some cases)
- Better TOML type support
- Future-proof configuration format
- Reserved by pytest team for enhanced features

**Migration**: Simply rename `[tool.pytest.ini_options]` to `[tool.pytest]` when upgrading to pytest 9.0+.

### Legacy pytest.ini (Not Recommended)

While `pytest.ini` still works, it's considered legacy. Use `pyproject.toml` instead for these benefits:
- Single configuration file for all Python tools
- Better IDE support
- TOML format is more readable
- Standardized by PEP 518 and PEP 621

### Command Line Configuration

```bash
# Common command line options
pytest -v                    # Verbose output
pytest -q                    # Quiet output
pytest -s                    # No capture, show print statements
pytest -x                    # Stop on first failure
pytest --maxfail=3          # Stop after 3 failures
pytest -k "user"            # Run tests matching "user"
pytest -m "not slow"        # Run tests not marked as slow
pytest --lf                 # Run last failed tests
pytest --ff                 # Run failed tests first
pytest --tb=short           # Short traceback format
pytest --tb=no              # No traceback
pytest --setup-show         # Show fixture setup/teardown
pytest --fixtures           # Show available fixtures
pytest --markers            # Show available markers
pytest --collect-only       # Only collect tests, don't run
pytest --cache-clear        # Clear cache before run
pytest --doctest-modules    # Run doctests
pytest --cov=myproject      # Coverage report
pytest --cov-report=html    # HTML coverage report
pytest --durations=10       # Show 10 slowest tests
pytest --pdb                # Drop to debugger on failure
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb  # Use IPython debugger
```

---

## Conftest.py: Shared Test Logic

### Fixture Sharing

```python
# tests/conftest.py - Available to all tests
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Shared test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

# tests/unit/conftest.py - Available to unit tests only
@pytest.fixture
def mock_database():
    """Mock database for unit tests."""
    return MockDatabase()

# tests/integration/conftest.py - Available to integration tests only
@pytest.fixture(scope="module")
def real_database():
    """Real database connection for integration tests."""
    db = Database()
    yield db
    db.cleanup()
```

### Hooks in conftest.py

```python
# Modify test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test file location
    for item in items:
        # Add markers based on location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Skip tests based on environment
        if "requires_gpu" in item.keywords and not has_gpu():
            item.add_marker(pytest.mark.skip(reason="GPU not available"))

# Custom command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )

# Configure based on options
def pytest_configure(config):
    """Configure pytest based on command line options."""
    if config.getoption("--run-slow"):
        config.option.markexpr = "slow"

# Custom markers registration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
```

#### Hook Execution Order Control

Control when your hooks run relative to other plugins:

```python
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Execute BEFORE other implementations."""
    # Priority operations here
    pass

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    """Execute AFTER other implementations."""
    # Cleanup or final modifications
    pass
```

#### Wrapper Hooks (Advanced)

Wrap other hook implementations for cross-cutting concerns:

```python
@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Wrap report generation for custom handling."""
    # Code before other hooks run
    outcome = yield  # Run wrapped hooks
    report = outcome.get_result()

    # Code after - modify or log report
    if report.when == "call" and report.failed:
        # Handle test failure
        log_failure(item.nodeid, report.longreprtext)

    return report

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    """Wrap setup with timing."""
    start = time.time()
    yield  # Run actual setup
    duration = time.time() - start
    item.setup_duration = duration
```

#### Storing Data Across Hooks

Use `item.stash` for type-safe data storage:

```python
from pytest import StashKey

# Define typed keys
phase_report_key = StashKey[dict]()
timing_key = StashKey[float]()

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Store reports for fixture access."""
    outcome = yield
    report = outcome.get_result()

    # Store in stash (type-safe)
    item.stash.setdefault(phase_report_key, {})[report.when] = report
    return report

@pytest.fixture
def test_outcome(request):
    """Fixture to access test outcome."""
    yield
    report = request.node.stash.get(phase_report_key, {}).get("call")
    if report and report.failed:
        # Handle failure in fixture teardown
        pass
```

#### Custom Report Sections

Add extra information to test reports:

```python
@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Add custom sections to report
    if report.when == "call":
        report.sections.append(
            ("Custom Info", f"Test: {item.nodeid}\nDuration: {call.duration:.2f}s")
        )

    return report
```

### Plugin Registration

```python
# Register external plugins
pytest_plugins = [
    "myproject.testing.fixtures",
    "myproject.testing.helpers",
]

# Conditional plugin loading
import sys
if sys.platform.startswith("win"):
    pytest_plugins.append("myproject.testing.windows")
```

---

## Mocking & Monkeypatching

### Using pytest-mock

```python
# Install: pip install pytest-mock

def test_with_mock(mocker):
    """Using pytest-mock plugin."""
    # Mock a module function
    mock_func = mocker.patch("mymodule.function")
    mock_func.return_value = 42

    # Mock an object method
    mock_method = mocker.patch.object(MyClass, "method")
    mock_method.return_value = "mocked"

    # Spy on a function
    spy = mocker.spy(mymodule, "function")
    mymodule.function()
    spy.assert_called_once()

# Using side effects
def test_side_effects(mocker):
    mock = mocker.patch("mymodule.function")
    mock.side_effect = [1, 2, 3]  # Returns different values each call

    assert mymodule.function() == 1
    assert mymodule.function() == 2
    assert mymodule.function() == 3

# Mock with exceptions
def test_mock_exception(mocker):
    mock = mocker.patch("mymodule.function")
    mock.side_effect = ValueError("Error!")

    with pytest.raises(ValueError):
        mymodule.function()
```

### Monkeypatch

```python
def test_monkeypatch_env(monkeypatch):
    """Monkeypatch environment variables."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.delenv("OLD_VAR", raising=False)

    assert os.environ["API_KEY"] == "test-key"
    assert "OLD_VAR" not in os.environ

def test_monkeypatch_attribute(monkeypatch):
    """Monkeypatch object attributes."""
    class MyClass:
        value = 10

    obj = MyClass()
    monkeypatch.setattr(obj, "value", 20)
    assert obj.value == 20

def test_monkeypatch_module(monkeypatch):
    """Monkeypatch module functions."""
    import time

    def mock_time():
        return 123456.0

    monkeypatch.setattr(time, "time", mock_time)
    assert time.time() == 123456.0

def test_monkeypatch_dict(monkeypatch):
    """Monkeypatch dictionary items."""
    config = {"url": "production.com"}
    monkeypatch.setitem(config, "url", "test.com")
    assert config["url"] == "test.com"
```

### Advanced Mocking Patterns

```python
# Context manager mocking
def test_context_manager(mocker):
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__.return_value = "resource"
    mock_cm.__exit__.return_value = None

    mocker.patch("mymodule.get_resource", return_value=mock_cm)

    with mymodule.get_resource() as resource:
        assert resource == "resource"

    mock_cm.__enter__.assert_called_once()
    mock_cm.__exit__.assert_called_once()

# Property mocking
def test_property_mock(mocker):
    mock_property = mocker.PropertyMock(return_value=42)
    mocker.patch("mymodule.MyClass.my_property", new_callable=mock_property)

    obj = mymodule.MyClass()
    assert obj.my_property == 42
    mock_property.assert_called_once()

# Async mocking
async def test_async_mock(mocker):
    mock_async = mocker.AsyncMock(return_value="async result")
    mocker.patch("mymodule.async_function", mock_async)

    result = await mymodule.async_function()
    assert result == "async result"
    mock_async.assert_awaited_once()
```

---

## Database Testing Patterns

Testing database interactions requires careful isolation and cleanup strategies.

### Transaction-Based Isolation

The most reliable approach is rolling back transactions after each test:

```python
import pytest

@pytest.fixture
def db_session(engine):
    """Create a transactional test session."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

def test_user_creation(db_session):
    """Test runs in transaction that gets rolled back."""
    user = User(name="test")
    db_session.add(user)
    db_session.flush()

    assert user.id is not None
    # Transaction rolled back - no cleanup needed
```

### pytest-django Database Access

```python
import pytest

# Mark test to enable database access
@pytest.mark.django_db
def test_user_creation():
    User.objects.create(username="testuser")
    assert User.objects.count() == 1

# Transaction testing (for testing transaction behavior)
@pytest.mark.django_db(transaction=True)
def test_atomic_operations():
    with transaction.atomic():
        User.objects.create(username="user1")
        # Test atomic behavior

# Multiple database support
@pytest.mark.django_db(databases=["default", "secondary"])
def test_multi_db():
    User.objects.using("secondary").create(username="remote_user")
```

### Database Blocker Pattern

Control database access at fixture level:

```python
@pytest.fixture
def setup_data(django_db_blocker):
    """Fixture that needs temporary DB access."""
    with django_db_blocker.unblock():
        # Database operations allowed here
        User.objects.create(username="fixture_user")
    # Database blocked again outside context

@pytest.fixture
def no_db_fixture(django_db_blocker):
    """Ensure no accidental DB access."""
    with django_db_blocker.block():
        yield  # DB access will raise error
```

### Query Count Assertions

Prevent N+1 query issues:

```python
def test_efficient_queries(django_assert_num_queries):
    """Assert exact number of queries."""
    with django_assert_num_queries(3):
        list(User.objects.all())
        list(Post.objects.all())
        list(Comment.objects.all())

def test_max_queries(django_assert_max_num_queries):
    """Assert maximum query count."""
    with django_assert_max_num_queries(5):
        # Complex operation that should be efficient
        process_users()
```

### SQLAlchemy Testing Patterns

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    return create_engine("sqlite:///:memory:")

@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables):
    """Create a new database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
```

### Factory Pattern for Test Data

```python
import pytest
from factory import Factory, Faker, SubFactory

class UserFactory(Factory):
    class Meta:
        model = User

    username = Faker("user_name")
    email = Faker("email")

class PostFactory(Factory):
    class Meta:
        model = Post

    title = Faker("sentence")
    author = SubFactory(UserFactory)

@pytest.fixture
def user_factory(db_session):
    """Factory fixture for creating test users."""
    def _create_user(**kwargs):
        user = UserFactory.build(**kwargs)
        db_session.add(user)
        db_session.flush()
        return user
    return _create_user

def test_user_posts(user_factory):
    author = user_factory(username="author")
    post = PostFactory.build(author=author)
    assert post.author.username == "author"
```

---

## Performance & Optimization

### Parallel Execution with pytest-xdist

```bash
# Install pytest-xdist
pip install pytest-xdist
```

#### Basic Usage

```bash
pytest -n auto          # Use all available CPUs
pytest -n 4             # Use 4 workers
pytest -n logical       # Use logical cores (requires psutil)
```

#### Distribution Strategies

Understanding distribution strategies is critical for efficient parallel testing:

```bash
# Load balancing (default) - distributes tests as workers become available
pytest -n auto --dist load

# Group by scope - keeps tests sharing fixtures on same worker (RECOMMENDED)
pytest -n auto --dist loadscope

# Group by file - all tests in a file run on same worker
pytest -n auto --dist loadfile

# Each test runs on every worker (for environment-specific testing)
pytest -n 2 --dist each
```

**When to Use Each Strategy**:

| Strategy | Use Case | Performance |
|----------|----------|-------------|
| `load` | Independent tests, no shared state | Best parallelization |
| `loadscope` | Tests sharing expensive fixtures | Balanced (recommended default) |
| `loadfile` | File-level isolation needed | Good for integration tests |
| `each` | Multi-environment testing | Multiplies test count |

#### Grouping Tests with xdist_group Marker

Force related tests to run on the same worker:

```python
import pytest

@pytest.mark.xdist_group(name="database")
def test_create_user():
    """Runs on same worker as other 'database' group tests."""
    db.create_user("alice")

@pytest.mark.xdist_group(name="database")
def test_query_user():
    """Guaranteed same worker as test_create_user."""
    user = db.get_user("alice")
    assert user is not None

@pytest.mark.xdist_group(name="api")
def test_api_endpoint():
    """Runs on potentially different worker."""
    pass
```

#### Session-Scoped Fixtures with Parallel Execution

Session-scoped fixtures require special handling in parallel execution to avoid race conditions:

```python
import json
from pathlib import Path
from filelock import FileLock  # pip install filelock

@pytest.fixture(scope="session")
def expensive_shared_data(tmp_path_factory, worker_id):
    """Thread-safe session fixture for parallel execution."""
    # Single worker mode - no synchronization needed
    if worker_id == "master":
        return generate_expensive_data()

    # Multi-worker mode - use file locking
    root_tmp = tmp_path_factory.getbasetemp().parent
    data_file = root_tmp / "shared_data.json"
    lock_file = str(data_file) + ".lock"

    with FileLock(lock_file):
        if data_file.is_file():
            # Another worker already created the data
            return json.loads(data_file.read_text())
        else:
            # First worker creates the data
            data = generate_expensive_data()
            data_file.write_text(json.dumps(data))
            return data

@pytest.fixture(scope="session")
def database_url(tmp_path_factory, worker_id):
    """Per-worker database for parallel isolation."""
    # Each worker gets its own database
    return f"sqlite:///test_db_{worker_id}.sqlite"
```

#### Configuration for Parallel Execution

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "-n", "auto",
    "--dist", "loadscope",
]
```

> ⚠️ **Warning**: Not all tests are parallelization-safe. Tests that modify global state, shared files, or external services may conflict. Use `xdist_group` or run such tests serially with `-n 0`.

### Test Duration Analysis

```python
# Show test durations
pytest --durations=10   # Show 10 slowest tests
pytest --durations=0    # Show all test durations

# In conftest.py - Custom timing
import time

@pytest.fixture(autouse=True)
def measure_test_time(request):
    start = time.time()
    yield
    duration = time.time() - start
    print(f"\n{request.node.name} took {duration:.2f}s")
```

### Caching

```python
# Using pytest cache
def test_expensive_computation(cache):
    # Check cache
    result = cache.get("computation_result", None)
    if result is None:
        # Compute and cache
        result = expensive_computation()
        cache.set("computation_result", result)

    assert result == expected_value

# Cache command line
pytest --cache-show     # Show cache contents
pytest --cache-clear    # Clear cache
```

### Fixture Optimization

```python
# Reuse expensive fixtures with broader scope
@pytest.fixture(scope="session")
def expensive_resource():
    """Create once, use many times."""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()

# Lazy fixture creation
@pytest.fixture
def maybe_expensive():
    """Only created if actually used by test."""
    return ExpensiveObject()

# Fixture factories for controlled creation
@pytest.fixture
def resource_factory():
    resources = []

    def _make_resource(**kwargs):
        resource = Resource(**kwargs)
        resources.append(resource)
        return resource

    yield _make_resource

    # Cleanup all at once
    for resource in resources:
        resource.cleanup()
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"

    - name: Run tests
      run: |
        pytest -v --cov=myproject --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Stages

```yaml
# Multi-stage testing
stages:
  - quick-tests
  - full-tests
  - integration-tests

quick-tests:
  script:
    - pytest -m "unit and not slow" --fail-fast

full-tests:
  script:
    - pytest -m "not integration"

integration-tests:
  script:
    - pytest -m integration
  only:
    - main
    - merge_requests
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = myproject
omit =
    */tests/*
    */venv/*
    */migrations/*
    */__init__.py

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

---

## Plugin Ecosystem

### Essential Plugins

```bash
# Coverage
pip install pytest-cov

# Parallel execution
pip install pytest-xdist

# Mocking
pip install pytest-mock

# Timeout
pip install pytest-timeout

# HTML reports
pip install pytest-html

# BDD
pip install pytest-bdd

# Benchmarking
pip install pytest-benchmark

# Django
pip install pytest-django

# Asyncio
pip install pytest-asyncio

# Flake8 integration
pip install pytest-flake8

# Order randomization
pip install pytest-randomly
```

### Plugin Usage Examples

```python
# pytest-timeout
@pytest.mark.timeout(10)  # 10 second timeout
def test_slow_operation():
    perform_slow_operation()

# pytest-benchmark
def test_performance(benchmark):
    result = benchmark(my_function, arg1, arg2)
    assert result == expected

# pytest-randomly (randomize test order)
# Just install and it works automatically
# Use --randomly-seed=1234 to reproduce order
```

### Async Testing with pytest-asyncio

#### Installation and Configuration

```bash
pip install pytest-asyncio
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Automatically handle async tests
```

#### Basic Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_operation()
    assert result == expected

@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager."""
    async with AsyncResource() as resource:
        result = await resource.fetch()
        assert result is not None
```

#### Async Fixtures

```python
@pytest.fixture
async def async_client():
    """Async fixture with proper cleanup."""
    client = await create_async_client()
    yield client
    await client.close()

@pytest.fixture(scope="session")
async def async_database():
    """Session-scoped async fixture."""
    db = await Database.connect()
    yield db
    await db.disconnect()

@pytest.mark.asyncio
async def test_with_async_fixtures(async_client, async_database):
    """Test using async fixtures."""
    result = await async_client.query(async_database)
    assert result is not None
```

#### Fixture Scopes for Async

```python
# Function scope (default) - new event loop per test
@pytest.fixture
async def function_resource():
    return await create_resource()

# Session scope - shared across tests
@pytest.fixture(scope="session")
async def session_resource():
    resource = await expensive_async_setup()
    yield resource
    await resource.cleanup()
```

> ⚠️ **Deprecation Warning**: Sync tests depending on async fixtures will warn in pytest 8.x and error in future versions. Always use `@pytest.mark.asyncio` for tests using async fixtures.

#### Event Loop Scope (pytest-asyncio 0.21+)

```python
# Control event loop scope
@pytest.fixture(scope="session")
def event_loop_policy():
    """Use uvloop for faster async."""
    import uvloop
    return uvloop.EventLoopPolicy()

# Or via configuration
# pyproject.toml
[tool.pytest.ini_options]
asyncio_default_fixture_loop_scope = "function"
```

---

## Snapshot & Regression Testing

Snapshot testing captures expected output and compares against future runs.

### Using syrupy (Recommended)

```bash
pip install syrupy
```

```python
def test_api_response(snapshot):
    """Compare API response against snapshot."""
    response = api_client.get("/users/1")
    assert response.json() == snapshot

def test_html_output(snapshot):
    """Compare rendered HTML."""
    html = render_template("user_profile.html", user=mock_user)
    assert html == snapshot

def test_complex_object(snapshot):
    """Snapshot complex data structures."""
    result = process_data(input_data)
    assert result == snapshot
```

### Snapshot Management

```bash
# Update all snapshots (after intentional changes)
pytest --snapshot-update

# Review snapshot changes interactively
pytest --snapshot-warn-unused

# CI mode - fail on snapshot mismatch
pytest  # Default behavior
```

### Custom Snapshot Serializers

```python
from syrupy.extensions.json import JSONSnapshotExtension

@pytest.fixture
def snapshot_json(snapshot):
    """Use JSON serialization for snapshots."""
    return snapshot.use_extension(JSONSnapshotExtension)

def test_json_api(snapshot_json):
    response = api.get("/data")
    assert response.json() == snapshot_json
```

### Inline Snapshots

```python
def test_inline(snapshot):
    """Snapshot stored in test file itself."""
    result = calculate_value()
    assert result == snapshot(result)  # First run creates snapshot
```

### Best Practices for Snapshot Testing

1. **Use for stable outputs**: HTML, JSON responses, serialized objects
2. **Avoid for volatile data**: Timestamps, random IDs, system-specific paths
3. **Review diffs carefully**: Snapshot updates should be intentional
4. **Combine with unit tests**: Snapshots complement, not replace, assertions
5. **Keep snapshots small**: Large snapshots are hard to review

---

## Property-Based Testing with Hypothesis

Property-based testing generates random inputs to find edge cases.

### Installation

```bash
pip install hypothesis
```

### Basic Property Tests

```python
from hypothesis import given, strategies as st

@given(st.integers())
def test_integer_properties(x):
    """Test properties that should hold for all integers."""
    assert x + 0 == x
    assert x * 1 == x
    assert x - x == 0

@given(st.lists(st.integers()))
def test_sort_is_idempotent(data):
    """Sorting twice equals sorting once."""
    assert sorted(data) == sorted(sorted(data))

@given(st.lists(st.integers()))
def test_sort_preserves_length(data):
    """Sorting doesn't change length."""
    assert len(sorted(data)) == len(data)

@given(st.text())
def test_string_roundtrip(s):
    """Encoding and decoding returns original."""
    assert s.encode("utf-8").decode("utf-8") == s
```

### Combining with pytest Fixtures

```python
@given(st.integers(min_value=1, max_value=100))
def test_with_fixture(db_session, quantity):
    """Property test with pytest fixture."""
    order = Order(quantity=quantity)
    db_session.add(order)
    db_session.flush()

    assert order.total == order.price * quantity

@pytest.mark.parametrize("discount", [0, 10, 25, 50])
@given(st.integers(min_value=1))
def test_parametrized_property(discount, price):
    """Combine parametrize with hypothesis."""
    discounted = apply_discount(price, discount)
    assert discounted <= price
```

### Custom Strategies

```python
from hypothesis import strategies as st

# Email strategy
emails = st.emails()

# Custom composite strategy
@st.composite
def user_data(draw):
    """Generate valid user data."""
    return {
        "username": draw(st.text(min_size=3, max_size=20)),
        "email": draw(st.emails()),
        "age": draw(st.integers(min_value=18, max_value=120)),
    }

@given(user_data())
def test_user_creation(data):
    user = User(**data)
    assert user.is_valid()
```

### Controlling Test Generation

```python
from hypothesis import given, settings, Verbosity

@given(st.integers())
@settings(
    max_examples=500,        # More thorough testing
    deadline=1000,           # 1 second timeout per example
    verbosity=Verbosity.verbose,
)
def test_thorough(x):
    assert some_property(x)

@given(st.integers())
@settings(max_examples=10)  # Quick smoke test
def test_quick(x):
    assert basic_property(x)
```

### Example Database for Reproducibility

```python
from hypothesis import given, settings, Phase

@given(st.integers())
@settings(
    database=None,  # Disable example database
    phases=[Phase.generate],  # Only generate, don't replay
)
def test_stateless(x):
    pass
```

### Best Practices

1. **Test properties, not examples**: Focus on invariants that always hold
2. **Keep tests fast**: Each example should be quick
3. **Use `@settings(deadline=None)`** for slow operations
4. **Review failing examples**: Hypothesis shows minimal failing case
5. **Combine with unit tests**: Property tests find edge cases, unit tests verify specific behavior

---

## Test Asset Generation & Management

Dynamic test asset generation ensures tests are self-contained, reproducible, and independent of external files. This is especially critical for ML/ONNX testing where models must be generated programmatically.

### Core Principle: Code-Generated Assets

**CARDINAL RULE**: Never rely on pre-existing files or LLM-generated test data. All test assets must be generated by code during test execution.

```python
# ❌ BAD: Relying on pre-existing files
def test_model_optimization():
    model = onnx.load("tests/fixtures/bert_model.onnx")  # External dependency!
    optimized = optimize(model)
    assert optimized is not None

# ✅ GOOD: Generate assets programmatically
def test_model_optimization(simple_model_fixture):
    """Model is generated by fixture - no external dependencies."""
    optimized = optimize(simple_model_fixture)
    assert optimized is not None
```

### Fixture-Based Asset Generation

#### Session-Scoped Expensive Assets

For expensive-to-generate assets, use session scope to generate once per test session:

```python
# conftest.py
import onnx
from onnx import helper, TensorProto
import numpy as np

@pytest.fixture(scope="session")
def base_model() -> onnx.ModelProto:
    """Generate a base ONNX model for testing.

    Session-scoped to avoid regenerating for every test.
    """
    # Create input
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128])

    # Create nodes
    nodes = [
        helper.make_node("Relu", ["input"], ["relu_out"], name="relu_1"),
        helper.make_node("Sigmoid", ["relu_out"], ["output"], name="sigmoid_1"),
    ]

    # Create output
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])

    # Build graph and model
    graph = helper.make_graph(nodes, "test_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    return model
```

#### Function-Scoped Mutable Assets

For assets that tests may modify, use function scope:

```python
@pytest.fixture(scope="function")
def mutable_model(base_model) -> onnx.ModelProto:
    """Create a fresh copy for tests that modify the model."""
    import copy
    return copy.deepcopy(base_model)
```

### Pattern-Specific Model Generation

Generate models containing specific patterns for targeted testing:

```python
# tests/optim/conftest.py

@pytest.fixture(scope="session")
def gelu_pattern_model() -> onnx.ModelProto:
    """Generate model with GELU approximation pattern.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    This pattern should be detected and fused by GELU fusion optimizers.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 768])

    # Create GELU approximation nodes
    nodes = [
        # x³
        helper.make_node("Pow", ["input", "three"], ["x_cubed"], name="pow_1"),
        # 0.044715 * x³
        helper.make_node("Mul", ["x_cubed", "coef"], ["scaled_cube"], name="mul_1"),
        # x + 0.044715 * x³
        helper.make_node("Add", ["input", "scaled_cube"], ["sum_1"], name="add_1"),
        # sqrt(2/π) * (x + 0.044715 * x³)
        helper.make_node("Mul", ["sum_1", "sqrt_2_pi"], ["tanh_input"], name="mul_2"),
        # tanh(...)
        helper.make_node("Tanh", ["tanh_input"], ["tanh_out"], name="tanh_1"),
        # 1 + tanh(...)
        helper.make_node("Add", ["one", "tanh_out"], ["one_plus_tanh"], name="add_2"),
        # 0.5 * x
        helper.make_node("Mul", ["half", "input"], ["half_x"], name="mul_3"),
        # 0.5 * x * (1 + tanh(...))
        helper.make_node("Mul", ["half_x", "one_plus_tanh"], ["output"], name="mul_4"),
    ]

    # Create initializers for constants
    initializers = [
        numpy_helper.from_array(np.array([3.0], dtype=np.float32), "three"),
        numpy_helper.from_array(np.array([0.044715], dtype=np.float32), "coef"),
        numpy_helper.from_array(np.array([0.7978845608], dtype=np.float32), "sqrt_2_pi"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), "half"),
    ]

    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 768])
    graph = helper.make_graph(nodes, "gelu_pattern", [X], [Y], initializers)

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.fixture(scope="session")
def matmul_add_pattern_model() -> onnx.ModelProto:
    """Generate model with MatMul+Add pattern for Gemm fusion testing."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])

    # Weight and bias initializers
    weight = numpy_helper.from_array(
        np.random.randn(512, 256).astype(np.float32), "weight"
    )
    bias = numpy_helper.from_array(
        np.random.randn(256).astype(np.float32), "bias"
    )

    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["matmul_out"], name="matmul_1"),
        helper.make_node("Add", ["matmul_out", "bias"], ["output"], name="add_1"),
    ]

    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])
    graph = helper.make_graph(nodes, "matmul_add_pattern", [X], [Y], [weight, bias])

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
```

### Multi-Pattern Test Models

For comprehensive testing, generate models with multiple patterns:

```python
@pytest.fixture(scope="session")
def all_patterns_model() -> onnx.ModelProto:
    """Generate model with ALL optimization patterns for comprehensive testing.

    Patterns included (with prefixes for identification):
    - p01_identity_: Identity elimination pattern
    - p02_dropout_: Dropout elimination pattern
    - p03_reshape_: Reshape fusion pattern
    - p04_transpose_: Transpose optimization pattern
    - p05_conv_: Conv optimization pattern
    - p06_matmuladdrelu_: MatMul+Add+Relu fusion pattern
    - p07_attention_: Attention pattern
    - p08_biasgelu_: Bias+GELU fusion pattern
    - p09_skiplayernorm_: SkipLayerNorm pattern

    Node naming convention: {pattern_prefix}{operation}_{index}
    Example: p06_matmuladdrelu_matmul_1
    """
    # Implementation generates all patterns in one model
    # Each pattern uses consistent naming for verification
    ...
```

### Conftest Hierarchy for Asset Sharing

Organize conftest files hierarchically for proper asset sharing:

```
tests/
├── conftest.py                    # Root: Core helpers (optimize_at_level, etc.)
├── optim/
│   ├── conftest.py               # Optim-wide: Base model fixtures
│   ├── capabilities/
│   │   ├── conftest.py           # Capability-specific: Pattern models, ORT names
│   │   ├── test_gelu_fusion.py
│   │   └── test_matmul_add.py
│   ├── pipes/
│   │   ├── conftest.py           # Pipe-specific: Pipe configs, mock models
│   │   ├── test_pipe_graph.py
│   │   └── test_pipe_fusion.py
│   └── integration/
│       ├── conftest.py           # Integration: Complex model fixtures
│       └── test_optimizer.py
```

#### Root conftest.py - Core Helpers

```python
# tests/conftest.py
"""Root conftest - Core testing utilities."""

import onnx
import onnxruntime as ort
import tempfile
from pathlib import Path

def optimize_at_level(
    model: onnx.ModelProto,
    level: int = 2,
    disabled_optimizers: list[str] | None = None,
) -> onnx.ModelProto:
    """Apply ORT graph optimization at specified level.

    This is the RAW ORT API helper - does NOT use Pipe classes.
    Use this in capability tests for isolation testing.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel(level)

    if disabled_optimizers:
        for name in disabled_optimizers:
            opts.add_session_config_entry(
                f"session.disable_specified_optimizers",
                ",".join(disabled_optimizers)
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.onnx"
        output_path = Path(tmpdir) / "output.onnx"

        onnx.save(model, str(input_path))
        opts.optimized_model_filepath = str(output_path)

        # Create session to trigger optimization
        ort.InferenceSession(str(input_path), opts)

        return onnx.load(str(output_path))
```

#### Domain conftest.py - Shared Fixtures

```python
# tests/optim/capabilities/conftest.py
"""Capability test fixtures - Pattern-specific models."""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx

# Import pattern model generators
from tests.optim.conftest import (
    gelu_pattern_model,
    matmul_add_pattern_model,
    all_patterns_model,
)

def get_all_ort_names() -> list[str]:
    """Get all registered ORT optimizer names for isolation testing."""
    return [
        "GeluFusionL2",
        "BiasGeluFusion",
        "MatMulAddFusion",
        "LayerNormFusion",
        # ... all 49+ ORT optimizer names
    ]

@pytest.fixture(scope="session")
def ort_optimizer_names() -> list[str]:
    """Fixture providing all ORT optimizer names."""
    return get_all_ort_names()
```

### Asset Verification Helpers

Create helpers to verify generated assets have expected structure:

```python
# tests/helpers/model_verification.py

def count_nodes_by_op(model: onnx.ModelProto, op_type: str) -> int:
    """Count nodes of specific operation type."""
    return sum(1 for n in model.graph.node if n.op_type == op_type)

def count_nodes_by_prefix(model: onnx.ModelProto, prefix: str) -> int:
    """Count nodes with name prefix (for pattern identification)."""
    return sum(1 for n in model.graph.node if n.name.startswith(prefix))

def count_nodes_by_prefix_and_op(
    model: onnx.ModelProto, prefix: str, op_type: str
) -> int:
    """Count nodes matching both prefix and operation type."""
    return sum(
        1 for n in model.graph.node
        if n.name.startswith(prefix) and n.op_type == op_type
    )

def verify_pattern_exists(
    model: onnx.ModelProto,
    pattern_prefix: str,
    expected_ops: list[str],
) -> bool:
    """Verify a pattern exists in the model with expected operations."""
    for op in expected_ops:
        if count_nodes_by_prefix_and_op(model, pattern_prefix, op) == 0:
            return False
    return True
```

### Differential Testing with Generated Assets

Test optimization effects by comparing before/after states:

```python
def test_gelu_fusion_effectiveness(gelu_pattern_model):
    """Test that GELU fusion actually reduces node count."""
    from tests.conftest import optimize_at_level
    from tests.helpers.model_verification import count_nodes_by_op

    # Before optimization
    before_tanh = count_nodes_by_op(gelu_pattern_model, "Tanh")
    before_mul = count_nodes_by_op(gelu_pattern_model, "Mul")

    # Apply optimization with GELU fusion enabled
    optimized = optimize_at_level(
        gelu_pattern_model,
        level=2,
        disabled_optimizers=[]  # All enabled
    )

    # After optimization - GELU pattern should be fused
    after_tanh = count_nodes_by_op(optimized, "Tanh")
    after_mul = count_nodes_by_op(optimized, "Mul")

    # Verify fusion occurred
    assert after_tanh < before_tanh, "GELU fusion should reduce Tanh nodes"
    assert after_mul < before_mul, "GELU fusion should reduce Mul nodes"
```

### Best Practices Summary

1. **Always generate assets in code**: Never rely on external files
2. **Use appropriate fixture scope**: Session for expensive, function for mutable
3. **Name patterns consistently**: Use prefixes for pattern identification
4. **Create verification helpers**: Standardize how you check asset structure
5. **Document pattern structure**: Explain what each generated model contains
6. **Test asset generation**: Verify fixtures produce expected structures
7. **Use conftest hierarchy**: Share assets at appropriate levels
8. **Prefer RAW APIs in unit tests**: Don't couple to higher-level abstractions

---

## Common Patterns & Anti-Patterns

### Patterns ✅

```python
# Good: Descriptive test names
def test_user_registration_sends_welcome_email():
    pass

# Good: Focused tests
def test_calculate_tax_for_standard_rate():
    income = 50000
    assert calculate_tax(income) == 10000

# Good: Using fixtures for setup
@pytest.fixture
def authenticated_client(client, user):
    client.login(username=user.username, password="password")
    return client

# Good: Parametrize instead of loops
@pytest.mark.parametrize("value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(value, expected):
    assert value ** 2 == expected

# Good: Clear test structure (Arrange-Act-Assert)
def test_user_creation():
    # Arrange
    data = {"username": "john", "email": "john@example.com"}

    # Act
    user = User.create(**data)

    # Assert
    assert user.username == "john"
    assert user.email == "john@example.com"
```

### Anti-Patterns ❌

```python
# Bad: Test doing too much
def test_everything():
    user = create_user()
    post = create_post(user)
    comment = create_comment(post)
    assert user.is_active
    assert post.author == user
    assert comment.post == post
    # Too many things tested at once

# Bad: Modifying global state
def test_with_global_state():
    global CONFIG
    CONFIG["debug"] = True  # Don't modify globals
    assert my_function() == expected

# Bad: Tests depending on order
def test_first():
    global shared_data
    shared_data = setup_data()

def test_second():
    # Depends on test_first running first
    assert shared_data.value == expected

# Bad: Catching all exceptions
def test_broad_exception():
    try:
        risky_operation()
    except Exception:  # Too broad
        pass  # Test passes even if unexpected error

# Bad: No assertion
def test_without_assertion():
    result = my_function()
    # No assert - test always passes
```

---

## Debugging & Troubleshooting

### Debugging Techniques

```python
# Drop into debugger on failure
pytest --pdb

# Drop into IPython debugger
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb

# Set breakpoint in code
def test_debug():
    value = calculate()
    import pdb; pdb.set_trace()  # or breakpoint() in Python 3.7+
    assert value == expected

# Print debugging (use -s flag)
def test_with_print():
    print("Debug info:", value)  # Visible with pytest -s
    assert value == expected

# Capture logs
def test_with_logging(caplog):
    with caplog.at_level(logging.INFO):
        my_function()
    assert "Expected message" in caplog.text

# Detailed failure info
pytest -vv  # Very verbose
pytest --tb=short  # Short traceback
pytest --tb=line   # One line per failure
pytest --tb=no     # No traceback
```

### Common Issues & Solutions

```python
# Issue: Import errors
# Solution: Check PYTHONPATH and use --import-mode
pytest --import-mode=importlib

# Issue: Fixture not found
# Solution: Check scope and conftest.py location
pytest --fixtures  # List available fixtures

# Issue: Tests not discovered
# Solution: Check naming conventions
pytest --collect-only  # See what's collected

# Issue: Flaky tests
# Solution: Use pytest-rerunfailures
pip install pytest-rerunfailures
pytest --reruns 3 --reruns-delay 1

# Issue: Test isolation
# Solution: Use fixtures and avoid global state
@pytest.fixture(autouse=True)
def reset_state():
    cleanup_before_test()
    yield
    cleanup_after_test()
```

---

---

## Deprecations & Migration Guide

### Deprecated Patterns to Avoid

Understanding deprecated patterns helps maintain forward compatibility.

#### Marker Access (Changed in pytest 4.0+)

```python
# ❌ DEPRECATED - will be removed
marker = item.get_marker("slow")

# ✅ CURRENT - use these instead
marker = item.get_closest_marker("slow")  # Single marker
markers = list(item.iter_markers("slow"))  # Multiple markers
```

#### Hook Decorators (Changed in pytest 7.0+)

```python
# ❌ DEPRECATED
@pytest.mark.tryfirst
def pytest_collection_modifyitems(items):
    pass

# ✅ CURRENT
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    pass
```

#### pytest_namespace Hook (Removed in pytest 8.0)

```python
# ❌ REMOVED - no longer works
def pytest_namespace():
    return {"my_value": 42}

# ✅ CURRENT - use pytest_configure instead
def pytest_configure(config):
    config.my_value = 42
```

#### Async Fixtures with Sync Tests (Warning in pytest 8.x+)

```python
# ❌ DEPRECATED - will warn and eventually error
@pytest.fixture
async def async_data():
    return await fetch_data()

def test_sync(async_data):  # Sync test using async fixture
    assert async_data is not None

# ✅ CURRENT - explicit async handling
@pytest.fixture
def async_data():
    import asyncio
    return asyncio.run(fetch_data())

def test_sync(async_data):
    assert async_data is not None

# OR use async test
@pytest.fixture
async def async_data():
    return await fetch_data()

@pytest.mark.asyncio
async def test_async(async_data):
    assert async_data is not None
```

#### yield_fixture Decorator (Removed)

```python
# ❌ REMOVED
@pytest.yield_fixture
def resource():
    r = acquire()
    yield r
    release(r)

# ✅ CURRENT - use regular fixture with yield
@pytest.fixture
def resource():
    r = acquire()
    yield r
    release(r)
```

### Migration Checklist

When upgrading pytest versions, check for:

- [ ] Replace `item.get_marker()` with `item.get_closest_marker()`
- [ ] Replace `@pytest.mark.tryfirst/trylast` with `@pytest.hookimpl(tryfirst=True/trylast=True)`
- [ ] Remove any `pytest_namespace` hooks
- [ ] Update async fixtures to use explicit handling
- [ ] Replace `@pytest.yield_fixture` with `@pytest.fixture`
- [ ] Check `--strict-config` passes with your configuration
- [ ] Review `filterwarnings` for any pytest deprecation warnings

### Version Compatibility Matrix

| Feature | Minimum Version | Notes |
|---------|-----------------|-------|
| `pyproject.toml` support | pytest 6.0 | `[tool.pytest.ini_options]` |
| Native TOML `[tool.pytest]` | pytest 9.0 | Cleaner syntax |
| `--import-mode=importlib` | pytest 6.0 | Recommended default |
| `@pytest.hookimpl` | pytest 7.0 | Replaces mark decorators |
| `item.iter_markers()` | pytest 4.0 | Replaces `get_marker()` |
| `required_plugins` | pytest 7.0 | With `--strict-config` |

## Best Practices Checklist

### ✅ DO's

1. **Write descriptive test names** that explain what is being tested
2. **Use fixtures** for setup and teardown
3. **Keep tests focused** - one concept per test
4. **Use parametrize** for data-driven tests
5. **Organize tests** to mirror source code structure
6. **Register custom markers** in pytest.ini
7. **Use appropriate scopes** for fixtures
8. **Mock external dependencies** in unit tests
9. **Run fastest tests first** in CI/CD
10. **Use pytest.raises** for exception testing
11. **Document complex test scenarios**
12. **Use tmp_path fixture** for file operations
13. **Configure pytest** in pyproject.toml or pytest.ini
14. **Use pytest plugins** to extend functionality
15. **Profile slow tests** and optimize
16. **Start without `__init__.py`** in test directories - add only when needed
17. **Use `--import-mode=importlib`** for modern import handling
18. **Declare `required_plugins`** for team/CI consistency
19. **Use `--strict-config`** to catch configuration errors early
20. **Handle async fixtures properly** with `@pytest.mark.asyncio`
21. **Use file locking** for session fixtures with parallel execution

### ❌ DON'Ts

1. **Don't write tests that depend on execution order**
2. **Don't use global state** that affects other tests
3. **Don't catch broad exceptions** without re-raising
4. **Don't hardcode paths** - use fixtures and tmp_path
5. **Don't skip writing tests** for "simple" functions
6. **Don't mix test types** in the same file
7. **Don't use production credentials** in tests
8. **Don't ignore flaky tests** - fix or mark them
9. **Don't write tests without assertions**
10. **Don't duplicate test logic** - use fixtures
11. **Don't test implementation details** - test behavior
12. **Don't use time.sleep** - use proper synchronization
13. **Don't modify source code** for testing - use mocks
14. **Don't run all tests locally** for every change
15. **Don't ignore test warnings** - fix or suppress explicitly
16. **Don't add `__init__.py` to tests by default** - pytest works without it
17. **Don't use deprecated marker access** - use `get_closest_marker()` not `get_marker()`
18. **Don't mix sync tests with async fixtures** - will warn/error in pytest 8+
19. **Don't ignore configuration file priority** - empty `pytest.ini` blocks `pyproject.toml`
20. **Don't use `@pytest.yield_fixture`** - use `@pytest.fixture` with yield
21. **Don't forget `xdist_group`** when tests must share state in parallel execution

### Final Recommendations

1. **Start Simple**: Begin with basic tests and add complexity as needed
2. **Test First**: Consider TDD for complex logic
3. **Continuous Integration**: Run tests automatically on every commit
4. **Code Coverage**: Aim for high coverage but focus on critical paths
5. **Performance**: Monitor and optimize test suite performance
6. **Documentation**: Document complex test scenarios and fixtures
7. **Maintenance**: Regularly update and refactor tests
8. **Team Standards**: Establish and follow team testing conventions

Remember: Good tests are as important as good code. They provide confidence, documentation, and safety for refactoring.
