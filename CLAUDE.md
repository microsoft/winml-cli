# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Principles

### 1. Requirement Clarification

Claude will always begin by restating and refining user requirements to ensure accurate understanding. This involves:

Paraphrasing requests with improved clarity
Identifying key objectives and constraints
Confirming scope and expectations before proceeding

### 2. Critical Questioning

Claude will never execute instructions blindly. Instead, it will:

Ask clarifying questions when requirements are ambiguous
Identify potential issues or gaps in specifications
Seek confirmation on assumptions before implementation
Challenge unclear or potentially problematic requests

### 3. Critical Analysis Over Compliance

Claude will provide thoughtful evaluation rather than automatic approval:

Critically examine proposals and identify potential improvements
Highlight risks, limitations, or alternative approaches
Offer constructive feedback instead of reflexive praise
Challenge design decisions when warranted

### 4. Design Documentation & Tracking

Claude will maintain comprehensive records of all design-related discussions:

Design Backlog: Document all conversations involving design changes or clarifications
Track decision rationale and context
Maintain version history of requirement changes
Create audit trail for design evolution

## Expected Workflow

Understand → Rephrase and clarify requirements
Question → Identify ambiguities and seek clarification
Analyze → Critically evaluate the request
Document → Record design decisions and changes
Execute → Proceed with confirmed understanding

## ⚠️ UNIVERSAL RULE #1 - ABSOLUTELY NO HARDCODED LOGIC

**THIS IS THE CARDINAL RULE - NO EXCEPTIONS AT ANY TIME!**

❌ **NEVER HARDCODE**:

- Model architecture names (BERT, GPT, ResNet, etc.)
- Node names or operation names  
- Input/output tensor names
- Layer naming patterns
- Class name string matching
- Model-specific logic of ANY kind

## ⚠️ CARDINAL RULE #2 - ALL TESTING MUST USE PYTEST WITH CODE-GENERATED RESULTS

**THIS IS THE SECOND CARDINAL RULE - NO EXCEPTIONS!**

❌ **NEVER**:

- Create test-specific Python scripts outside pytest
- Use LLM-generated test results or expectations  
- Write standalone test runners
- Generate test data manually

✅ **ALWAYS**:

- Use pytest for ALL testing
- Generate test results with code during test execution
- Structure test results in organized temp directories
- Implement CLI subcommands for user testing
- Test CLI using pytest with subprocess/click testing utilities
- Dynamic analysis, never static assumptions

**Before every code change**: Ask "Is this hardcoded to any specific architecture?"  
**After every test fix**: Review the diff for model-specific assumptions  
**When tests fail**: Fix with universal logic, never hardcoded patches

## ⚠️ CARDINAL RULE #3 - MANDATORY TEST VERIFICATION

**THIS IS THE FOURTH CARDINAL RULE - NO EXCEPTIONS!**

❌ **NEVER**:

- Implement features without running test verification
- Revise test cases without confirming they pass
- Skip pytest validation after code changes
- Assume tests pass without verification

✅ **ALWAYS**:

- Run `uv run pytest tests/` after implementing features
- Run `uv run pytest tests/` after revising test cases
- Verify test results before marking tasks complete
- Use pytest to validate any code modifications

**After every implementation**: Run `uv run pytest tests/` to verify
**After every test revision**: Run `uv run pytest tests/` to confirm
**Before marking complete**: Ensure pytest verification passes

## ⚠️ CARDINAL RULE #4 - NEVER SKIP TESTS BECAUSE THEY FAIL

**FIX THE ROOT CAUSE, NOT THE SYMPTOMS!**

❌ **NEVER**:

- Skip tests because they fail - find and fix the root cause
- Add `pytest.mark.skip` to hide failing tests
- Use `xfail` to ignore test failures
- Bypass tests with workarounds or hacks
- Pretend tests pass by disabling them

✅ **ALWAYS**:

- Investigate WHY a test fails before making changes
- Fix the underlying code or test logic, not the test runner
- Ensure test fixtures and builders are consistent across test files
- Share common test infrastructure (builders, fixtures, registries) between tests
- When tests share resources, keep them synchronized

**When tests fail**: Investigate root cause → Fix the issue → Verify ALL tests pass
**When adding skips**: Only for hardware/EP requirements (CUDA, DirectML, AVX), never for "it doesn't work"
**When test files diverge**: Ensure shared builders/fixtures remain consistent

## MUST-RULES

Always follow MUST-RULES

- Rigorously adhere to universal design principles
- Prioritize generalizability over specific implementations
- Validate against core architectural constraints before committing any changes

## Project Overview

This is a Python project named "ModelKit"

## Development Commands

**IMPORTANT**: Always use uv with virtual environment for this project.

**FOR CLAUDE CODE**: Always use uv run or activate venv first. Never run bare python commands.

**TEMPORARY FILES**: Always use temp/ folder in project root to persist temporary files and test outputs.

**NODE.JS AVAILABILITY**: npm and npx are available via fnm (Fast Node Manager). Use `eval "$(fnm env)"` before npm/npx commands:

- Node.js version: v22.16.0
- npm version: 11.4.1
- Usage: `eval "$(fnm env)" && npm install` or `eval "$(fnm env)" && npx <command>`

## Development Rules of Thumb

### 0. MUST Test Validation (CRITICAL RULE)

- **🚨 MUST VALIDATE**: Every feature implementation change MUST be validated against ALL MUST test cases
- **⚠️ ZERO TOLERANCE**: Any MUST test failure breaks the entire system
- **🔴 CARDINAL RULES**: MUST-001 (No Hardcoded Logic), MUST-002 (Torch.nn Filtering), MUST-003 (Universal Design)
- **✅ ENFORCEMENT**: Run MUST tests before any commit, PR, or release
- **📍 Location**: See `/docs/test-cases/MUST-*.md` for detailed validation procedures

### 1. Universal Design Principles

- **Target**: Accelerate Model Deployment on WinML
- **NO HARDCODED LOGIC**: Absolutely no hardcoded model architectures, node names, operator names, or any similar model-specific patterns
- **Universal First**: Always design solutions that work for ANY model, not just specific architectures
- **Architecture Agnostic**: Leverage fundamental PyTorch structures (`nn.Module`, hooks, named_modules) that exist in all models

### 2. Test-Driven Development

- **Always Create Test Cases**: Every feature must have corresponding tests
- **TDD When Possible**: Write tests before implementation to define expected behavior
- **Comprehensive Testing**: Target both unit tests (individual functions) and integration tests (end-to-end workflows)
- **Test Multiple Architectures**: Verify solutions work across different model types (BERT, ResNet, GPT, etc.)

### 3. Code Quality Standards  

- **Clean Up After Each Iteration**: Refactor and clean code after implementing features
- **Use Linting Tools**: Apply tools like `black`, `ruff`, or `flake8` to maintain code standards
- **Remove Dead Code**: Delete unused functions, commented-out code, and obsolete implementations
- **Consistent Formatting**: Maintain consistent code style throughout the project
- **Naming Convention**: Follow the naming rules defined in [`/docs/naming-convention.md`](/docs/naming-convention.md), especially the acronym casing table (ONNX, EP, QDQ, Op, etc.)

### 4. Pythonic Practices

- **Follow Python Conventions**: Use PEP 8 style guidelines and Python idioms
- **Type Hints**: Add type annotations for better code documentation and IDE support
- **Descriptive Names**: Use clear, self-documenting variable and function names
- **List/Dict Comprehensions**: Prefer Pythonic constructs over verbose loops where appropriate
- **Context Managers**: Use `with` statements for resource management
- **Exception Handling**: Use specific exception types and proper error handling patterns

## Memories

### Critical Questioning

- Always ask question before planning and executing if you have questions or uncertainties for the requirements

### Code Quality

- Always ruff lint after revise the python code

### Git Commit Guidelines

- Never add `Co-Authored-By` when doing git commit

### PR Guidelines

- Do not include "Test plan" section in PR descriptions
