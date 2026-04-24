# ModelKit Design Document Standard

**Version**: 1.1
**Date**: 2026-04-19
**Status**: Active
**Supersedes**: `docs/design/optracing/learnings_from_232.md` (survey; promoted to normative spec)
**Owner**: `docs/` CODEOWNERS (repository maintainers)

---

## 1. Scope and Applicability

### 1.1 Purpose

This document defines the mandatory structure, content, and lifecycle of all design documents checked into the ModelKit repository. It is normative: a pull request whose design docs violate this standard MAY be rejected on that basis alone.

### 1.2 RFC 2119 Terminology

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in this document are to be interpreted as described in RFC 2119. In brief:

- **MUST** / **MUST NOT** — absolute requirement / prohibition
- **SHOULD** / **SHOULD NOT** — strong recommendation; exceptions permitted with written rationale in the doc
- **MAY** — optional; author's discretion

### 1.3 When a Design Doc Is Required

A design doc **MUST** be authored before implementation when any of the following applies:

- Adding a new module under `src/winml/modelkit/` (a new top-level package directory)
- Introducing a new public API surface (new `__init__.py` exports)
- Changing the behavior of an existing public API in a backward-incompatible way
- Refactoring that spans three or more modules or deletes an existing package
- Introducing a new CLI command under `wmk`
- Changing a persisted on-disk format (cache layout, config schema, output file format)

A design doc **MAY** be skipped for:

- Bug fixes that preserve public API
- Documentation-only changes
- Test additions that do not change production code
- Dependency bumps that do not alter behavior
- Refactoring confined to a single file

### 1.4 Document Types

Three document types are defined. A feature **MUST** have at least `1_prd.md`. It **SHOULD** have `2_coreloop.md` when the implementation is non-trivial (>1 file or >200 lines of code). It **MAY** have one or more `3_design_<topic>.md` files for detailed component specifications.

| File | Role | Required for | Audience |
|------|------|---------------|----------|
| `1_prd.md` | Product Requirements Document — WHAT the system does and WHY | Every feature | PMs, architects, reviewers |
| `2_coreloop.md` | Core Loop Design — HOW to build it (architecture, API, flow) | Every non-trivial feature | Implementers, reviewers |
| `3_design_<topic>.md` | Design Detail — internal APIs, call graphs, test strategy for one component | Optional, split from `2_coreloop.md` when a component exceeds ~300 lines of spec | Implementers, reviewers |

### 1.5 Path Conventions

Design docs **MUST** live under `docs/design/<module>/` where `<module>` matches the primary source directory name under `src/winml/modelkit/`.

```
docs/design/<module>/
├── 1_prd.md                      (required)
├── 2_coreloop.md                 (required if non-trivial)
├── 3_design_<topic>.md           (optional; split from coreloop when needed)
└── iterations/                   (optional; brainstorming record)
    ├── 01.md
    └── 02.md
```

The `iterations/` subdirectory **MAY** be used to record the brainstorming history leading to the final design. Iteration files are informational, not normative, and are not subject to this spec's structural rules.

Cross-module features (affecting multiple `<module>` directories) **SHOULD** be documented in the directory of the primary module, with references from other affected modules' docs.

### 1.5.1 Transitional Locations (refactor exception)

A refactor that renames or relocates the primary source directory **MAY** keep its design docs at the original directory name for the duration of one release cycle, provided the docs include a **Transitional Location** note immediately after the metadata header. The note **MUST** contain:

- The current doc directory (legacy name)
- The target `<module>` value declared in the `Module` metadata field
- A commitment to relocate the docs under `docs/design/<target_module>/` in a named follow-up PR or within a stated timeframe

Example note:

```markdown
**Transitional Location**: This doc lives at `docs/design/<legacy>/` while its `Module` field declares the post-refactor target `<target>`. Relocation to `docs/design/<target>/` is scheduled for PR #<N> (tracked in issue #<M>).
```

The exception expires when the refactor implementation lands. At that point the docs **MUST** be moved to comply with §1.5 proper.

---

## 2. Metadata Header (MANDATORY)

Every design doc **MUST** begin with a metadata header immediately after the H1 title. The header **MUST** use bold key-value syntax and include all required fields.

### 2.1 Required Fields

```markdown
# <Title>

**Version**: <semver, start at 1.0>
**Date**: <YYYY-MM-DD>
**Status**: <one of: Draft | Active | Implemented | Deprecated>
**Module**: <primary module name, matches docs/design/<module>/>
```

### 2.2 Optional Fields

```markdown
**Supersedes**: <path to previous doc>
**Depends-On**: <comma-separated paths to upstream docs>
**Author**: <name or handle>
**Related Documents**: see §<N> (when a table appears later)
```

### 2.3 Status Values — Semantics

| Status | Meaning |
|--------|---------|
| **Draft** | Under active authoring; not yet reviewed. Implementation **MUST NOT** begin. |
| **Active** | Reviewed and approved. Implementation in progress. |
| **Implemented** | Feature is shipped; doc reflects current reality. |
| **Deprecated** | Feature removed or superseded. Doc retained for history. **MUST** include `Supersedes` pointing to the replacement (or `Removed-In: <commit hash>`). |

### 2.4 Example

```markdown
# WinMLSession — Core Loop Design

**Version**: 2.1
**Date**: 2026-04-19
**Status**: Active
**Module**: session
**Supersedes**: docs/design/session/archive/2_coreloop_v2.md
**Depends-On**: docs/design/compiler/1_prd.md, docs/design/session/monitor/2_coreloop.md
```

---

## 3. Structural Conventions

### 3.1 Section Numbering

Every section at H2 and H3 levels **MUST** be numbered. Deeper nesting **MAY** omit numbers.

```markdown
## 1. Overview                  ✓ MUST
### 1.1 Purpose                 ✓ MUST
#### Why this matters           ✓ MAY (unnumbered OK at H4+)
```

### 3.2 Table of Contents

Docs longer than 200 lines **MUST** include a Table of Contents as the first `## ` section (before §1). TOC entries **MUST** be anchor-linked. Docs shorter than 200 lines **MAY** omit the TOC.

### 3.3 Related Documents Table

`2_coreloop.md` and `3_design_*.md` **MUST** include a "Related Documents" section as §0 (before §1), formatted as a table:

```markdown
## 0. Related Documents

| Document | Path | Purpose |
|----------|------|---------|
| PRD | `./1_prd.md` | Feature requirements |
| Upstream | `../foo/2_coreloop.md` | Provides X API |
| Parallel | `./3_design_bar.md` | Details of the Bar component |
```

`1_prd.md` **MAY** include a Related Documents table but it is not required.

### 3.4 Requirement ID Prefixes

Requirements, stories, constraints, and criteria **MUST** use the following ID prefixes where applicable:

| Prefix | Meaning | Scope |
|--------|---------|-------|
| `US-N` | User Story | `1_prd.md` §3 |
| `FR-N` | Functional Requirement | `1_prd.md` §4 |
| `NFR-N` | Non-Functional Requirement | `1_prd.md` §5 |
| `C-N` | Design Constraint | `1_prd.md` §7 |
| `R-N` / `M-N` | Risk / Mitigation pair | `1_prd.md` §8 |
| `SC-N` | Success Criterion | `1_prd.md` §1.3 or §10 |
| `FP-N` | Forbidden Pattern | this doc §8 only |

IDs **MUST** be unique within a doc and **SHOULD** be stable across versions (do not renumber unless you renumber the entire set).

### 3.5 Versioning Suffix Rule

Architectural revisions that change the document's direction (not just fixes) **MUST** bump the Version field. When a revision is large enough that cross-references to the old version would be confusing, the author **MAY** create `<N>_<type>_v2.md` as a new file and set `Supersedes` on the new file. In that case, the old file's Status **MUST** be updated to `Deprecated` and **MUST** include `Supersedes` pointing to the new file.

Minor edits (typos, clarifications, link fixes) **SHOULD NOT** bump the version.

---

## 4. PRD Skeleton (`1_prd.md`)

A PRD **MUST** include the mandatory sections below, in order. A PRD's substance is its Requirements (§4 and §5) — earlier sections provide context, later sections capture risk and history. A lightweight PRD for an internal refactor **MAY** have a one-paragraph Executive Summary but **MUST NOT** omit the Requirements sections.

### 4.1 Mandatory Sections

```
## 1. Executive Summary
   1.1 Purpose
   1.2 Problem Statement
   1.3 Success Metrics                 (SC-N)

## 2. Scope
   2.1 In Scope
   2.2 Out of Scope

## 4. Functional Requirements           (FR-N)

## 5. Non-Functional Requirements       (NFR-N)
   5.1 Performance
   5.2 Reliability
   5.3 Usability
   5.4 Compatibility

## 10. Appendix
   10.1 Glossary
   10.2 References
   10.3 Document History
```

### 4.2 Conditional Sections

Include these only when applicable. Omission is permitted but **SHOULD** be justified in the Open Questions (§9) if the reader might expect them.

```
## 3. User Stories                      (US-N)      — when there is a PM / end-user audience
## 6. Technical Design (high-level)                 — when architectural overview aids comprehension
## 7. Design Constraints                (C-N)       — when external constraints bind the design
## 8. Risks and Mitigations             (R-N/M-N)   — when non-obvious failure modes exist
## 9. Open Questions                                — when the design has unresolved points
```

### 4.3 Exemplar

See §7.1 of this spec for the canonical PRD reference.

---

## 5. Coreloop Skeleton (`2_coreloop.md`)

`2_coreloop.md` describes the architecture and core execution flow. It **MUST** reference `1_prd.md` via the Related Documents table (§3.3 of this spec).

### 5.1 Mandatory Sections

```
## 0. Related Documents                               (MUST, per §3.3)

## 1. Design Philosophy
   1.1 Purpose
   1.2 Core Principles
   1.3 Design Pattern (if applicable)

## 2. Module Structure
   2.1 Component Diagram or File Layout
   2.2 Key Dependencies

## 3. Core Loop Implementation
   3.1 High-Level Flow
   3.2 Call Sequence / Data Flow

## 4. API Design
   4.1 Public Functions / Classes
   4.2 Function Signatures
   4.3 Return Types / Data Structures

## 7. Error Handling

## 8. Testing Strategy
   8.1 Unit Tests
   8.2 Integration Tests

## 11. Revision History
```

### 5.2 Conditional Sections

```
## 0.5 I/O Dependencies                               — when the module orchestrates 3+ upstream modules (see config/2_coreloop.md §0 for a canonical example)
## 5. CLI Design / Integration                        — when the feature ships a CLI command
## 6. Configuration / Data Structures                 — when configuration format is non-trivial
## 9. Integration Points                              — when downstream modules consume this one
## 10. Future Work                                    — when forward-looking notes matter
```

### 5.3 Data Flow Diagrams

Large coreloops **SHOULD** include at least one data flow diagram. Format **MAY** be ASCII box art or Mermaid. A doc **MUST NOT** mix formats within a single diagram.

---

## 6. Design Detail Skeleton (`3_design_<topic>.md`)

### 6.1 When to Create

A `3_design_<topic>.md` **SHOULD** be created when a single component's detailed design would push `2_coreloop.md` above ~900 lines, or when the component has a well-bounded internal API worth documenting separately (e.g., `export/3_design_io.md`, `loader/3_design_task.md`).

### 6.2 Naming

The filename **MUST** be `3_design_<topic>.md` where `<topic>` is a snake_case noun phrase describing the component (e.g., `3_design_qnn_monitor.md`, `3_design_io.md`, `3_design_task.md`).

### 6.3 Mandatory Sections

```
## 0. Related Documents                               (MUST, per §3.3)

## 1. Purpose
   1.1 What problem does this solve?
   1.2 Scope

## 2. Public API
   2.1 Classes / Functions
   2.2 Signatures and contracts

## 3. Internal Implementation
   3.1 Internal functions (prefix `_`)
   3.2 Call graph
   3.3 Design rationale

## 4. Resolution / Scenario Flows                     (when multiple code paths exist)

## 5. Integration / Override Mechanism                (when extensible by users)

## 7. Test Strategy
   (test cases tied to specific API functions — see §8.3 of this spec)

## 8. Future Extensions
```

---

## 7. Exemplars

The following documents are the canonical style references. If a future doc of the same type conflicts with this spec, the spec wins; but within the freedoms the spec permits, these are the style anchors.

**Branch note**: the exemplars below live on branch `232` (see `D:\BYOM\ModelKit_PRs\232\docs\design\`). They are not present on `feat/mvp`. Authors on `feat/mvp` **SHOULD** consult them via the `232` branch checkout. When `232` merges into `feat/mvp` (or into `main`), this section's paths become relative.

### 7.1 Best PRD

`docs/design/build/1_prd.md` (branch `232`) — paired user stories (US-1 to US-6), explicit two-step workflow scoping, detailed output directory contract.

### 7.2 Best Coreloop

`docs/design/config/2_coreloop.md` (branch `232`) — §0 I/O Dependencies section (upstream-first), four-tier priority system, call sequence diagrams, scenario-driven flows.

### 7.3 Best Design Detail

`docs/design/export/3_design_io.md` (branch `232`) — public vs internal function split, current-vs-proposed call graph, test strategy tied to specific API functions.

### 7.4 First compliant exemplar on `feat/mvp`

`docs/design/session/monitor/1_prd.md` + `docs/design/session/monitor/2_coreloop.md` — the first doc pair authored against v1.0 of this spec. Use the 232 exemplars above for depth and pattern; use the optracing pair to see how the spec's rules apply in practice on this branch.

---

## 8. Forbidden Patterns (MUST NOT)

### FP-1. Multiple approaches without designating canonical

A doc **MUST NOT** present multiple implementation approaches as equally viable. One **MUST** be designated canonical; others **MUST** be labeled `Rejected Alternative` with rationale. Violation example: `module/1_prd.md` on branch `232` presents three approaches without sequencing.

### FP-2. Circular cross-references

Doc A referencing Doc B referencing Doc C referencing Doc A **MUST NOT** occur. Cross-references **MUST** flow downward in the dependency graph: PRD → Coreloop → Design Detail. Upstream docs may reference downstream docs only via the Related Documents table, never inline.

### FP-3. Test strategy disconnected from API

A Testing Strategy section **MUST NOT** list generic test categories. It **MUST** map specific test files or test cases to specific API functions or classes. Example of compliant form: *"`tests/session/test_perf.py::test_auto_reset_fires_when_options_differ` validates `WinMLSession.perf().__enter__` in §4.5."*

### FP-4. Missing success criteria

A PRD **MUST NOT** ship without at least one `SC-N` success criterion. Features without measurable completion criteria cannot be verified as done.

### FP-5. Silent supersession

A doc **MUST NOT** replace another doc without adding a `Supersedes` field to the new doc and updating the old doc's Status to `Deprecated`. Silent replacement breaks the doc history chain.

### FP-6. Undocumented abbreviations

Module-specific acronyms (QNN, HTP, QDQ, EPContext, PDH, QHAS, etc.) **MUST** be defined in §10 (Appendix Glossary) of the PRD. A doc **MUST NOT** use such acronyms without either glossary entry or inline expansion on first use.

### FP-7. Mixed markdown diagram formats

A single diagram **MUST NOT** mix ASCII and Mermaid. A document **MAY** use both formats for different diagrams, but each diagram **MUST** be internally consistent.

---

## 9. Deprecation and Lifecycle

### 9.1 Marking a Doc as Deprecated

When a feature is removed or replaced:

1. Set the old doc's `Status: Deprecated` in the metadata header.
2. Add `Supersedes: <path to replacement>` OR `Removed-In: <commit hash / PR number>` field.
3. Add a prominent H2 section immediately after the header:
   ```markdown
   ## ⚠ Deprecated
   This document describes a superseded design. See the replacement at <path>. Retained for historical context only.
   ```

### 9.2 Superseding a Doc

The new doc **MUST**:

1. Include `Supersedes: <path to old doc>` in its metadata header.
2. State in its §1.2 Problem Statement how it differs from the predecessor.
3. Migrate any still-relevant cross-references (e.g., other docs pointing at the old file) in the same PR.

### 9.3 Archiving

A deprecated doc **SHOULD** be retained in place for 6 months after supersession, then **MAY** be moved to `docs/design/<module>/archive/`. Archive files **MUST** keep their metadata header. They **MUST NOT** be deleted from git history.

---

## 10. Spec Governance

### 10.1 Owner

Changes to this spec are approved by the `docs/` CODEOWNERS (currently the repository maintainers). In the absence of an explicit CODEOWNERS file, the reviewer of the latest merged PR touching `docs/` is de-facto owner.

### 10.2 Change Process

Amendments to this spec **MUST** be made via pull request and **MUST**:

1. Update the `Version` field (semver: MAJOR = breaking rule change; MINOR = new rule; PATCH = clarification).
2. Update the `Date` field.
3. Append an entry to the Revision History (§11).
4. Receive at least one approval from someone who has authored a design doc under this spec.

### 10.3 Review Cadence

This spec **SHOULD** be reviewed annually and **SHOULD** be revisited after the first three new modules adopt it, to incorporate real-world feedback.

### 10.4 Enforcement

Reviewers **MAY** reject a PR whose design docs violate this spec, citing the violated section. Authors **MAY** appeal by proposing an amendment to the spec itself.

Exceptions to **SHOULD**-level rules **MUST** be justified inline in the doc (e.g., "This doc omits §5 because no CLI integration exists; see §9 Open Questions"). Exceptions to **MUST**-level rules **MUST** be resolved either by fixing the doc or amending this spec.

---

## 11. Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-04-19 | Initial version. Promoted from `docs/design/optracing/learnings_from_232.md` (descriptive survey) to normative spec. Added RFC 2119 vocabulary, metadata header requirement, `3_design_*.md` skeleton, Forbidden Patterns, Deprecation protocol, and Governance section. |
| 1.1 | 2026-04-19 | Post-audit amendments: (a) added §1.5.1 "Transitional Locations" as a principled exception mechanism for refactors that rename or relocate their primary source directory — addresses the first compliant doc pair's need to sit at its legacy path until implementation lands; (b) §7 now explicitly notes that the canonical exemplars live on branch `232` (not on `feat/mvp`), with cross-branch consultation guidance and a local `feat/mvp` reference to the first compliant doc pair. |
