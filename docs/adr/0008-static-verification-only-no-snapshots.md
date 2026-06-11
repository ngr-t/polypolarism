# ADR-0008: Stay purely static — no snapshot mechanism (for now)

Date: 2026-06-12
Status: accepted (deliberate non-implementation; revivable)
Backlog: closes C-12b and C-11 tier 3 as deferred

## Context

Two roadmap items pointed beyond static analysis:

- **C-12b**: a `polypolarism snapshot` command reading parquet/IPC
  footer schemas into a committed path→schema file, turning ADR-0006's
  open-frame source assumptions into verified facts without check-time
  IO.
- **C-11 tier 3**: genuinely dynamic schemas (config/env-driven class
  bodies, `type(...)` construction), where the only complete resolution
  is executing user code and reading `Model.to_schema()`.

A unified design was sketched: one snapshot mechanism with two sources
(parquet footers, dynamic-schema evaluation), executing only at
explicit `snapshot` time so checks stay hermetic, with a hash-based
staleness warning. The open design questions were the trust model
(blind trust vs hash verification vs check-time IO fallback) and the
nullability mapping for file-backed schemas (Arrow marks nearly
everything nullable; null presence is data-dependent; claiming either
direction without a runtime validation backing is unsound in one
direction or the other).

## Decision

polypolarism stays a **purely static verifier** in its current scope:
no check-time IO beyond the analyzed source files, no execution of user
code (the runtime differential test harness remains the only, internal,
exception), and no snapshot artifacts. C-12b and C-11 tier 3 are
**deliberately not implemented**.

What covers the gap instead — all already shipped:

- unannotated / file-backed sources bind as ADR-0006 open frames:
  everything the code itself determines is checked, source facts are
  assumptions;
- dynamic schemas degrade loudly (issue #90: unresolved object schemas
  bind as open assumption frames with a PLW011 advisory);
- users who want verified source schemas write them down — a pandera
  schema plus `Schema.validate(...)` narrowing IS the runtime-backed
  version of a snapshot, with better guarantees (validation actually
  runs).

## Rationale

- The snapshot file is a second source of truth that can silently drift
  from the data; the honest mitigations (hash checks, re-snapshot
  discipline) import operational complexity into a tool whose value is
  exactly that it runs anywhere, instantly, with no state.
- The nullability question has no sound static answer for file data —
  any choice manufactures either false positives or unbacked non-null
  claims. pandera validation answers it correctly at runtime, and the
  validate-narrowing path already rewards that pattern.
- Scope discipline: the static core still has cheaper precision wins
  available; an IO/execution subsystem would dominate maintenance cost.

## Revival conditions

Revisit if (any of): users demonstrably accumulate large bodies of
literal-path `read_parquet` code they cannot annotate; the LSP grows a
workspace-trust model that makes snapshot-time execution natural; or
the nullability gap gains a type-system answer (an explicit
"nullability unknown" wildcard) that removes the unsound choice. The
unified one-command/two-sources design sketch above remains the
starting point.
