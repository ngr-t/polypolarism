# ADR-0002: Compiler-Style Test Harness (Golden Fixtures + Property-Based Type Laws)

- **Status**: Accepted
- **Date**: 2026-06-10
- **Deciders**: @negotetsu

## Context

polypolarism is structurally a compiler front-end: it parses Python source,
runs type inference over a dtype lattice, and emits diagnostics with stable
codes (`pple-*` / `pplw-*`). Its test suite, however, grew as a conventional
unit-test suite:

- `tests/fixtures/{valid,invalid,warning}/` held end-to-end input programs,
  but each fixture needed a **hand-written companion assertion** in
  `test_cli.py`. Fixtures without one were silently untested, and assertions
  checked only "some error containing `pple-non-boolean-predicate` exists" — message wording,
  error counts, and co-occurring diagnostics were unverified.
- Nothing guaranteed that every diagnostic code defined in `diagnostics.py`
  was exercised end-to-end. When this ADR was implemented, the new coverage
  check immediately found **seven codes with no fixture at all** (two of
  them now folded into `pple-column-not-found`: the `cast` and
  `drop_nulls`-subset misses, plus pple-column-name-collision, pple-unpivot,
  pple-lazy-only-method, pplw-unresolved-pipe, pplw-unknown-schema).
- The type-system algebra (`_is_subtype`, `promote_types`, `unify_types`,
  `supertype`, `_is_frame_subtype`) was tested only on hand-picked examples,
  although call sites implicitly rely on algebraic laws (commutativity,
  reflexivity, transitivity) holding across the whole dtype lattice —
  including nested `List` / `Nullable` combinations no one wrote out by hand.

Mature compiler projects solve these problems with two standard techniques:

1. **Golden / snapshot testing** (rustc UI tests, LLVM lit + FileCheck,
   CPython's `test_syntax`): every input program in the test tree is
   discovered automatically, compiled, and its *full* diagnostic output is
   compared byte-for-byte against a checked-in expected file. Updating
   expectations is a deliberate, reviewable act (`--bless` in rustc).
2. **Property-based testing of the type lattice** (QuickCheck-style, as used
   for GHC's and many SMT solvers' algebraic cores): generate random types
   and assert the laws the implementation is supposed to satisfy, instead of
   enumerating examples.

## Decision

### 1. Golden-file fixture harness (`tests/test_fixtures.py`)

- Every `*.py` under `tests/fixtures/{valid,invalid,warning}/` is
  **auto-discovered** via pytest parametrization. Adding a fixture file *is*
  adding a test; no companion assertion code is required.
- Each fixture has a `<name>.expected` golden file next to it containing the
  rendered diagnostic report (per-function `OK`/`WARN`/`FAIL` status plus
  every error and warning message). The test compares the live output
  byte-for-byte.
- Regeneration is explicit, never implicit:

  ```bash
  POLYPOLARISM_UPDATE_EXPECTED=1 uv run pytest tests/test_fixtures.py
  ```

  The regenerated goldens must be reviewed in the git diff like any code
  change — a diagnostic regression then shows up as a readable diff in the
  PR instead of passing unnoticed.
- **Category invariants** are asserted independently of the golden contents,
  so a stale or blindly-regenerated golden can never bless a category
  violation: `valid/` must fully pass, `invalid/` must contain at least one
  failure, `warning/` must pass with at least one warning.
- An orphan check fails if a `.expected` file outlives its fixture.

### 2. Diagnostic-code coverage gate

`test_every_diagnostic_code_is_exercised_by_a_fixture` introspects
`diagnostics.ALL_CODES` for all defined `pple-*`/`pplw-*` codes and requires
each to appear in at least one golden file. Codes that genuinely cannot fire from a
self-contained fixture (currently only `pplw-unsupported-version`, the environment version
floor) live in an explicit allowlist that must name the unit test covering
them instead — and the test also fails if an allowlisted code *becomes*
fixture-covered, keeping the allowlist minimal.

This turns "we added a diagnostic but never tested it end-to-end" into a CI
failure.

### 3. Property-based tests for the type algebra (`tests/test_properties.py`)

Using `hypothesis` (new dev dependency), random dtypes — including nested
`List[...]` and `Nullable[...]` — are generated and the intended laws are
asserted:

- `_is_subtype` is a **partial order** (reflexive, transitive along
  `Nullable`-widening chains incl. inside `List`, antisymmetric) on the
  Unknown-free fragment, with one-way `T <: Nullable[T]` widening.
  `Unknown` is excluded from the order laws on purpose: gradual typing's
  consistency relation is not transitive by design.
- `promote_types`, `unify_types`, `supertype` are **commutative** (including
  the error cases), so no diagnostic can depend on operand source order.
- `promote_types` / `unify_types` are **idempotent** on Unknown-free types.
- `infer_cast` keeps the target base type and never loses nullability.
- `_is_frame_subtype` implements row-polymorphic width subtyping:
  reflexivity, extra-column tolerance (non-strict) / rejection (strict),
  missing-required-column rejection, and depth widening to `Nullable`.

Properties were validated against the implementation exhaustively over the
full simple-dtype pair matrix before being adopted, so they document real
guarantees rather than aspirations.

## Consequences

- The fixture suite is now self-maintaining: a new fixture is automatically
  tested, its full output is pinned, and forgetting a golden is a test
  failure with regeneration instructions.
- Diagnostic wording is now under test. Changing a message intentionally
  requires regenerating goldens — small extra friction, traded for catching
  unintentional message/severity/count regressions.
- The existing hand-written fixture tests in `test_cli.py` remain; they
  encode *intent* ("this fixture exists to cover issue #28") while goldens
  pin *behavior*. New fixtures generally don't need a `test_cli.py` entry.
- `hypothesis` runs add ~2–3 s to the suite. Acceptable.
- Future work, not in scope here: differential testing of the
  `supertype` table against live polars (`pl.when/then/otherwise` probing)
  as an opt-in/nightly job, and source-program fuzzing of the analyzer.
