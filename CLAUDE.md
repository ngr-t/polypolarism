# polypolarism - Claude Code Project Guide

## Overview

Polars DataFrame static type checker, inspired by row polymorphism
(the implementation is open-record structural subtyping + gradual `Unknown`;
genuine row variables are scaffolded but not yet active — see `RowVar`).

## Commands

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=polypolarism

# Run specific test file
uv run pytest tests/test_analyzer.py -v

# Run type checker on fixtures
uv run polypolarism tests/fixtures/valid/
uv run polypolarism tests/fixtures/invalid/

# Build package
uv build
```

## Project Structure

```
src/polypolarism/
├── types.py               # DataType, ColumnSpec, FrameType definitions
├── pandera_dtype.py       # AST type expression -> ColumnSpec translator
├── pandera_schema.py      # Pandera DataFrameModel class registry
├── pandera_annotation.py  # DataFrame[Schema] annotation detection
├── patito_dtype.py        # Patito field -> ColumnSpec translator (ADR-0010)
├── patito_schema.py       # Patito Model class registry (import-anchored)
├── expr_infer.py          # Expression type inference
├── ops/
│   ├── join.py            # Join operation type inference
│   └── groupby.py         # GroupBy/Agg type inference
├── analyzer.py            # AST analysis, data flow tracking, validate-narrowing
├── checker.py             # Declared vs inferred type comparison
└── cli.py                 # Command-line interface
```

## Development Notes

### TDD Workflow

This project follows TDD (t-wada style):
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Commit incrementally

### Key Design Decisions

- **Schema declaration**: Pandera class-based `pa.DataFrameModel` with `DataFrame[Schema]` / `LazyFrame[Schema]` annotations. The legacy `DF["{...}"]` DSL has been removed.
- **Patito frontend** (ADR-0010): a second schema dialect, `patito.Model` + `pt.DataFrame[Model]`. It reuses the dialect-neutral `Schema` / `SchemaRegistry` / `FrameType` IR, checker, and narrowing; only detection (`patito_schema.py`, **import-anchored** because `Model` is collision-prone) and field translation (`patito_dtype.py`) are Patito-specific. Patito semantics that differ from Pandera: `Optional[T]` → **value nullable** (column required), the inverse of Pandera's "column may be absent"; `int`/`float`/`Literal` map to a `DataTypeGroup` (accept any width / String-or-Enum) collapsed to a canonical representative for inference math via `types.collapse_groups`; `pt.Field(dtype=...)` forces an exact dtype; nested `Model` → `Struct`; models bind `strict=True`. **Single-dialect assumption**: a file mixing Patito and Pandera is unsupported. Runtime-differential coverage for Patito is deferred (the harness's input synthesis is Pandera-specific; Patito functions are naturally out of scope).
- **Nullable subtyping**: `T` is a subtype of `Nullable[T]`, but `Nullable[T]` is NOT a subtype of `T`. Value nullability is encoded by wrapping a column's dtype in `Nullable(...)` (declared via `pa.Field(nullable=True)`).
- **Optional columns**: A column declared `Optional[T]` carries `ColumnSpec(dtype=T, required=False)` and is allowed to be absent in the inferred frame. An inferred optional cannot satisfy a required declared slot.
- **Strict schemas**: `class Config: strict = True` rejects extra columns at every position the schema is the "expected" side. Default (non-strict) allows structural subtyping.
- **Validation as narrowing**: `Schema.validate(df)`, `df.pipe(Schema.validate)`, `Schema.validate(lf).collect()` all retype the downstream variable. Bare-statement narrowing only fires at the function body's top level.
- **Join nullability**: left join makes right columns nullable, right join makes left columns nullable, full join makes both sides nullable.
- **AST analysis**: Method chains are analyzed by recursively inferring receiver types.

### Gotchas / Lessons Learned

1. **AST method chains**: When analyzing `df.join(...).group_by(...).agg(...)`, need to handle `.agg()` specially because it follows `.group_by()` which doesn't return a FrameType directly.

3. **Git worktree for parallel work**: Can use `git worktree add` to create parallel working directories for subagents. Remember to clean up with `git worktree remove`.

4. **uv build backend**: The `uv_build` backend works for `pip install .` but may need `hatchling` or `setuptools` for broader compatibility if issues arise.

5. **Python 3.14 compatibility**: Tests run on Python 3.14 locally. CI tests 3.11-3.13. Be careful with new syntax features.

6. **Aggregation function signatures**: Different aggregations have different return types:
   - `count()` → `UInt32`
   - `sum(Int64)` → `Int64`
   - `mean(Int64)` → `Float64`
   - `list(T)` → `List[T]`

### Testing

The suite combines unit tests with two compiler-style layers
(see `docs/adr/0002-compiler-style-test-harness.md` for the full rationale):

**Golden-file fixtures** (`tests/test_fixtures.py`, rustc-UI-test style):

- Fixtures live in `tests/fixtures/{valid,invalid,warning}/` and are
  auto-discovered — adding a `.py` fixture file IS adding a test.
- Each fixture has a `<name>.expected` golden file next to it with the full
  diagnostic report; the test compares byte-for-byte.
- Category invariants are enforced independently of the goldens:
  `valid/` passes cleanly, `invalid/` has ≥1 error, `warning/` passes with
  ≥1 warning.
- After intentionally changing diagnostics, regenerate and **review the
  golden diff in git** — never regenerate to silence a failure you don't
  understand:

  ```bash
  POLYPOLARISM_UPDATE_EXPECTED=1 uv run pytest tests/test_fixtures.py
  ```

- Every diagnostic code (`pple-*` errors / `pplw-*` warnings, registered in
  `diagnostics.ALL_CODES`) must appear in at least
  one golden file (coverage gate). When adding a new diagnostic code, add a
  minimal fixture demonstrating it; only codes that cannot fire from a
  self-contained file go in `FIXTURE_EXEMPT_CODES` with a pointer to their
  unit test.

**Property-based type-algebra tests** (`tests/test_properties.py`, hypothesis):

- Random (nested) dtypes verify the laws the inference engine relies on:
  `_is_subtype` partial order + one-way `T <: Nullable[T]` widening;
  commutativity of `promote_types` / `unify_types` / `supertype`;
  idempotence; `infer_cast` nullability preservation; row-polymorphic
  width subtyping of `_is_frame_subtype`.
- `Unknown` is excluded from the order laws by design (gradual-typing
  consistency is not transitive). When extending the type system, add the
  new dtype to the strategies here and check which laws it must satisfy.

Hand-written fixture assertions in `test_cli.py` encode *intent* (e.g.
"covers issue #28"); goldens pin *behavior*. New fixtures normally don't
need a `test_cli.py` entry.

## Git Conventions

- Commit messages in English
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Include emoji footer for Claude Code generated commits
