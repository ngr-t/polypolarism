# polypolarism - Claude Code Project Guide

## Overview

Polars DataFrame static type checker based on row polymorphism.

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
├── types.py        # DataType, FrameType definitions
├── dsl.py          # Schema DSL parser: DF["{col: Type}"]
├── expr_infer.py   # Expression type inference
├── ops/
│   ├── join.py     # Join operation type inference
│   └── groupby.py  # GroupBy/Agg type inference
├── analyzer.py     # AST analysis, data flow tracking
├── checker.py      # Declared vs inferred type comparison
└── cli.py          # Command-line interface
```

## Development Notes

### TDD Workflow

This project follows TDD (t-wada style):
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Commit incrementally

### Key Design Decisions

- **Nullable subtyping**: `T` is a subtype of `Nullable[T]`, but `Nullable[T]` is NOT a subtype of `T`
- **Join nullability**: left join makes right columns nullable, right join makes left columns nullable
- **AST analysis**: Method chains are analyzed by recursively inferring receiver types

### Gotchas / Lessons Learned

1. **DSL Parser whitespace**: The schema DSL `{col: Type}` allows flexible whitespace. Parser uses recursive descent.

2. **AST method chains**: When analyzing `df.join(...).group_by(...).agg(...)`, need to handle `.agg()` specially because it follows `.group_by()` which doesn't return a FrameType directly.

3. **Git worktree for parallel work**: Can use `git worktree add` to create parallel working directories for subagents. Remember to clean up with `git worktree remove`.

4. **uv build backend**: The `uv_build` backend works for `pip install .` but may need `hatchling` or `setuptools` for broader compatibility if issues arise.

5. **Python 3.14 compatibility**: Tests run on Python 3.14 locally. CI tests 3.11-3.13. Be careful with new syntax features.

6. **Aggregation function signatures**: Different aggregations have different return types:
   - `count()` → `UInt32`
   - `sum(Int64)` → `Int64`
   - `mean(Int64)` → `Float64`
   - `list(T)` → `List[T]`

### Testing

- Test fixtures in `tests/fixtures/valid/` and `tests/fixtures/invalid/`
- Valid fixtures should pass type checking
- Invalid fixtures should fail with specific errors
- 211 tests total as of Phase 4 completion

## Git Conventions

- Commit messages in English
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Include emoji footer for Claude Code generated commits
