# ADR-0010: Patito (pydantic-for-polars) as a Second Schema Frontend

- **Status**: Accepted (implemented)
- **Date**: 2026-06-27
- **Deciders**: @negotetsu

## Context

polypolarism currently has exactly one schema-declaration frontend:
Pandera class-based `pa.DataFrameModel` + `DataFrame[Schema]` /
`LazyFrame[Schema]` annotations (plus the object API). The request is to
also recognize **pydantic**-flavored schema declarations.

"pydantic" is ambiguous. Plain `pydantic.BaseModel` describes a single
*record*, not a DataFrame, and has no canonical Polars DataFrame
annotation. [Patito](https://github.com/JakobGM/patito) is the
purpose-built "pydantic for polars" layer: models subclass `patito.Model`
(a `pydantic.BaseModel` subclass) and double as Polars DataFrame schemas.
**Decision scope is Patito** (user-selected); plain `BaseModel` is out of
scope.

**Single-dialect assumption (user-confirmed):** a file/project mixing
Patito *and* Pandera schemas is not a supported case. This simplifies
base-class detection (see Decision 2).

### What the analyzer already does that carries over for free

The narrowing and checking machinery is **dialect-neutral** — it operates
on the `Schema` / `SchemaRegistry` / `FrameType` intermediate
representation, not on Pandera specifics:

- Bare-statement narrowing (`analyzer.py:5666`) keys on method name
  `validate` + a `schema_registry` lookup. It does not care whether the
  schema came from Pandera or Patito.
- The `DataFrame[X]` / `LazyFrame[X]` annotation detector
  (`pandera_annotation.py`) matches the **attribute tail** `DataFrame` /
  `LazyFrame` and extracts the subscript name `X`.
- The dtype-expression core (`pandera_dtype.py:_parse_plain_dtype` /
  `_parse_annotated`) is pure AST → `DataType`, with no Pandera-specific
  wrapper semantics in the leaf cases.

So the cost of Patito support concentrates in a new **frontend**
(base-class detection + field → `ColumnSpec` translation with Patito
semantics) plus **one** type-system addition (numeric/Literal group
acceptance, Decision 4). The checker, narrowing, and `FrameType` are
reused unchanged.

### Probe results (patito 0.8.6, polars 1.42.0, pydantic 2.13.4)

Probed via `Model.dtypes` (canonical dtype), `Model.valid_dtypes`
(accepted set), `Model.nullable_columns`, and `Model.validate(df)`
behavior. `Model.dtypes` / `valid_dtypes` is the authoritative spec the
static translator must reproduce.

| Annotation | canonical dtype | accepted set (`valid_dtypes`) | nullable | column may be absent |
|---|---|---|---|---|
| `int` | `Int64` | **all 10 integer dtypes** (Int8..128, UInt8..128) | no | no (required) |
| `float` | `Float64` | **Float16/32/64** | no | no |
| `str` | `String` | `String` | no | no |
| `bool` | `Boolean` | `Boolean` | no | no |
| `Literal["a","b"]` | `Enum(categories=[...])` | **`{String, Enum}`** | no | no |
| `Optional[T]` / `T \| None` | `T` | `T` | **yes** | no |
| `pt.Field(dtype=pl.UInt16)` | `UInt16` | **`{UInt16}` (forced)** | no | no |
| `datetime.date` | `Date` | `Date` | no | no |
| `list[str]` | `List(String)` | `List(String)` | no | no |
| nested `Inner(pt.Model)` | **`Struct({...})`** | `Struct({...})` | no | no |

Behavioral facts:

- **`Model.validate(df)` returns the validated frame** (a model-bound
  `ProductDataFrame`), *not* `None`. The library docs say "returns None";
  that is stale. Consequence: BOTH narrowing forms work — bare
  `Model.validate(df)` (narrows `df`) and `out = Model.validate(df)`
  (binds `out`), exactly like Pandera.
- **Validation is strict by default**: an extra column raises
  `DataFrameValidationError`. A missing declared column also raises.
- **`Optional[T]` / `T | None` means the value is nullable**, with the
  column still required (`nullable_columns == {the Optional fields}`).
  This is the **opposite** of Pandera/polypolarism's `Optional[T]`, which
  means `required=False` (column may be absent — `pandera_dtype.py:143`).
- **`pt.DataFrame[Model]` / `pt.LazyFrame[Model]` exist** as real generic
  annotations (and `Model.DataFrame` / `Model.LazyFrame` as per-model
  subclasses).
- `pt.Field` forwards `*args, **kwargs` to `pydantic.Field`; the only
  *type-affecting* kwarg is `dtype=`. `unique=` and value constraints are
  runtime-only (ignored statically, like Pandera's `pa.Field` extras).

The "all-null Optional column rejected" probe result was an artifact (an
all-`None` polars `Series` has dtype `Null`, so it failed the dtype check,
not the nullability check); `nullable_columns` is the authoritative
source.

## Decision

1. **Add Patito as a parallel frontend; do not overload the Pandera one.**
   The wrapper semantics diverge (Optional, strict default), so merging
   risks regressing Pandera. Share only the dialect-neutral pieces:
   `_parse_plain_dtype` / `_parse_annotated` (dtype-expression core), the
   `Schema` / `SchemaRegistry` / `FrameType` dataclasses, the checker, and
   the narrowing + annotation detectors.

   New modules: `compat/patito_api.py` (name tables) and `patito_dtype.py`
   (field → `ColumnSpec` with Patito semantics). A light collector
   registers Patito models into the existing `SchemaRegistry`.

2. **Base-class detection is import-anchored.** Unlike `DataFrameModel`,
   the name `Model` is not distinctive (collision-prone). A `class
   X(Model)` / `class X(pt.Model)` is treated as a Patito schema only when
   `Model` provably resolves to Patito — `import patito as pt` +
   `pt.Model`, or `from patito import Model` (alias-aware, reusing
   `_collect_name_aliases`). The single-dialect assumption means we do not
   need to disambiguate Patito-vs-Pandera within one file, but we DO need
   to avoid mis-claiming unrelated `class X(Model)` declarations (ADR-0009:
   no false-positive-prone diagnostics).

3. **Map field annotations with Patito semantics:**
   - `Optional[T]` / `T | None` → `Nullable(T)`, `required=True` (the
     inverse of the Pandera frontend).
   - `pt.Field(dtype=pl.X)` → force dtype `X`, overriding the annotation's
     default mapping.
   - `Literal[str, ...]` → `Enum(categories=tuple(...))` (canonical), with
     group acceptance per Decision 4.
   - nested registered model `Inner` → `Struct(Inner.columns)`.
   - all columns `required=True` (Patito has no `required=False` column in
     the basic case; deferred probe: Python defaults / a superfluous-column
     config).
   - Patito models bind **`strict=True`** (closed frames) — extra columns
     are rejected at runtime, so undeclared-column lookups are
     `pple-column-not-found` proofs, and the Pandera non-strict
     checked-island (`nonstrict_schema`) machinery is not needed.

4. **Add group acceptance for `int` / `float` / `Literal`.** Patito's
   `int` accepts any of 10 integer dtypes, `float` any float dtype, and a
   `Literal[str]` accepts `String` or `Enum`. Mapping `int` → `Int64` and
   relying on exact-dtype subtyping would **falsely reject** a `UInt32`
   column in an `int` slot (ADR-0009 violation). The minimal sound
   addition: a Patito-scoped relaxation in `checker._subtype_verdict`
   reusing `types.NUMERIC_DTYPES` so that integer↔integer and float↔float
   (within family) is accepted, and `String`/`Enum` are interchangeable
   for a `Literal` slot. (See Alternatives for why not `coerce=True`.)

5. **Annotation support comes for free.** `pt.DataFrame[Model]` /
   `pt.LazyFrame[Model]` already resolve through
   `pandera_annotation.extract_dataframe_annotation` (attribute-tail match
   on `DataFrame`/`LazyFrame` + subscript name), once `Model` is in the
   registry. Only the bare model-qualified `Model.DataFrame` attribute form
   needs new handling and is deferred to a follow-up.

## Rationale

### Why a parallel frontend, not a generalized one

The two dialects disagree on `Optional` (absent column vs nullable value)
and on the strict default (open vs closed). A shared parser with a dialect
flag threaded through every wrapper case is more error-prone than two thin
frontends over a shared dtype core — and the Pandera frontend is already
large and load-bearing. Keep the blast radius small.

### Why import-anchored detection

`Model` is a common base-class name. Matching it on the attribute tail the
way `DataFrameModel` is matched would mis-claim arbitrary `class X(Model)`
declarations as Patito schemas and emit phantom diagnostics — exactly the
false-positive class ADR-0009 forbids. Anchoring to the `patito` import is
the only sound trigger.

### Why group acceptance instead of `coerce=True`

`coerce=True` already tolerates numeric↔numeric differences
(`checker.py:335`), which would cover `int`/`float`. But Patito does **not**
coerce — it validates strictly (a `str` field rejects an int column;
`product_id: int` rejects a `Float64` column). Binding Patito frames as
`coerce=True` would also tolerate non-numeric mismatches (anything →
String is an "always" cast), producing false negatives far beyond the
int-family relaxation Patito actually grants. A narrow, Patito-scoped
integer-family / float-family / String-or-Enum relaxation matches the
probed `valid_dtypes` groups precisely.

### Why `strict=True` for Patito frames

Patito's `validate` rejects extra columns by default. Modeling Patito
frames as closed is faithful and *simpler* than the Pandera default — it
skips the entire `nonstrict_schema` checked-island provenance path.

## Alternatives considered

- **Plain `pydantic.BaseModel` as a row schema** — rejected (out of scope).
  No canonical Polars DataFrame annotation; would require inventing a
  convention. Patito already is that convention.
- **Map `int` → `Int64` with exact subtyping** — rejected: false-positive
  on any non-`Int64` integer column (ADR-0009).
- **Map `int` → `Unknown`** — sound (no false positive) but throws away
  all precision (a `str` column would satisfy an `int` slot; downstream
  dtype checks degrade). Group acceptance keeps precision.
- **Bind Patito frames `coerce=True`** — rejected: too lenient on
  non-numeric mismatches (see Rationale).
- **Generalize the Pandera frontend with a dialect flag** — rejected:
  threads divergent `Optional`/strict semantics through a large existing
  module; higher regression risk than a thin parallel frontend.

## Consequences

### Positive

- Patito users get parameter/return typing (`pt.DataFrame[Model]`),
  `Model.validate(df)` narrowing, and dtype/nullability checking, reusing
  the entire existing checker.
- Patito's strict default makes its frames closed — undeclared-column
  lookups are hard `pple-column-not-found` errors, not soft interface
  lints.
- No new diagnostic codes are expected; existing codes
  (column-not-found, undeclared-column, return-type, nullable mismatch)
  apply. (To be confirmed against the coverage gate when fixtures land.)

### Negative / limitations

- Group acceptance is a real (if small) type-system change touching
  `_subtype_verdict`; its laws must be added to `tests/test_properties.py`.
- The `int`-family relaxation accepts a `Float64` column in an `int` slot
  (Patito rejects it) — a deliberate false *negative* (sound under
  ADR-0009's no-false-positive rule), unless we split integer-family and
  float-family acceptance (the plan does split them — int accepts ints
  only, float accepts floats only).
- Single-dialect assumption: a file mixing both is unsupported; behavior
  there is undefined and not tested.

### Deferred probes (do not block v1)

- Is there a Patito config to allow superfluous columns (open frames)?
- Does a Python default value (`x: int = 5`) make a column absent-allowed
  (i.e. `required=False`)?
- `Field(nullable=...)` — does Patito accept it, or is `Optional` the only
  nullability path?

## Implementation outline

| Step | Action | Files | Status |
|---|---|---|---|
| 1 | `compat/patito_api.py`: `Model` base name, `patito` package anchor, `Field` callable + `dtype` keyword. | `compat/patito_api.py` | **Done** |
| 2 | `patito_dtype.py`: field → `(ColumnSpec, nested_model)`. Reuses `pandera_dtype._parse_plain_dtype` for leaves; adds `Optional→Nullable` (required), `Field(dtype=)` override, `int/float→DataTypeGroup`, `Literal→{String,Enum} group`, `list[T]→List`, nested-model marker. | `patito_dtype.py` | **Done** |
| 3 | `patito_schema.py`: import-anchored `scan_patito_imports` + `collect_patito_schemas`; registers `Schema(strict=True)` into the shared `SchemaRegistry`; inheritance via the reused `_topo_sort`; two-pass nested-model → open `Struct` resolution. | `patito_schema.py` | **Done** |
| 4 | Wire `collect_patito_schemas` into `collect_schemas` (lazy import to avoid a module cycle); no-op unless `patito` is imported. | `pandera_schema.py` | **Done** |
| 5 | Group acceptance: new `DataTypeGroup` dtype + `collapse_groups`; `checker._subtype_verdict` accepts any group member on the declared side; the four `expr_infer` combine primitives (`promote_types`/`unify_types`/`supertype`/`infer_cast`) collapse groups to their representative for inference math. | `types.py`, `checker.py`, `expr_infer.py` | **Done** |
| 6 | 5 `valid/` + 3 `invalid/` Patito fixtures + goldens; unit tests (`test_patito_dtype.py`, `test_patito_schema.py`); group laws in `test_checker.py` / `test_types.py`. patito added to the `runtime` dependency group so the differential harness can import the fixtures. | `tests/`, `pyproject.toml` | **Done** |
| 7 | README + CLAUDE.md: document the frontend, the semantic differences, and the single-dialect assumption. | `README.md`, `CLAUDE.md` | **Done** |

Annotation support (`pt.DataFrame[Model]` / `pt.LazyFrame[Model]`) needed no
step — it resolves through the existing detector once Patito models are
registered (Decision 5). The `Model.DataFrame` bare-attribute form is a
follow-up.

### Design note: group acceptance is two-sided

`DataTypeGroup` lives on the DECLARED side (acceptance). But a Patito-typed
parameter's group columns also flow into the body's expression inference,
where arithmetic / casts must produce concrete dtypes (e.g. `int * 1.0` →
`Float64`). The group therefore carries a `canonical` representative, and
`collapse_groups` swaps it in at the entry of the dtype-combining primitives.
Pass-through projections keep the group, which still matches the same declared
group under single-dialect Patito. This is why the fix touches `expr_infer`
and not just the checker.

## Verification

- `valid/patito_basic` — `pt.DataFrame[In] -> pt.DataFrame[Out]` with a body
  producing `Out`'s columns type-checks clean.
- `valid/patito_group_widths` vs `invalid/patito_dtype_mismatch` — a `UInt32`
  column satisfies an `int` field (no error); a `String` column does NOT (real
  error). The two are explicit false-positive twins.
- `valid/patito_optional_nullable` — a non-null `T` column satisfies a nullable
  `T | None` slot; the column is required.
- `valid/patito_validate_narrow` — `S.validate(df)` at the body top level
  narrows `df`.
- `valid/patito_field_and_nested` — `Field(dtype=UInt16)` is exact; a nested
  model becomes a `Struct`.
- `invalid/patito_missing_column` / `invalid/patito_strict_unknown_column` —
  a dropped declared column and an undeclared-column lookup on a strict frame
  produce the expected `pple-return-type` / `pple-column-not-found`.
- `uv run pytest` green (3941 passed), `ruff` + `pyright` clean.

### Deferred (follow-ups)

- Runtime-differential coverage for Patito (the harness's input synthesis is
  Pandera-specific; Patito functions are naturally out of scope today).
- The `Model.DataFrame` / `Model.LazyFrame` bare-attribute annotation forms.
- Cross-file Patito inheritance where the base is imported (the inherited-
  subclass fixpoint pass is Pandera-base-aware).
- The probes listed above (superfluous-column config, Python-default columns,
  `Field(nullable=...)`).
