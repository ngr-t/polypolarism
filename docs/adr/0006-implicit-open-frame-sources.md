# ADR-0006: Implicit open-frame sources (bare annotations and `pl.read_*`)

Date: 2026-06-11
Status: accepted
Backlog: C-12 (half 1 of 2 — the snapshot half stays open as C-12b)

## Context

Until now a function was only checked when it carried a
`DataFrame[Schema]` / `LazyFrame[Schema]` annotation, and a frame was
only trackable when it originated from such an annotation (or a
`Schema.validate` narrowing). Code that reads a parquet file, or a
helper annotated with plain `pl.DataFrame`, was invisible — even though
much of what such code does downstream is statically determinable.

The `FrameType` machinery already supports *open frames* — `rest:
RowVar` marks "this frame may hold extra columns we don't know" — used
today for `unnest` of unknown structs and opaque `.name.*` outputs. The
expression layer resolves references to unpinned columns on an open
frame as `Unknown`, and the checker records open-frame passes as
leniency notes (ADR-0003 visibility).

## Decision

1. **Bare frame annotations opt a function in.** A parameter annotated
   `pl.DataFrame` / `pl.LazyFrame` (or `polars.DataFrame` /
   `polars.LazyFrame`; no subscript) binds an empty open frame
   `FrameType({}, rest=RowVar(param), is_lazy=...)` and makes the
   function checkable. Other prefixes (`pd.DataFrame`, bare
   `DataFrame`) are NOT recognized — they may be pandas. A bare frame
   *return* annotation also opts the function in but declares nothing
   to check (no schema claim; see Future work for the eager/lazy bit).

2. **`pl.read_*` / `pl.scan_*` produce open frames.** Calls whose
   polars return annotation is unconditionally `DataFrame` /
   `LazyFrame` (probed 1.41.2; union-typed readers like `read_excel`
   are excluded, mirroring the `EAGER_FRAME_RETURNING_METHODS` policy)
   infer an empty open frame with the matching laziness. The file's
   actual schema is unknown — and deliberately not read at check time
   (hermetic checks; the snapshot design is C-12b).

3. **Assumption semantics, not verification.** On an open frame, an
   operation that *requires* a column (`select("a")`, `pl.col("a")`,
   join keys, `unpivot(on=...)`) is assumed to have succeeded: the
   column is pinned (dtype `Unknown` unless the operation determines
   it) and **no diagnostic fires**, because its absence is not provable.
   This is the gradual-typing boundary rule: code downstream of line N
   only executes if line N succeeded at runtime, so the static verdict
   is *conditional* — "no new type errors beyond what the data source
   itself determines". polypolarism proves consistency of everything
   the code itself determines; it does not verify the source.

   The alternative — flagging every unverifiable column reference —
   was rejected: it would fire on essentially every line of
   open-frame code, violating the no-false-positive philosophy that
   every error is a proof.

4. **Operations must not manufacture proofs from open frames.** Every
   inference rule that errors on a missing column, or closes a result
   frame, must be open-frame aware:
   - *Determined-shape outputs close the frame*: `select`,
     `group_by().agg()`, `unpivot` output exactly the columns the call
     names — the result is closed (fully checkable downstream) even
     when the input was open. A selector argument (`cs.*` /
     `pl.all()` / `pl.exclude`) on an open frame cannot be enumerated,
     so it keeps the result open instead.
   - *Shape-preserving outputs stay open*: `with_columns`, `filter`,
     `drop`, `rename`, `cast`, identity methods carry `rest` through.
   - *Existence errors are suppressed on the open side* (join keys,
     `rename`, `cast`, `drop_nulls` subset, `unpivot`, vertical-concat
     set equality); a closed frame in the same operation still proves
     its own mismatches.
   - *Mixed concat contributions*: per column, a frame contributes its
     pinned dtype, `Unknown` if it is open and does not pin the
     column, and a provable error if it is closed and lacks it.

## What is guaranteed (and what is not)

Guaranteed (errors are proofs, conditional on reaching the line):
- dtype rules on every column the code itself pins — arithmetic
  validity, comparisons, casts, namespace receivers, aggregation
  signatures, join-key compatibility, nullability flow;
- column-set correctness downstream of any shape-determining call;
- eager/lazy method discipline (`collect` on eager, `to_pandas` on
  lazy) from the annotation's laziness;
- declared `DataFrame[Schema]` returns: pinned columns are checked
  exactly; unpinned ones pass with a visible leniency note.

Not guaranteed: anything about columns that only the data source
determines — existence and dtypes of unpinned columns. Discharging
those assumptions for file-backed sources is C-12b (committed schema
snapshots read from parquet/IPC footers).

## Amendments

- **Absence tracking ("lacks" constraints)** — implemented (issue #78).
  `FrameType.absent` carries the open frame's negative knowledge: names
  provably removed by `drop` / renamed away by `rename` (enumerable
  targets only; selector-based drops keep assumption semantics). A later
  reference is a provable error at every lookup site (expressions,
  select, drop/rename/cast/drop_nulls, join keys, the declared-return
  missing-column check, frame-argument subtyping); reintroducing the
  name (`with_columns`, a rename target, a join's other side) clears the
  mark. `FrameType.lacks(name)` is the shared "provably not a column"
  predicate.
- **Open-left join pins are collision-aware** (issue #79). A right-side
  column pinned into a join result with an OPEN left frame may collide
  with a left-rest extra at runtime — polars would suffix the *right*
  column away and the unsuffixed name would be the left column. The name
  provably exists either way, but the dtype is conditional: it degrades
  to `Unknown` (no manufactured proofs). Collisions with PINNED left
  names keep deterministic suffixes and precise dtypes; a closed left
  frame keeps every pin precise.

- **Non-strict return schemas bind open at call sites** (issue #81).
  `strict=False` is pandera's "at least these columns": `check_types`
  passes the caller's extras through, so the row-polymorphic helper
  pattern (`∀r. {price|r} → {price,total|r}`) binds its call result as
  an open frame — the caller keeps using its own columns (dtype
  degrades to Unknown through the call; pandera cannot share the row
  variable between input and output). `strict=True` returns stay
  closed — that closure is what makes select-style proofs possible.
- **Strict parameter schemas reject provable extras at call sites**
  (issue #82): a pinned extra column passed into a `strict=True`
  parameter is flagged (check_types validates arguments); unknown
  open-frame extras stay lenient.

- **Backward narrowing** — implemented. An assumption lookup on an
  open frame (expression `pl.col`, string selections) pins the column
  INTO the frame as `Unknown`: if line N succeeded, the column exists
  on every execution reaching line N+1, and object identity carries the
  positive knowledge forward (e.g. to strict-extra proofs at later call
  sites and extra-column return checks). Island lookups error instead
  of assuming, so they never pin; join keys and unpivot arguments do
  not pin (minor, assumption-only paths).
- **`pl.DataFrame(non_literal)` binds open** — implemented. The
  constructor provably builds SOME frame; untracked `None` lost all
  downstream checking. The no-args constructor `pl.DataFrame()` is the
  provably EMPTY closed frame (column references are proofs), and a
  readable data dict / `schema=` keeps the exact closed inference.
- **Bare return annotations check the eager/lazy bit** — implemented.
  `-> pl.DataFrame` with an inferred LazyFrame return (or vice versa)
  is PLY032; no schema claim is made, so an uninferable body stays
  silent (no could-not-infer error).
- **Checked-island semantics for non-strict declared schemas** (issue
  #83, design decision). A `strict=False` schema on YOUR OWN parameter
  (or a `Schema.validate` narrowing) binds CLOSED: the declaration is
  the function's interface, and referencing an undeclared column is
  flagged — but as `PLY042` with honest wording (the schema admits
  caller extras at runtime; the declaration just doesn't promise the
  column), not `PLY001`'s runtime certainty. The frame carries
  `nonstrict_schema` provenance through shape-preserving operations;
  shape-determining calls (`select`, aggregations) re-anchor exactness
  and PLY001 proofs. The alternative — binding non-strict schemas open
  (assumption semantics) — was prototyped and rejected: it silences the
  declared-interface checking that is the tool's core value for the
  default (non-strict) schema style, invalidating 159 tests/25 fixtures
  of intent. The asymmetry with call RESULTS (issue #81: a CALLEE's
  non-strict return binds open at the caller) is principled and mirrors
  structural typing precedent (TypeScript): your own annotation is your
  contract; someone else's return annotation is a lower bound on the
  value you receive. Row-polymorphic helpers have two honest spellings:
  declare the columns you touch, or take a bare `pl.DataFrame`.

## Future work

- **C-12b**: `polypolarism snapshot` writing committed path→schema
  snapshots from parquet/IPC metadata, turning source assumptions into
  verified facts without check-time IO.
