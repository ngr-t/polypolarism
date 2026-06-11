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

## Future work

- **Absence tracking ("lacks" constraints)**: after `drop("a")` on an
  open frame, a later reference to `"a"` is provably dead (drop
  succeeded ⟹ `a` existed ⟹ it is now gone) — needs negative row
  knowledge the FrameType does not carry yet. Same for `rename`.
- **Backward narrowing**: `df.select("a")` succeeding implies `df`
  itself has `a` for *subsequent* statements. Currently unobservable
  (the pin would be `Unknown` and the frame stays open) — revisit if
  absence tracking lands.
- **`pl.DataFrame(non_literal)`** could produce an open frame instead
  of untracked; left out to keep this slice focused on declared intent
  (annotations) and file sources.
- **Bare return annotations** could check the eager/lazy bit (a
  `-> pl.DataFrame` function returning a LazyFrame is a real bug) —
  needs an "empty declared schema" that does not trigger the
  could-not-infer error.
- **C-12b**: `polypolarism snapshot` writing committed path→schema
  snapshots from parquet/IPC metadata, turning source assumptions into
  verified facts without check-time IO.
