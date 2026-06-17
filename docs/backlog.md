# Backlog

Codebase assessment backlog (2026-06-11). Not bugs — intentional limits,
deferred work, and improvement candidates. GitHub issues were all closed
(#62 and earlier) at the time of this survey.

Status legend: `[ ]` open / `[x]` done / `[-]` deliberately deferred.

## A. Short-term (low cost, ready to pick up)

- [x] **A-1: Finish valid/invalid fixture pairing (7 known gaps)**
  Done 2026-06-11: 9 invalid twins added (pivot, partition_by, landmark
  dtypes, frame literals, pl constructors, variable annotations, plural
  col, struct rename, hstack); pair audit table updated. Discovered
  **N-1** below in the process.
- [x] **A-2: Centralize duplicated numeric type sets**
  Done 2026-06-11: `unwrap_nullable`/`wrap_nullable` moved to `types.py`.
  The three numeric sets are intentionally different (13-width coercion
  set / 4-width promotion lattice / 6-width probed agg subset) and now
  carry comments saying why widening each needs probing first.
- [x] **A-3: Document Unknown-fallback (leniency) points**
  Done 2026-06-11: README section "Leniency: when 'no error' is not a
  proof".

## B. Mid-term (precision improvements)

- [x] **B-4: Warn on unsupported/unprobed Polars methods**
  Done 2026-06-11: PLW007 fires at the expression-chain and namespace
  fall-throughs when the receiver dtype was precisely known (degraded
  receivers stay silent — no cascade; a `.cast(...)` directly after the
  call retracts the warning). Frame-level methods deliberately NOT
  covered: warning there needs a probed terminal-method table
  (`to_dicts`, `write_*`, `item`, ... legitimately return non-frames) or
  every terminal call becomes noise — recorded as **N-3** below.
- [x] **B-5: Rolling-window inference with non-literal args**
  Done 2026-06-11 — see decision notes below. The reported fallback was
  stale (fixed since issue #57); int-constant propagation added for
  `min_samples`/`window_size`/`ddof`. Discovered **N-2** below.
- [x] **B-6: Fill probed-matrix gaps in namespace methods**
  Done 2026-06-11: `arr.eval(as_list=True)` → `List(body dtype)`;
  `str.to_datetime` resolves `Datetime[UTC]` for all chrono offset
  directives (`%z`, `%:z`, `%::z`, `%:::z`, `%#z`). Non-literal
  `format`/`time_zone` stay Unknown (no "tz wildcard" in `types.Datetime`
  — imprecision-pinned with upgrade triggers).

## N. Discovered while working the backlog (2026-06-11)

- [x] **N-1: Variable annotation contradicting an inferable RHS passes
  silently (false negative).** Done 2026-06-11 per **ADR-0005**
  (two-direction rule; annotation still wins downstream):
  - [x] **N-1a (案1)**: PLW008 on every provable contradiction
    (warn-only phase), via the checker verdict engine — pivot/Unknown
    annotations stay silent.
  - [x] **N-1b (案3)**: reverse-direction classification — pure
    narrowing (incl. optional→required) keeps PLW008 with the
    `Schema.validate` remedy; neither-direction becomes PLY033.
  - [x] **N-1c**: fixtures `invalid/variable_annotation_contradiction`
    (+ runtime SKIP: annotations are inert at runtime) and
    `warning/annotation_narrowing`.
  - [x] **N-1d**: docs (README code tables, fixtures pair audit,
    CHANGELOG).
- [x] **N-2: rolling_mean/std/var/median/quantile on Float32 infer Float64
  (wrong width, false positive).** Done 2026-06-11: width follows the
  receiver in the rolling family AND the select/group_by reductions
  (shared `_float_reduction_width`, probed 1.41.2). Float16 deliberately
  NOT width-preserved in rolling (probed: widens to Float64). Spawned
  N-4/N-5 below.
- [x] **N-4: mean_horizontal width.** Done 2026-06-11: Float32 iff every
  operand is Float32 (probed 1.41.2); pinned in the
  `float32_reduction_width` fixture pair.
- [x] **N-5: groupby NUMERIC_TYPES width/coverage gap.** Done 2026-06-11:
  the agg receiver matrix is widened to all 13 numeric widths with a full
  probe (1.41.2) — small ints sum/product → Int64 (UInt8/16 land SIGNED),
  Float16 keeps width through every select reduction, 128-bit preserved.
  `infer_agg_result_type` gained `context="select"|"agg"`; the four
  probed rust-PANIC cells ({mean,median,quantile}×Float16,
  product×UInt128) are PLY011 errors in grouped contexts (group_by/agg
  chains/`.over()`), valid in select. Runtime harness now catches
  `PanicException` (BaseException) as the predicted crash. Pair:
  `valid/small_int_float16_reductions` + 2 invalid twins.
- [x] **N-3: PLW007 for unmodeled FRAME-level methods.** Done 2026-06-11:
  probed frame-returning sets (`EAGER/LAZY_FRAME_RETURNING_METHODS`,
  73/66 names from signature return annotations on 1.41.2) gate the
  warning — terminal methods and unknown names stay silent.
  `Schema.validate(<unmodeled call>)` retracts the warning (the frame
  variant of the cast retraction). Per-call-site dedupe prevents
  double-firing on re-analyzed nodes. Follow-up noted by the
  implementation: `df.x().pipe(Schema.validate)` does not retract (no
  evidence of need; revisit if it shows up).

## C. Long-term (design decisions required)

- [x] **C-7: Array width tracking** — Done 2026-06-11. `Array` carries
  `width: int | None`; widths parse from annotations/cast targets
  (`shape=` keyword, 1-tuples; multi-dim/non-literal → None wildcard).
  Probed (1.41.2): pandera rejects width mismatches (coerce cannot
  repair), width-change casts raise in both strict modes (PLY013), arr
  namespace + arr.eval preserve the receiver width. Unknown widths pass
  with a "via: unknown Array width" leniency note. Fixture pair:
  `invalid/array_width_mismatch` + existing `valid/array_dtype`.
- [-] **C-8: `pivot()` output schema inference** — data-dependent,
  fundamentally not inferable; current PLW005 warning + user annotation
  is the accepted design.
- [x] **C-9: Open-struct semantics** — Done 2026-06-12 (user-approved
  design). The original framing ("tighten = close, trading FN for FP")
  predated ADR-0006; the open-frame machinery enables the middle ground:
  bare `pl.Struct` (and unreadable `pl.Struct(...)` constructions) parse
  to `Struct({}, open=True)` instead of `Unknown` — the struct-ness is
  provable (probed: pandera's bare declaration validates any struct and
  rejects non-structs; `.str` on a struct is a runtime SchemaError), so
  wrong-namespace accessors became proofs (PLY012), while field lookups
  get ADR-0006 assumption semantics (`struct.field` pins Unknown,
  `unnest` opens the frame with pinned fields registered). Checker
  verdict mirrors Array-width/Enum-categories wildcards: overlapping
  pins compared, a pin provably absent from a CLOSED other side fails
  (struct dtypes are exact at runtime), otherwise leniency. Closed
  structs keep exact field proofs (typo detection unchanged). The full
  "checked island" tightening (require field declarations) remains
  possible but unattractive — bare `pl.Struct` is chosen precisely to
  avoid enumerating fields. Pair: `valid/open_struct` +
  `invalid/open_struct_misuse`.
- [-] **C-10: when/then mixed non-literal int branches degrade to Int32**
  (`expr_infer.py`, issue #40 area) — polars itself has no integer
  literal type; accepted leniency.
- [ ] **C-11: Programmatic / dynamic schema generation** (user request
  2026-06-11). Three feasibility tiers, cheapest first:
  1. [x] *Statically-readable object API* — Done 2026-06-12:
     module-level `NAME = pa.DataFrameSchema({"a": pa.Column(int,
     nullable=..., required=...)}, strict=..., coerce=...)` registers
     like a class schema keyed by the variable name; `schema.validate`
     narrowing, checked-island provenance, cross-file imports and
     PLW011 loud-degrade (string dtype aliases, non-literal kwargs)
     apply uniformly. Module-level only; `update_columns` /
     `rename_columns` and dotted `mod.schema.validate` not modeled.
  2. [x] *Constant-foldable construction* — Done 2026-06-12: dict
     comprehensions over literal/module-const string lists (key = bare
     loop var, no conditions; a loop-var-referencing value registers
     the known keys as Unknown + PLW011), `**` spreads of module-level
     column dicts, direct `Name` column-dict arguments, and
     `add_columns` / `remove_columns` derivation (probed immutable).
     Statement-level loops (`for c in ...: cols[c] = ...`) are NOT
     folded — comprehension-only by design.
  3. [-] *Genuinely dynamic* — deliberately NOT implemented per
     **ADR-0008** (2026-06-12). The open-frame degrade option shipped
     via issue #90 (unresolved schemas bind as open assumption frames +
     PLW011); the execute-user-code "schema provider" stays out of
     scope. Revival conditions in the ADR.
- [ ] **C-12: Implicit schemas for unannotated frames** (user request
  2026-06-11). Two independent halves:
  1. [x] *Open-frame propagation from unannotated sources* — Done
     2026-06-11 per **ADR-0006**: bare `pl.DataFrame` / `pl.LazyFrame`
     params bind empty open frames; `pl.read_*` / `pl.scan_*` infer
     open frames; join/rename/cast/drop_nulls/selectors/concat/unpivot
     made open-frame aware (assumption semantics — errors only on
     provable contradictions; provable conflicts still fire). Fixture
     pair `valid/open_frame_sources` + `invalid/open_frame_
     contradictions`. Follow-ups recorded in the ADR: absence tracking
     ("lacks" constraints for drop/rename), backward narrowing,
     `pl.DataFrame(non_literal)` as open frame, eager/lazy check for
     bare return annotations.
  2. [-] *Source-schema snapshots* (C-12b) — deliberately NOT
     implemented per **ADR-0008** (2026-06-12): polypolarism stays a
     purely static verifier; open-frame assumption semantics (ADR-0006)
     plus validate-narrowing cover the gap. Revival conditions recorded
     in the ADR.

- [ ] **C-13: import/annotation recognition gaps** (found while
  isolating the 2026-06-12 dotted-import report; both reproduce):
  1. Imports nested under `if TYPE_CHECKING:` are not followed —
     `_merge_imports` / `_merge_module_imports` only scan `tree.body`,
     so the canonical annotation-only schema import hits PLW006.
     Static-analysis convention is to treat `TYPE_CHECKING` as True.
  2. String annotations (`def f(df: "DataFrame[S]")`) are not parsed —
     the function is **silently skipped** (no functions found), which
     combines badly with (1) since TYPE_CHECKING imports force quoted
     annotations on Python < 3.12 without `from __future__ import
     annotations`. Parse `ast.Constant(str)` annotations via
     `ast.parse(..., mode="eval")` at the detection sites.

- [-] **C-15: Class-body statements unanalyzed (issue #110 remainder).**
  Issue #110 extended provable-error checking (missing-column references,
  dtype misuse) to statements outside a frame-typed function signature:
  module top level, top-level `if __name__` / compound blocks, and
  frame-untyped module-level functions (`def main() -> None:`) are now
  analyzed, seeded only from provably-known frames (a typed same-module
  call's closed return, `Schema.validate(...)`, or a `pl.DataFrame`
  literal). **Class-body statements outside methods remain deferred** —
  a frame pipeline written directly in a class body (rare) is not checked.
  Deferred deliberately, not for soundness (the reused analyzer only fires
  on provably-closed frames) but for scope: a class body needs its own
  name environment and interacts with `self`/`cls` attribute resolution
  (issues #104/#108). To cover it, drive a `FunctionBodyAnalyzer` over the
  `ClassDef` body statements with a fresh seed, mirroring the `<module>`
  pass in `_analyze_out_of_function_scopes`, and decide how class-level
  bindings should (or should not) feed method analysis. Low priority —
  driver/demo pipelines live at module / `main()` level, which is covered.

- [ ] **C-14: True row polymorphism via `Annotated`-carried row variables**
  (user request 2026-06-15). Research-grade precision/soundness upgrade for
  column-preserving helpers. Today an open frame's `rest` is an *anonymous*
  marker (`RowVar` exists but is inert — its name is never unified; see
  `_is_frame_subtype`, which consults `rest` only as an open/closed flag).
  So a non-strict in/out helper *accepts* wide inputs and lets the caller's
  extra columns pass downstream, but only via `Unknown` leniency: their
  dtypes are lost (no downstream checking) and preservation is not enforced
  (a helper that silently drops them still type-checks — a runtime
  `ColumnNotFoundError` waiting to happen). Genuine row polymorphism gives
  the `rest` an *identity* threaded input→output, recovering both.

  **Hard constraint (non-negotiable): do not deviate from Pandera.** Pandera
  stays the runtime authority and the de-facto surface; in practice users
  validate with Pandera and that is enough. The row variable is a
  *static-only* annotation that Pandera ignores at runtime. Surface:
  `Annotated[DataFrame[InId], Row("R")]` on params / returns — the base
  (`DataFrame[InId]`, open) is a real Pandera schema it still validates;
  `Row("R")` is runtime-inert metadata only polypolarism reads. No standard
  Python typing feature can express a threadable *named-field* row variable
  (`TypeVar` gives whole-schema identity only; `Concatenate`/PEP 612 is the
  right shape but scoped to callable params; PEP 728 `extra_items` is a
  *uniform* rest, not a variable; Python has no intersection / mapped types
  / record spread) — hence the `Annotated` side-channel. Accepted cost: it
  is a polypolarism *dialect* (mypy / pyright / Pandera see a plain open
  DataFrame); the threading is checked by polypolarism only.

  Tiers, cheapest / highest-de-risk first; each independently shippable and
  **opt-in** (no `Row(...)` ⇒ today's behavior byte-for-byte — golden
  fixtures for unannotated code must not move):
  **Surface as built (Tiers 1–5, on branch `feat/row-polymorphism`,
  2026-06-17):** the de-risk picked a runtime-inert **decorator**
  `@rowpoly("R")` beside a bare `DataFrame[Schema]` annotation — NOT
  `Annotated[..., Row("R")]`, which pandera 0.31 refuses to unwrap on a frame
  annotation, silently disabling `@pa.check_types`. See `src/polypolarism/
  rowpoly.py` and the C-14 Tier 1 notes at the bottom.
  1. [x] *De-risk + decide.* Done 2026-06-15/17 (commit 022c621 notes +
     `3b938da`). Decorator surface selected and pinned by a
     `tests/test_runtime_differential.py` case (a missing-column frame still
     raises `SchemaError` under `@pa.check_types` + `@rowpoly`, both decorator
     orders). A standalone ADR was not written — the rationale lives in the
     backlog Tier 1 notes; promote to `docs/adr/` if the feature graduates
     from research to shipped.
  2. [x] *Surface + binding, no semantics.* Done 2026-06-17 (`1b909de`).
     `analyze_function` recognizes `@rowpoly("R")` and records
     `FunctionAnalysis.row_var` / `FunctionSignature.row_var`; metadata-only,
     verdict unchanged.
  3. [x] *Precision (call-site instantiation).* Done 2026-06-17 (`a313a89`).
     `_thread_row_poly_extras` adds the caller's extras (arg columns − declared
     param columns) to the call result with their real dtypes; a wrong-dtype
     declaration of a preserved column now fails PLY040 where the open-frame
     Unknown leniency used to accept it.
  4. [x] *Soundness (preservation check).* Done 2026-06-17 (`f6028bb`, PLY043).
     `_check_row_preservation` skolemizes the row variable (sentinel column)
     and flags a return point that provably drops it. Static-only — the
     invalid fixture is runtime-SKIPped (caller-relative property).
  5. [x] *Row algebra + relations — partial.* Done 2026-06-17 (`b2e5a69`):
     per-parameter row variables `@rowpoly(a="R1", b="R2")` (multi-frame
     threading + per-variable preservation, so a join helper preserves both
     sides), plus threading-law property tests in `test_properties.py`.
     **Tier-5 remainder, 2026-06-17:**
     - [x] *Row add/drop/rename tracking — no false positive.* Audited
       whether `_check_row_preservation` (PLY043) wrongly flags
       legitimately-preserving bodies (`with_columns`, `drop("realcol")`,
       `rename({"realcol": ...})`, `select(pl.all())`, `select(cs.all())`,
       conditional/early-return, `pl.concat([df, df])`). **No false positive
       exists today** — the skolem sentinel survives all of them (the
       all-columns selectors resolve to the full column set including the
       sentinel; drop/rename only touch named real columns). Pinned by tests
       in `test_rowpoly_preservation.py` and the
       `valid/rowpoly_select_all_preserves` fixture. No new machinery was
       needed, so the `lacks`-as-polymorphic-constraint tracking is NOT
       built — it would be infrastructure with no defect to fix.
     - [-] *Explicit `R1 # R2` disjointness diagnostic — DEFERRED (cannot be
       made sound/useful).* The proposal was a definition-time warning when
       two `@rowpoly(a="R1", b="R2")` params' declared schemas share a
       "non-key" column name. It is dropped for three reasons:
       1. **No key signal at definition time.** `ColumnSpec` carries no
          key/unique marker and the body's join keys are not consulted by a
          schema-only check, so a shared column cannot be classified as key
          vs non-key. Every join helper shares its key (e.g. `id` in
          `rowpoly_multivar_join`), so any "shared column" warning is a
          **false positive** on the most common, correct case.
       2. **Shared declared columns are not row-variable extras.** `R1`/`R2`
          capture `arg.columns − param.columns` (`_thread_row_poly_extras`),
          i.e. columns *beyond* the declared schema. A column present in
          *both* declared schemas is excluded from both extras by
          construction, so it can never cause an `R1 # R2` collision — the
          premise of the check is conceptually wrong.
       3. **The real collision is already covered.** A genuine non-key
          overlap surfaces at the join (polars suffixes the right side,
          `tag` → `tag_right`) and is checked against the declared return by
          the existing PLY040 return-type comparison. A separate
          definition-time warning would be redundant where it is right and a
          false positive where it is not. A type checker must not flag
          correct code; this diagnostic stays deferred. Revive only if a
          schema-level key marker lands AND threading is extended so that
          row-variable extras (not declared columns) can provably collide.
  6. [~] *Ergonomics / tooling — partial.* Done 2026-06-17:
     - [x] JSON output (`--format json`) exposes each helper's bound row
       variable(s): `"row_var": "R"` for `@rowpoly("R")`,
       `"param_row_vars": {...}` for the keyword form (added only when
       present, so the payload stays backward-compatible). `FunctionAnalysis`
       gained `param_row_vars` (it already carried `row_var`).
     - [x] Docs: `docs/row-polymorphism.md` documents the dialect (surface,
       threading precision, `PLY043`, JSON exposure, the static-only /
       Pandera-runtime-authority constraint), with every example verified by
       running the CLI. Linked from `docs/README.md` / `README.md`; `PLY043`
       added to the `docs/diagnostics.md` code table.
     - Golden fixtures already exist (`valid/rowpoly_*`,
       `invalid/rowpoly_drops_row_variable`, plus the new
       `valid/rowpoly_select_all_preserves`). A *warning*-category fixture is
       not added: the only deferred row diagnostic (`R1 # R2` disjointness) is
       deferred as unsound (see Tier 5), so there is no `@rowpoly` warning to
       demonstrate.
     - [ ] **Still deferred:** optional inference of `R` for untyped helpers
       (no annotation / decorator) — would let polypolarism guess a row
       variable for a bare `pl.DataFrame` helper. Not built; needs a sound
       trigger that does not regress the zero-config open-frame behavior.

  Invariants across all tiers: Pandera is the runtime authority and the
  marker is runtime-inert (re-gated by a runtime-differential test each
  tier); the feature is opt-in with zero regression for pandera-only code.
  Honest framing: practical users stay on Pandera; this is a theoretical
  upper bound on static precision, not a daily-driver requirement.

## D. Tooling / distribution

- [x] **D-11: VS Code extension feature gaps** — done 2026-06-12:
  schema hover shipped (per-parameter / declared / inferred frames from
  the `functions` array `--format json` emits since 73c13a6); bundled
  install fixed (`nox -s setup` vendors polypolarism from GitHub main —
  hash-pinned requirements.txt can't carry a VCS dep — verified in the
  rebuilt vsix); `@vscode/test-electron` harness instantiated (`npm
  test` green: download VS Code, install ms-python.python, smoke
  suite); toolchains refreshed superseding 8 stale dependabot PRs
  (closed); LSP e2e suite extended with a hover test (sample now pins
  PLY042 after the checked-island re-bundle).
  - [ ] **D-11b (follow-up): QuickFix / rename** — blocked on core:
    diagnostics report function-body line spans, not expression-level
    column positions. QuickFix ("declare column on schema", "switch to
    bare param") and rename-aware narrowing need per-expression ranges
    in the JSON output first; design that core change before extension
    work resumes.
- [ ] **D-12: PyPI publication** — `publish.yml` is complete (build +
  Trusted Publishing via OIDC, `pypi` environment, v-tag or manual
  dispatch). The remaining steps are owner actions, not code:
  1. On pypi.org → account → Publishing → add a *pending publisher*:
     project `polypolarism`, owner `ngr-t`, repo `polypolarism`,
     workflow `publish.yml`, environment `pypi`.
  2. Create the `pypi` environment in the GitHub repo settings
     (optionally with a required reviewer as a publish gate).
  3. Cut the release: bump `version` in pyproject.toml if needed, tag
     `v0.1.0`, push the tag (or run the workflow manually).
  Publishing also fixes the extension's bundled-install story (D-11).

## Non-issues (verified healthy)

- Test discipline: 146+ golden fixture pairs, hypothesis type-algebra
  laws, runtime differential harness with a justified skip list.
- No bare excepts, no TODO/FIXME comments, zero runtime dependencies.
- Polars support floor fixed at 1.37 with corpus evidence (ADR-0004).

---

# Decision notes

Records for backlog items that did not warrant a full ADR
(see `docs/adr/` for the convention on larger design decisions).

## B-5 — rolling_* inference with non-literal window args (2026-06-11)

**Investigated.** The reported symptom ("non-literal `window_size` /
`min_samples` falls back to `Nullable[Float64]`") was stale: since issue
#57 (commit 8b10020) the dtype family is determined by (method, receiver
dtype) on every path — `rolling_sum(Int64, n)` infers `Int64?`, never
`Float64?` — and only nullability widens, which is the sound upper bound
(`T <: Nullable[T]`). Regression tests now pin this.

**Improved:** int-constant propagation for `min_samples` / `window_size` /
`ddof` (commit f6df123). A function-local `ms = 1` or module-level
`MIN_SAMPLES = 1` resolves like a literal through the existing
`var_consts` / `module_consts` machinery (issue #39), so const-bound
totality (`min_samples<=1`, `window_size=1`, `ddof=0`) is recognized.
Probed (polars 1.41.2): `min_samples in (0, 1)` fills every window for any
accepted `window_size` (`window_size=0` is an expanding window with no
nulls; negative raises before producing a frame).

**Deliberately skipped:** flow-sensitive constant tracking. The const
machinery is visit-order based and a rebinding inside an `if`/`for` branch
overwrites the recorded value — the same accepted hazard as the string
constants that resolve join keys. Building per-branch environments would be
new infrastructure for a niche gain (window constants are rarely
conditionally rebound) and stays out.

## C-14 Tier 1 — row-variable surface de-risk (2026-06-15)

**Goal:** decide a surface for the static row variable that does NOT deviate
from Pandera (Pandera must stay the runtime authority validating the base
schema). Probed against pandera 0.31.1 / polars 1.41.2 (the `runtime`
group).

**Finding — the naive `Annotated` surface FAILS the hard constraint.** With
`@pa.check_types def f(df: Annotated[DataFrame[S], marker]) -> ...`, pandera
does **not** unwrap `Annotated` on a frame annotation: it stops recognizing
the parameter as a pandera DataFrame and **skips validation entirely**. A
frame missing the required `id` column passed cleanly (vs the bare
`DataFrame[S]` baseline, which raised `SchemaError: column 'id' not in
dataframe`). Return validation was skipped too. So wrapping the frame
annotation in `Annotated` silently disables Pandera's runtime check — a
direct violation of the constraint. (The type hint is preserved —
`get_type_hints(include_extras=True)` shows the `Annotated[...]` — pandera
just doesn't look through it.)

**Decision — runtime-inert decorator surface.** Keep the annotation a bare,
pandera-validated `DataFrame[S]` and carry the row variable in an
orthogonal decorator:

```python
@pa.check_types
@pp.rowpoly("R")                      # ships from polypolarism; returns fn unchanged
def add_score(df: DataFrame[S]) -> DataFrame[OutScore]: ...
```

Probed: pandera validation fires on a bad frame in **both** decorator orders
(`@pa.check_types` outer or inner), because the decorator is identity and
the annotation is untouched. The marker is reachable at runtime
(`fn.__pp_rowpoly__`) and, more importantly, in the AST polypolarism reads.

**Static side already supports it.** Running the current analyzer:
`Annotated[DataFrame[S], ...]` params are NOT recognized today (the function
is silently skipped — frame-level annotation detection doesn't unwrap
`Annotated`, unlike field-dtype parsing in `pandera_dtype.py`), whereas a
bare `DataFrame[S]` carrying an unknown `@rowpoly("R")` decorator analyzes
exactly as today (unknown decorators are ignored; a return-type mismatch on
such a function still fires PLY040). So the decorator surface is purely
additive on the static side — no regression, and the row-var reader is a
new decorator-recognition pass, not a rework of annotation handling.

**Why the decorator over `Annotated`, beyond the constraint:** it is
*version-robust* — it sidesteps whatever any pandera version does with
`Annotated`, because the annotation stays the plain form pandera already
validates. `Annotated` remains the right carrier for *field-level* dtype
metadata (pandera already uses it there); it is wrong only for the
*frame-level* row variable.

**Scope note (common case needs no extra algebra):** the "add a column"
part of a preserving helper is already captured by the difference between
the param schema (`InId`) and the return schema (`OutScore`); a single
`@pp.rowpoly("R")` (all open frames in the signature share `R`) covers
preserve+add. Per-position binding (`@pp.rowpoly(df="R", returns="R")`) and
explicit drop/rename are only needed once joins / two independent rests
arrive — deferred to Tiers 4–5.

**Remaining for Tier 1:** write the ADR recording this decision; pin the
constraint with a `tests/test_runtime_differential.py` case (bad frame
under `@pa.check_types` + `@pp.rowpoly` must still raise `SchemaError`).
