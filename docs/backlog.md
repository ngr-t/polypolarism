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
- [ ] **N-5: groupby NUMERIC_TYPES width/coverage gap.** `ops/groupby.py`
  rejects Float16/Int8/Int16/UInt8/UInt16/Int128/UInt128 receivers for
  numeric aggs, so e.g. select-context `mean(Int8)` (valid → Float64) and
  `mean(Float16)` (valid in select → Float16) are falsely flagged.
  CAUTION (probed 1.41.2): `group_by().agg(mean)` on Float16 PANICS in
  rust — any fix must distinguish select vs agg contexts.
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
- [ ] **C-9: Stricter open-struct semantics** — bare `pl.Struct` means
  "any fields"; field-name typos surface only at runtime. Tightening
  trades false negatives for false positives.
- [-] **C-10: when/then mixed non-literal int branches degrade to Int32**
  (`expr_infer.py`, issue #40 area) — polars itself has no integer
  literal type; accepted leniency.

## D. Tooling / distribution

- [ ] **D-11: VS Code extension feature gaps** (vscode-polypolarism,
  v0.1.0 preview) — QuickFix / hover / rename not implemented; bundled
  install path broken (works only with `importStrategy:
  "fromEnvironment"`).
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
