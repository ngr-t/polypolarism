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

- [ ] **N-1: Variable annotation contradicting an inferable RHS passes
  silently (false negative).** `visit_AnnAssign`
  (`src/polypolarism/analyzer.py`, ~3853) lets the annotation win
  unconditionally: `x: DataFrame[A] = df.select(...)` where the select
  infers B≠A produces zero diagnostics. Design question to settle first:
  is the annotation a *checked* declaration (error/warn on contradiction)
  or a trusted assertion like validate-narrowing? Deserves an ADR.
- [ ] **N-2: rolling_mean/std/var/median/quantile on Float32 infer Float64
  (wrong width, false positive).** Probed (polars 1.41.2): these return
  **Float32** on a Float32 receiver. Affects literal-arg calls too — a
  `Float32` declaration over a Float32 rolling mean is falsely rejected.
  Needs a full float-family receiver probe before fixing.
- [ ] **N-3: PLW007 for unmodeled FRAME-level methods.** The B-4 warning
  covers expression chains and namespaces only. An unmodeled frame method
  (`df.unknown_method()`) silently untracks the variable. Warning there
  requires a probed table of terminal methods that legitimately return
  non-frames (`to_dicts`, `write_*`, `item`, `height`, ...), otherwise
  every terminal call warns.

## C. Long-term (design decisions required)

- [ ] **C-7: Array width tracking** — `Array[Int64, 3]` vs `Array[Int64, 5]`
  compare equal (`types.py`). Needs Decimal-style parametrization; low
  user impact, high cost.
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
- [ ] **D-12: PyPI publication** — `publish.yml` exists but is not wired
  up. Publishing also fixes the extension's bundled-install story (D-11).

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
