# Backlog

Codebase assessment backlog (2026-06-11). Not bugs — intentional limits,
deferred work, and improvement candidates. GitHub issues were all closed
(#62 and earlier) at the time of this survey.

Status legend: `[ ]` open / `[x]` done / `[-]` deliberately deferred.

## A. Short-term (low cost, ready to pick up)

- [ ] **A-1: Finish valid/invalid fixture pairing (7 known gaps)**
  `tests/fixtures/README.md` lists rules without a valid/invalid twin
  (pivot, partition_by, landmark dtypes, frame literals, plural col,
  hstack, ...). Each is a small per-rule fixture + golden addition
  (ADR-0003 pairing convention).
- [ ] **A-2: Centralize duplicated numeric type sets**
  Same numeric-dtype sets defined in `expr_infer.py`, `ops/groupby.py`,
  and `analyzer.py`. Consolidate into one shared module.
- [ ] **A-3: Document Unknown-fallback (leniency) points**
  `analyzer.py` has ~15 explicit `return Unknown()` sites plus many
  `return None` paths that degrade to Unknown. All intentional, but the
  "where we give up" list is not user-visible. Add a README section so
  users can tell "checked" from "passed via Unknown".

## B. Mid-term (precision improvements)

- [ ] **B-4: Warn on unsupported/unprobed Polars methods**
  Methods absent from the dispatch tables silently fall through to
  Unknown. Add a warning diagnostic (new PLW code) so drift against new
  polars releases becomes visible instead of silent.
- [ ] **B-5: Rolling-window inference with non-literal args**
  `rolling_*` falls back to `Nullable[Float64]` when
  `window_size`/`min_samples` are not literals (`analyzer.py` ~1143).
  Constant propagation for simple cases would tighten the result.
- [ ] **B-6: Fill probed-matrix gaps in namespace methods**
  - `arr.eval(as_list=True)` → Unknown (polars 1.41 arg, issue #53 area)
  - `str.to_datetime(format=<non-literal>)` → Unknown
  - `dt.replace_time_zone/convert_time_zone(<non-literal tz>)` → Unknown

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

- Test discipline: 146 golden fixture pairs, hypothesis type-algebra
  laws, runtime differential harness at 96.1% with a justified skip list.
- No bare excepts, no TODO/FIXME comments, zero runtime dependencies.
- Polars support floor fixed at 1.37 with corpus evidence (ADR-0004).
