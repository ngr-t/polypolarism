# Leniency: when "no error" is not a proof

Polypolarism is a **gradual** type checker: under uncertainty it stays
silent rather than risk a false positive. A clean run therefore means
"no mismatch was *provable*", not "every column was checked precisely".
Three leniency rules can let a column pass without a precise check:

| Rule | When it applies |
|---|---|
| **Unknown compatibility** | a column's inferred dtype degraded to `Unknown` somewhere upstream; `Unknown` satisfies any declared dtype (and vice versa), at any nesting depth (`List[Unknown]` satisfies `List[T]`) |
| **Open frame** | open-record width subtyping: after an operation that can carry extra unknown columns, a declared column that is missing from the inferred frame is "not provably absent" — skipped, not an error |
| **Coerce** | with `class Config: coerce = True`, dtype differences that pandera would cast away at validation time (e.g. `UInt32` vs `Int64`) are not errors |

The main constructs whose inference intentionally degrades to `Unknown`:

- **Unmodeled methods** — a method the analyzer does not model (including
  methods added in polars releases newer than the probed set) yields an
  `Unknown` result rather than a guess. Downstream of such a call, dtype
  errors are no longer detectable. When the receiver dtype was precisely
  known, the degradation is surfaced as `pplw-unmodeled-method`; a `.cast(...)` directly
  after the call both restores precision and retracts the warning.
- **Value-dependent arguments** — when a result dtype depends on a
  *runtime value* that is not a literal in the source (a dtype passed via
  a variable to `cast`, a non-literal `time_zone=` / `format=` argument,
  …), the result degrades to `Unknown`. Using a literal restores precise
  checking.
- **Data-dependent schemas** — `pivot()` output columns depend on the
  data; this one is surfaced as `pplw-data-dependent-schema` instead of silent leniency (see
  the warning table in [Diagnostics](diagnostics.md)).
- **External / untyped helpers** — `pipe`/calls into code the analyzer
  cannot see degrade to best-effort inference and warn (`pplw-unresolved-pipe`–`pplw-untyped-callable`).
- **Open-frame column reads** — `pl.col("x")` on an open frame where `x`
  is not statically known resolves to `Unknown` (it may be one of the
  unknown extra columns).

How to keep leniency visible:

- Where a concrete source change restores precision, polypolarism emits a
  `pplw-*` warning naming it (see
  [Apply-style helpers and warning codes](diagnostics.md#apply-style-helpers-and-warning-codes)).
- In this repo's own test suite, every golden fixture report renders an
  indented `via:` note for each leniency-mediated pass, and unit tests
  pinning an `Unknown` fallback carry `@pytest.mark.imprecision` with an
  upgrade trigger — so the exact set of lenient spots is reviewable in
  `tests/fixtures/*/*.expected` (grep for `via:`).
