# ADR-0003: Dual-Direction Fixture Testing (Leniency Made Visible)

- **Status**: Accepted
- **Date**: 2026-06-11
- **Deciders**: @negotetsu

## Context

polypolarism is deliberately lenient under uncertainty (gradual typing):
an `Unknown` dtype satisfies any declaration, an open inferred frame
(`rest is not None`) excuses missing columns, and `Config.coerce` tolerates
coercible dtype differences. Leniency keeps false positives near zero — but
the test suite repeatedly interacted badly with it, in two distinct ways:

1. **Leniency-pinning tests break on precision upgrades.** Tests asserting
   "this expression degrades to Unknown" or "this passes via the open-frame
   rule" pin *today's imprecision*. Each time inference later became
   precise, those tests broke and had to be hand-triaged with no marker
   telling the triager which way to resolve them: `pl.len()`
   (Unknown → UInt32), when/then/otherwise (Unknown → supertype, #40),
   `.arr` on a List column (accepted → flagged pple-wrong-namespace-dtype). The dangerous
   failure mode in triage is "weaken the new feature until the old test
   passes" instead of "upgrade the assertion".

2. **Worse: valid fixtures that pass *via* leniency mask false negatives.**
   A `valid/` fixture proves nothing about the rule it appears to cover if
   its pass is leniency-mediated. Concrete incidents:
   - **#47**: `pl.struct(x=...)` silently produced `Struct{}` for months.
     The surrounding when/then degraded to Unknown and the open-frame rule
     let every downstream declaration through — the valid struct fixture
     kept passing the whole time, *via leniency instead of via the rule*.
   - **#55**: `list.sum()` on `List(String)` regressed to accept-anything
     when a probed dtype table left invalid cells silently Unknown; again
     no fixture failed, because acceptance was indistinguishable from a
     correct pass.

   The common mechanism: **a pass does not say why it passed.** Nothing
   distinguished "passed because inference proved the dtype" from "passed
   because inference gave up".

ADR-0002 established the golden-fixture harness, the diagnostic-code
coverage gate, and property-based type-law tests. This ADR builds on it to
guard both directions of every rule and to make leniency-mediated passes
impossible to miss.

## Decision

### (a) Pairing convention: every rule ships a valid + invalid fixture pair

Documented and audited in `tests/fixtures/README.md`. Every inference rule
gets:

- a **valid** fixture: the correct declaration passes (no false positive);
- an **invalid** twin: a *wrong declaration of the same operation* fails
  (no false negative).

The invalid twin is the only artifact that detects the #47/#55 failure
mode: if inference degrades to Unknown, the valid fixture keeps passing,
but the wrong declaration starts passing too — and the `invalid/` category
invariant ("at least one function fails", ADR-0002) fails the build.

The README contains the corpus audit: the pairs that exist, five
previously-missing high-value twins added with this ADR (struct field
dtype — the #47 rule itself; left-join nullability; selector expansion;
`map_elements` `return_dtype`; validate-narrowing), and the remaining gaps
as an explicit backlog ("add the invalid twin when touching the rule").

### (b) `imprecision` marker: leniency-pinning tests carry an upgrade contract

A pytest marker registered in `pyproject.toml`:

> test pins current inference imprecision (Unknown fallback / leniency);
> when the construct gains precise inference, UPGRADE the assertion
> instead of weakening the feature.

Applied (50 tests at adoption time) to tests asserting Unknown-degradation
or leniency-acceptance — open-frame skips, `interpolate()`-style
un-inferable chains, silent cells of the probed arithmetic / comparison /
`is_in` / cast matrices — each with a one-line comment naming the upgrade
trigger (e.g. "open frames gain row-var bounds"). When a precision upgrade
breaks a marked test, the resolution is mechanical: upgrade the assertion.

Checker-level rule-definition tests (e.g. `TestUnknownCompatibility`) are
deliberately *not* marked: they construct `Unknown` directly to define the
leniency rule's semantics and cannot break from construct-level precision
gains.

### (c) Leniency tracing baked into the goldens

`CheckResult` gains a `leniency: list[str]` field (default empty, never
affects `passed`). `checker.py` records a note whenever the
declared-vs-inferred check passes via a leniency rule:

- **Unknown-dtype compatibility** (either side, including `List`/`Array`
  recursion): `column 'x': passed via Unknown`. Implemented by refactoring
  the subtype check into `_subtype_verdict` returning a `Verdict`
  NamedTuple (`ok` + optional `reason`); `_is_subtype` remains as a boolean
  wrapper so existing call sites keep working.
- **Open-frame missing-column skip**: `column 'y': not provably absent
  (open frame)`.
- **Coerce-tolerated dtype difference**: `column 'z': UInt32 -> Int64 via
  coerce`.

The golden renderer appends an indented `via:` line per note. Regenerating
all goldens surfaced the corpus's entire current leniency surface as a
single reviewable diff — nine `invalid/` fixtures whose error-poisoned
downstream columns register as Unknown next to the real error, plus the
intentionally lenient `valid/` fixtures (`coerce_len_agg`,
`agg_len_filter`, `unknown_dtype_tracking`, `container_dtypes`). From now
on, **any new dependence on leniency shows up as a golden diff** and must
be justified in review. The CLI's human output is unchanged; leniency notes
are a harness/golden concern.

### (d) Runtime differential harness (implemented separately)

A test module executes the fixture corpus against **real polars + pandera**:
inputs are synthesized from each function's declared input schemas
(best-effort; fixtures whose inputs cannot be synthesized go on an explicit
skip-list with a reason). Valid fixtures must succeed at runtime; invalid
fixtures must fail. This is the ground-truth complement to (a)–(c): it
catches cases where polypolarism's model and polars' actual behavior drift
apart — including drift introduced by new polars/pandera versions.

Design constraints (decision level; implementation in a concurrent change):

- lives behind a dedicated `runtime` dependency group and pytest marker so
  the core suite stays dependency-free and fast;
- best-effort input synthesis with an explicit, reasoned skip-list rather
  than silent skips;
- a dedicated CI job runs it against the supported polars version window.

## Consequences

- **Golden diffs become the review surface for leniency.** A fixture whose
  pass flips from precise to leniency-mediated produces a visible `via:`
  diff; blindly regenerating goldens without reading that diff is the new
  (much smaller) residual risk, mitigated by the `invalid/` category
  invariant and the pairing convention.
- **Leniency-pinning tests carry an upgrade contract.** Precision upgrades
  still break them — by design — but the marker plus trigger comment makes
  triage mechanical and biases resolution toward upgrading assertions, not
  weakening features.
- **The corpus audit is honest about gaps.** Missing invalid twins are a
  documented backlog instead of an unknown unknown; the convention makes
  the twin part of the definition of done for new rules.
- **Runtime drift across polars versions becomes detectable** once (d)
  lands, at the cost of a slower, dependency-carrying CI job that is kept
  out of the default test run.
- `CheckResult` grows a field; downstream consumers that construct it by
  position are unaffected (keyword field with a default), and the CLI
  output is unchanged.
- Golden regeneration now happens for *any* change to the traced leniency
  rules, adding small churn — accepted, as that churn is exactly the
  visibility this ADR exists to create.
