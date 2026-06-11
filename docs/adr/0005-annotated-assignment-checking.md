# ADR-0005: Check Annotated Assignments Against the Inferred RHS

- **Status**: Accepted
- **Date**: 2026-06-11
- **Deciders**: @negotetsu

## Context

`x: DataFrame[A] = df.select(...)` is currently a *trusted assertion*:
`visit_AnnAssign` adopts the annotation as the variable's static type
unconditionally and discards the RHS's inferred frame type (the RHS is
walked only so expression-level warnings/errors surface). While pairing
the fixture corpus (backlog A-1) this was found to hide a real false
negative: an annotation that contradicts a *precisely inferable* RHS —
`x: DataFrame[A] = expr` where the expression provably infers `B ≠ A` —
produces zero diagnostics, and every downstream check then trusts the
wrong schema.

Two legitimate uses pull in opposite directions:

- **Checked declaration** (mypy/pyright intuition: `x: int = "s"` is an
  error). polypolarism's entire value proposition is comparing declared
  against inferred schemas; an unchecked declaration position is a hole
  in that loop.
- **Deliberate re-typing**. `PLW005` explicitly tells users to annotate
  the result of `pivot()` (whose schema is data-dependent), and users
  legitimately assert *narrower* types than the conservative inference —
  the canonical case is annotating a post-left-join column non-nullable
  (`Int64` over an inferred `Int64?`) when they know every key matches.

A blanket "error on contradiction" would break the second use and would
amplify every analyzer imprecision into a user-facing false positive at
each annotation site. A blanket "trust the annotation" keeps the false
negative. Note that the `PLW005`/pivot workflow is *not* actually in
tension: there the inference degrades to `Unknown`, and the comparison
engine's Unknown-compatibility leniency (ADR-0003 family) already treats
that as non-contradictory — only *precise* disagreements are at stake.

## Decision

Annotated assignments are checked with the same verdict engine as
declared return types (`checker._subtype_verdict` semantics: Unknown
compatibility, open-frame skips, `coerce` differences), classified by a
**two-direction rule**:

1. **Forward holds** (`inferred <: declared`): pass, exactly as today.
2. **Forward fails, reverse holds** (`declared <: inferred` — a pure
   *narrowing assertion*: non-null over nullable, required over
   optional): allowed, surfaced as **`PLW008`** naming the runtime-backed
   upgrade (`Schema.validate(...)` retypes with an actual check).
3. **Neither direction holds** (unrelated dtype, a column provably
   absent from a closed inferred frame, extra columns under a `strict`
   annotation, eager/lazy mismatch): **`PLY033` error** — the annotation
   re-interprets the frame as something it provably is not.

In every case the annotation still wins for the variable's downstream
type: one clear diagnostic at the assignment, no cascading errors (the
same convention as `PLY013`'s degrade-after-error).

Rollout is phased in two steps within the same release train so each
step's golden diff stays reviewable: first `PLW008` fires on *every*
provable contradiction (warn-only, exit code unchanged); then the
unrelated-contradiction class is promoted to `PLY033`, with `PLW008`
remaining on the narrowing class.

## Alternatives considered

- **Status quo** (annotation always wins, silently): keeps the false
  negative; rejected.
- **Warn-only forever**: a provable re-interpretation to an unrelated
  type is a bug, not a style issue; warnings are routinely ignored and
  the CLI exits 0. Kept only as the transitional phase.
- **Error on every contradiction**: kills the narrowing assertion (the
  post-join non-null case would force a runtime `Schema.validate` or a
  weaker annotation everywhere) and turns any future inference bug into
  immediate false positives at every annotation site; rejected.

## Consequences

- The `variable_annotation` invalid-twin gap in
  `tests/fixtures/README.md` ("annotation contradicting the assigned
  expression") becomes fixable: the contradiction now fails, so the twin
  pins it.
- The runtime differential harness needs a justified SKIP entry for the
  contradiction fixture: annotations are inert at runtime (pandera
  validates only via `validate`/`check_types`), so a static `PLY033`
  FAIL with a runtime success is by design — same family as the existing
  `PLY032` annotation-contract skip.
- The narrowing direction remains runtime-unverified *by construction*;
  `PLW008`'s remedy text makes `Schema.validate` the blessed escape
  hatch when the assertion actually matters.
- `typing.cast(...)` stays a passthrough (`_unwrap_cast`) and is NOT an
  escape hatch for frame re-typing; the annotation (with its PLW008
  trace) or `Schema.validate` (with its runtime check) are.
