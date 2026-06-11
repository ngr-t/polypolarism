# ADR-0007: Unrecognized Field Annotations Degrade to Unknown-Dtype Columns with PLW011

- **Status**: Accepted
- **Date**: 2026-06-11
- **Deciders**: @negotetsu

## Context

Issue #77: a `pa.DataFrameModel` field annotation that
`parse_field_annotation` cannot translate was silently DROPPED — the column
simply did not exist in the resolved `FrameType`. Both failure modes
followed:

- **FP** (strict schema): correct code was rejected — every returned column
  was "extra" against the emptied declaration.
- **FN** (non-strict schema): a wrong dtype against the dropped column
  passed statically and failed at runtime.

The immediate trigger was two missing stdlib aliases (`decimal.Decimal`,
`datetime.time` — fixed in the alias maps), but the *silent drop* is the
structural hazard: any future gap in the parser reproduces both failure
modes invisibly.

What should an unparseable annotation do instead? Pandera ground truth
(probed, pandera 0.31.1 + polars 1.41.2):

- A genuinely-unresolvable annotation (`x: SomeUserClass`, `import time;
  x: time` — a module) raises `TypeError: cannot parse input ... into
  Polars data type` at the FIRST USE of the schema (`to_schema()` /
  `validate` / `@pa.check_types`), not at the class statement.
- But a bare name is not provably unresolvable from the AST: `MyAlias =
  pl.Int64; x: MyAlias` resolves fine at runtime — pandera evaluates the
  annotation object, polypolarism only sees the NAME.

## Decision

An annotation `parse_field_annotation` rejects registers the column anyway,
with `Unknown` dtype (`unrecognized_field_spec`; `Optional[...]` /
`Series[...]` wrappers still contribute requiredness), and the schema
records a per-field `definition_warnings` entry that the analyzer surfaces
as **PLW011** (warning, once per schema per function) on every function
referencing the schema — mirroring the PLY041 `definition_errors` plumbing
from issue #69.

Wrong-arity `Annotated` forms are excluded: PLY041 already carries their
verdict (those ARE provably broken), and a second PLW011 on the same field
would be noise.

## Alternatives considered

- **Keep the silent drop** — rejected: hides parser gaps as wrong verdicts
  in both directions (the issue itself).
- **Hard error (PLY041 family)** — rejected: pandera's own error is real
  only for *genuinely* unresolvable annotations, and unresolvability is not
  provable statically (the `MyAlias = pl.Int64` counterexample). PLY041 is
  reserved for provable runtime crashes; a hard error here would reject
  working code.
- **Silent `Unknown` (no diagnostic)** — rejected: the issue explicitly
  asks for a loud degrade, and a genuinely-broken schema (pandera TypeError
  at first use) would otherwise pass with zero signal. The PLW011 text
  names the field, the annotation source, and the runtime consequence.

## Consequences

- Strict schemas with an unrecognized field accept correct code (the
  declared slot exists; `Unknown` is the usual gradual-typing wildcard, and
  the ADR-0003 leniency notes make the `passed via Unknown` path visible in
  goldens).
- A wrong dtype against the degraded column still passes statically
  (Unknown accepts everything) — the FN narrows from "any code shape" to
  the ordinary, visible Unknown leniency, with PLW011 telling the user
  exactly which annotation to fix to restore precision.
- Inheritance/repair semantics mirror PLY041: warnings are inherited from
  parents; a child re-declaring the field with a recognized annotation
  clears the entry.
