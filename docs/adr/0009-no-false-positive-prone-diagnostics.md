# ADR-0009: No false-positive-prone diagnostics (and the deferred row-variable disjointness check)

Date: 2026-06-17
Status: accepted (deliberate non-implementation; revivable)
Backlog: closes C-14 item 5 (polymorphic lacks / disjointness) as deferred

## Context

polypolarism is a **gradual, soundness-first** checker: under uncertainty
it stays silent rather than risk a false positive (see ADR-0006's
open-frame assumption semantics and `docs/leniency.md`, "when 'no error'
is not a proof"). A clean run means "no mismatch was *provable*", not
"every column was checked". Across the issue work (#109–#112) and the
`@rowpoly` row-polymorphism tiers (C-14), every diagnostic was held to the
same bar: it fires only on a **provable** runtime failure, and degrades to
silence (or `Unknown`) when it cannot prove one.

C-14 item 5 proposed "polymorphic lacks / disjointness" static constraints.
Investigating it (2026-06-17) showed the **practical core is already
shipped** and the **pure residual cannot be added soundly**:

- *Monomorphic lacks* — a concrete frame tracking columns provably removed
  by `drop`/`rename` — already exists (`FrameType.absent`, ADR-0006;
  exercised by #109/#110).
- *Exact-column lacks* — `class Config: strict = True` already means "these
  columns and no others", the strongest practical "frame lacks X".
- *Collision-as-runtime-error* — `rename` to an existing/duplicate target
  and duplicate `select`/`alias` outputs are caught (pple-duplicate-column, extended for
  rename); the join non-key `_right` suffix is already modeled in the join
  result schema (verified).
- *Declared restriction of the rest* — `@rowpoly("R", drops=<selector>)`
  declares an intended pattern-drop and is checked precisely (Feature 4).

What remains of item 5 is the **definition-time `R1 # R2` disjointness
diagnostic**: warn, where a helper is `@rowpoly(a="R1", b="R2")`, that the
two row variables might overlap. This was already deferred once (the C-14
Tier 5 "1b" investigation) and the reasoning holds:

1. **Shared declared columns are not row-variable extras.** A row variable
   captures `arg.columns − declared param columns` (`_thread_row_poly_extras`).
   A column present in *both* declared parameter schemas is excluded from
   *both* extras by construction, so it can never cause an `R1 # R2`
   collision — the premise of the check is conceptually wrong.
2. **No key signal at definition time.** `ColumnSpec` carries no key/unique
   marker, and the body's join keys are not consulted by a schema-only
   check. The one shared column a join helper always has is its key — so any
   "shared column" warning is a **false positive on the most common, correct
   case** (e.g. `id` in `merge(a, b)` joining on `id`).
3. **Caller frames are unknown at definition time.** Whether two *actual*
   arguments' extras overlap is a per-call fact. At a call site the frames
   are known, but there polars merely **suffixes** a non-key collision
   (`c` / `c_right`) — not a runtime error — so there is nothing to flag;
   the suffix is already reflected in the inferred result schema.

A separate, *sound* residual exists — when two callers' extras collide by
name through a `@rowpoly` join helper, threading degrades the colliding
column to `Unknown` instead of modeling the `c` / `c_right` suffix. That is
an **imprecision (false negative via `Unknown`), not a false positive**, on
a contrived case (a row-poly join helper whose two callers happen to share
an extra column name). It is deferred as niche, not as unsound.

## Decision

**Do not ship a diagnostic that produces false positives, even if it would
catch some real issues.** In a soundness-first checker a false positive
(rejecting correct code) is worse than a missed case: it trains users to
ignore or suppress diagnostics and erodes trust in every other one. When a
property cannot be *proven* from the statically-available facts, the
checker stays silent (or degrades to `Unknown` / an open frame), optionally
emitting a `PLW###` advisory that names a concrete source change which would
make precise checking possible — never a hard error on unprovable ground.

Concretely:

- The **`@rowpoly` definition-time `R1 # R2` disjointness diagnostic is
  deliberately not implemented** — it is unsound (false-positive on every
  join helper's key; shared declared columns cannot collide by
  construction; caller overlap is not a runtime error). C-14 item 5's
  disjointness sub-item is closed as **deferred (unsound)**.
- The **niche `@rowpoly`-join colliding-extras suffix precision** (model
  `c` / `c_right` instead of degrading to `Unknown`) is deferred as **niche
  (sound but low-value)**, not built now.
- A **polymorphic "open-but-lacks-X" annotation** is not added — it has no
  consumer: the runtime errors it could guard (`rename`-to-existing) are
  already covered (pple-duplicate-column) and the rest (`with_columns` overwrite, join
  suffix) are not errors. `strict` and `FrameType.absent` cover the real
  needs.

This ADR generalizes the rule already applied piecemeal: PLW-advisories
over hard errors under uncertainty (`docs/diagnostics.md`), open-frame
leniency (ADR-0006), the `Unknown`-degradation points (`docs/leniency.md`),
and the conservative boundaries of pple-rowpoly-not-preserved / the duplicate-column and
pattern-drop checks.

## Consequences

- Item 5 is closed: its practical core is shipped; its unsound part is
  permanently deferred (this ADR); its niche part is deferred pending need.
- New diagnostics must demonstrate they fire only on a **provable** failure
  (the dual-direction fixture harness, ADR-0003, and the runtime-
  differential harness are how that is shown — an `invalid/` fixture's
  flagged operation must actually raise at runtime, or be a declared-
  contract violation with a documented static-only justification).
- Reviewers should reject a proposed diagnostic that cannot be shown free
  of false positives, and prefer a `PLW###` advisory or silence instead.

## Revival conditions

Revive the disjointness diagnostic only if BOTH: (a) a schema-level
key/unique marker lands so a join key can be excluded from the "shared
column" signal, AND (b) threading is extended so that *row-variable extras*
(not declared columns) can be shown to provably collide — at which point
the check would fire on real overlaps, not on keys. Revive the suffix
precision when a real `@rowpoly`-join helper with colliding caller extras
appears in practice.
