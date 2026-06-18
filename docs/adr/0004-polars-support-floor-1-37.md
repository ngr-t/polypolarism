# ADR-0004: Fix the Polars Support Floor at 1.37

- **Status**: Accepted (supersedes the "latest two 1.x minors" window of
  ADR-0001, Decision item 1 / item 3's floor rule)
- **Date**: 2026-06-11
- **Deciders**: @negotetsu

## Context

ADR-0001 defined the supported polars range as a *moving window*: the
latest two 1.x minor releases, computed as
`POLARS_LATEST_KNOWN.minor - 1`. That was a placeholder policy chosen
before any empirical data existed about where polypolarism's analysis
claims actually stop holding on older minors.

That data now exists. A 35-example corpus of realistic polars+pandera
programs was executed against each minor from 1.41 downward:

| Version   | Newly broken examples (cause)                                  |
|-----------|----------------------------------------------------------------|
| 1.37–1.41 | all 35 OK (only two `hasattr` guards: `gather` 1.41, `bin.get` 1.38) |
| 1.36      | upsample row counts; Bool/String schema (behavior changes)     |
| 1.35      | + `over` inside `agg` (behavior); `show`; `bin.slice`          |
| 1.34      | + `name.replace`; `glimpse(return_type=)`                      |
| ≤1.31     | + `pl.Categories`; `arr.mean`                                  |

Two distinct failure classes emerged:

- **Method availability** (`gather`, `bin.get`, `name.replace`, …):
  these only affect code that uses the newer method, degrade loudly at
  runtime, and could in principle be papered over per-method.
- **Behavior changes** (`upsample` row counts, `over` inside `agg`,
  Bool/String schema differences): the *same code* produces *different
  schemas or errors* across the boundary. polypolarism's dispatch tables
  and probed rule matrices (pple-incompatible-operands/pple-invalid-cast/pple-non-numeric-operand layers, supertype,
  rolling/cum nullability, …) are all probed against ≥1.37 polars; below
  the boundary the analyzer would assert claims that are simply wrong,
  and no `hasattr`-style guard can absorb a behavior change.

The behavior-change class has its floor at 1.36–1.37, which caps how far
the supported range can usefully extend regardless of how many
method-availability guards are added.

## Decision

1. **`POLARS_FLOOR` is a fixed minor: `1.37.0`.** It no longer derives
   from `POLARS_LATEST_KNOWN`. Bumping `POLARS_LATEST_KNOWN` on new
   polars releases does not move the floor; the floor moves only as a
   deliberate decision backed by corpus evidence (a future ADR or an
   amendment here).
2. **Everything below 1.37 is best-effort**, surfaced via the existing
   `pplw-unsupported-version` warning on exact-source detections. The pre-1.0 surface
   remains explicitly out of scope (unchanged from ADR-0001).
3. **The probed rule matrices target ≥1.37 semantics.** When a probe is
   version-sensitive within the supported range, the newest behavior is
   encoded (consistent with ADR-0001's "recognizing newer names on older
   polars is harmless" stance); behavior differences below the floor are
   not modeled.

## Consequences

- Users on 1.37–1.38 stop receiving a spurious `pplw-unsupported-version` (under the old
  window they were "unsupported" the moment 1.40/1.41 shipped, despite
  the analysis being accurate for them).
- The floor stops silently ratcheting upward with every
  `POLARS_LATEST_KNOWN` bump — support promises only change when someone
  decides they should.
- The corpus run (35 examples × minor) becomes the floor's evidence
  base; re-running it is the prerequisite for any future floor move, in
  either direction.
- `version_check.POLARS_SUPPORT_NOTE`, the README support section, and
  the landmark version tests now pin 1.36 → warn / 1.37 → silent.
