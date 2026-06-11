# Backlog notes

Decision records for backlog items that did not warrant a full ADR
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

**Follow-up finding (not B-5, unfixed):** probed (polars 1.41.2),
`rolling_mean/std/var/median/quantile` on a Float32 receiver return
**Float32**, not Float64 — the analyzer's `_ROLLING_FLOAT_METHODS` branch
currently infers `Float64`(?) for every receiver, a wrong width that
affects literal-arg calls too (a `Float32` declaration over a Float32
rolling mean is falsely rejected). Worth its own item with a full
float-family receiver probe.
