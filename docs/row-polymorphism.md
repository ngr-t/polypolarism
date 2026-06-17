# Row polymorphism (`@rowpoly`)

Column-preserving helpers — `add a column`, `join two frames`, `rename a
field` — keep every column the caller passes in, beyond the columns named in
their parameter schema. With a plain non-strict `DataFrame[Schema]` annotation,
those extra columns *do* flow downstream, but only through gradual `Unknown`
leniency: their dtypes are lost (no downstream checking) and there is nothing
stopping the helper from silently dropping them (a runtime
`ColumnNotFoundError` waiting to happen).

The `@rowpoly` decorator recovers both. It names a **row variable** — the
helper's "rest" row — and threads it from input to output with real dtypes,
while a soundness check (`PLY043`) confirms the body actually preserves it.

> **Opt-in dialect, static only.** `@rowpoly` is a **polypolarism dialect**:
> mypy, pyright and Pandera all see a plain open `DataFrame`. The threading is
> checked by polypolarism alone. Code without `@rowpoly` behaves exactly as
> before — the feature is purely additive.

## The hard constraint: Pandera stays the runtime authority

`@rowpoly` is a **runtime no-op**. It returns the decorated function unchanged,
so Pandera's `@pa.check_types` validation is completely untouched — the
annotation stays a bare, Pandera-recognized `DataFrame[Schema]`:

```python
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class InId(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class OutScore(pa.DataFrameModel):
    id: int
    score: float

    class Config:
        strict = False


@pa.check_types
@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))
```

```text
$ polypolarism add_score.py
  add_score (line 25): OK
```

The decorator sits **beside** the annotation, not wrapped around it. Wrapping
the frame annotation in `Annotated[DataFrame[InId], ...]` would make Pandera
stop recognizing the parameter and **skip validation entirely** (probed against
pandera 0.31); the decorator surface sidesteps that, and is robust to whatever
any pandera version does with `Annotated`. The marker is reachable at runtime
as `fn.__pp_rowpoly__` for optional introspection, but it changes no runtime
behavior. This inertness is pinned by a runtime-differential test
(`test_rowpoly_decorator_does_not_disable_pandera_validation`): a frame missing
a required column still raises `SchemaError` under `@pa.check_types` + `@rowpoly`
in both decorator orders.

## The decorator surface

### Single row variable — `@rowpoly("R")`

One shared row variable across the single frame parameter. Use it for the
common "preserve every input column and add some" helper. The example above is
the canonical form: `InId` declares only `id`, `OutScore` adds `score`, and the
caller's other columns ride through `R`.

### Per-parameter row variables — `@rowpoly(a="R1", b="R2")`

One row variable per named frame parameter — needed once a helper has two
independent "rests", e.g. a join helper that must preserve **both** sides:

```python
import pandera.polars as pa
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class A(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class B(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


class Joined(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


@rowpoly(a="R1", b="R2")
def merge(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Joined]:
    return a.join(b, on="id", how="inner")
```

```text
$ polypolarism merge.py
  merge (line 31): OK
```

Each named parameter's extras are threaded independently, and preservation is
checked per parameter (with a distinct skolem sentinel per side), so the
diagnostic can name exactly which side a buggy body dropped.

## Threading precision

At a call site, `@rowpoly` adds the caller's *extra* columns — `(argument
columns) − (declared parameter columns)` — to the call result **with their real
dtypes** instead of degrading to `Unknown`. A downstream operation on a threaded
column is then checked precisely:

```python
class Wide(pa.DataFrameModel):
    id: int
    label: str

    class Config:
        strict = False


@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    return df.with_columns(score=pl.col("id").cast(pl.Float64))


def caller(wide: DataFrame[Wide]) -> DataFrame[OutScore]:
    # `wide` carries an extra `label: str` beyond InId. @rowpoly threads it into
    # the result with its real dtype, so this arithmetic on a Utf8 column is a
    # provable error — without @rowpoly, `label` would be Unknown and the bug
    # would slip through.
    scored = add_score(wide)
    return scored.with_columns(bad=pl.col("label") + 1)
```

```text
$ polypolarism threading.py
  add_score (line 32): OK
  caller (line 36): FAIL
    - [PLY009] arithmetic 'Utf8 + Int64' is not supported — polars raises InvalidOperationError at runtime; cast an operand first
```

Threading rules:

- **Declared-return columns win** on a name collision — the helper's explicit
  output contract takes precedence over a threaded extra.
- **The same extra name from two arguments** (keyword form) is unified; with no
  unifier it falls back to `Unknown`.
- **A `strict` declared return carries no extras.** `@pa.check_types` validates
  the return against the strict schema and rejects any column beyond it, so a
  `@rowpoly` on a strict-return helper is a no-op (the surface is only
  meaningful with a non-strict / open return).

## `PLY043` — the preservation check

Threading *trusts* that the body preserves the row variable. The preservation
check verifies it: it skolemizes the row variable by injecting a distinct
sentinel column into the parameter frame, re-analyzes the body, and flags
`PLY043` for any return point that **provably drops** the sentinel (a closed
frame lacking it — e.g. a `select` of a fixed column set, or a
`group_by().agg()` that collapses the schema):

```python
@rowpoly("R")
def add_score(df: DataFrame[InId]) -> DataFrame[OutScore]:
    # select("id") drops the caller's extra columns -> breaks the @rowpoly promise.
    return df.select("id").with_columns(score=pl.col("id").cast(pl.Float64))
```

```text
$ polypolarism ply043.py
  add_score (line 24): FAIL
    - [PLY043] @rowpoly: helper does not preserve row variable 'R' (parameter 'df') — the frame returned at line 26 provably drops it. A row-polymorphic helper must keep every column of 'df' (use with_columns / pl.all() rather than selecting a fixed set), or remove 'R' from @rowpoly.
```

The check is conservative — it only fires on a **provable** drop:

- `with_columns(...)`, `drop("realcol")`, `rename({"realcol": ...})`,
  `select(pl.all())` / `select(cs.all())`, conditional/early-return bodies, and
  `pl.concat([df, df])` all keep every column (sentinel included) and stay
  silent. An all-columns selector resolves to the full column set, so it
  preserves the row variable; only a *fixed* set of named columns drops it.
- An **open** result is not a provable drop (gradual), so it stays silent.

Because the property is relative to the *caller* (did the helper keep columns
the caller supplied?), Pandera cannot check it at runtime — an input
synthesized from the parameter schema alone has no extras to drop. `PLY043` is
therefore **static-only**, and the invalid fixture is skipped by the
runtime-differential harness.

To fix a `PLY043`: keep every input column (use `with_columns` /
`select(pl.all())` instead of selecting a fixed set), or — if the helper
genuinely does not preserve the row — drop the row variable from `@rowpoly`.

## Editor / JSON integration

`--format json` exposes each function's bound row variable(s) in its
`functions` entry, so editors can show them:

```jsonc
// @rowpoly("R")
{ "name": "add_score", "row_var": "R" }

// @rowpoly(a="R1", b="R2")
{ "name": "merge", "param_row_vars": { "a": "R1", "b": "R2" } }
```

The fields are added only when present — a function with neither decorator form
carries neither key, so the JSON stays backward-compatible for pandera-only
code.

## Honest framing

`@rowpoly` is a theoretical upper bound on static precision for
column-preserving helpers, not a daily-driver requirement. Practical users stay
on Pandera, which validates the base schema at runtime regardless. Reach for
`@rowpoly` when you want polypolarism to *statically* check that a generic
helper preserves and threads the caller's columns — and accept that the
threading guarantee is a polypolarism dialect the rest of the toolchain ignores.

See the [backlog](backlog.md) entry **C-14** for the tiered design history and
the deferred row-algebra work (explicit `R1 # R2` disjointness diagnostics).
