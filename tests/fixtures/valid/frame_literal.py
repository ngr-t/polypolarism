"""Valid: frame-literal schema inference (issue #25).

``pl.DataFrame({...})`` / ``pl.LazyFrame({...})`` constructors are
inferable: column names come from the dict keys, dtypes from the values
(constant lists, eager range constructors, broadcast scalars) or from an
explicit ``schema=`` / ``schema_overrides=``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Empty(pa.DataFrameModel):
    pass


class Lit(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


@pa.check_types
def pure_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
    return pl.DataFrame({"a": [1, 2, 3]})


class Cal(pa.DataFrameModel):
    d: pl.Date
    year: pl.Int32

    class Config:
        strict = True


@pa.check_types
def build_calendar(df: DataFrame[Empty]) -> DataFrame[Cal]:
    cal = pl.DataFrame({"d": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True)})
    return cal.with_columns(year=pl.col("d").dt.year())


class Mixed(pa.DataFrameModel):
    name: str
    score: pl.Float64
    note: str = pa.Field(nullable=True)
    source: str


@pa.check_types
def mixed_literal(df: DataFrame[Empty]) -> DataFrame[Mixed]:
    return pl.DataFrame(
        {
            "name": ["ann", "bob"],
            "score": [1.0, 2.5],
            "note": ["good", None],
            "source": "manual",
        }
    )


class Typed(pa.DataFrameModel):
    a: pl.Int32
    b: pl.Int8


@pa.check_types
def explicit_schema(df: DataFrame[Empty]) -> DataFrame[Typed]:
    return pl.DataFrame(
        {"a": [1], "b": [2]},
        schema={"a": pl.Int32, "b": pl.Int8},
    )


@pa.check_types
def lazy_literal_collected(df: DataFrame[Empty]) -> DataFrame[Lit]:
    return pl.LazyFrame({"a": [1, 2, 3]}).collect()


NAMES = ["x", "y", "z"]


class Ev(pa.DataFrameModel):
    name: str
    v: int

    class Config:
        coerce = True


class Joined(pa.DataFrameModel):
    """Issue #39: a column whose values come from a module constant types
    like the literal-list case and joins cleanly."""

    step: int
    name: str
    v: int = pa.Field(nullable=True)

    class Config:
        strict = True


@pa.check_types
def skeleton_join(ev: DataFrame[Ev]) -> DataFrame[Joined]:
    sk = pl.DataFrame({"step": [1, 2, 3], "name": NAMES})
    return sk.join(ev, on="name", how="left")
