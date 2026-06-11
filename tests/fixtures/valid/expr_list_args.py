"""Valid: list-literal expression args for multi-expression helpers (issue #16).

``pl.struct([...])`` / ``pl.coalesce([...])`` / ``pl.concat_str([...])`` /
``pl.concat_list([...])`` / ``pl.sum_horizontal([...])`` etc. are analyzed
the same as their varargs forms; bare string elements are column references.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: int

    class Config:
        coerce = True


class StructOut(pa.DataFrameModel):
    a: int
    b: int

    class Config:
        coerce = True


class CoOut(pa.DataFrameModel):
    c: int

    class Config:
        coerce = True


@pa.check_types
def struct_list(df: DataFrame[In]) -> DataFrame[StructOut]:
    return df.select(s=pl.struct([pl.col("a"), pl.col("b")])).unnest("s")


@pa.check_types
def struct_strings(df: DataFrame[In]) -> DataFrame[StructOut]:
    return df.select(s=pl.struct(["a", "b"])).unnest("s")


@pa.check_types
def struct_mixed(df: DataFrame[In]) -> DataFrame[StructOut]:
    # Mixed *element* kinds inside one list: a bare column name next to an
    # expression. (Mixing a list with further positional args, e.g.
    # ``pl.struct("a", [pl.col("b")])``, raises TypeError at runtime on
    # polars 1.41.2 — the list is parsed as a nested literal.)
    return df.select(s=pl.struct(["a", pl.col("b")])).unnest("s")


@pa.check_types
def coalesce_list(df: DataFrame[In]) -> DataFrame[CoOut]:
    return df.select(c=pl.coalesce([pl.col("a"), pl.col("b")]))


class Names(pa.DataFrameModel):
    first: str
    last: str


class FullName(pa.DataFrameModel):
    full: str


def concat_str_list(df: DataFrame[Names]) -> DataFrame[FullName]:
    return df.select(full=pl.concat_str([pl.col("first"), pl.col("last")], separator=" "))


class Paired(pa.DataFrameModel):
    pair: pl.List(pl.Int64) = pa.Field()


def concat_list_list(df: DataFrame[In]) -> DataFrame[Paired]:
    return df.select(pair=pl.concat_list(["a", "b"]))


class Horizontal(pa.DataFrameModel):
    total: int
    lo: int
    hi: int
    avg: float


def horizontal_list(df: DataFrame[In]) -> DataFrame[Horizontal]:
    return df.select(
        total=pl.sum_horizontal([pl.col("a"), pl.col("b")]),
        lo=pl.min_horizontal(["a", "b"]),
        hi=pl.max_horizontal(["a", "b"]),
        avg=pl.mean_horizontal([pl.col("a"), pl.col("b")]),
    )
