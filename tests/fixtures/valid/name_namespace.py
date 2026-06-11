""".name.* manipulates output column names, dtype unchanged (issue #56).

Probed (polars 1.41.2): ``prefix`` / ``suffix`` / ``to_uppercase`` apply to
the expression's CURRENT output name (an earlier ``.alias`` included);
``keep`` restores the chain's ROOT column name, overriding any earlier
``.alias``; ``prefix_fields`` renames Struct FIELD names (the output column
name stays). A selector-rooted chain (``pl.all().name.prefix(...)``) runs
once per selected column — the original issue repro.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: str


class PreO(pa.DataFrameModel):
    pre_a: int
    pre_b: str

    class Config:
        strict = True


@pa.check_types
def ok_prefix_all_columns(df: DataFrame[In]) -> DataFrame[PreO]:
    # The issue #56 repro: previously "Could not infer return type".
    return df.select(pl.all().name.prefix("pre_"))


class Doubled(pa.DataFrameModel):
    a_x2: int

    class Config:
        strict = True


@pa.check_types
def ok_suffix_on_arithmetic(df: DataFrame[In]) -> DataFrame[Doubled]:
    return df.select((pl.col("a") * 2).name.suffix("_x2"))


class RootA(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


@pa.check_types
def ok_keep_restores_root_name(df: DataFrame[In]) -> DataFrame[RootA]:
    # Probed: keep returns the ROOT column name, overriding the alias.
    return df.select((pl.col("a") * 2).alias("z").name.keep())


class UpperA(pa.DataFrameModel):
    A: int

    class Config:
        strict = True


@pa.check_types
def ok_to_uppercase(df: DataFrame[In]) -> DataFrame[UpperA]:
    return df.select(pl.col("a").name.to_uppercase())


class WithStruct(pa.DataFrameModel):
    s: pl.Struct({"x": pl.Int64, "y": pl.Utf8})


class PrefixedFields(pa.DataFrameModel):
    p_x: int
    p_y: str

    class Config:
        strict = True


@pa.check_types
def ok_prefix_fields_then_unnest(df: DataFrame[WithStruct]) -> DataFrame[PrefixedFields]:
    # prefix_fields renames the struct FIELD names (dtype transform);
    # the unnest round trip surfaces them as columns.
    return df.select(pl.col("s").name.prefix_fields("p_")).unnest("s")
