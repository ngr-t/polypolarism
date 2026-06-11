"""Landmark dtypes are distinct types, not interchangeable look-alikes.

False-negative twin of ``valid/dtype_enum.py``, ``valid/dtype_float16.py``,
``valid/dtype_int128.py`` and ``valid/dtype_uint128.py`` (combined file —
each wrong declaration fails for its own dtype-specific reason).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class OrderSchema(pa.DataFrameModel):
    order_id: int
    status: pl.Enum(["new", "paid"])
    priority: pl.Enum


class OrderWrong(pa.DataFrameModel):
    order_id: int
    status: pl.Categorical  # WRONG: Enum (1.25+) is distinct from Categorical
    priority: pl.Enum


def enum_declared_categorical(df: DataFrame[OrderSchema]) -> DataFrame[OrderWrong]:
    return df.filter(pl.col("order_id") > 0)


class FeaturesSchema(pa.DataFrameModel):
    feat_id: int
    embedding_dim_0: pl.Float16
    embedding_dim_1: pl.Float16


class FeaturesWrong(pa.DataFrameModel):
    feat_id: int
    embedding_dim_0: pl.Float32  # WRONG: Float16 (1.36+) is not Float32
    embedding_dim_1: pl.Float16


def float16_declared_float32(df: DataFrame[FeaturesSchema]) -> DataFrame[FeaturesWrong]:
    return df.filter(pl.col("feat_id") > 0)


class LedgerSchema(pa.DataFrameModel):
    txn_id: int
    amount_micros: pl.Int128


class LedgerWrong(pa.DataFrameModel):
    txn_id: int
    amount_micros: pl.Int64  # WRONG: Int128 (1.18+) is not Int64


def int128_declared_int64(df: DataFrame[LedgerSchema]) -> DataFrame[LedgerWrong]:
    return df.filter(pl.col("amount_micros") > 0)


class HashIndexSchema(pa.DataFrameModel):
    key: pl.UInt128
    value: pl.Int64


class HashIndexWrong(pa.DataFrameModel):
    key: pl.Int128  # WRONG: UInt128 (1.34+) is not Int128
    value: pl.Int64


def uint128_declared_int128(df: DataFrame[HashIndexSchema]) -> DataFrame[HashIndexWrong]:
    return df.filter(pl.col("value") > 0)
