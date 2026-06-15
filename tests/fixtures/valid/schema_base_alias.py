"""Schema base class recognized via import-as and assignment aliases (issue #98)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.polars import DataFrameModel as DFM
from pandera.typing.polars import DataFrame

# Assignment alias: BAlias = pa.DataFrameModel
BAlias = pa.DataFrameModel


class SchemaFromImportAlias(DFM):
    x: int


class SchemaFromAssignAlias(BAlias):
    y: float


def fn_import_alias(df: DataFrame[SchemaFromImportAlias]) -> DataFrame[SchemaFromImportAlias]:
    return df.with_columns(pl.col("x").cast(pl.Int64))


def fn_assign_alias(df: DataFrame[SchemaFromAssignAlias]) -> DataFrame[SchemaFromAssignAlias]:
    return df.with_columns(pl.col("y").cast(pl.Float64))
