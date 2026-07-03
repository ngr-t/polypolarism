"""Valid: pandera Field(alias=) and leading-underscore attributes (issue #148).

pandera validates a column under ``pa.Field(alias=...)`` when present (not the
attribute name), and its metaclass omits leading-underscore attributes from the
schema entirely. polypolarism must match: an ``alias`` renames the column, and a
``_private`` attribute is not a required column.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Renamed(pa.DataFrameModel):
    val: int = pa.Field(alias="actual_name")

    class Config:
        strict = True


class Src(pa.DataFrameModel):
    actual_name: int

    class Config:
        strict = True


def alias_is_column_name(df: DataFrame[Src]) -> DataFrame[Renamed]:
    return df


class WithPrivate(pa.DataFrameModel):
    _private: int  # ignored by pandera's metaclass — not a column
    a: int

    class Config:
        strict = True


class AIn(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


def underscore_attr_not_column(df: DataFrame[AIn]) -> DataFrame[WithPrivate]:
    return df
