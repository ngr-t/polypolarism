"""Invalid fixture for issue #69 (PLY040): ``Annotated`` dtype forms with the
wrong metadata arity crash pandera at runtime, so every function referencing
such a schema is dead on arrival.

pandera maps the ``Annotated[pl.<Dtype>, ...]`` metadata 1:1 onto the dtype
class's ``__init__`` parameters and requires EXACTLY all of them
(``get_dtype_kwargs``); a mismatch is a deferred TypeError ("Annotation
'Array' requires all positional arguments ['inner', 'shape', 'width']")
raised the first time the schema is used (``to_schema`` / ``validate`` /
``@pa.check_types``). The class statement itself imports cleanly, so a
checker that only parses is the user's only early warning.

Probed required arities (pandera 0.31.1 / polars 1.41.2): List 1 (inner);
Array 3 (inner, shape, width); Struct 1 (fields); Datetime 2 (time_unit,
time_zone); Duration 1 (time_unit); Decimal 2 (precision, scale); Enum 1
(categories); Categorical 2 (categories, ordering); simple scalars 0.
Exact-arity legal twins live in ``valid/dtype_annotated_params.py``.

Runtime differential: the return-side functions self-verify (the harness's
return validation hits the TypeError); ``broken_schema_as_input`` needs a
SKIP entry because the input frame cannot even be synthesized from an
un-buildable schema.
"""

from __future__ import annotations

import typing

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class ArrReadme(pa.DataFrameModel):
    # The form polypolarism's README used to document: missing shape + width.
    v: typing.Annotated[pl.Array, pl.Int64(), 2]

    class Config:
        strict = True


def readme_array_missing_width(df: DataFrame[Src]) -> DataFrame[ArrReadme]:
    # Issue #69 repro: the body runs fine, but pandera raises TypeError the
    # moment ArrReadme is used (here: the return validation).
    return df.select(v=pl.concat_list(pl.col("a"), pl.col("a")).cast(pl.Array(pl.Int64, 2)))


class DtUnitOnly(pa.DataFrameModel):
    t: typing.Annotated[pl.Datetime, "us"]  # missing time_zone

    class Config:
        strict = True


def datetime_missing_time_zone(df: DataFrame[Src]) -> DataFrame[DtUnitOnly]:
    return df.select(t=pl.col("a"))


class DecPrecisionOnly(pa.DataFrameModel):
    d: typing.Annotated[pl.Decimal, 12]  # missing scale

    class Config:
        strict = True


def decimal_missing_scale(df: DataFrame[Src]) -> DataFrame[DecPrecisionOnly]:
    return df.select(d=pl.col("a"))


class ListExtraMeta(pa.DataFrameModel):
    xs: typing.Annotated[pl.List, pl.Int64(), 3]  # List takes only inner

    class Config:
        strict = True


def list_extra_metadata(df: DataFrame[Src]) -> DataFrame[ListExtraMeta]:
    return df.select(xs=pl.concat_list(pl.col("a")))


class ScalarWithMeta(pa.DataFrameModel):
    a: typing.Annotated[pl.Int64, 5]  # pl.Int64 takes no arguments

    class Config:
        strict = True


def scalar_with_metadata(df: DataFrame[Src]) -> DataFrame[ScalarWithMeta]:
    return df.select("a")


def broken_schema_as_input(df: DataFrame[DtUnitOnly]) -> DataFrame[Src]:
    # The parameter side is just as dead: @pa.check_types would raise the
    # TypeError while validating the *input*.
    return df.select(a=pl.lit(1))


def broken_schema_via_validate(df: DataFrame[Src]) -> DataFrame[Src]:
    # Body-only use: the signature is healthy, but the validate call crashes
    # at runtime the same way.
    checked = DtUnitOnly.validate(df.select(t=pl.col("a")))  # noqa: F841
    return df
