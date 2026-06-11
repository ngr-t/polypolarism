"""Tests for issue #5: group_by/agg shorthand forms.

Three forms that should be recognised:
1. List-form group_by keys: ``df.group_by(["x"])``.
2. Top-level ``pl.<agg>("col")`` shorthand inside ``agg`` / ``select`` /
   ``with_columns``.
3. Kwarg-form ``agg(name=expr)`` / ``select(name=expr)`` /
   ``with_columns(name=expr)``.
"""

from __future__ import annotations

import textwrap

from polypolarism.checker import check_source

COMMON = """
import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame
from typing import cast


class Sales(pa.DataFrameModel):
    model_code: str
    sales: int


class Out(pa.DataFrameModel):
    sales: int
"""


def _check(body: str) -> list:
    src = textwrap.dedent(COMMON) + textwrap.dedent(body)
    return check_source(src)


def _all_pass(body: str) -> bool:
    results = _check(body)
    return all(r.passed for r in results)


class TestListFormGroupByKeys:
    def test_list_single_key(self):
        # ``group_by(["model_code"])`` should keep model_code.
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by(["model_code"]).agg(pl.col("sales").sum())
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(pl.col("model_code"))))
            """
        )

    def test_list_multiple_keys(self):
        # ``group_by(["a", "b"])`` should keep both keys.
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                a: str
                b: str
                c: int


            def f(df: DataFrame[S]) -> DataFrame[S]:
                return df.group_by(["a", "b"]).agg(pl.col("c").sum())
            """
        )

    def test_tuple_form(self):
        # Tuple form should also work.
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by(("model_code",)).agg(pl.col("sales").sum())
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(pl.col("model_code"))))
            """
        )

    def test_multi_positional_still_works(self):
        # Existing form ``group_by("a", "b")`` shouldn't regress.
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                a: str
                b: str
                c: int


            def f(df: DataFrame[S]) -> DataFrame[S]:
                return df.group_by("a", "b").agg(pl.col("c").sum())
            """
        )


class TestTopLevelAggShorthand:
    def test_pl_sum_shorthand(self):
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by("model_code").agg(pl.sum("sales"))
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(pl.col("sales"))))
            """
        )

    def test_pl_min_with_alias(self):
        # The aggregated frame must actually carry Out's ``sales`` column:
        # the original body validated {model_code, sales_min} against
        # Out{sales} — a guaranteed runtime SchemaError that the issue #89
        # validate-input check now correctly flags. The alias resolution
        # under test is unchanged.
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by("model_code").agg(pl.min("sales").alias("sales_min"))
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(sales=pl.col("sales_min"))))
            """
        )

    def test_pl_mean_shorthand(self):
        # mean is a numeric->Float64 reduction; output column is named after
        # the input column. Verify by chaining a reference to the column
        # against the expected dtype.
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                k: str
                v: int


            class MeanOut(pa.DataFrameModel):
                k: str
                v: pl.Float64


            def f(df: DataFrame[S]) -> DataFrame[MeanOut]:
                return df.group_by("k").agg(pl.mean("v"))
            """
        )

    def test_pl_count_shorthand(self):
        # count returns UInt32 named after the input column.
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                k: str
                v: int


            class CountOut(pa.DataFrameModel):
                k: str
                v: pl.UInt32


            def f(df: DataFrame[S]) -> DataFrame[CountOut]:
                return df.group_by("k").agg(pl.count("v"))
            """
        )

    def test_pl_sum_in_select(self):
        # The shorthand should also be recognised in ``select(...)``.
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                a: str
                b: int


            def f(df: DataFrame[S]) -> DataFrame[Out]:
                return df.select(pl.col("a").alias("model_code"), pl.sum("b").alias("sales"))
            """
        )


class TestKwargForm:
    def test_kwarg_agg(self):
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by("model_code").agg(sales=pl.col("sales").sum())
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(pl.col("sales"))))
            """
        )

    def test_kwarg_overrides_inner_alias(self):
        # Polars: kwarg name wins over an inner ``.alias(...)``. The output
        # column should be ``sales``, not ``ignored``.
        assert _all_pass(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Out]:
                df_g = df.group_by("model_code").agg(
                    sales=pl.col("sales").sum().alias("ignored")
                )
                return cast(DataFrame[Out], Out.validate(df_g.with_columns(pl.col("sales"))))
            """
        )

    def test_kwarg_select(self):
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                a: str
                b: int


            def f(df: DataFrame[S]) -> DataFrame[S]:
                return df.select(pl.col("a"), b=pl.col("b") * 2)
            """
        )

    def test_kwarg_with_columns(self):
        assert _all_pass(
            """
            class S(pa.DataFrameModel):
                a: str
                b: int


            def f(df: DataFrame[S]) -> DataFrame[S]:
                return df.with_columns(b=pl.col("b") * 2)
            """
        )
