"""Tests for AST analyzer."""

import textwrap

import pytest

from polypolarism.analyzer import (
    FunctionBodyAnalyzer,
    _is_column_subtype,
    _is_frame_subtype,
    analyze_source,
)
from polypolarism.types import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float16,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Null,
    Nullable,
    RowVar,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
    Utf8,
)
from polypolarism.types import (
    List as ListT,
)

PANDERA_HEADER = """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame
"""


class TestAnalyzeSourceBasic:
    """Test basic source code analysis."""

    def test_finds_function_with_df_annotation(self):
        """Find function with Pandera DataFrame[Schema] annotations."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"

    def test_extracts_input_frame_types(self):
        """Extract input FrameType from parameter annotations."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        assert "data" in results[0].input_types
        input_type = results[0].input_types["data"]
        assert input_type == FrameType({"id": Int64(), "name": Utf8()})

    def test_extracts_return_frame_type(self):
        """Extract return FrameType from annotation."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        expected = FrameType({"id": Int64(), "name": Utf8()})
        assert results[0].declared_return_type == expected

    def test_multiple_input_parameters(self):
        """Handle multiple DataFrame parameters."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class L(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class R(pa.DataFrameModel):
                id: int
                name: str

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            def merge(
                left: DataFrame[L],
                right: DataFrame[R],
            ) -> DataFrame[Out]:
                return left.join(right, on="id")
        """
        )

        results = analyze_source(source)

        assert len(results[0].input_types) == 2
        assert "left" in results[0].input_types
        assert "right" in results[0].input_types

    def test_ignores_functions_without_df_annotations(self):
        """Ignore functions that don't use DataFrame types."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def helper(x: int) -> int:
                return x + 1

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"


class TestAnalyzeJoinOperations:
    """Test join operation analysis."""

    def test_infers_inner_join_result(self):
        """Infer result type of inner join."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class Users(pa.DataFrameModel):
                user_id: int
                name: str

            class Orders(pa.DataFrameModel):
                order_id: int
                user_id: int
                amount: pl.Float64

            class Out(pa.DataFrameModel):
                user_id: int
                name: str
                order_id: int
                amount: pl.Float64

            def merge(
                users: DataFrame[Users],
                orders: DataFrame[Orders],
            ) -> DataFrame[Out]:
                return users.join(orders, on="user_id", how="inner")
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "user_id": Int64(),
                "name": Utf8(),
                "order_id": Int64(),
                "amount": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_infers_left_join_with_nullable(self):
        """Left join makes right columns nullable."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class Users(pa.DataFrameModel):
                user_id: int
                name: str

            class Orders(pa.DataFrameModel):
                user_id: int
                amount: pl.Float64

            class Out(pa.DataFrameModel):
                user_id: int
                name: str
                amount: pl.Float64 = pa.Field(nullable=True)

            def merge(
                users: DataFrame[Users],
                orders: DataFrame[Orders],
            ) -> DataFrame[Out]:
                return users.join(orders, on="user_id", how="left")
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "user_id": Int64(),
                "name": Utf8(),
                "amount": Nullable(Float64()),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_detects_join_key_missing_error(self):
        """Detect when join key is missing from right frame."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class Users(pa.DataFrameModel):
                user_id: int
                name: str

            class Orders(pa.DataFrameModel):
                order_id: int
                amount: pl.Float64

            class Out(pa.DataFrameModel):
                user_id: int
                name: str
                order_id: int
                amount: pl.Float64

            def bad_join(
                users: DataFrame[Users],
                orders: DataFrame[Orders],
            ) -> DataFrame[Out]:
                return users.join(orders, on="user_id", how="inner")
        """
        )

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("user_id" in str(e) for e in results[0].errors)


class TestAnalyzeGroupByOperations:
    """Test group_by().agg() operation analysis."""

    def test_infers_groupby_agg_result(self):
        """Infer result type of group_by().agg()."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                category: str
                amount: pl.Float64

            class Out(pa.DataFrameModel):
                category: str
                total: pl.Float64

            def summarize(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.group_by("category").agg(
                    pl.col("amount").sum().alias("total"),
                )
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "category": Utf8(),
                "total": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_infers_count_returns_uint32(self):
        """count() aggregation returns UInt32."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                category: str
                value: int

            class Out(pa.DataFrameModel):
                category: str
                count: pl.UInt32

            def count_by_category(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.group_by("category").agg(
                    pl.col("value").count().alias("count"),
                )
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "category": Utf8(),
                "count": UInt32(),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_temporal_mean_preserves_receiver_dtype(self):
        """Grouped mean on Datetime preserves unit+tz; Duration median keeps
        the unit (issue #85). Probed on polars 1.41.2."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                k: str
                ts: pl.Datetime("ms", time_zone="UTC")
                dur: pl.Duration("ns")

            class Out(pa.DataFrameModel):
                k: str
                mean_ts: pl.Datetime("ms", time_zone="UTC")
                med_dur: pl.Duration("ns")

            def summarize(data: DataFrame[In]) -> DataFrame[Out]:
                return data.group_by("k").agg(
                    pl.col("ts").mean().alias("mean_ts"),
                    pl.col("dur").median().alias("med_dur"),
                )
        """
        )

        results = analyze_source(source)

        assert results[0].errors == []
        expected = FrameType(
            {
                "k": Utf8(),
                "mean_ts": Datetime(unit="ms", tz="UTC"),
                "med_dur": Duration(unit="ns"),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_detects_groupby_nonexistent_column(self):
        """Detect when group_by uses non-existent column."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                category: str
                total: pl.Float64

            def bad_groupby(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.group_by("category").agg(
                    pl.col("value").sum().alias("total"),
                )
        """
        )

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("category" in str(e) for e in results[0].errors)


class TestPlLenAggregation:
    """Zero-arg ``pl.len()`` / ``pl.count()`` recognition (issue #9)."""

    def test_pl_len_in_agg_kwarg_form(self):
        """``agg(n=pl.len())`` infers column n: UInt32."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                g: str
                n: pl.UInt32

            def agg(data: DataFrame[In]) -> DataFrame[Out]:
                return data.group_by("g").agg(n=pl.len())
        """
        )

        results = analyze_source(source)

        expected = FrameType({"g": Utf8(), "n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_pl_len_in_agg_bare_form(self):
        """Bare ``agg(pl.len())`` defaults the output column name to 'len'."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                g: str
                len: pl.UInt32

            def agg(data: DataFrame[In]) -> DataFrame[Out]:
                return data.group_by("g").agg(pl.len())
        """
        )

        results = analyze_source(source)

        expected = FrameType({"g": Utf8(), "len": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_pl_len_in_agg_with_alias(self):
        """``agg(pl.len().alias("n"))`` names the output column 'n'."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                g: str
                n: pl.UInt32

            def agg(data: DataFrame[In]) -> DataFrame[Out]:
                return data.group_by("g").agg(pl.len().alias("n"))
        """
        )

        results = analyze_source(source)

        expected = FrameType({"g": Utf8(), "n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_zero_arg_pl_count_in_agg(self):
        """Zero-arg ``pl.count()`` behaves like ``pl.len()``."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                g: str
                n: pl.UInt32

            def agg(data: DataFrame[In]) -> DataFrame[Out]:
                return data.group_by("g").agg(n=pl.count())
        """
        )

        results = analyze_source(source)

        expected = FrameType({"g": Utf8(), "n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_pl_len_in_select(self):
        """``select(pl.len())`` infers column len: UInt32."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            class Out(pa.DataFrameModel):
                len: pl.UInt32

            def row_count(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.len())
        """
        )

        results = analyze_source(source)

        expected = FrameType({"len": UInt32()})
        assert results[0].inferred_return_type == expected


class TestExprLenAggregation:
    """``pl.col("v").len()`` — the count-including-nulls variant (issue #23)."""

    def test_expr_len_in_agg_with_alias(self):
        """``agg(pl.col("v").len().alias("n"))`` infers n: UInt32."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("v").len().alias("n"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"g": Utf8(), "n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_expr_len_in_agg_kwarg_form(self):
        """``agg(n=pl.col("v").len())`` infers n: UInt32."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(n=pl.col("v").len())
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"g": Utf8(), "n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_expr_len_in_select(self):
        """``select(n=pl.col("v").len())`` infers UInt32 outside agg context."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def count_rows(data: DataFrame[In]):
                return data.select(n=pl.col("v").len())
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"n": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_expr_len_on_missing_column_errors(self):
        """``pl.col("missing").len()`` still surfaces the column error."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("missing").len().alias("n"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestExprFilterChain:
    """``Expr.filter(...)`` is row-subsetting and dtype-preserving (issue #23)."""

    def test_filter_then_sum_in_agg(self):
        """Conditional aggregation: ``filter(...).sum()`` resolves to Int64."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(
                    pl.col("v").filter(pl.col("v") > 0).sum().alias("fs")
                )
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"g": Utf8(), "fs": Int64()})
        assert results[0].inferred_return_type == expected

    def test_filter_then_mean_in_agg(self):
        """``filter(...).mean()`` resolves to Float64."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(
                    m=pl.col("v").filter(pl.col("v") > 0).mean()
                )
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"g": Utf8(), "m": Float64()})
        assert results[0].inferred_return_type == expected

    def test_filter_preserves_dtype_in_select(self):
        """``select(pl.col("v").filter(...))`` keeps the receiver dtype."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int
                keep: bool

            def f(data: DataFrame[In]):
                return data.select(pl.col("v").filter(pl.col("keep")).alias("x"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"x": Int64()})
        assert results[0].inferred_return_type == expected

    def test_filter_preserves_nullable_receiver(self):
        """Filtering does not strip a Nullable wrapper (nulls may survive)."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int = pa.Field(nullable=True)
                keep: bool

            def f(data: DataFrame[In]):
                return data.select(pl.col("v").filter(pl.col("keep")).alias("x"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"x": Nullable(Int64())})
        assert results[0].inferred_return_type == expected

    def test_filter_predicate_missing_column_errors(self):
        """A predicate referencing a missing column surfaces PLY001."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(
                    pl.col("v").filter(pl.col("missing") > 0).sum().alias("fs")
                )
        """
        )

        results = analyze_source(source)

        assert any("PLY042" in e for e in results[0].errors)
        assert any("missing" in e for e in results[0].errors)

    def test_filter_kwarg_constraint_missing_column_is_validated(self):
        """Kwarg-value expressions are validated through the analyser too."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(
                    pl.col("v").filter(g=pl.col("missing")).sum().alias("fs")
                )
        """
        )

        results = analyze_source(source)

        assert any("PLY042" in e for e in results[0].errors)


class TestExprDropNullsChain:
    """``Expr.drop_nulls()`` strips the Nullable wrapper (issue #23)."""

    def test_drop_nulls_strips_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int = pa.Field(nullable=True)

            def f(data: DataFrame[In]):
                return data.select(pl.col("v").drop_nulls().alias("x"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"x": Int64()})
        assert results[0].inferred_return_type == expected

    def test_drop_nulls_on_non_nullable_is_identity(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int

            def f(data: DataFrame[In]):
                return data.select(pl.col("v").drop_nulls().alias("x"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"x": Int64()})
        assert results[0].inferred_return_type == expected

    def test_drop_nulls_then_sum_in_agg(self):
        """``drop_nulls().sum()`` on a nullable column aggregates non-null Int64."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: int = pa.Field(nullable=True)

            def agg(data: DataFrame[In]):
                return data.group_by("g").agg(
                    s=pl.col("v").drop_nulls().sum()
                )
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False, results[0].errors
        expected = FrameType({"g": Utf8(), "s": Int64()})
        assert results[0].inferred_return_type == expected


class TestRankMethod:
    """``Expr.rank()`` dtype inference (issue #9)."""

    def test_rank_default_returns_float64(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int

            class Out(pa.DataFrameModel):
                r: pl.Float64

            def ranked(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("v").rank().alias("r"))
        """
        )

        results = analyze_source(source)

        expected = FrameType({"r": Float64()})
        assert results[0].inferred_return_type == expected

    def test_rank_dense_returns_uint32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int

            class Out(pa.DataFrameModel):
                r: pl.UInt32

            def ranked(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("v").rank(method="dense").alias("r"))
        """
        )

        results = analyze_source(source)

        expected = FrameType({"r": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_rank_positional_method_returns_uint32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int

            class Out(pa.DataFrameModel):
                r: pl.UInt32

            def ranked(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("v").rank("ordinal").alias("r"))
        """
        )

        results = analyze_source(source)

        expected = FrameType({"r": UInt32()})
        assert results[0].inferred_return_type == expected

    def test_rank_preserves_nullable_receiver(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: int = pa.Field(nullable=True)

            class Out(pa.DataFrameModel):
                r: pl.Float64 = pa.Field(nullable=True)

            def ranked(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("v").rank().alias("r"))
        """
        )

        results = analyze_source(source)

        expected = FrameType({"r": Nullable(Float64())})
        assert results[0].inferred_return_type == expected


class TestAnalyzeChainedOperations:
    """Test chained DataFrame operations."""

    def test_infers_join_then_groupby(self):
        """Infer result of join followed by group_by."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class Orders(pa.DataFrameModel):
                order_id: int
                customer_id: int
                amount: pl.Float64

            class Customers(pa.DataFrameModel):
                customer_id: int
                region: str

            class Out(pa.DataFrameModel):
                region: str
                total_sales: pl.Float64

            def sales_summary(
                orders: DataFrame[Orders],
                customers: DataFrame[Customers],
            ) -> DataFrame[Out]:
                return (
                    orders.join(customers, on="customer_id", how="inner")
                    .group_by("region")
                    .agg(pl.col("amount").sum().alias("total_sales"))
                )
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "region": Utf8(),
                "total_sales": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected


class TestAnalyzeSelectOperations:
    """Test select operation analysis."""

    def test_infers_select_with_col(self):
        """Infer result of select with pl.col()."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                name: str
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def select_columns(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.select(pl.col("id"), pl.col("value"))
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "id": Int64(),
                "value": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected

    def test_detects_select_nonexistent_column(self):
        """Detect when select references non-existent column."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                amount: pl.Float64

            def bad_select(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.select(
                    pl.col("id"),
                    pl.col("amount"),
                )
        """
        )

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("amount" in str(e) for e in results[0].errors)


class TestAnalyzeWithColumn:
    """Test with_columns operation analysis."""

    def test_infers_with_columns_adds_column(self):
        """with_columns adds new column to existing ones."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64
                doubled: pl.Float64

            def add_doubled(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.with_columns(
                    (pl.col("value") * 2).alias("doubled"),
                )
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "id": Int64(),
                "value": Float64(),
                "doubled": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected


class TestM1IdentityMethods:
    """Schema-preserving (identity-typed) DataFrame methods."""

    @pytest.mark.parametrize(
        "method_call",
        [
            'filter(pl.col("id") > 0)',
            'sort("id")',
            "head(10)",
            "tail(5)",
            "limit(100)",
            "slice(0, 10)",
            "reverse()",
            "sample(n=3)",
            "unique()",
            "clone()",
            "lazy()",
            'set_sorted("id")',
        ],
    )
    def test_identity_method_preserves_schema(self, method_call: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def keep(data: DataFrame[S]) -> DataFrame[S]:
                return data.{method_call}
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        assert results[0].inferred_return_type == FrameType(
            {
                "id": Int64(),
                "value": Float64(),
            }
        )

    def test_filter_chains_with_other_methods(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def pick(data: DataFrame[S]) -> DataFrame[S]:
                return data.filter(pl.col("id") > 0).head(10)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False


class TestM1Drop:
    def test_drop_single_column(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.drop("name")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        assert results[0].inferred_return_type == FrameType(
            {
                "id": Int64(),
                "value": Float64(),
            }
        )

    def test_drop_multiple_columns_via_list(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            class Out(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.drop(["name", "value"])
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        assert results[0].inferred_return_type == FrameType({"id": Int64()})

    def test_drop_multiple_columns_via_varargs(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            class Out(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.drop("name", "value")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        assert results[0].inferred_return_type == FrameType({"id": Int64()})

    def test_drop_unknown_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]) -> DataFrame[S]:
                return data.drop("missing")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM1Rename:
    def test_rename_mapping(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                user_id: int
                amount: pl.Float64

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.rename({"id": "user_id", "value": "amount"})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        assert results[0].inferred_return_type == FrameType(
            {
                "user_id": Int64(),
                "amount": Float64(),
            }
        )

    def test_rename_unknown_source_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]) -> DataFrame[S]:
                return data.rename({"nope": "x"})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("nope" in e for e in results[0].errors)

    def test_rename_preserves_nullability_and_required(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int = pa.Field(nullable=True)

            def f(data: DataFrame[In]):
                return data.rename({"id": "user_id"})
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "user_id" in ft.columns
        spec = ft.columns["user_id"]
        assert isinstance(spec.dtype, Nullable)


class TestM1Cast:
    def test_cast_dict_form(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: pl.Int32
                value: pl.Float64

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.cast({"id": pl.Int32})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int32()
        assert ft.columns["value"].dtype == Float64()

    def test_cast_preserves_nullability(self):
        from polypolarism.types import Int32 as _Int32

        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int = pa.Field(nullable=True)

            def f(data: DataFrame[In]):
                return data.cast({"id": pl.Int32})
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Nullable(_Int32())

    def test_cast_unknown_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]) -> DataFrame[S]:
                return data.cast({"missing": pl.Int32})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True


class TestM1DropNulls:
    def test_drop_nulls_strips_nullable_on_all(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int = pa.Field(nullable=True)
                value: pl.Float64 = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.drop_nulls()
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["value"].dtype == Float64()

    def test_drop_nulls_subset(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int = pa.Field(nullable=True)
                value: pl.Float64 = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.drop_nulls(subset=["id"])
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["value"].dtype == Nullable(Float64())


class TestM1WithRowIndex:
    def test_with_row_index_default_name(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                value: pl.Float64

            def f(data: DataFrame[In]):
                return data.with_row_index()
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["index"].dtype == UInt32()
        assert ft.columns["value"].dtype == Float64()

    def test_with_row_index_custom_name(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                value: pl.Float64

            def f(data: DataFrame[In]):
                return data.with_row_index(name="row_nr")
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "row_nr" in ft.columns


class TestM2BooleanPredicates:
    """`pl.col(...).is_*()` methods must produce Boolean."""

    @pytest.mark.parametrize(
        "method",
        [
            "is_null()",
            "is_not_null()",
            "is_nan()",
            "is_not_nan()",
            "is_finite()",
            "is_infinite()",
            "is_unique()",
            "is_duplicated()",
            "is_first_distinct()",
            "is_last_distinct()",
        ],
    )
    def test_unary_predicate_returns_boolean(self, method: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                value: pl.Float64

            class Out(pa.DataFrameModel):
                flag: bool

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("value").{method}.alias("flag"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"flag": Boolean()})

    def test_is_in_returns_boolean(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int

            class Out(pa.DataFrameModel):
                flag: bool

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("id").is_in([1, 2, 3]).alias("flag"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"flag": Boolean()})

    def test_is_between_returns_boolean(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                value: pl.Float64

            class Out(pa.DataFrameModel):
                flag: bool

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("value").is_between(0.0, 1.0).alias("flag"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_predicate_on_missing_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]):
                return data.select(pl.col("missing").is_null().alias("flag"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM2CompareAndLogical:
    """Comparison and logical operators must produce Boolean."""

    @pytest.mark.parametrize(
        "predicate",
        [
            'pl.col("v") > 0',
            'pl.col("v") >= 0',
            'pl.col("v") < 0',
            'pl.col("v") <= 0',
            'pl.col("v") == 0',
            'pl.col("v") != 0',
        ],
    )
    def test_compare_returns_boolean(self, predicate: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                v: pl.Float64

            class Out(pa.DataFrameModel):
                flag: bool

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(({predicate}).alias("flag"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"flag": Boolean()})

    def test_and_or_invert_return_boolean(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: pl.Float64
                w: pl.Float64

            class Out(pa.DataFrameModel):
                a: bool
                b: bool
                c: bool

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(
                    ((pl.col("v") > 0) & (pl.col("w") > 0)).alias("a"),
                    ((pl.col("v") > 0) | (pl.col("w") > 0)).alias("b"),
                    (~pl.col("v").is_null()).alias("c"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType(
            {
                "a": Boolean(),
                "b": Boolean(),
                "c": Boolean(),
            }
        )

    def test_filter_predicate_validates_column(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]) -> DataFrame[S]:
                return data.filter(pl.col("missing") > 0)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM2FillNull:
    def test_fill_null_strips_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: pl.Float64 = pa.Field(nullable=True)

            class Out(pa.DataFrameModel):
                v_clean: pl.Float64

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.select(pl.col("v").fill_null(0.0).alias("v_clean"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"v_clean": Float64()})


class TestM2NewAggregations:
    @pytest.mark.parametrize(
        "agg, expected_type",
        [
            # std/var: ddof=1 default is null for singleton groups (probed,
            # polars 1.41.2; issue #60) -> Nullable even on non-null input.
            ("std", Nullable(Float64())),
            ("var", Nullable(Float64())),
            # median/quantile: total on non-empty, non-null groups (probed).
            ("median", Float64()),
            ("quantile", Float64()),
        ],
    )
    def test_groupby_agg_float_result(self, agg: str, expected_type):
        # quantile takes an arg; pad it for that case
        call = f"{agg}(0.5)" if agg == "quantile" else f"{agg}()"
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(data: DataFrame[In]):
                return data.group_by("k").agg(pl.col("v").{call}.alias("agg"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["agg"].dtype == expected_type

    def test_groupby_product_preserves_dtype(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                k: str
                v: int

            def f(data: DataFrame[In]):
                return data.group_by("k").agg(pl.col("v").product().alias("p"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["p"].dtype == Int64()


class TestM2StdVarDdof:
    """std/var nullability and the explicit ``ddof=0`` refinement (issue #60).

    Probed (polars 1.41.2): ``std()``/``var()`` with the default ``ddof=1``
    are null whenever only one sample is available (singleton group, 1-row
    frame), so the result is Nullable(Float64). An explicit literal
    ``ddof=0`` is total on non-empty input (singleton group -> 0.0), so the
    Nullable wrap is dropped — the receiver's own nullability still wins.
    """

    def _select(self, schema_field: str, expr: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                g: str
                {schema_field}

            def f(data: DataFrame[In]):
                return data.select({expr})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_select_std_is_nullable(self):
        ft = self._select("v: pl.Float64", 'pl.col("v").std().alias("s")')
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_select_std_ddof_zero_keyword_non_nullable(self):
        ft = self._select("v: pl.Float64", 'pl.col("v").std(ddof=0).alias("s")')
        assert ft.columns["s"].dtype == Float64()

    def test_select_var_ddof_zero_positional_non_nullable(self):
        # Expr.var(ddof) takes ddof as its first positional parameter.
        ft = self._select("v: int", 'pl.col("v").var(0).alias("s")')
        assert ft.columns["s"].dtype == Float64()

    def test_select_std_ddof_zero_nullable_receiver_stays_nullable(self):
        # An all-null window is still null even with ddof=0.
        ft = self._select(
            "v: pl.Float64 = pa.Field(nullable=True)",
            'pl.col("v").std(ddof=0).alias("s")',
        )
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_select_std_explicit_ddof_one_stays_nullable(self):
        ft = self._select("v: pl.Float64", 'pl.col("v").std(ddof=1).alias("s")')
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_select_std_non_literal_ddof_stays_nullable(self):
        # A non-literal ddof cannot be proven 0 -> conservative Nullable.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[In], d: int):
                return data.select(pl.col("v").std(ddof=d).alias("s"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_agg_std_ddof_zero_non_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: pl.Float64

            def f(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("v").std(ddof=0).alias("s"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Float64()

    def test_agg_var_ddof_zero_positional_non_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: pl.Float64

            def f(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("v").var(0).alias("s"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Float64()

    def test_std_over_stays_nullable(self):
        # .std().over(g): singleton partitions broadcast their null (probed).
        ft = self._select("v: pl.Float64", 'pl.col("v").std().over("g").alias("s")')
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_pl_std_shorthand_select_nullable(self):
        ft = self._select("v: pl.Float64", 'pl.std("v")')
        assert ft.columns["v"].dtype == Nullable(Float64())


class TestM3StrNamespace:
    """`pl.col("x").str.<method>(...)` dispatch tables."""

    @pytest.mark.parametrize(
        "expr, expected_type",
        [
            ('pl.col("name").str.contains("a")', Boolean()),
            ('pl.col("name").str.starts_with("a")', Boolean()),
            ('pl.col("name").str.ends_with("a")', Boolean()),
            ('pl.col("name").str.is_empty()', Boolean()),
            ('pl.col("name").str.lower()', Utf8()),
            ('pl.col("name").str.upper()', Utf8()),
            ('pl.col("name").str.to_lowercase()', Utf8()),
            ('pl.col("name").str.to_uppercase()', Utf8()),
            ('pl.col("name").str.strip_chars()', Utf8()),
            ('pl.col("name").str.replace("a", "b")', Utf8()),
            ('pl.col("name").str.replace_all("a", "b")', Utf8()),
            ('pl.col("name").str.zfill(3)', Utf8()),
            ('pl.col("name").str.len_chars()', UInt32()),
            ('pl.col("name").str.len_bytes()', UInt32()),
            ('pl.col("name").str.count_matches("a")', UInt32()),
            ('pl.col("name").str.split(",")', ListT(Utf8())),
            ('pl.col("name").str.to_date()', Date()),
            ('pl.col("name").str.to_datetime()', Datetime()),
            # Issue #19 — parse helpers
            ('pl.col("name").str.to_integer()', Int64()),
            ('pl.col("name").str.to_integer(base=10)', Int64()),
            # Issue #61 — to_decimal reads the literal scale: Decimal(38, scale)
            ('pl.col("name").str.to_decimal(scale=0)', Decimal(38, 0)),
            ('pl.col("name").str.to_decimal(scale=2)', Decimal(38, 2)),
            ('pl.col("name").str.to_time()', Time()),
        ],
    )
    def test_str_method_return_type(self, expr: str, expected_type):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                name: str

            def f(data: DataFrame[S]):
                return data.select(({expr}).alias("out"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == expected_type, expr

    @pytest.mark.parametrize(
        "expr",
        [
            # ``scale`` is keyword-only AND required on polars 1.41 — both
            # forms raise TypeError at runtime before any frame exists.
            'pl.col("name").str.to_decimal()',
            'pl.col("name").str.to_decimal(2)',
            # A non-literal scale is unknowable — never claim a fixed scale
            # (the issue #61 false positive).
            'pl.col("name").str.to_decimal(scale=s)',
        ],
    )
    # upgrade trigger: flag the crashing missing/positional-scale forms
    @pytest.mark.imprecision
    def test_str_to_decimal_unreadable_scale_degrades_to_unknown(self, expr: str):
        """Issue #61 — no literal ``scale=`` keyword -> Unknown, silently."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                name: str

            s = 2

            def f(data: DataFrame[S]):
                return data.select(({expr}).alias("out"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Unknown()

    def test_str_to_decimal_wraps_nullable_receiver(self):
        """Issue #61 — the parsed Decimal keeps the receiver's nullability."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                name: str = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(pl.col("name").str.to_decimal(scale=2).alias("out"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Nullable(Decimal(38, 2))

    def test_str_to_integer_wraps_nullable_receiver(self):
        """Issue #19 — a Nullable receiver wraps the parse result in Nullable."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                name: str = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(pl.col("name").str.to_integer().alias("out"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Nullable(Int64())

    def test_str_method_on_missing_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                name: str

            def f(data: DataFrame[S]):
                return data.select(pl.col("missing").str.lower().alias("x"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM3DtNamespace:
    """`pl.col("ts").dt.<method>(...)` dispatch tables."""

    @pytest.mark.parametrize(
        "expr, expected_type",
        [
            ('pl.col("ts").dt.year()', Int32()),
            ('pl.col("ts").dt.iso_year()', Int32()),
            ('pl.col("ts").dt.month()', Int8()),
            ('pl.col("ts").dt.day()', Int8()),
            ('pl.col("ts").dt.hour()', Int8()),
            ('pl.col("ts").dt.minute()', Int8()),
            ('pl.col("ts").dt.second()', Int8()),
            ('pl.col("ts").dt.weekday()', Int8()),
            ('pl.col("ts").dt.quarter()', Int8()),
            ('pl.col("ts").dt.week()', Int8()),
            ('pl.col("ts").dt.ordinal_day()', Int16()),
            ('pl.col("ts").dt.date()', Date()),
        ],
    )
    def test_dt_method_return_type(self, expr: str, expected_type):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                ts: pl.Datetime

            def f(data: DataFrame[S]):
                return data.select(({expr}).alias("out"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == expected_type, expr

    def test_dt_truncate_preserves_receiver_type(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                ts: pl.Datetime

            def f(data: DataFrame[S]):
                return data.select(pl.col("ts").dt.truncate("1d").alias("ts"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ts"].dtype == Datetime()


class TestM3ListNamespace:
    """`pl.col("xs").list.<method>(...)` requires List receiver."""

    def test_list_get_returns_element_type(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                xs: Annotated[pl.List, pl.Int64()]

            def f(data: DataFrame[S]):
                return data.select(pl.col("xs").list.get(0).alias("first"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["first"].dtype == Int64()

    def test_list_len_returns_uint32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                xs: Annotated[pl.List, pl.Int64()]

            def f(data: DataFrame[S]):
                return data.select(pl.col("xs").list.len().alias("n"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["n"].dtype == UInt32()

    def test_list_sum_returns_element_type(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                xs: Annotated[pl.List, pl.Float64()]

            def f(data: DataFrame[S]):
                return data.select(pl.col("xs").list.sum().alias("total"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["total"].dtype == Float64()

    def test_list_unique_preserves(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                xs: Annotated[pl.List, pl.Int64()]

            def f(data: DataFrame[S]):
                return data.select(pl.col("xs").list.unique().alias("u"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["u"].dtype == ListT(Int64())


class TestM4Explode:
    """`explode("xs")` turns List[T] into T."""

    def test_explode_single_column(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class In(pa.DataFrameModel):
                user_id: int
                tags: Annotated[pl.List, pl.Utf8()]

            class Out(pa.DataFrameModel):
                user_id: int
                tags: str

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.explode("tags")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["tags"].dtype == Utf8()
        assert ft.columns["user_id"].dtype == Int64()

    def test_explode_multiple_columns_via_list(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class In(pa.DataFrameModel):
                tags: Annotated[pl.List, pl.Utf8()]
                scores: Annotated[pl.List, pl.Float64()]

            def f(data: DataFrame[In]):
                return data.explode(["tags", "scores"])
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["tags"].dtype == Utf8()
        assert ft.columns["scores"].dtype == Float64()

    def test_explode_non_list_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.explode("v")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("v" in e and "List" in e for e in results[0].errors)

    def test_explode_unknown_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.explode("missing")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM4Concat:
    """`pl.concat([f1, f2], how=...)`."""

    def test_concat_vertical_unifies_schemas(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Out]:
                return pl.concat([a, b])
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType(
            {
                "id": Int64(),
                "value": Float64(),
            }
        )

    def test_concat_vertical_mismatched_columns_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                id: int
                extra: str

            def f(a: DataFrame[A], b: DataFrame[B]):
                return pl.concat([a, b])
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True

    def test_concat_horizontal_merges_columns(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                v: pl.Float64

            class B(pa.DataFrameModel):
                name: str

            class Out(pa.DataFrameModel):
                id: int
                v: pl.Float64
                name: str

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Out]:
                return pl.concat([a, b], how="horizontal")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_concat_horizontal_overlap_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                id: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return pl.concat([a, b], how="horizontal")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("id" in e for e in results[0].errors)

    def test_concat_diagonal_makes_missing_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                v: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                w: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return pl.concat([a, b], how="diagonal")
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        # v is in A only, so it becomes Nullable in the union
        assert ft.columns["v"].dtype == Nullable(Float64())
        assert ft.columns["w"].dtype == Nullable(Float64())


class TestM4VHStack:
    def test_vstack_unifies(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(a: DataFrame[S], b: DataFrame[S]) -> DataFrame[S]:
                return a.vstack(b)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"id": Int64()})

    def test_hstack_merges(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                name: str

            class Out(pa.DataFrameModel):
                id: int
                name: str

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Out]:
                return a.hstack(b)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors


class TestM4Unpivot:
    def test_unpivot_default_names(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                a: pl.Float64
                b: pl.Float64

            def f(data: DataFrame[In]):
                return data.unpivot(index=["id"], on=["a", "b"])
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["variable"].dtype == Utf8()
        assert ft.columns["value"].dtype == Float64()

    def test_unpivot_custom_names(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                a: pl.Float64
                b: pl.Float64

            def f(data: DataFrame[In]):
                return data.unpivot(
                    index=["id"], on=["a", "b"],
                    variable_name="metric", value_name="amount",
                )
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "metric" in ft.columns
        assert ft.columns["metric"].dtype == Utf8()
        assert ft.columns["amount"].dtype == Float64()

    def test_unpivot_mixed_int_str_value_columns_supertype(self):
        # Issue #41 repro. Probed (polars 1.41.2):
        # df.unpivot(index="id", on=["a", "s"]) with a: Int64, s: String
        # -> Schema({'id': Int64, 'variable': String, 'value': String}).
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                a: int
                s: str

            def f(data: DataFrame[In]):
                return data.unpivot(index="id", on=["a", "s"])
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["value"].dtype == Utf8()

    def test_unpivot_temporal_numeric_value_columns_supertype(self):
        # Probed: Date + Int64 -> Int64 (physical-repr promotion).
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                d: pl.Date
                n: int

            def f(data: DataFrame[In]):
                return data.unpivot(index=["id"], on=["d", "n"])
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["value"].dtype == Int64()

    def test_unpivot_value_columns_without_supertype_error(self):
        # Probed surviving PLY022 case: List + scalar has no polars
        # supertype — unpivot raises
        # "InvalidOperationError: 'unpivot' not supported for dtype: list[i64]".
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                xs: pl.List(pl.Int64) = pa.Field()
                s: str

            def f(data: DataFrame[In]):
                return data.unpivot(index=["id"], on=["xs", "s"])
        """
        )
        results = analyze_source(source)
        assert any("PLY022" in e and "incompatible" in e for e in results[0].errors), results[
            0
        ].errors

    def test_unpivot_unknown_value_column_stays_silent(self):
        # An Unknown-dtyped value column must not raise PLY022 — the value
        # dtype degrades to Unknown (gradual typing, no false positives).
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                a: int
                s: str

            def f(data: DataFrame[In]):
                df2 = data.with_columns(u=pl.col("a").interpolate())
                return df2.unpivot(index=["id"], on=["u", "s"])
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["value"].dtype == Unknown()


class TestM5Cumulative:
    """Cumulative methods preserve dtype; cum_count → UInt32."""

    @pytest.mark.parametrize(
        "method",
        ["cum_sum", "cum_max", "cum_min", "cum_prod"],
    )
    def test_cum_method_preserves_dtype(self, method: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.col("v").{method}().alias("v"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Float64()

    def test_cum_count_returns_uint32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.col("v").cum_count().alias("n"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["n"].dtype == UInt32()


class TestCumulativeStrictDtypes:
    """Issue #49: ``cum_sum``/``cum_prod``/``cum_min``/``cum_max`` are
    strictly typed instead of blindly dtype-preserving.

    Probed (polars 1.41.2) receiver-dtype matrix:

    - invalid receivers raise InvalidOperationError at runtime — flagged
      PLY016, output degrades to Unknown:
        cum_sum:  Utf8, Date, Datetime, Time, List, Struct, Categorical,
                  Enum, Null
        cum_prod: the cum_sum set plus Duration and Decimal
        cum_min / cum_max: Utf8, List, Struct, Null
    - probed-valid non-preserving cells:
        cum_sum:  Int8/Int16/UInt8/UInt16 -> Int64 (overflow guard),
                  Boolean -> UInt32, Decimal(p, s) -> Decimal(38, s)
        cum_prod: every int dtype narrower than UInt64 plus Boolean ->
                  Int64 (UInt64/Int128/UInt128 keep their dtype)
    - every other probed cell preserves the receiver dtype.
    - ``cum_count`` returns UInt32 for EVERY receiver dtype (even the
      ones the other cumulatives reject) and never raises.
    """

    def _run(self, receiver, method: str):
        frame = FrameType({"v": receiver})
        return _run_body(frame, f'out = df.select(c=pl.col("v").{method}())')

    @pytest.mark.parametrize(
        ("method", "receiver"),
        [
            ("cum_sum", Utf8()),
            ("cum_sum", Date()),
            ("cum_sum", Datetime()),
            ("cum_sum", Datetime(tz="UTC")),
            ("cum_sum", Time()),
            ("cum_sum", ListT(Int64())),
            ("cum_sum", Struct({"x": Int64()})),
            ("cum_sum", Categorical()),
            ("cum_sum", Enum()),
            ("cum_sum", Null()),
            ("cum_prod", Utf8()),
            ("cum_prod", Date()),
            ("cum_prod", Time()),
            ("cum_prod", Duration()),
            ("cum_prod", Decimal(10, 2)),
            ("cum_prod", ListT(Int64())),
            ("cum_prod", Null()),
            ("cum_min", Utf8()),
            ("cum_min", ListT(Int64())),
            ("cum_min", Struct({"x": Int64()})),
            ("cum_min", Null()),
            ("cum_max", Utf8()),
            ("cum_max", Struct({"x": Int64()})),
        ],
        ids=lambda p: str(p),
    )
    def test_invalid_receiver_flags_ply016_and_degrades(self, method, receiver):
        analyzer = self._run(receiver, method)
        assert len(analyzer.errors) == 1, analyzer.errors
        err = analyzer.errors[0]
        assert "PLY016" in err and method in err, err
        assert "InvalidOperationError" in err, err
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    @pytest.mark.parametrize(
        ("method", "receiver", "expected"),
        [
            ("cum_sum", Int8(), Int64()),
            ("cum_sum", Int16(), Int64()),
            ("cum_sum", UInt8(), Int64()),
            ("cum_sum", UInt16(), Int64()),
            ("cum_sum", Boolean(), UInt32()),
            ("cum_sum", Decimal(10, 2), Decimal(38, 2)),
            ("cum_prod", Int8(), Int64()),
            ("cum_prod", Int16(), Int64()),
            ("cum_prod", Int32(), Int64()),
            ("cum_prod", UInt8(), Int64()),
            ("cum_prod", UInt16(), Int64()),
            ("cum_prod", UInt32(), Int64()),
            ("cum_prod", Boolean(), Int64()),
        ],
        ids=lambda p: str(p),
    )
    def test_probed_widening_cells(self, method, receiver, expected):
        analyzer = self._run(receiver, method)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == expected

    @pytest.mark.parametrize(
        ("method", "receiver"),
        [
            ("cum_sum", Int32()),
            ("cum_sum", Int64()),
            ("cum_sum", Int128()),
            ("cum_sum", UInt32()),
            ("cum_sum", UInt64()),
            ("cum_sum", UInt128()),
            ("cum_sum", Float16()),
            ("cum_sum", Float32()),
            ("cum_sum", Float64()),
            ("cum_sum", Duration()),
            ("cum_prod", Int64()),
            ("cum_prod", UInt64()),
            ("cum_prod", Int128()),
            ("cum_prod", UInt128()),
            ("cum_prod", Float32()),
            ("cum_prod", Float64()),
            ("cum_min", Int8()),
            ("cum_min", UInt8()),
            ("cum_min", Float64()),
            ("cum_min", Boolean()),
            ("cum_min", Date()),
            ("cum_min", Datetime(tz="UTC")),
            ("cum_min", Time()),
            ("cum_min", Duration()),
            ("cum_min", Decimal(10, 2)),
            ("cum_min", Categorical()),
            ("cum_min", Enum()),
            ("cum_max", Int64()),
            ("cum_max", Boolean()),
            ("cum_max", Date()),
            ("cum_max", Categorical()),
        ],
        ids=lambda p: str(p),
    )
    def test_probed_preserving_cells(self, method, receiver):
        analyzer = self._run(receiver, method)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == receiver

    def test_nullable_receiver_widening_preserves_wrapper(self):
        analyzer = self._run(Nullable(Int8()), "cum_sum")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(Int64())

    def test_nullable_receiver_preserving_keeps_wrapper(self):
        analyzer = self._run(Nullable(Date()), "cum_min")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(Date())

    def test_nullable_invalid_receiver_still_flags(self):
        # Nullability does not rescue an invalid base dtype.
        analyzer = self._run(Nullable(Utf8()), "cum_sum")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY016" in analyzer.errors[0]
        assert "Utf8" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    def test_unknown_receiver_stays_silent(self):
        analyzer = self._run(Unknown(), "cum_sum")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    def test_cum_count_never_flags_even_on_string(self):
        # Probed: cum_count is UInt32 for every receiver dtype.
        analyzer = self._run(Utf8(), "cum_count")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == UInt32()

    def test_regression_agg_sum_on_string_still_errors(self):
        # ``pl.col(str).sum()`` goes through the aggregation chain path —
        # must keep erroring (independent of the new cum_* rule).
        analyzer = _run_body(FrameType({"s": Utf8()}), 'out = df.select(c=pl.col("s").sum())')
        assert any("sum" in e for e in analyzer.errors), analyzer.errors

    def test_regression_rolling_mean_on_string_stays_silent(self):
        # Probed: polars ACCEPTS rolling_mean on String (all-null Float64
        # output) — the rolling family must not be flagged (issue #49 note).
        analyzer = _run_body(
            FrameType({"s": Utf8()}),
            'out = df.select(c=pl.col("s").rolling_mean(window_size=2))',
        )
        assert analyzer.errors == [], analyzer.errors


class TestM5ShiftDiff:
    @pytest.mark.parametrize("method", ["shift", "diff", "pct_change"])
    def test_shift_like_makes_nullable(self, method: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.col("v").{method}(1).alias("v"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Nullable(Float64())


class TestDiffTemporalAndUnsigned:
    """Issue #46: ``diff()`` on temporal columns yields Duration; unsigned
    int receivers widen to a signed dtype.

    Probed (polars 1.41.2):
    - Date.diff / Datetime.diff (any tz) / Time.diff / Duration.diff
      -> Duration (nullable: head slot is null), keeping the receiver's
      time unit (issue #66) — Date is us-based, Time ns-based
    - UInt8.diff -> Int16, UInt16.diff -> Int32, UInt32.diff -> Int64,
      UInt64.diff -> Int64; UInt128.diff stays UInt128
    - signed ints / floats keep their dtype (Int8 -> Int8, Float32 -> Float32)
    """

    @pytest.mark.parametrize(
        ("receiver", "expected_unit"),
        [
            (Date(), "us"),
            (Datetime(), "us"),
            (Datetime(tz="UTC"), "us"),
            (Datetime(unit="ns"), "ns"),
            (Time(), "ns"),
            (Duration(), "us"),
            (Duration(unit="ms"), "ms"),
        ],
        ids=["date", "datetime", "datetime_tz", "datetime_ns", "time", "duration", "duration_ms"],
    )
    def test_temporal_diff_is_nullable_duration(self, receiver, expected_unit):
        frame = FrameType({"t": receiver})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("t").diff())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(
            Duration(unit=expected_unit)
        )

    def test_nullable_temporal_diff_is_nullable_duration(self):
        frame = FrameType({"t": Nullable(Date())})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("t").diff())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Duration())

    @pytest.mark.parametrize(
        ("receiver", "expected_inner"),
        [
            (UInt8(), Int16()),
            (UInt16(), Int32()),
            (UInt32(), Int64()),
            (UInt64(), Int64()),
        ],
        ids=["u8", "u16", "u32", "u64"],
    )
    def test_unsigned_diff_widens_to_signed(self, receiver, expected_inner):
        frame = FrameType({"v": receiver})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("v").diff())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(expected_inner)

    @pytest.mark.parametrize(
        "receiver",
        [UInt128(), Int8(), Int64(), Float32(), Float64()],
        ids=["u128", "i8", "i64", "f32", "f64"],
    )
    def test_other_numeric_diff_keeps_dtype(self, receiver):
        # UInt128 has no wider signed dtype — polars keeps it (probed).
        frame = FrameType({"v": receiver})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("v").diff())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(receiver)

    def test_unknown_receiver_diff_stays_unknown(self):
        frame = FrameType({"v": Unknown()})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("v").diff())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Unknown())

    def test_shift_on_temporal_is_untouched(self):
        # Regression guard: only ``diff`` maps to Duration — ``shift``
        # keeps the receiver dtype.
        frame = FrameType({"t": Date()})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("t").shift(1))')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Date())

    def test_pct_change_on_unsigned_is_float64(self):
        # ``pct_change`` divides — no diff-style unsigned widening, the
        # result is Float64 (issue #71; full matrix in TestPctChangeDtype).
        frame = FrameType({"v": UInt32()})
        analyzer = _run_body(frame, 'out = df.select(d=pl.col("v").pct_change())')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Float64())


class TestM5Over:
    def test_over_preserves_receiver_type(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.with_columns(
                    pl.col("v").mean().over("k").alias("v_mean"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_mean"].dtype == Float64()


class TestM5Rolling:
    @pytest.mark.parametrize("method", ["rolling_sum", "rolling_min", "rolling_max"])
    def test_rolling_preserve_methods(self, method: str):
        # Dtype-preserving on Float64, but the default min_samples
        # (= window_size) leaves leading nulls (probed; issue #57).
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.col("v").{method}(window_size=3).alias("v"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Nullable(Float64())

    @pytest.mark.parametrize(
        "method",
        ["rolling_mean", "rolling_std", "rolling_var", "rolling_median"],
    )
    def test_rolling_float_methods(self, method: str):
        # Default min_samples (= window_size) leaves the first
        # window_size-1 rows null (probed 1.41.2; issue #57).
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                v: int

            def f(data: DataFrame[S]):
                return data.select(pl.col("v").{method}(window_size=3).alias("v"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Nullable(Float64())


class TestM5RollingNullability:
    """Rolling-window nullability (issue #57; probed on polars 1.41.2).

    Rows whose window holds fewer than ``min_samples`` (default:
    ``window_size``) non-null values are null, so rolling outputs are
    Nullable by default. An explicit literal ``min_samples<=1`` (or
    ``window_size=1`` with min_samples unset) makes the window total —
    except rolling_std/rolling_var, whose default ``ddof=1`` is null on
    1-sample windows, and String receivers, which are all-null always.
    """

    def _select(self, schema_field: str, expr: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                {schema_field}

            def f(data: DataFrame[S]):
                return data.select({expr})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_rolling_quantile_default_nullable(self):
        ft = self._select(
            "v: pl.Float64",
            'pl.col("v").rolling_quantile(quantile=0.5, window_size=3).alias("q")',
        )
        assert ft.columns["q"].dtype == Nullable(Float64())

    @pytest.mark.parametrize("kwargs", ["min_samples=1", "min_samples=0", "min_periods=1"])
    def test_rolling_mean_min_samples_one_total(self, kwargs: str):
        # min_samples<=1 -> every window has enough values (probed); the
        # deprecated pre-1.21 spelling min_periods= still works.
        ft = self._select(
            "v: pl.Float64",
            f'pl.col("v").rolling_mean(window_size=3, {kwargs}).alias("m")',
        )
        assert ft.columns["m"].dtype == Float64()

    def test_rolling_mean_window_size_one_total(self):
        # min_samples defaults to window_size, so window_size=1 is total.
        ft = self._select("v: pl.Float64", 'pl.col("v").rolling_mean(1).alias("m")')
        assert ft.columns["m"].dtype == Float64()

    def test_rolling_std_min_samples_one_still_nullable(self):
        # ddof=1 (default) on a 1-sample window is null (probed).
        ft = self._select(
            "v: pl.Float64",
            'pl.col("v").rolling_std(window_size=3, min_samples=1).alias("s")',
        )
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_rolling_std_min_samples_one_ddof_zero_total(self):
        ft = self._select(
            "v: pl.Float64",
            'pl.col("v").rolling_std(window_size=3, min_samples=1, ddof=0).alias("s")',
        )
        assert ft.columns["s"].dtype == Float64()

    def test_rolling_var_window_size_one_nullable(self):
        # Probed: rolling_var(window_size=1) is ALL-null with ddof=1.
        ft = self._select("v: pl.Float64", 'pl.col("v").rolling_var(1).alias("s")')
        assert ft.columns["s"].dtype == Nullable(Float64())

    def test_rolling_var_window_size_one_ddof_zero_total(self):
        ft = self._select("v: pl.Float64", 'pl.col("v").rolling_var(1, ddof=0).alias("s")')
        assert ft.columns["s"].dtype == Float64()

    def test_rolling_mean_nullable_receiver_stays_nullable(self):
        # A window of only-null values is null even with min_samples=1.
        ft = self._select(
            "v: pl.Float64 = pa.Field(nullable=True)",
            'pl.col("v").rolling_mean(window_size=3, min_samples=1).alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_rolling_mean_string_receiver_all_null(self):
        # Probed: rolling_mean on String is accepted but ALL-null Float64,
        # so min_samples=1 must not strip the Nullable wrap.
        ft = self._select(
            "v: str",
            'pl.col("v").rolling_mean(window_size=3, min_samples=1).alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_rolling_mean_non_literal_min_samples_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S], n: int):
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=n).alias("m")
                )
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_rolling_mean_non_literal_window_size_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S], n: int):
                return data.select(pl.col("v").rolling_mean(n).alias("m"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["m"].dtype == Nullable(Float64())


class TestRollingIntConstArgs:
    """Int-constant ``min_samples``/``window_size``/``ddof`` resolve like
    literals (backlog B-5).

    A function-local ``ms = 1`` or module-level ``MIN_SAMPLES = 1`` binding
    feeds the same totality rules as a literal argument, through the same
    constant machinery that resolves string column-spec args (function
    locals shadow module constants; any reassignment invalidates the
    binding). Unresolvable names stay conservative (Nullable).
    """

    def _ft(self, source: str):
        results = analyze_source(textwrap.dedent(PANDERA_HEADER + source))
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_local_const_min_samples_one_total(self):
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                ms = 1
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=ms).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Float64()

    def test_module_const_min_samples_one_total(self):
        ft = self._ft(
            """
            MIN_SAMPLES = 1

            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=MIN_SAMPLES).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Float64()

    def test_local_const_min_samples_two_stays_nullable(self):
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                ms = 2
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=ms).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_local_const_window_size_one_total(self):
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                w = 1
                return data.select(pl.col("v").rolling_mean(w).alias("m"))
        """
        )
        assert ft.columns["m"].dtype == Float64()

    def test_reassigned_const_invalidated_stays_nullable(self):
        # ``ms`` is re-bound to a non-constant before the call — the
        # earlier literal must not leak through (same invalidation rule as
        # the string constants).
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S], n: int):
                ms = 1
                ms = n
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=ms).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_rolling_sum_const_min_samples_preserves_dtype_total(self):
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: int

            def f(data: DataFrame[S]):
                ms = 1
                return data.select(
                    pl.col("v").rolling_sum(window_size=3, min_samples=ms).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Int64()

    def test_rolling_std_const_ddof_zero_total(self):
        ft = self._ft(
            """
            DDOF = 0

            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(
                    pl.col("v").rolling_std(window_size=3, min_samples=1, ddof=DDOF).alias("s")
                )
        """
        )
        assert ft.columns["s"].dtype == Float64()

    def test_bool_const_is_not_an_int(self):
        # ``ms = True`` would be accepted by polars (bool is int at
        # runtime) but the analyzer's int-literal rule excludes bools —
        # stays conservative.
        ft = self._ft(
            """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                ms = True
                return data.select(
                    pl.col("v").rolling_mean(window_size=3, min_samples=ms).alias("m")
                )
        """
        )
        assert ft.columns["m"].dtype == Nullable(Float64())


class TestRollingStrictDtypes:
    """``rolling_sum``/``rolling_min``/``rolling_max`` are strictly typed
    instead of blindly dtype-preserving (issue #57; #49 deferred this family).

    Probed (polars 1.41.2) receiver-dtype matrix:

    - invalid receivers raise InvalidOperationError at runtime — flagged
      PLY016, output degrades to Unknown:
        rolling_sum:      Utf8, Date, Datetime, Time, Duration, Decimal,
                          List, Array, Struct, Categorical, Enum, Null
        rolling_min/max:  Utf8, Decimal, List, Array, Struct, Categorical,
                          Enum, Null  (temporals and Boolean are accepted)
    - probed-valid non-preserving cells (mirrors cum_sum's overflow guard):
        rolling_sum: Int8/Int16/UInt8/UInt16 -> Int64, Boolean -> UInt32
    - every other probed cell preserves the receiver dtype
      (Int32/Int64/UInt32/UInt64/Int128/UInt128/Float32/Float64; plus
      Boolean/Date/Datetime/Time/Duration for rolling_min/max).
    - nullability follows the rolling-window rule (issue #57): Nullable
      unless an explicit literal min_samples<=1 / window_size=1 fills
      every window; a Nullable receiver always stays Nullable.
    """

    def _run(self, receiver, method: str, call: str = "(window_size=3)"):
        frame = FrameType({"v": receiver})
        return _run_body(frame, f'out = df.select(c=pl.col("v").{method}{call})')

    @pytest.mark.parametrize(
        ("method", "receiver"),
        [
            ("rolling_sum", Utf8()),
            ("rolling_sum", Date()),
            ("rolling_sum", Datetime()),
            ("rolling_sum", Time()),
            ("rolling_sum", Duration()),
            ("rolling_sum", Decimal(10, 2)),
            ("rolling_sum", Categorical()),
            ("rolling_sum", Enum()),
            ("rolling_sum", ListT(Int64())),
            ("rolling_sum", Array(Int64())),
            ("rolling_sum", Struct({"x": Int64()})),
            ("rolling_sum", Null()),
            ("rolling_min", Utf8()),
            ("rolling_min", Decimal(10, 2)),
            ("rolling_min", ListT(Int64())),
            ("rolling_max", Struct({"x": Int64()})),
            ("rolling_max", Categorical()),
            ("rolling_max", Null()),
        ],
        ids=lambda p: str(p),
    )
    def test_invalid_receiver_flags_ply016_and_degrades(self, method, receiver):
        analyzer = self._run(receiver, method)
        assert len(analyzer.errors) == 1, analyzer.errors
        err = analyzer.errors[0]
        assert "PLY016" in err and method in err, err
        assert "InvalidOperationError" in err, err
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    @pytest.mark.parametrize(
        ("receiver", "expected"),
        [
            (Int8(), Int64()),
            (Int16(), Int64()),
            (UInt8(), Int64()),
            (UInt16(), Int64()),
            (Boolean(), UInt32()),
        ],
        ids=lambda p: str(p),
    )
    def test_rolling_sum_widening_cells(self, receiver, expected):
        analyzer = self._run(receiver, "rolling_sum")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(expected)

    @pytest.mark.parametrize(
        ("method", "receiver"),
        [
            ("rolling_sum", Int32()),
            ("rolling_sum", UInt64()),
            ("rolling_sum", Int128()),
            ("rolling_sum", Float32()),
            ("rolling_min", Boolean()),
            ("rolling_min", Date()),
            ("rolling_min", Int8()),
            ("rolling_max", Duration()),
            ("rolling_max", Datetime(tz="UTC")),
            ("rolling_max", Float64()),
        ],
        ids=lambda p: str(p),
    )
    def test_dtype_preserving_cells_nullable_by_default(self, method, receiver):
        analyzer = self._run(receiver, method)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(receiver)

    def test_rolling_sum_min_samples_one_total(self):
        analyzer = self._run(Int64(), "rolling_sum", "(window_size=3, min_samples=1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Int64()

    def test_rolling_min_window_size_one_total(self):
        analyzer = self._run(Float64(), "rolling_min", "(1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Float64()

    def test_rolling_sum_nullable_receiver_stays_nullable(self):
        # min_samples=1 cannot help an all-null window; upcast still applies.
        analyzer = self._run(Nullable(Int8()), "rolling_sum", "(window_size=3, min_samples=1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(Int64())

    @pytest.mark.parametrize(
        ("method", "receiver", "expected"),
        [
            ("rolling_sum", Int64(), Int64()),
            ("rolling_sum", Int8(), Int64()),  # overflow-guard upcast survives
            ("rolling_min", Float32(), Float32()),
            ("rolling_max", Date(), Date()),
        ],
        ids=lambda p: str(p),
    )
    def test_non_literal_window_preserves_dtype_family(self, method, receiver, expected):
        # Backlog B-5 regression pin: an unresolvable ``window_size`` only
        # loses nullability precision (Nullable is the sound upper bound,
        # T <: Nullable[T]); the dtype family stays determined by
        # (method, receiver dtype) — probed (polars 1.41.2): rolling_sum
        # dtype is identical across window_size/min_samples combinations.
        # It must NOT degrade to Nullable[Float64] (pre-issue-#57 behavior).
        analyzer = self._run(receiver, method, "(n)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(expected)

    def test_literal_min_samples_one_with_non_literal_window_total(self):
        # Probed (polars 1.41.2): min_samples=1 fills every window for any
        # accepted window_size (window_size=0 is an expanding window with 0
        # nulls; negative raises OverflowError before producing a frame),
        # so a literal min_samples<=1 is total even when window_size is
        # unresolvable.
        analyzer = self._run(Int64(), "rolling_sum", "(window_size=n, min_samples=1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Int64()

    def test_rolling_sum_unknown_receiver_stays_silent(self):
        analyzer = self._run(Unknown(), "rolling_sum")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()


class TestNumericElementwiseStrictDtypes:
    """``round``/``floor``/``ceil``/``clip``/``abs``/``sign``/``neg`` are
    strictly typed instead of blindly dtype-preserving (issue #62).

    Probed (polars 1.41.2) receiver-dtype matrix — invalid receivers raise
    InvalidOperationError at runtime, flagged PLY016, output degrades to
    Unknown:

    - round/floor/ceil: every non-numeric dtype (Utf8, Binary, Boolean,
      Date, Datetime, Time, Duration, Categorical, Enum, List, Array,
      Struct, Null); Decimal and all ints/uints/floats are accepted.
    - clip: "only supports physical numeric types" — rejects Utf8, Binary,
      Boolean, List, Array, Struct, Null even with matching-dtype literal
      bounds; temporals / Decimal / Categorical / Enum are physically
      numeric and accepted.
    - abs: rejects the round set minus Duration/Decimal (both preserved).
    - sign: rejects the round set (Duration included); Decimal preserved.
    - neg: rejects the abs set PLUS every unsigned int and Int128
      ("`neg` operation not supported for dtype `u128`").
    - shrink_dtype stays unchecked: deprecated no-op in 1.41.2, accepts
      every dtype.
    """

    def _run(self, receiver, call: str):
        frame = FrameType({"v": receiver})
        return _run_body(frame, f'out = df.select(c=pl.col("v").{call})')

    @pytest.mark.parametrize(
        ("call", "receiver"),
        [
            ("round(1)", Utf8()),
            ("round(1)", Boolean()),
            ("round(1)", Datetime()),
            ("round(1)", Duration()),
            ("round(1)", Null()),
            ("floor()", Utf8()),
            ("ceil()", ListT(Int64())),
            ("clip(0, 1)", Utf8()),
            ("clip(0, 1)", Boolean()),
            ("clip(0, 1)", Null()),
            ("abs()", Utf8()),
            ("abs()", Date()),
            ("abs()", Binary()),
            ("sign()", Duration()),
            ("neg()", UInt32()),
            ("neg()", UInt64()),
            ("neg()", Int128()),
            ("neg()", Categorical()),
        ],
        ids=lambda p: str(p),
    )
    def test_invalid_receiver_flags_ply016_and_degrades(self, call, receiver):
        analyzer = self._run(receiver, call)
        assert len(analyzer.errors) == 1, analyzer.errors
        err = analyzer.errors[0]
        method = call.split("(")[0]
        assert "PLY016" in err and method in err, err
        assert "InvalidOperationError" in err, err
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    def test_nullable_invalid_receiver_still_flags(self):
        analyzer = self._run(Nullable(Utf8()), "round(1)")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY016" in analyzer.errors[0], analyzer.errors

    def test_unknown_receiver_stays_silent(self):
        analyzer = self._run(Unknown(), "round(1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    @pytest.mark.parametrize(
        ("call", "receiver"),
        [
            # round/floor/ceil are int-identity and Decimal-preserving
            # (probed: round(1) on Decimal(10, 2) keeps precision AND scale).
            ("round(1)", Decimal(10, 2)),
            ("round(1)", Float32()),
            ("round(1)", UInt128()),
            ("floor()", Decimal(10, 2)),
            ("ceil()", Int64()),
            # clip accepts everything physically numeric.
            ("clip(0, 1)", Date()),
            ("clip(0, 1)", Duration()),
            ("clip(0, 1)", Decimal(10, 2)),
            ("clip(0, 1)", Categorical()),
            ("clip(0, 1)", Enum()),
            # abs/neg accept Duration and Decimal (preserving).
            ("abs()", Duration()),
            ("abs()", Decimal(10, 2)),
            ("abs()", UInt128()),
            # sign keeps the float dtype in 1.41.2 (no Int8 cast).
            ("sign()", Decimal(10, 2)),
            ("sign()", Float32()),
            ("neg()", Duration()),
            ("neg()", Decimal(10, 2)),
            # shrink_dtype is a deprecated no-op: any dtype, identity.
            ("shrink_dtype()", Utf8()),
        ],
        ids=lambda p: str(p),
    )
    def test_probed_preserving_cells(self, call, receiver):
        analyzer = self._run(receiver, call)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == receiver

    def test_nullable_preserving_keeps_wrapper(self):
        analyzer = self._run(Nullable(Float64()), "round(1)")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(Float64())


class TestFloatReturnStrictDtypes:
    """The Float64-return family (``log``/``log10``/``log1p``/``exp``/
    ``sqrt``/``cbrt``/``entropy``) is strictly typed (issue #62).

    Probed (polars 1.41.2) receiver-dtype matrix — the error class varies
    per cell (InvalidOperationError / ComputeError / rust panic), all
    flagged PLY016 with output degrading to Unknown:

    - log/log10:   reject Binary, Categorical, Enum, List, Array, Struct
    - log1p/exp:   reject Binary, List, Array, Struct
                   (Categorical/Enum are ACCEPTED -> Float64)
    - sqrt/cbrt:   reject Binary, Categorical, Enum, List, Array, Struct
    - entropy:     rejects Utf8, Binary, Boolean, List, Array, Struct,
                   Null but ACCEPTS temporals/Decimal/Categorical/Enum

    Every other accepted receiver — String, Boolean, temporals, Decimal,
    Null included (polars casts them into Float64 non-strictly) — stays
    silent and yields Float64; a Float32 receiver keeps Float32 (probed).
    """

    def _run(self, receiver, call: str):
        frame = FrameType({"v": receiver})
        return _run_body(frame, f'out = df.select(c=pl.col("v").{call})')

    @pytest.mark.parametrize(
        ("call", "receiver"),
        [
            ("log()", Categorical()),
            ("log()", ListT(Int64())),
            ("log()", Binary()),
            ("log10()", Enum()),
            ("log1p()", ListT(Int64())),
            ("exp()", Struct({"x": Int64()})),
            ("sqrt()", Categorical()),
            ("cbrt()", Enum()),
            ("entropy()", Utf8()),
            ("entropy()", Boolean()),
            ("entropy()", Null()),
            ("entropy()", Array(Int64())),
        ],
        ids=lambda p: str(p),
    )
    def test_invalid_receiver_flags_ply016_and_degrades(self, call, receiver):
        analyzer = self._run(receiver, call)
        assert len(analyzer.errors) == 1, analyzer.errors
        err = analyzer.errors[0]
        method = call.split("(")[0]
        assert "PLY016" in err and method in err, err
        assert analyzer.var_types["out"].columns["c"].dtype == Unknown()

    @pytest.mark.parametrize(
        ("call", "receiver"),
        [
            # polars casts these receivers into Float64 non-strictly —
            # accepted at runtime, so they MUST stay silent.
            ("log()", Utf8()),
            ("log1p()", Categorical()),
            ("exp()", Enum()),
            ("sqrt()", Utf8()),
            ("entropy()", Categorical()),
            ("entropy()", Date()),
        ],
        ids=lambda p: str(p),
    )
    def test_accepted_nonnumeric_receivers_stay_silent(self, call, receiver):
        analyzer = self._run(receiver, call)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Float64()

    def test_float32_receiver_keeps_float32(self):
        analyzer = self._run(Float32(), "sqrt()")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Float32()

    def test_nullable_float32_keeps_wrapper_and_width(self):
        analyzer = self._run(Nullable(Float32()), "log()")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["c"].dtype == Nullable(Float32())


class TestM5GroupByDynamic:
    def test_group_by_dynamic_then_agg(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                ts: pl.Datetime
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.group_by_dynamic("ts", every="1d").agg(
                    pl.col("v").mean().alias("v_mean"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ts"].dtype == Datetime()
        assert ft.columns["v_mean"].dtype == Float64()

    def test_rolling_then_agg(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                ts: pl.Datetime
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.rolling("ts", period="1d").agg(
                    pl.col("v").sum().alias("v_sum"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_sum"].dtype == Float64()


class TestM5JoinAsof:
    def test_join_asof_makes_right_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class L(pa.DataFrameModel):
                ts: pl.Datetime
                user_id: int

            class R(pa.DataFrameModel):
                ts: pl.Datetime
                price: pl.Float64

            def f(left: DataFrame[L], right: DataFrame[R]):
                return left.join_asof(right, on="ts")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        # right-side columns become Nullable (left-asof semantics)
        assert ft.columns["price"].dtype == Nullable(Float64())
        assert ft.columns["user_id"].dtype == Int64()


class TestM6PlExprConstructors:
    def test_pl_concat_str_returns_utf8(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                first: str
                last: str

            def f(data: DataFrame[S]):
                return data.select(
                    pl.concat_str([pl.col("first"), pl.col("last")], separator=" ").alias("full"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["full"].dtype == Utf8()

    def test_pl_format_returns_utf8(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: int

            def f(data: DataFrame[S]):
                return data.select(pl.format("v={}", pl.col("v")).alias("s"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Utf8()

    def test_pl_coalesce_unifies(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: pl.Float64 = pa.Field(nullable=True)
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.coalesce(pl.col("a"), pl.col("b")).alias("c"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        # b is non-null so the coalesced result is non-null Float64
        assert ft.columns["c"].dtype == Float64()

    def test_pl_struct_returns_struct(self):
        from polypolarism.types import Struct

        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(
                    pl.struct(pl.col("a"), pl.col("b")).alias("ab"),
                )
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ab"].dtype == Struct({"a": Int64(), "b": Float64()})


class TestPlStructKeywordFields:
    """Issue #47: ``pl.struct(name=expr)`` keyword args name struct fields."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class AB(pa.DataFrameModel):
                a: int
                b: pl.Float64
"""
    )

    def _infer(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[AB]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_kwarg_struct_field(self):
        from polypolarism.types import Struct

        ft = self._infer('df.select(w=pl.struct(x=pl.col("a")))')
        assert ft.columns["w"].dtype == Struct({"x": Int64()})

    def test_kwarg_struct_unnest_round_trip(self):
        ft = self._infer('df.select(w=pl.struct(x=pl.col("a"))).unnest("w")')
        assert ft.columns["x"].dtype == Int64()

    def test_kwarg_struct_field_access(self):
        ft = self._infer('df.select(x=pl.struct(x=pl.col("a")).struct.field("x"))')
        assert ft.columns["x"].dtype == Int64()

    def test_mixed_positional_and_kwarg_fields(self):
        from polypolarism.types import Struct

        ft = self._infer('df.select(w=pl.struct(pl.col("a"), y=pl.col("b")))')
        assert ft.columns["w"].dtype == Struct({"a": Int64(), "y": Float64()})

    def test_kwarg_expression_value(self):
        from polypolarism.types import Struct

        ft = self._infer('df.select(w=pl.struct(x=pl.col("a") + 1))')
        assert ft.columns["w"].dtype == Struct({"x": Int64()})

    def test_kwarg_uninferable_value_registers_unknown_field(self):
        from polypolarism.types import Struct

        ft = self._infer('df.select(w=pl.struct(x=pl.col("a").interpolate()))')
        assert ft.columns["w"].dtype == Struct({"x": Unknown()})


class TestExprListArgs:
    """Issue #16: a list/tuple literal of expressions is equivalent to varargs
    for multi-expression helpers (pl.struct, pl.coalesce, pl.concat_str, ...)."""

    _ISSUE_SCHEMAS = """
            class In(pa.DataFrameModel):
                a: int
                b: int

            class StructOut(pa.DataFrameModel):
                a: int
                b: int

            class CoOut(pa.DataFrameModel):
                c: int
    """

    def test_struct_list_then_unnest(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._ISSUE_SCHEMAS
            + """
            def f(df: DataFrame[In]) -> DataFrame[StructOut]:
                return df.select(s=pl.struct([pl.col("a"), pl.col("b")])).unnest("s")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Int64()
        assert ft.columns["b"].dtype == Int64()

    def test_struct_varargs_then_unnest(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._ISSUE_SCHEMAS
            + """
            def f(df: DataFrame[In]) -> DataFrame[StructOut]:
                return df.select(s=pl.struct(pl.col("a"), pl.col("b"))).unnest("s")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Int64()
        assert ft.columns["b"].dtype == Int64()

    def test_coalesce_list(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._ISSUE_SCHEMAS
            + """
            def f(df: DataFrame[In]) -> DataFrame[CoOut]:
                return df.select(c=pl.coalesce([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["c"].dtype == Int64()

    def test_coalesce_varargs(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._ISSUE_SCHEMAS
            + """
            def f(df: DataFrame[In]) -> DataFrame[CoOut]:
                return df.select(c=pl.coalesce(pl.col("a"), pl.col("b")))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["c"].dtype == Int64()

    def test_struct_bare_string_names(self):
        from polypolarism.types import Struct

        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.struct(["a", "b"]).alias("ab"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ab"].dtype == Struct({"a": Int64(), "b": Float64()})

    def test_struct_bare_string_names_unnest_roundtrip(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._ISSUE_SCHEMAS
            + """
            def f(df: DataFrame[In]) -> DataFrame[StructOut]:
                return df.select(s=pl.struct(["a", "b"])).unnest("s")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Int64()
        assert ft.columns["b"].dtype == Int64()

    def test_struct_mixed_varargs_and_list_flags_ply017(self):
        # Issue #59: mixing a list literal with further positional args does
        # NOT flatten — polars raises TypeError ("Nested object types") at
        # runtime, so the pre-#59 flatten-as-varargs typing was a false
        # negative. The mix is flagged and the output degrades to Unknown.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.struct("a", [pl.col("b")]).alias("ab"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("PLY017" in e and "struct" in e for e in results[0].errors)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ab"].dtype == Unknown()

    def test_struct_unknown_string_column_ply001(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int

            def f(data: DataFrame[S]):
                return data.select(pl.struct(["missing"]).alias("s"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("PLY042" in e and "missing" in e for e in results[0].errors)

    def test_coalesce_list_default_output_name(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: pl.Float64 = pa.Field(nullable=True)
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(pl.coalesce([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        # First flattened element supplies the default output name; b is
        # non-nullable so the coalesced result is non-nullable.
        assert ft.columns["a"].dtype == Float64()

    def test_coalesce_bare_string_names(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: pl.Float64 = pa.Field(nullable=True)
                b: pl.Float64 = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(pl.coalesce(["a", "b"]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        # All operands nullable -> result stays nullable; string element
        # yields the default output name like pl.col does.
        assert ft.columns["a"].dtype == Nullable(Float64())

    def test_concat_str_list_form(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                first: str
                last: str

            def f(data: DataFrame[S]):
                return data.select(full=pl.concat_str([pl.col("first"), pl.col("last")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["full"].dtype == Utf8()

    def test_concat_str_list_form_missing_column_ply001(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                first: str

            def f(data: DataFrame[S]):
                return data.select(full=pl.concat_str([pl.col("first"), pl.col("nope")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("PLY042" in e and "nope" in e for e in results[0].errors)


class TestMixedListArgsPLY017:
    """Issue #59: a list/tuple literal mixed with further positional args.

    Probed (polars 1.41.2) for EVERY multi-expression helper (pl.struct,
    pl.coalesce, pl.concat_str, pl.format, pl.concat_list, pl.*_horizontal):
    a mixed call either raises at runtime (TypeError "Nested object types"
    for expression-bearing lists; SchemaError / InvalidOperationError for
    string-only lists in coalesce / concat_str / horizontal) or silently
    misparses the list as a nested *literal* column (string-only lists in
    struct / concat_list) — never the flatten polypolarism assumed. The mix
    is flagged PLY017 and the output degrades to Unknown.
    """

    _SCHEMA = """
            class S(pa.DataFrameModel):
                a: int
                b: int
    """

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.struct('a', [pl.col('b')])",  # the issue #59 repro
            "pl.struct([pl.col('a')], pl.col('b'))",
            "pl.struct([pl.col('a')], [pl.col('b')])",  # two lists crash too
            "pl.struct(('a',), pl.col('b'))",  # tuple-of-strings misparses
            "pl.coalesce(pl.col('a'), [pl.col('b')])",
            "pl.coalesce(['a'], 'b')",
            "pl.concat_str([pl.col('a')], pl.col('b'))",
            "pl.format('{}', [pl.col('a')])",
            "pl.concat_list(['a'], 'b')",
            "pl.sum_horizontal([pl.col('a')], pl.col('b'))",
            "pl.min_horizontal(['a'], 'b')",
            "pl.max_horizontal(pl.col('a'), ['b'])",
            "pl.mean_horizontal([pl.col('a')], pl.col('b'))",
        ],
    )
    def test_mixed_list_args_flag_ply017_and_degrade_to_unknown(self, expr: str) -> None:
        source = textwrap.dedent(
            PANDERA_HEADER
            + self._SCHEMA
            + f"""
            def f(data: DataFrame[S]):
                return data.select(out=({expr}))
        """
        )
        results = analyze_source(source)
        assert any("PLY017" in e for e in results[0].errors), results[0].errors
        assert len(results[0].errors) == 1, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Unknown()

    def test_single_list_with_keyword_fields_is_not_mixed(self) -> None:
        # Probed: ``pl.struct(["a"], x=pl.col("b"))`` is fine at runtime —
        # keyword args name extra fields; only positional mixes misparse.
        from polypolarism.types import Struct

        source = textwrap.dedent(
            PANDERA_HEADER
            + self._SCHEMA
            + """
            def f(data: DataFrame[S]):
                return data.select(out=pl.struct(["a"], x=pl.col("b")))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Struct({"a": Int64(), "x": Int64()})


class TestConcatListAndHorizontal:
    """Issue #16 (continued): pl.concat_list and the horizontal helpers
    (sum/min/max/mean_horizontal) infer through list-literal and varargs args."""

    def test_concat_list_returns_list_dtype(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: int

            def f(data: DataFrame[S]):
                return data.select(xs=pl.concat_list([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["xs"].dtype == ListT(Int64())

    def test_concat_list_unifies_numeric(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(xs=pl.concat_list(pl.col("a"), pl.col("b")))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["xs"].dtype == ListT(Float64())

    def test_concat_list_bare_string_names(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: int

            def f(data: DataFrame[S]):
                return data.select(xs=pl.concat_list(["a", "b"]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["xs"].dtype == ListT(Int64())

    def test_concat_list_unresolvable_elements_is_list_unknown(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int

            def f(data: DataFrame[S]):
                return data.select(xs=pl.concat_list([pl.col("a").mystery_method()]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["xs"].dtype == ListT(Unknown())

    @pytest.mark.parametrize("func", ["sum_horizontal", "min_horizontal", "max_horizontal"])
    def test_horizontal_promotes_int_float(self, func):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64

            def f(data: DataFrame[S]):
                return data.select(out=pl.{func}([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Float64()

    @pytest.mark.parametrize("func", ["sum_horizontal", "min_horizontal", "max_horizontal"])
    def test_horizontal_same_int_types(self, func):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                a: int
                b: int

            def f(data: DataFrame[S]):
                return data.select(out=pl.{func}(["a", "b"]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Int64()

    def test_horizontal_all_nullable_wraps_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int = pa.Field(nullable=True)
                b: int = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(out=pl.sum_horizontal([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Nullable(Int64())

    def test_horizontal_any_non_nullable_strips_nullable(self):
        # Horizontal ops skip nulls — the result is null only when *every*
        # input is null, so one non-Nullable operand makes it non-Nullable.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int = pa.Field(nullable=True)
                b: int

            def f(data: DataFrame[S]):
                return data.select(out=pl.sum_horizontal([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Int64()

    def test_horizontal_non_numeric_is_unknown(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                s: str

            def f(data: DataFrame[S]):
                return data.select(out=pl.sum_horizontal([pl.col("a"), pl.col("s")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Unknown()

    def test_mean_horizontal_returns_float64(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: int = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(out=pl.mean_horizontal([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Float64()

    def test_mean_horizontal_all_float32_keeps_width(self):
        # Probed (polars 1.41.2; backlog N-4): mean_horizontal returns
        # Float32 iff every operand is Float32; any other operand (int,
        # Float64) widens the result to Float64.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: pl.Float32
                b: pl.Float32

            def f(data: DataFrame[S]):
                return data.select(out=pl.mean_horizontal([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Float32()

    def test_mean_horizontal_float32_int_mix_widens(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: pl.Float32
                b: int

            def f(data: DataFrame[S]):
                return data.select(out=pl.mean_horizontal([pl.col("a"), pl.col("b")]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Float64()

    def test_mean_horizontal_all_nullable(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int = pa.Field(nullable=True)
                b: int = pa.Field(nullable=True)

            def f(data: DataFrame[S]):
                return data.select(out=pl.mean_horizontal(["a", "b"]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Nullable(Float64())

    def test_horizontal_missing_column_ply001(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int

            def f(data: DataFrame[S]):
                return data.select(out=pl.sum_horizontal(["a", "nope"]))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("PLY042" in e and "nope" in e for e in results[0].errors)


class TestM6Selectors:
    def test_cs_numeric_in_select(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(cs.numeric())
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" in ft.columns and "value" in ft.columns
        assert "name" not in ft.columns

    def test_cs_by_dtype_in_drop(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                code: int
                name: str

            def f(data: DataFrame[S]):
                return data.drop(cs.integer())
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" not in ft.columns
        assert "code" not in ft.columns
        assert "name" in ft.columns

    def test_cs_starts_with(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                price_a: pl.Float64
                price_b: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(cs.starts_with("price_"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "price_a" in ft.columns
        assert "price_b" in ft.columns
        assert "name" not in ft.columns


class TestM6DiagnosticCodes:
    """Errors carry a stable PLY### prefix for IDE / CI consumption."""

    def test_column_not_found_has_code(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]):
                return data.select(pl.col("missing"))
        """
        )
        results = analyze_source(source)
        assert any("[PLY042]" in e for e in results[0].errors)

    def test_drop_unknown_has_code(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(data: DataFrame[S]):
                return data.drop("missing")
        """
        )
        results = analyze_source(source)
        assert any("[PLY002]" in e for e in results[0].errors)


class TestM7PipeRegistry:
    """``df.pipe(typed_helper)`` should use the helper's declared return type."""

    def test_pipe_typed_helper(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64
                doubled: pl.Float64

            def double_value(df: DataFrame[S]) -> DataFrame[Out]:
                return df.with_columns((pl.col("value") * 2).alias("doubled"))

            def via_pipe(df: DataFrame[S]) -> DataFrame[Out]:
                return df.pipe(double_value)
        """
        )
        results = analyze_source(source)
        via_pipe = next(r for r in results if r.name == "via_pipe")
        assert via_pipe.has_errors is False, via_pipe.errors
        assert via_pipe.inferred_return_type is not None
        assert "doubled" in via_pipe.inferred_return_type.columns

    def test_pipe_untyped_helper_propagates(self):
        """Untyped helper called via pipe still propagates types via body inference."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def add_one(df):
                return df.with_columns((pl.col("value") + 1).alias("value"))

            def via_pipe(df: DataFrame[S]) -> DataFrame[S]:
                return df.pipe(add_one)
        """
        )
        results = analyze_source(source)
        via_pipe = next(r for r in results if r.name == "via_pipe")
        assert via_pipe.has_errors is False, via_pipe.errors

    def test_pipe_unknown_callable_warns(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from somewhere import external_helper

            class S(pa.DataFrameModel):
                id: int

            def f(df: DataFrame[S]):
                return df.pipe(external_helper)
        """
        )
        results = analyze_source(source)
        f = next(r for r in results if r.name == "f")
        assert any("PLW002" in w for w in f.warnings)
        assert any("external_helper" in w for w in f.warnings)


class TestM7MapElementsReturnDtype:
    def test_map_elements_with_return_dtype(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64
                doubled: pl.Float64

            def f(df: DataFrame[S]) -> DataFrame[Out]:
                return df.with_columns(
                    pl.col("value").map_elements(
                        lambda v: v * 2.0, return_dtype=pl.Float64
                    ).alias("doubled")
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["doubled"].dtype == Float64()

    def test_map_elements_without_return_dtype_warns(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(df: DataFrame[S]):
                return df.with_columns(
                    pl.col("value").map_elements(lambda v: v * 2.0).alias("v2")
                )
        """
        )
        results = analyze_source(source)
        f = results[0]
        assert any("PLW001" in w for w in f.warnings)
        assert any("return_dtype" in w for w in f.warnings)


class TestAggUdfImplicitListWrap:
    """UDF expressions inside ``group_by().agg()`` (issue #86).

    A non-aggregating UDF result is implicitly list-aggregated in grouped
    context (probed identical on polars 1.41.2 and 1.37.0):

    - ``map_elements(f, return_dtype=T)`` -> ``List(T)`` regardless of
      ``returns_scalar=`` (deprecated since polars 1.32.0 and ignored);
    - ``map_batches`` / ``pl.map_groups`` -> ``List(T)`` unless
      ``returns_scalar=True`` (then scalar ``T``);
    - a native aggregation chained after the UDF reduces as usual;
    - select/with_columns contexts keep the scalar dtype (covered by
      TestM7MapElementsReturnDtype).
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class Src(pa.DataFrameModel):
                g: str
                v: int
        """
    )

    def _inferred(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[Src]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return results[0], ft

    def test_map_elements_in_agg_is_list_wrapped(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_map_elements_returns_scalar_true_is_ignored(self):
        # ``returns_scalar`` on map_elements is deprecated (1.32.0) and has
        # no effect — probed on 1.41.2 and 1.37.0: still List(Float64).
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64, returns_scalar=True))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_map_batches_in_agg_is_list_wrapped(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_batches(lambda s: s * 2.0, return_dtype=pl.Float64))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_map_batches_returns_scalar_true_stays_scalar(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_batches(lambda s: s.mean(), return_dtype=pl.Float64, returns_scalar=True))'
        )
        assert ft.columns["x"].dtype == Float64()

    def test_map_batches_returns_scalar_false_is_list_wrapped(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_batches(lambda s: s * 2.0, return_dtype=pl.Float64, returns_scalar=False))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_native_agg_after_udf_reduces_as_usual(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64).sum())'
        )
        assert ft.columns["x"].dtype == Float64()

    def test_alias_form_is_list_wrapped(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(pl.col("v").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64).alias("x"))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_no_return_dtype_fallback_is_list_wrapped_and_warns(self):
        # The PLW001 receiver-dtype fallback is still a guess, but the
        # implicit list aggregation applies to it like any other dtype.
        f, ft = self._inferred(
            'df.group_by("g").agg(x=pl.col("v").map_elements(lambda v: v * 2.0))'
        )
        assert any("PLW001" in w for w in f.warnings)
        assert ft.columns["x"].dtype == ListT(Int64())

    def test_pl_map_groups_in_agg_is_list_wrapped(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.map_groups(exprs=["v"], function=lambda ss: ss[0] * 2.0, return_dtype=pl.Float64))'
        )
        assert ft.columns["x"].dtype == ListT(Float64())

    def test_pl_map_groups_returns_scalar_true_stays_scalar(self):
        _, ft = self._inferred(
            'df.group_by("g").agg(x=pl.map_groups(exprs=["v"], function=lambda ss: ss[0].mean(), return_dtype=pl.Float64, returns_scalar=True))'
        )
        assert ft.columns["x"].dtype == Float64()

    def test_pl_map_groups_default_output_name_is_first_expr(self):
        # Probed: the output column takes the first input expression's name.
        _, ft = self._inferred(
            'df.group_by("g").agg(pl.map_groups(exprs=["v"], function=lambda ss: ss[0] * 2.0, return_dtype=pl.Float64))'
        )
        assert ft.columns["v"].dtype == ListT(Float64())


class TestM7ExternalHelperWarning:
    def test_unknown_function_call_warns(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from othermodule import process

            class S(pa.DataFrameModel):
                id: int

            def f(df: DataFrame[S]):
                return process(df)
        """
        )
        results = analyze_source(source)
        f = results[0]
        assert any("PLW003" in w for w in f.warnings)
        assert any("process" in w for w in f.warnings)


class TestM8AggChains:
    """Inside ``.agg(...)`` we now accept post-aggregation method chains."""

    def test_post_agg_dt_year(self):
        """``pl.col("ts").max().dt.year()`` returns Int32 in agg context."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                user_id: int
                ts: pl.Datetime

            def f(data: DataFrame[S]):
                return data.group_by("user_id").agg(
                    pl.col("ts").max().dt.year().alias("last_year"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["last_year"].dtype == Int32()

    def test_post_agg_cum_sum_last(self):
        """``count().cum_sum().last()`` is UInt32 (count) → UInt32 (cum) → UInt32 (last)."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                user_id: int

            def f(data: DataFrame[S]):
                return data.group_by("user_id").agg(
                    pl.col("user_id").count().cum_sum().last().alias("running"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["running"].dtype == UInt32()

    def test_post_agg_arithmetic(self):
        """``sum().alias()`` plus arithmetic still resolves."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                user_id: int
                amount: pl.Float64

            def f(data: DataFrame[S]):
                return data.group_by("user_id").agg(
                    (pl.col("amount").sum() * 2).alias("doubled_total"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["doubled_total"].dtype == Float64()

    def test_post_agg_str_namespace(self):
        """``first().str.to_uppercase()`` in agg context returns Utf8."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                user_id: int
                tag: str

            def f(data: DataFrame[S]):
                return data.group_by("user_id").agg(
                    pl.col("tag").first().str.to_uppercase().alias("tag_upper"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["tag_upper"].dtype == Utf8()

    def test_chain_on_unknown_column_errors(self):
        """A chain that references a missing column still surfaces PLY001."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                user_id: int

            def f(data: DataFrame[S]):
                return data.group_by("user_id").agg(
                    pl.col("missing").max().dt.year().alias("y"),
                )
        """
        )
        results = analyze_source(source)
        assert any("PLY042" in e for e in results[0].errors)
        assert any("missing" in e for e in results[0].errors)


class TestM9StructAccess:
    """``pl.col("s").struct.field("x")`` returns the inner field's dtype."""

    def test_struct_field_access(self):
        from polypolarism.types import Struct

        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                user: Annotated[pl.Struct, {"id": pl.Int64(), "name": pl.Utf8()}]

            def f(data: DataFrame[S]):
                return data.select(
                    pl.col("user").struct.field("id").alias("user_id"),
                    pl.col("user").struct.field("name").alias("user_name"),
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["user_id"].dtype == Int64()
        assert ft.columns["user_name"].dtype == Utf8()
        # Round-trip touches Struct itself
        del Struct  # only imported to make the test name dependency explicit

    def test_struct_field_unknown_field_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class S(pa.DataFrameModel):
                user: Annotated[pl.Struct, {"id": pl.Int64()}]

            def f(data: DataFrame[S]):
                return data.select(pl.col("user").struct.field("missing").alias("x"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestStructRenameFields:
    """Issue #48: ``struct.rename_fields([...])`` retypes the Struct.

    Probed (polars 1.41.2): the new names are applied positionally to the
    existing fields, and length mismatches do NOT raise — fewer names
    truncate the struct to the renamed prefix, surplus names are ignored
    (plain ``zip`` semantics). Non-literal names degrade to Unknown:
    keeping the original field names was the false positive being fixed.
    """

    def _frame(self) -> FrameType:
        return FrameType({"s": Struct({"x": Int64(), "y": Utf8()})})

    def test_rename_fields_applies_names_in_order(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("s").struct.rename_fields(["p", "q"]))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Struct({"p": Int64(), "q": Utf8()})

    def test_rename_fields_tuple_form(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("s").struct.rename_fields(("p", "q")))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Struct({"p": Int64(), "q": Utf8()})

    def test_rename_fields_fewer_names_truncates(self):
        # Probed: polars drops the un-named trailing fields without error.
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("s").struct.rename_fields(["p"]))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Struct({"p": Int64()})

    def test_rename_fields_more_names_ignores_extras(self):
        # Probed: surplus names are silently ignored.
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("s").struct.rename_fields(["p", "q", "r"]))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Struct({"p": Int64(), "q": Utf8()})

    def test_rename_fields_non_literal_names_degrade_to_unknown(self):
        # A variable name list is unresolvable — the column must degrade to
        # Unknown, NOT keep the stale field names (issue #48).
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("s").struct.rename_fields(names))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Unknown()

    def test_rename_fields_unknown_receiver_stays_silent(self):
        analyzer = _run_body(
            FrameType({"s": Unknown()}),
            'out = df.select(pl.col("s").struct.rename_fields(["p", "q"]))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Unknown()

    def test_rename_fields_nullable_receiver_preserves_wrapper(self):
        analyzer = _run_body(
            FrameType({"s": Nullable(Struct({"x": Int64()}))}),
            'out = df.select(pl.col("s").struct.rename_fields(["p"]))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == Nullable(Struct({"p": Int64()}))


class TestM9Unnest:
    """``df.unnest("col")`` flattens a Struct column into its field columns."""

    def test_unnest_single_struct_column(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class In(pa.DataFrameModel):
                id: int
                user: Annotated[pl.Struct, {"name": pl.Utf8(), "age": pl.Int64()}]

            class Out(pa.DataFrameModel):
                id: int
                name: str
                age: int

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                return data.unnest("user")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_unnest_list_form(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            class In(pa.DataFrameModel):
                a: Annotated[pl.Struct, {"x": pl.Int64()}]
                b: Annotated[pl.Struct, {"y": pl.Float64()}]

            def f(data: DataFrame[In]):
                return data.unnest(["a", "b"])
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["x"].dtype == Int64()
        assert ft.columns["y"].dtype == Float64()
        assert "a" not in ft.columns
        assert "b" not in ft.columns

    def test_unnest_non_struct_column_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.unnest("v")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("Struct" in e for e in results[0].errors)


class TestM9PluralCol:
    """``pl.col("a", "b", ...)`` selects multiple columns in select / with_columns."""

    def test_plural_col_in_select(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: pl.Float64
                c: str

            def f(data: DataFrame[S]):
                return data.select(pl.col("a", "b"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Int64()
        assert ft.columns["b"].dtype == Float64()
        assert "c" not in ft.columns

    def test_plural_col_unknown_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int

            def f(data: DataFrame[S]):
                return data.select(pl.col("a", "missing"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True
        assert any("missing" in e for e in results[0].errors)


class TestM10SelectorArithmetic:
    """``cs.exclude(...)`` and selector ``|`` / ``&`` / ``-`` / ``~``."""

    def test_cs_exclude_names(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(cs.exclude("name"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" in ft.columns
        assert "value" in ft.columns
        assert "name" not in ft.columns

    def test_cs_exclude_selector(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                price: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(cs.exclude(cs.string()))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" in ft.columns
        assert "price" in ft.columns
        assert "name" not in ft.columns

    def test_selector_subtraction(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                price: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(cs.numeric() - cs.by_name("id"))
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "price" in ft.columns
        assert "id" not in ft.columns
        assert "name" not in ft.columns

    def test_selector_union_and_intersection(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                price_a: pl.Float64
                price_b: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(
                    cs.numeric() & cs.starts_with("price_"),  # intersection
                )

            def g(data: DataFrame[S]):
                return data.select(cs.starts_with("price_") | cs.by_name("name"))
        """
        )
        results = analyze_source(source)
        f = next(r for r in results if r.name == "f")
        g = next(r for r in results if r.name == "g")
        f_ft = f.inferred_return_type
        g_ft = g.inferred_return_type
        assert f_ft is not None
        assert "price_a" in f_ft.columns and "price_b" in f_ft.columns
        assert "id" not in f_ft.columns and "name" not in f_ft.columns
        assert g_ft is not None
        assert "price_a" in g_ft.columns and "name" in g_ft.columns
        assert "id" not in g_ft.columns

    def test_selector_complement(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            import polars.selectors as cs

            class S(pa.DataFrameModel):
                id: int
                price: pl.Float64
                name: str

            def f(data: DataFrame[S]):
                return data.select(~cs.string())
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" in ft.columns
        assert "price" in ft.columns
        assert "name" not in ft.columns


class TestM12Pivot:
    """``df.pivot(...)`` is data-dependent — we warn and require a typed annotation."""

    def test_pivot_emits_plw005_with_actionable_message(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                cat: str
                v: pl.Float64

            def f(data: DataFrame[S]):
                return data.pivot(on="cat", index=["k"], values="v")
        """
        )
        results = analyze_source(source)
        f = results[0]
        assert any("PLW005" in w for w in f.warnings)
        # The message should mention pivot and a typed-annotation hint.
        assert any("pivot" in w and "DataFrame[" in w for w in f.warnings)

    def test_pivot_assigned_to_typed_var_uses_annotation(self):
        """``result: DataFrame[Out] = df.pivot(...)`` lets the annotation win."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                k: str
                cat: str
                v: pl.Float64

            class Out(pa.DataFrameModel):
                k: str
                A: pl.Float64
                B: pl.Float64

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                result: DataFrame[Out] = data.pivot(on="cat", index=["k"], values="v")
                return result
        """
        )
        results = analyze_source(source)
        # The pivot itself emits PLW005 …
        assert any("PLW005" in w for w in results[0].warnings)
        # … but the function passes because the annotation gave us the schema.
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["A"].dtype == Float64()
        assert ft.columns["B"].dtype == Float64()


class TestNullCountInference:
    """``df.null_count()`` (issue #74): same column names, every dtype
    UInt32 (the per-column null tally), one row.

    Probed (polars 1.41.2, identical on 1.37.0): eager and lazy receivers
    both map every column — Nullable or not, temporal or not — to a
    non-null UInt32 column of the same name.
    """

    def test_null_count_maps_every_column_to_uint32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                a: int
                b: str = pa.Field(nullable=True)
                t: pl.Datetime

            def f(data: DataFrame[In]):
                return data.null_count()
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert set(ft.columns) == {"a", "b", "t"}
        for name in ("a", "b", "t"):
            assert ft.columns[name].dtype == UInt32(), name
        assert ft.is_lazy is False

    def test_null_count_on_lazyframe_stays_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame, LazyFrame

            class In(pa.DataFrameModel):
                a: int

            def f(data: LazyFrame[In]):
                return data.null_count()
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == UInt32()
        assert ft.is_lazy is True

    def test_null_count_preserves_open_frame(self):
        # The non-literal name.prefix opens the frame; unknown extra
        # columns also count nulls at runtime, so openness is preserved.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            P = get_prefix()

            class In(pa.DataFrameModel):
                a: int

            def f(data: DataFrame[In]):
                return data.select(pl.col("a").name.prefix(P)).null_count()
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.rest is not None


class TestUpsampleInference:
    """``df.upsample(time_column, every=...)`` (issue #74): identity
    columns, but non-key columns become Nullable — the inserted gap rows
    are null-filled in every column except the time column and the
    ``group_by`` keys.

    Probed (polars 1.41.2, identical on 1.37.0): gap rows null only the
    non-key columns; ``group_by`` columns are forward-filled per group;
    the time column keeps its dtype (unit included). Eager-only —
    ``pl.LazyFrame`` has no ``upsample`` attribute.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                t: pl.Datetime
                g: str
                v: int
        """
    )

    def _frame(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(data: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_non_key_columns_become_nullable(self):
        ft = self._frame('data.upsample(time_column="t", every="30m")')
        assert ft.columns["t"].dtype == Datetime()
        assert ft.columns["g"].dtype == Nullable(Utf8())
        assert ft.columns["v"].dtype == Nullable(Int64())

    def test_positional_time_column(self):
        ft = self._frame('data.upsample("t", every="30m")')
        assert ft.columns["t"].dtype == Datetime()
        assert ft.columns["v"].dtype == Nullable(Int64())

    def test_group_by_keys_stay_non_nullable(self):
        ft = self._frame('data.upsample(time_column="t", every="30m", group_by="g")')
        assert ft.columns["g"].dtype == Utf8()
        assert ft.columns["v"].dtype == Nullable(Int64())

    def test_group_by_list_keys_stay_non_nullable(self):
        ft = self._frame('data.upsample(time_column="t", every="30m", group_by=["g"])')
        assert ft.columns["g"].dtype == Utf8()
        assert ft.columns["v"].dtype == Nullable(Int64())

    def test_already_nullable_column_is_not_double_wrapped(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                t: pl.Datetime
                v: int = pa.Field(nullable=True)

            def f(data: DataFrame[In]):
                return data.upsample("t", every="30m")
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Nullable(Int64())

    def test_upsample_on_lazyframe_is_ply030(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame, LazyFrame

            class In(pa.DataFrameModel):
                t: pl.Datetime
                v: int

            def f(data: LazyFrame[In]):
                return data.upsample("t", every="30m")
        """
        )
        results = analyze_source(source)
        assert any("PLY030" in e for e in results[0].errors), results[0].errors


class TestJoinWhereDegradation:
    """``df.join_where(other, *predicates)`` (issue #74): polars documents
    the method as experimental ("may be changed at any point"), so instead
    of encoding its schema we degrade to an OPEN frame and surface PLW007 —
    correct code passes (no more hard "Could not infer return type"), and
    the degradation is visible.

    Probed (polars 1.41.2, identical on 1.37.0): the observed schema is
    left + right columns with ``_right`` suffix on collisions, on both
    DataFrame and LazyFrame — a candidate for precise inference if/when
    polars stabilizes the API.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class LeftK(pa.DataFrameModel):
                k: int
                x: int

            class RightK(pa.DataFrameModel):
                k: int
                y: int
        """
    )

    def test_join_where_returns_open_frame_with_plw007(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(a: DataFrame[LeftK], b: DataFrame[RightK]):
                return a.join_where(b, pl.col("x") < pl.col("y"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns == {}
        assert ft.rest is not None
        plw007 = [w for w in results[0].warnings if "PLW007" in w]
        assert len(plw007) == 1, results[0].warnings
        assert "join_where" in plw007[0] and "experimental" in plw007[0]

    def test_join_where_on_lazyframes_stays_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame, LazyFrame

            class LeftK(pa.DataFrameModel):
                k: int
                x: int

            class RightK(pa.DataFrameModel):
                k: int
                y: int

            def f(a: LazyFrame[LeftK], b: LazyFrame[RightK]):
                return a.join_where(b, pl.col("x") < pl.col("y"))
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.is_lazy is True

    def test_validate_retracts_the_join_where_plw007(self):
        # Schema.validate is exactly the repair the warning recommends.
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                k: int
                x: int
                k_right: int
                y: int

            def f(a: DataFrame[LeftK], b: DataFrame[RightK]) -> DataFrame[Out]:
                return Out.validate(a.join_where(b, pl.col("x") < pl.col("y")))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        assert not any("PLW007" in w for w in results[0].warnings), results[0].warnings

    def test_join_where_warning_is_deduped_through_agg_chains(self):
        # .agg() chains analyze the grouped receiver twice (laziness probe
        # + _infer_agg_call) — the warning must fire once per call site.
        source = self.HEADER + textwrap.dedent(
            """
            def f(a: DataFrame[LeftK], b: DataFrame[RightK]):
                return a.join_where(b, pl.col("x") < pl.col("y")).group_by("k").agg(pl.len())
            """
        )
        results = analyze_source(source)
        plw007 = [w for w in results[0].warnings if "PLW007" in w]
        assert len(plw007) == 1, results[0].warnings


class TestToDummiesWarning:
    """``df.to_dummies(...)`` (issue #74): output column names depend on
    runtime values (``c`` -> ``c_a``, ``c_b``, ... UInt8), exactly like
    pivot — so it gets the same PLW005 annotate-the-result nudge instead
    of dying silently into "Could not infer return type"."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                c: str
                n: int
        """
    )

    def test_to_dummies_emits_plw005_with_actionable_message(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(data: DataFrame[In]):
                return data.to_dummies("c")
            """
        )
        results = analyze_source(source)
        f = results[0]
        assert f.errors == [], f.errors
        assert any("PLW005" in w for w in f.warnings), f.warnings
        assert any("to_dummies" in w and "DataFrame[" in w for w in f.warnings)
        # The closed receiver yields a copy-pasteable hint: passthrough
        # columns keep their dtype, dummied columns are UInt8-per-value.
        assert any("n: pl.Int64" in w and "UInt8" in w for w in f.warnings), f.warnings

    def test_to_dummies_without_columns_dummies_everything(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(data: DataFrame[In]):
                return data.to_dummies()
            """
        )
        results = analyze_source(source)
        f = results[0]
        assert any("PLW005" in w for w in f.warnings), f.warnings
        # No passthrough column survives — the hint must not claim one.
        assert not any("n: pl.Int64" in w for w in f.warnings), f.warnings

    def test_to_dummies_assigned_to_typed_var_uses_annotation(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                c_a: pl.UInt8
                c_b: pl.UInt8
                n: int

            def f(data: DataFrame[In]) -> DataFrame[Out]:
                result: DataFrame[Out] = data.to_dummies("c")
                return result
            """
        )
        results = analyze_source(source)
        assert any("PLW005" in w for w in results[0].warnings)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["c_a"].dtype == UInt8()


class TestGroupByMapGroupsWarning:
    """``group_by(...).map_groups(fn)`` (issue #87): the output schema
    depends on the group function's body — statically unknowable, same
    family as pivot/to_dummies — so it gets the PLW005 annotate-the-result
    nudge instead of the generic "Could not infer return type".
    (``GroupBy.apply`` no longer exists on probed polars 1.37.0/1.41.2,
    so there is no alias to cover.)"""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class Src(pa.DataFrameModel):
                g: str
                v: int
        """
    )

    def test_map_groups_emits_plw005(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[Src]):
                return df.group_by("g").map_groups(lambda gdf: gdf.head(1))
            """
        )
        results = analyze_source(source)
        f = results[0]
        assert f.errors == [], f.errors
        assert any("PLW005" in w for w in f.warnings), f.warnings
        assert any("map_groups" in w and "DataFrame[" in w for w in f.warnings)

    def test_map_groups_lazy_receiver_suggests_lazyframe(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[Src]):
                return df.lazy().group_by("g").map_groups(lambda gdf: gdf.head(1), schema=None)
            """
        )
        results = analyze_source(source)
        f = results[0]
        assert any("PLW005" in w and "LazyFrame[" in w for w in f.warnings), f.warnings

    def test_map_groups_assigned_to_typed_var_uses_annotation(self):
        source = self.HEADER + textwrap.dedent(
            """
            class Out(pa.DataFrameModel):
                g: str
                v: int
                n: pl.Int32

            def f(df: DataFrame[Src]) -> DataFrame[Out]:
                x: DataFrame[Out] = df.group_by("g").map_groups(
                    lambda gdf: gdf.head(1).with_columns(n=pl.lit(1, dtype=pl.Int32))
                )
                return x
            """
        )
        results = analyze_source(source)
        assert any("PLW005" in w for w in results[0].warnings)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["n"].dtype == Int32()


class TestM13LazyFrame:
    """LazyFrame-specific methods: identity passthrough + sinks + collect_async."""

    LF_HEADER = """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame, LazyFrame
"""

    def test_collect_async_preserves_schema(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(lf: LazyFrame[S]) -> DataFrame[S]:
                return lf.collect_async()
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()

    @pytest.mark.parametrize(
        "method",
        ["cache()", "first()", "last()", "inspect()", "top_k(5, by='id')", "bottom_k(5, by='id')"],
    )
    def test_lazy_identity_methods(self, method: str):
        source = textwrap.dedent(
            self.LF_HEADER
            + f"""
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(lf: LazyFrame[S]) -> LazyFrame[S]:
                return lf.{method}
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "id" in ft.columns and "value" in ft.columns

    def test_sink_csv_keeps_chain(self):
        """sink_csv terminates lazily but we keep the schema for chain continuity."""
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(lf: LazyFrame[S]) -> LazyFrame[S]:
                return lf.sink_csv("out.csv", lazy=True)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_collect_batches_preserves_schema(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(lf: LazyFrame[S]) -> DataFrame[S]:
                return lf.collect_batches()
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_full_lazy_pipeline(self):
        """Lazy chain end-to-end: filter → with_columns → group_by → collect."""
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class In(pa.DataFrameModel):
                k: str
                v: pl.Float64

            class Out(pa.DataFrameModel):
                k: str
                v_sum: pl.Float64

            def f(lf: LazyFrame[In]) -> DataFrame[Out]:
                return (
                    lf.filter(pl.col("v") > 0)
                      .with_columns(pl.col("v").fill_null(0.0).alias("v"))
                      .group_by("k")
                      .agg(pl.col("v").sum().alias("v_sum"))
                      .collect()
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_sum"].dtype == Float64()


class TestM14PartitionBy:
    """``df.partition_by("k")`` returns ``list[FrameType]`` — element-typed."""

    def test_partition_subscript_first_element(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(df: DataFrame[S]) -> DataFrame[S]:
                parts = df.partition_by("k")
                return parts[0]
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "k" in ft.columns and "v" in ft.columns

    def test_partition_for_loop_propagates_element_type(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            class Out(pa.DataFrameModel):
                v: pl.Float64

            def f(df: DataFrame[S]):
                last: DataFrame[Out] = df  # placeholder so we can read it back
                for part in df.partition_by("k"):
                    last = part.select(pl.col("v"))
                return last
        """
        )
        results = analyze_source(source)
        # No PLY001 — pl.col("v") was resolved using the loop var's element type.
        assert not any("PLY001" in e for e in results[0].errors), results[0].errors

    def test_partition_include_key_false_drops_key(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(df: DataFrame[S]):
                parts = df.partition_by("k", include_key=False)
                first = parts[0]
                return first.select(pl.col("v"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors

    def test_partition_include_key_false_drops_key_errors_on_key_ref(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(df: DataFrame[S]):
                parts = df.partition_by("k", include_key=False)
                first = parts[0]
                return first.select(pl.col("k"))
        """
        )
        results = analyze_source(source)
        # k was excluded from each partition; selecting it now raises PLY001.
        assert any("PLY001" in e and "'k'" in e for e in results[0].errors)


class TestM15LazyEagerDistinction:
    """LazyFrame[T] is statically distinct from DataFrame[T]."""

    LF_HEADER = """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame, LazyFrame
"""

    def test_writing_csv_on_lazy_emits_ply030(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(lf: LazyFrame[S]):
                return lf.write_csv("out.csv")
        """
        )
        results = analyze_source(source)
        assert any("PLY030" in e and "write_csv" in e for e in results[0].errors)
        assert any("collect" in e for e in results[0].errors)

    def test_sink_on_eager_emits_ply031(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(df: DataFrame[S]):
                return df.sink_csv("out.csv")
        """
        )
        results = analyze_source(source)
        assert any("PLY031" in e and "sink_csv" in e for e in results[0].errors)

    def test_collect_on_eager_emits_ply031(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(df: DataFrame[S]):
                return df.collect()
        """
        )
        results = analyze_source(source)
        assert any("PLY031" in e for e in results[0].errors)

    def test_function_call_lazy_arg_to_eager_param_errors(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def helper(df: DataFrame[S]) -> DataFrame[S]:
                return df

            def caller(lf: LazyFrame[S]):
                return helper(lf)
        """
        )
        results = analyze_source(source)
        caller = next(r for r in results if r.name == "caller")
        assert any("PLY032" in e for e in caller.errors)
        assert any(".collect()" in e for e in caller.errors)

    def test_return_lazy_when_eager_declared_errors(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(lf: LazyFrame[S]) -> DataFrame[S]:
                return lf
        """
        )
        # check_function adds the return-type lazy mismatch.
        from polypolarism.checker import check_source

        results_chk = check_source(source)
        assert any("PLY032" in str(e) for e in results_chk[0].errors)

    def test_lazy_to_eager_via_collect_is_clean(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(lf: LazyFrame[S]) -> DataFrame[S]:
                return lf.collect()
        """
        )
        from polypolarism.checker import check_source

        results_chk = check_source(source)
        assert results_chk[0].passed is True, results_chk[0].errors

    def test_eager_to_lazy_via_lazy_method(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def f(df: DataFrame[S]) -> LazyFrame[S]:
                return df.lazy()
        """
        )
        from polypolarism.checker import check_source

        results_chk = check_source(source)
        assert results_chk[0].passed is True, results_chk[0].errors

    def test_chain_lazy_through_filter_select_collect(self):
        source = textwrap.dedent(
            self.LF_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(lf: LazyFrame[S]) -> DataFrame[S]:
                return lf.filter(pl.col("value") > 0).select(pl.col("id"), pl.col("value")).collect()
        """
        )
        from polypolarism.checker import check_source

        results_chk = check_source(source)
        assert results_chk[0].passed is True, results_chk[0].errors


class TestFunctionAnalysisDataClass:
    """Test FunctionAnalysis data class."""

    def test_has_errors_property(self):
        """has_errors returns True when errors exist."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def bad_col(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.select(pl.col("missing"))
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is True

    def test_has_errors_false_when_valid(self):
        """has_errors returns False when no errors."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def identity(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        assert results[0].has_errors is False


class TestAnalyzeIntermediateVariables:
    """Test analysis with intermediate variable assignments."""

    def test_tracks_variable_assignment(self):
        """Track DataFrame type through variable assignment."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class Users(pa.DataFrameModel):
                user_id: int
                name: str

            class Orders(pa.DataFrameModel):
                order_id: int
                user_id: int
                amount: pl.Float64

            class Out(pa.DataFrameModel):
                user_id: int
                name: str
                order_id: int
                amount: pl.Float64

            def with_intermediate(
                users: DataFrame[Users],
                orders: DataFrame[Orders],
            ) -> DataFrame[Out]:
                joined = users.join(orders, on="user_id", how="inner")
                return joined
        """
        )

        results = analyze_source(source)

        expected = FrameType(
            {
                "user_id": Int64(),
                "name": Utf8(),
                "order_id": Int64(),
                "amount": Float64(),
            }
        )
        assert results[0].inferred_return_type == expected


class TestSourceLocation:
    """Test source location information in analysis results."""

    def test_function_analysis_has_lineno(self):
        """FunctionAnalysis should include function definition line number."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        """
        )

        results = analyze_source(source)

        assert len(results) == 1
        # Function def line is determined by the schema preamble.
        assert results[0].lineno > 0

    def test_multiple_functions_have_correct_linenos(self):
        """Multiple functions should each have their correct line numbers."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                name: str

            def first(df: DataFrame[A]) -> DataFrame[A]:
                return df

            def second(df: DataFrame[B]) -> DataFrame[B]:
                return df
        """
        )

        results = analyze_source(source)

        assert len(results) == 2
        results.sort(key=lambda r: r.lineno)
        assert results[0].name == "first"
        assert results[1].name == "second"
        # Second function comes after the first
        assert results[1].lineno > results[0].lineno


class TestUnknownColumnSubtype:
    """analyzer._is_column_subtype: Unknown is compatible in both directions."""

    def test_unknown_actual_passes_any_expected(self):
        assert _is_column_subtype(Unknown(), Int64())
        assert _is_column_subtype(Unknown(), Utf8())
        assert _is_column_subtype(Unknown(), Nullable(Int64()))

    def test_any_actual_passes_unknown_expected(self):
        assert _is_column_subtype(Int64(), Unknown())
        assert _is_column_subtype(Nullable(Utf8()), Unknown())

    def test_nullable_unknown_actual_passes_non_nullable_expected(self):
        assert _is_column_subtype(Nullable(Unknown()), Int64())

    def test_regular_mismatch_still_fails(self):
        assert not _is_column_subtype(Utf8(), Int64())
        assert not _is_column_subtype(Nullable(Int64()), Int64())


class TestOpenFrameSubtype:
    """analyzer._is_frame_subtype: open actual frames may satisfy extra columns."""

    def test_missing_required_column_allowed_on_open_actual(self):
        actual = FrameType({"id": Int64()}, rest=RowVar("r"))
        expected = FrameType({"id": Int64(), "qty": Int64()})
        assert _is_frame_subtype(actual, expected)

    def test_missing_required_column_rejected_on_closed_actual(self):
        actual = FrameType({"id": Int64()})
        expected = FrameType({"id": Int64(), "qty": Int64()})
        assert not _is_frame_subtype(actual, expected)

    def test_present_column_still_type_checked_on_open_actual(self):
        actual = FrameType({"id": Utf8()}, rest=RowVar("r"))
        expected = FrameType({"id": Int64()})
        assert not _is_frame_subtype(actual, expected)


def _run_body(frame: FrameType, body: str):
    """Drive FunctionBodyAnalyzer directly with a pre-built input frame.

    Open frames cannot yet be written down in a Pandera schema, so these
    tests inject them as the type of ``df`` and analyze a small body.
    """
    import ast

    errors: list[str] = []
    analyzer = FunctionBodyAnalyzer({"df": frame}, errors)
    tree = ast.parse(textwrap.dedent(body))
    for stmt in tree.body:
        analyzer.visit(stmt)
    return analyzer


class TestOpenFrameMethodCalls:
    """Column references that are missing on an open frame are not errors."""

    def _open_frame(self) -> FrameType:
        return FrameType({"id": Int64()}, rest=RowVar("r"))

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_drop_missing_column_on_open_frame_no_error(self):
        analyzer = _run_body(self._open_frame(), "out = df.drop('ghost')")
        assert analyzer.errors == []
        assert "ghost" not in analyzer.var_types["out"].columns

    def test_drop_missing_column_on_closed_frame_still_errors(self):
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.drop('ghost')")
        assert any("ghost" in e for e in analyzer.errors)

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_explode_missing_column_on_open_frame_becomes_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.explode('items')")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["items"].dtype == Unknown()

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_unnest_missing_column_on_open_frame_no_error(self):
        analyzer = _run_body(self._open_frame(), "out = df.unnest('s')")
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.rest is not None
        assert "s" not in out.columns

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_select_col_expr_missing_on_open_frame_registers_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.select(pl.col('ghost'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["ghost"].dtype == Unknown()

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_select_plural_col_missing_on_open_frame_registers_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.select(pl.col('a', 'b'))")
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.columns["a"].dtype == Unknown()
        assert out.columns["b"].dtype == Unknown()

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_with_columns_plural_col_missing_on_open_frame_no_error(self):
        analyzer = _run_body(self._open_frame(), "out = df.with_columns(pl.col('a', 'b'))")
        assert analyzer.errors == []

    def test_select_result_is_closed_even_from_open_input(self):
        """A projection enumerates its output — openness does not survive."""
        analyzer = _run_body(self._open_frame(), "out = df.select(pl.col('id'))")
        assert analyzer.var_types["out"].rest is None


class TestRestPropagation:
    """rest / strict survive shape-preserving operations."""

    def test_with_columns_propagates_rest_and_strict(self):
        frame = FrameType({"id": Int64()}, strict=True, rest=RowVar("r"))
        analyzer = _run_body(frame, "out = df.with_columns(doubled=pl.col('id') * 2)")
        out = analyzer.var_types["out"]
        assert out.rest is not None
        assert out.strict is True
        assert out.columns["doubled"].dtype == Int64()


class TestSelectStringColumns:
    """Issue #7: bare string column names in select / with_columns."""

    def _source(self, call: str) -> str:
        return textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                a: int
                b: str
                c: pl.Float64

            class Out(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[In]) -> DataFrame[Out]:
                return df.{call}
        """
        )

    def test_select_positional_strings(self):
        results = analyze_source(self._source('select("a", "b")'))
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_select_list_of_strings(self):
        results = analyze_source(self._source('select(["a", "b"])'))
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_select_string_missing_column_errors(self):
        results = analyze_source(self._source('select("a", "ghost")'))
        assert any("ghost" in str(e) for e in results[0].errors)

    def test_select_list_with_missing_column_errors(self):
        results = analyze_source(self._source('select(["a", "ghost"])'))
        assert any("ghost" in str(e) for e in results[0].errors)

    def test_select_mixes_strings_and_exprs(self):
        results = analyze_source(self._source('select("a", pl.col("c").alias("b"))'))
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Float64()})

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_select_string_on_open_frame_registers_unknown(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.select("ghost")')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["ghost"].dtype == Unknown()

    def test_select_kwarg_string_is_column_reference(self):
        # ``select(x="a")`` — polars parses a bare string in expression
        # position as a column name, not a Utf8 literal: x takes a's dtype.
        frame = FrameType({"a": Int64(), "s": Utf8()})
        analyzer = _run_body(frame, 'out = df.select(x="a")')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["x"].dtype == Int64()

    def test_with_columns_kwarg_string_is_column_reference(self):
        frame = FrameType({"a": Int64(), "s": Utf8()})
        analyzer = _run_body(frame, 'out = df.with_columns(y="a")')
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.columns["y"].dtype == Int64()
        assert out.columns["a"].dtype == Int64()

    def test_select_kwarg_string_missing_column_errors(self):
        frame = FrameType({"a": Int64()})
        analyzer = _run_body(frame, 'out = df.select(x="ghost")')
        assert any("ghost" in e for e in analyzer.errors)

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_select_kwarg_string_missing_on_open_frame_registers_unknown(self):
        frame = FrameType({"a": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.select(x="ghost")')
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["x"].dtype == Unknown()

    def test_with_columns_string_keeps_schema(self):
        results = analyze_source(self._source('with_columns("a")'))
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType(
            {"a": Int64(), "b": Utf8(), "c": Float64()}
        )

    def test_with_columns_string_list_keeps_schema(self):
        results = analyze_source(self._source('with_columns(["a", "b"])'))
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType(
            {"a": Int64(), "b": Utf8(), "c": Float64()}
        )

    def test_with_columns_string_missing_column_errors(self):
        results = analyze_source(self._source('with_columns("ghost")'))
        assert any("ghost" in str(e) for e in results[0].errors)

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_with_columns_string_missing_on_open_frame_no_error(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.with_columns("ghost")')
        assert analyzer.errors == []


class TestUnknownColumnRegistration:
    """Issue #8: columns added by un-inferable expressions stay in the schema."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                v: int
                ts: pl.Datetime
"""
    )

    def test_when_then_otherwise_column_registers_and_chains(self):
        # Since #40 the when/then/otherwise dtype is inferred precisely
        # (then(1).otherwise(0) -> Int64); the column must still register
        # and chain downstream.
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(
                    a=pl.when(pl.col("v") > 0).then(1).otherwise(0)
                ).with_columns(b=pl.col("a") + 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["a"].dtype == Int64()
        assert inferred.columns["b"].dtype == Int64()

    def test_cast_after_uninferable_method_pins_dtype(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(
                    a=pl.col("v").interpolate().cast(pl.Int64)
                ).with_columns(b=pl.col("a") + 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["a"].dtype == Int64()
        assert inferred.columns["b"].dtype == Int64()

    # upgrade trigger: interpolate() gains a return-dtype inference rule
    @pytest.mark.imprecision
    def test_uninferable_chain_alias_survives_in_select(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.select(pl.col("v").interpolate().alias("x"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["x"].dtype == Unknown()

    # upgrade trigger: interpolate() gains a return-dtype inference rule
    @pytest.mark.imprecision
    def test_uninferable_chain_keeps_receiver_name_in_with_columns(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(pl.col("v").interpolate())
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["v"].dtype == Unknown()

    def test_dt_strftime_returns_utf8(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(ym=pl.col("ts").dt.strftime("%Y-%m"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["ym"].dtype == Utf8()

    def test_dt_to_string_returns_utf8(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.with_columns(ym=pl.col("ts").dt.to_string("%Y-%m"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["ym"].dtype == Utf8()

    # upgrade trigger: interpolate() gains a return-dtype inference rule
    @pytest.mark.imprecision
    def test_unknown_column_usable_as_group_by_key(self):
        # ``interpolate`` is not in the inference tables, so the kwarg
        # column registers as Unknown and stays usable as a group_by key.
        # (This used to use a when/then/otherwise expression, which since
        # #40 infers precisely.)
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return (
                    df.with_columns(ym=pl.col("v").interpolate())
                    .group_by("ym")
                    .agg(total=pl.col("v").sum())
                )
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["ym"].dtype == Unknown()
        assert inferred.columns["total"].dtype == Int64()

    def test_agg_kwarg_uninferable_registers_unknown(self):
        # ``pl.int_range(...)`` is not an aggregation polypolarism models —
        # the kwarg output column must still register (as Unknown).
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.group_by("v").agg(n=pl.int_range(10))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["n"].dtype == Unknown()

    def test_select_kwarg_uninferable_registers_unknown(self):
        # ``interpolate`` is not in the inference tables — the kwarg output
        # column must still register (as Unknown). (This used to use a
        # when/then/otherwise expression, which since #40 infers precisely.)
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.select(a=pl.col("v").interpolate())
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["a"].dtype == Unknown()


class TestContainerFieldAnalysis:
    """Issue #10: container-typed schema fields work with explode/unnest/.arr."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                id: int
                vals: pl.List(pl.Int64) = pa.Field()
                items: pl.List(pl.Struct) = pa.Field()
                q: pl.Array(pl.Int64, 4) = pa.Field()
"""
    )

    def test_explode_list_field_yields_element_type(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.explode("vals")
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["vals"].dtype == Int64()

    def test_explode_then_unnest_unknown_struct_opens_frame(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.explode("items").unnest("items")
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert "items" not in inferred.columns
        assert inferred.rest is not None

    def test_arr_sum_on_array_field_infers_element_type(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                return df.select("id", total=pl.col("q").arr.sum())
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["id"].dtype == Int64()
        assert inferred.columns["total"].dtype == Int64()


class TestJoinSuffix:
    """#11: ``join(..., suffix=...)`` must rename overlapping right columns."""

    def test_join_custom_suffix_renames_overlap(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                g: int
                v: int

            class B(pa.DataFrameModel):
                g: int
                v: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="g", suffix="_new")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_new"].dtype == Float64()
        assert "v_right" not in ft.columns

    def test_join_default_suffix_still_right(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                g: int
                v: int

            class B(pa.DataFrameModel):
                g: int
                v: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="g")
        """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_right"].dtype == Float64()

    def test_join_asof_custom_suffix(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class L(pa.DataFrameModel):
                ts: pl.Datetime
                v: int

            class R(pa.DataFrameModel):
                ts: pl.Datetime
                v: pl.Float64

            def f(left: DataFrame[L], right: DataFrame[R]):
                return left.join_asof(right, on="ts", suffix="_r")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        # join_asof is left-join shaped: right conflict gets the suffix and
        # becomes nullable.
        assert ft.columns["v_r"].dtype == Nullable(Float64())
        assert "v_right" not in ft.columns


class TestJoinMultiKey:
    """``join(on=["x", "y"])`` resolves instead of raising a false PLY010."""

    def test_join_on_list_literal(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                x: int
                y: str
                a: pl.Float64

            class B(pa.DataFrameModel):
                x: int
                y: str
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on=["x", "y"], how="inner")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["x"].dtype == Int64()
        assert ft.columns["y"].dtype == Utf8()
        assert ft.columns["a"].dtype == Float64()
        assert ft.columns["b"].dtype == Float64()
        assert "x_right" not in ft.columns
        assert "y_right" not in ft.columns


class TestConstantResolution:
    """#12: column-spec args passed via simple constants must resolve."""

    def test_join_on_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "id"

            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on=KEY, how="inner")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["a"].dtype == Float64()
        assert ft.columns["b"].dtype == Float64()

    def test_join_on_module_list_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            ON_COLS = ["x", "y"]

            class A(pa.DataFrameModel):
                x: int
                y: str
                a: pl.Float64

            class B(pa.DataFrameModel):
                x: int
                y: str
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on=ON_COLS, how="inner")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Float64()
        assert ft.columns["b"].dtype == Float64()

    def test_join_suffix_via_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            SUFFIX = "_b"

            class A(pa.DataFrameModel):
                g: int
                v: int

            class B(pa.DataFrameModel):
                g: int
                v: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="g", suffix=SUFFIX)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v_b"].dtype == Float64()

    def test_local_constant_resolves(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                key = "id"
                return a.join(b, on=key, how="inner")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["b"].dtype == Float64()

    def test_local_constant_shadows_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "nope"

            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                KEY = "id"
                return a.join(b, on=KEY, how="inner")
        """
        )
        results = analyze_source(source)
        # The local "id" shadows the bogus module-level "nope" — no errors.
        assert results[0].has_errors is False, results[0].errors

    def test_reassignment_invalidates_local_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def compute_key():
                return "id"

            def f(a: DataFrame[A], b: DataFrame[B]):
                key = "id"
                key = compute_key()
                return a.join(b, on=key, how="inner")
        """
        )
        results = analyze_source(source)
        # ``key`` is no longer a known constant; the join keys are
        # unresolvable, so the original PLY010 fires.
        assert any("PLY010" in e for e in results[0].errors), results[0].errors

    def test_unpivot_on_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            ON_COLS = ["a", "b"]

            class Wide(pa.DataFrameModel):
                id: int
                a: pl.Float64
                b: pl.Float64

            def f(wide: DataFrame[Wide]):
                return wide.unpivot(index=["id"], on=ON_COLS)
        """
        )
        results = analyze_source(source)
        assert not any("PLY022" in e for e in results[0].errors), results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["variable"].dtype == Utf8()
        assert ft.columns["value"].dtype == Float64()

    def test_group_by_key_via_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            GROUP_KEY = "g"

            class S(pa.DataFrameModel):
                g: str
                v: pl.Float64

            def f(df: DataFrame[S]):
                return df.group_by(GROUP_KEY).agg(pl.col("v").sum().alias("total"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["g"].dtype == Utf8()
        assert ft.columns["total"].dtype == Float64()

    def test_drop_targets_via_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            DROP_COLS = ["b"]

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.drop(DROP_COLS)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "b" not in ft.columns
        assert ft.columns["a"].dtype == Int64()

    def test_explode_target_via_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from typing import Annotated

            LIST_COL = "tags"

            class S(pa.DataFrameModel):
                id: int
                tags: Annotated[pl.List, pl.Utf8()]

            def f(df: DataFrame[S]):
                return df.explode(LIST_COL)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["tags"].dtype == Utf8()

    def test_drop_nulls_subset_via_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            SUBSET = ["v"]

            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64 = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.drop_nulls(subset=SUBSET)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["v"].dtype == Float64()

    def test_partition_by_keys_via_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            PART_KEY = "k"

            class S(pa.DataFrameModel):
                k: str
                v: pl.Float64

            def f(df: DataFrame[S]):
                parts = df.partition_by(PART_KEY, include_key=False)
                first = parts[0]
                return first.select(pl.col("k"))
        """
        )
        results = analyze_source(source)
        # k was excluded from each partition; selecting it raises PLY001.
        assert any("PLY001" in e and "'k'" in e for e in results[0].errors)

    def test_annotated_local_constant_resolves(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def f(a: DataFrame[A], b: DataFrame[B]):
                key: str = "id"
                return a.join(b, on=key, how="inner")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["b"].dtype == Float64()

    def test_constant_resolution_in_untyped_helper(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "id"

            class A(pa.DataFrameModel):
                id: int
                a: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                b: pl.Float64

            def helper(a, b):
                return a.join(b, on=KEY, how="inner")

            def f(a: DataFrame[A], b: DataFrame[B]):
                return helper(a, b)
        """
        )
        results = analyze_source(source)
        f_result = next(r for r in results if r.name == "f")
        assert f_result.has_errors is False, f_result.errors
        ft = f_result.inferred_return_type
        assert ft is not None
        assert ft.columns["a"].dtype == Float64()
        assert ft.columns["b"].dtype == Float64()


class TestBareConstantLiterals:
    """Bare Python constants in expressions are typed like ``pl.lit`` values.

    ``df.select(x=1)`` is a literal column in polars; recognising bare
    constants also gives binary operators a right-operand type for
    expressions like ``pl.col("a") * 2``.
    """

    def test_select_kwarg_int_literal(self):
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.select(x=1)")
        assert analyzer.var_types["out"].columns["x"].dtype == Int64()

    def test_select_kwarg_float_literal(self):
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.select(x=2.5)")
        assert analyzer.var_types["out"].columns["x"].dtype == Float64()

    def test_select_kwarg_bool_literal_is_boolean_not_int(self):
        """bool is a subclass of int — must check bool first."""
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.select(x=True)")
        assert analyzer.var_types["out"].columns["x"].dtype == Boolean()

    def test_select_kwarg_none_literal_is_null(self):
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.select(x=None)")
        assert analyzer.var_types["out"].columns["x"].dtype == Null()

    def test_select_kwarg_str_is_column_ref_not_literal(self):
        """Polars parses a bare string in expression position as a column
        name — ``select(x="id")`` selects ``id`` under the name ``x``.
        Only operator operands treat strings as implicit literals."""
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.select(x='id')")
        assert analyzer.var_types["out"].columns["x"].dtype == Int64()

    def test_bare_string_positional_still_selects_column(self):
        """Positional bare strings remain column selections, not literals."""
        frame = FrameType({"id": Int64(), "name": Utf8()})
        analyzer = _run_body(frame, "out = df.select('name')")
        out = analyzer.var_types["out"]
        assert list(out.columns) == ["name"]
        assert out.columns["name"].dtype == Utf8()


class TestArithmeticBinOpInference:
    """Issue #14: binary arithmetic uses polars promotion rules, not left-type.

    True division ``/`` always yields Float64 (int/int included); floor
    division ``//`` keeps the integer dtype; other ops promote numerically
    and propagate Nullable from either operand.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "a": Int64(),
                "b": Int64(),
                "f": Float64(),
                "n": Nullable(Int64()),
                "u": Unknown(),
                "s": Utf8(),
                "t": Nullable(Utf8()),
            }
        )

    # -- true division ------------------------------------------------------

    def test_int_truediv_int_is_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') / pl.col('b'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_int_truediv_int_literal_is_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') / 2)")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_float_truediv_float_is_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('f') / pl.col('f'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_truediv_nullable_operand_is_nullable_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') / pl.col('n'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Float64())

    def test_truediv_unknown_operand_is_unknown(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') / pl.col('u'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_truediv_nullable_unknown_operand_is_unknown(self):
        """Unknown is detected on the base type, after the Nullable unwrap."""
        frame = FrameType({"a": Int64(), "nu": Nullable(Unknown())})
        analyzer = _run_body(frame, "out = df.select(r=pl.col('a') / pl.col('nu'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_truediv_one_resolved_operand_is_float64(self):
        """A single resolved operand is enough — / yields Float64 regardless."""
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') / helper())")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_truediv_no_resolved_operands_registers_unknown(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=helper() / other())")
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    # -- floor division and other arithmetic --------------------------------

    def test_int_floordiv_int_stays_int64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') // pl.col('b'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Int64()

    def test_int_plus_float_promotes_to_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') + pl.col('f'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_int_plus_float_literal_promotes_to_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') + 2.5)")
        assert analyzer.var_types["out"].columns["r"].dtype == Float64()

    def test_int_plus_nullable_int_is_nullable_int64(self):
        """Issue #18 (arithmetic side): Nullable propagates through +."""
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') + pl.col('n'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Int64())

    def test_unknown_plus_int_is_unknown(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('u') + 1)")
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_only_left_resolved_keeps_left_type(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') + helper())")
        assert analyzer.var_types["out"].columns["r"].dtype == Int64()

    def test_only_right_resolved_keeps_right_type(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=helper() + pl.col('a'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Int64()

    def test_output_name_defaults_to_left_column(self):
        analyzer = _run_body(self._frame(), "out = df.select(pl.col('a') / pl.col('b'))")
        out = analyzer.var_types["out"]
        assert list(out.columns) == ["a"]
        assert out.columns["a"].dtype == Float64()

    # -- string concat fallback ---------------------------------------------

    def test_utf8_concat_keeps_utf8(self):
        """Utf8 + Utf8 is concat — promotion fails, left type wins."""
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + pl.col('s'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Utf8()

    def test_utf8_concat_with_nullable_operand_is_nullable_utf8(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + pl.col('t'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Utf8())


class TestArithmeticIncompatibleDtypes:
    """Issue #30: arithmetic between incompatible dtypes flags PLY009.

    Ground truth verified against polars 1.41.2 by driving the full
    8-dtype x {+, -, *, /, //, %, **} product through ``df.select``
    (probe script output in the implementing commit message). Allowed
    cells — every other pairing within the closed set
    {numeric, Utf8, Boolean, Date, Datetime, Time, Duration} raises
    InvalidOperationError / SchemaError at runtime:

    ``+``  num+num -> promoted; bool acts as int with num; bool+bool -> UInt32;
           str+str / str+bool / bool+str -> String (bool casts to string);
           Date+Duration -> Date; Datetime+Duration -> Datetime (tz kept);
           Duration+Duration -> Duration. NOT Time+Duration.
    ``-``  num/bool as ``+`` except bool-bool errors;
           {Date,Datetime} - {Date,Datetime} -> Duration; Time-Time -> Duration;
           Date-Duration -> Date; Datetime-Duration -> Datetime;
           Duration-Duration -> Duration. NOT Duration-Date/Datetime/Time,
           Time-Duration.
    ``*``  num/bool as ``-`` (bool*bool errors); Duration*num and
           num*Duration -> Duration. NOT Duration*bool, Duration*Duration.
    ``/``  any num/bool mix -> Float64; Duration/num -> Duration;
           Duration/Duration -> Float64. NOT Duration/bool, num/Duration,
           and no str or Date/Datetime/Time operand anywhere.
    ``//`` and ``%``  num mixes only (bool acts as int with num,
           bool-bool errors); every str/temporal operand errors
           (even Duration // int).
    ``**`` plain numerics only — bool, str and temporal operands all error.

    Operands outside the closed set (Unknown, unresolved, List, ...)
    and Null literals keep the legacy silent fallback — false positives
    are worse than false negatives here. Decimal has its own probed arm
    (issue #52, ``TestDecimalArithmetic``).
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "i": Int64(),
                "f": Float64(),
                "s": Utf8(),
                "b": Boolean(),
                "d": Date(),
                "dt": Datetime(),
                "tz": Datetime(tz="UTC"),
                "t": Time(),
                "du": Duration(),
                "ni": Nullable(Int64()),
                "nd": Nullable(Date()),
                "ns": Nullable(Utf8()),
                "u": Unknown(),
            }
        )

    # -- allowed combinations -------------------------------------------------

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # string concat (+ only); a Boolean operand casts to its string repr
            ("pl.col('s') + pl.col('s')", Utf8()),
            ("pl.col('s') + pl.col('b')", Utf8()),
            ("pl.col('b') + pl.col('s')", Utf8()),
            # boolean acts numeric when mixed with numerics
            ("pl.col('b') + pl.col('i')", Int64()),
            ("pl.col('i') + pl.col('b')", Int64()),
            ("pl.col('b') + pl.col('f')", Float64()),
            ("pl.col('b') + pl.col('b')", UInt32()),
            ("pl.col('b') / pl.col('b')", Float64()),
            ("pl.col('i') / pl.col('b')", Float64()),
            ("pl.col('b') // pl.col('i')", Int64()),
            ("pl.col('b') % pl.col('i')", Int64()),
            # temporal differences -> Duration (Date is us-based, Time
            # ns-based — issue #66)
            ("pl.col('d') - pl.col('d')", Duration()),
            ("pl.col('d') - pl.col('dt')", Duration()),
            ("pl.col('dt') - pl.col('d')", Duration()),
            ("pl.col('dt') - pl.col('dt')", Duration()),
            ("pl.col('tz') - pl.col('tz')", Duration()),
            ("pl.col('t') - pl.col('t')", Duration(unit="ns")),
            # date/datetime shifted by a duration
            ("pl.col('d') + pl.col('du')", Date()),
            ("pl.col('du') + pl.col('d')", Date()),
            ("pl.col('d') - pl.col('du')", Date()),
            ("pl.col('dt') + pl.col('du')", Datetime()),
            ("pl.col('du') + pl.col('dt')", Datetime()),
            ("pl.col('dt') - pl.col('du')", Datetime()),
            ("pl.col('tz') + pl.col('du')", Datetime(tz="UTC")),
            ("pl.col('du') + pl.col('tz')", Datetime(tz="UTC")),
            ("pl.col('tz') - pl.col('du')", Datetime(tz="UTC")),
            # duration arithmetic
            ("pl.col('du') + pl.col('du')", Duration()),
            ("pl.col('du') - pl.col('du')", Duration()),
            ("pl.col('du') * 2", Duration()),
            ("2 * pl.col('du')", Duration()),
            ("pl.col('du') * pl.col('f')", Duration()),
            ("pl.col('du') / 2", Duration()),
            ("pl.col('du') / pl.col('f')", Duration()),
            ("pl.col('du') / pl.col('du')", Float64()),
            # pow is numeric-only
            ("pl.col('i') ** pl.col('i')", Int64()),
            ("pl.col('i') ** pl.col('f')", Float64()),
        ],
    )
    def test_allowed_combination_infers_dtype(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    # -- known-invalid combinations -------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # string with numeric / string non-concat ops
            "pl.col('s') + pl.col('i')",
            "pl.col('i') + pl.col('s')",
            "pl.col('s') + 1",
            "pl.col('s') - pl.col('s')",
            "pl.col('s') * pl.col('i')",
            "pl.col('s') / pl.col('s')",
            "pl.col('s') // pl.col('i')",
            "pl.col('s') % pl.col('s')",
            "pl.col('s') ** pl.col('s')",
            "pl.col('s') + pl.col('d')",
            # boolean-only arithmetic beyond + and /
            "pl.col('b') - pl.col('b')",
            "pl.col('b') * pl.col('b')",
            "pl.col('b') // pl.col('b')",
            "pl.col('b') % pl.col('b')",
            "pl.col('b') ** pl.col('i')",
            "pl.col('i') ** pl.col('b')",
            # temporal combinations polars rejects
            "pl.col('d') + pl.col('d')",
            "pl.col('d') + 1",
            "pl.col('dt') + pl.col('dt')",
            "pl.col('d') + pl.col('dt')",
            "pl.col('t') + pl.col('t')",
            "pl.col('t') + pl.col('du')",
            "pl.col('du') + pl.col('t')",
            "pl.col('t') - pl.col('du')",
            "pl.col('du') - pl.col('d')",
            "pl.col('du') - pl.col('dt')",
            "pl.col('d') - pl.col('t')",
            "pl.col('d') - 1",
            "pl.col('i') + pl.col('d')",
            "pl.col('i') - pl.col('du')",
            "pl.col('b') + pl.col('d')",
            "pl.col('du') * pl.col('du')",
            "pl.col('du') * pl.col('b')",
            "pl.col('d') * 2",
            "pl.col('d') / pl.col('d')",
            "pl.col('dt') / 2",
            "pl.col('t') / 2",
            "pl.col('i') / pl.col('du')",
            "pl.col('du') / pl.col('b')",
            "pl.col('du') // 2",
            "pl.col('du') % pl.col('du')",
            "pl.col('d') % pl.col('d')",
            "pl.col('du') ** pl.col('i')",
        ],
    )
    def test_invalid_combination_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        # The error is the signal — the output registers as Unknown.
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_error_message_names_dtypes_and_op(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + pl.col('i'))")
        assert "Utf8 + Int64" in analyzer.errors[0]

    # -- nullability propagation on allowed temporal/concat results -----------

    def test_nullable_date_difference_is_nullable_duration(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nd') - pl.col('d'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Duration())

    def test_duration_times_nullable_int_is_nullable_duration(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('du') * pl.col('ni'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Duration())

    def test_nullable_string_concat_with_bool_is_nullable_utf8(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('ns') + pl.col('b'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Utf8())

    def test_invalid_pair_detected_under_nullable_wrappers(self):
        """Classification unwraps Nullable: Nullable[Utf8] + Nullable[Int64] errors."""
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('ns') + pl.col('ni'))")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        assert "Utf8 + Int64" in analyzer.errors[0]

    # -- silent fallback: not fully understood => no error ---------------------

    # upgrade trigger: Unknown operand becomes inferable upstream
    @pytest.mark.imprecision
    def test_unknown_operand_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('u') + pl.col('s'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    # upgrade trigger: unresolved call expressions gain inference
    @pytest.mark.imprecision
    def test_unresolved_operand_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + helper())")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Utf8()

    # upgrade trigger: List operands get probed PLY009 cells in the arithmetic table
    @pytest.mark.imprecision
    def test_list_operand_is_silent(self):
        frame = FrameType({"xs": ListT(Int64()), "i": Int64()})
        analyzer = _run_body(frame, "out = df.select(r=pl.col('xs') + pl.col('i'))")
        assert analyzer.errors == []

    def test_null_literal_keeps_promote_path(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('i') + None)")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Int64())

    def test_null_literal_with_string_keeps_promote_path(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + None)")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Utf8())


class TestDecimalArithmetic:
    """Issue #52: Decimal arithmetic propagates polars' precision growth.

    Ground truth probed on polars 1.41.2 by driving Decimal x {Decimal,
    every int width incl. 128-bit, every float width incl. Float16,
    Boolean, Utf8, Null literal, Date/Datetime/Time/Duration,
    Categorical/Enum/List} through ``df.select`` with all seven
    operators, in both operand orders. Eager results and
    ``LazyFrame.collect()`` agree on every cell; the *lazy*
    ``collect_schema()`` reports a stale (pre-growth) precision for
    Decimal x integer and Decimal x Null cells — polypolarism claims the
    materialized dtype.

    ``+ - * /``  Decimal x Decimal -> Decimal(38, max(scales)) — the
                 precision saturates to 38 for ANY input precisions,
                 ``*`` does NOT add scales and ``/`` stays Decimal;
                 Decimal x int -> Decimal(38, dec.scale) for every
                 signed/unsigned width, either order (int literals
                 behave like Int64 columns);
                 Decimal x Null literal -> all-null Decimal(38, scale).
    ``// % **``  raise InvalidOperationError for Decimal x
                 {Decimal, int, Null} -> PLY009.
    x float      -> Float64 for every operator except ``**`` (error),
                 both widths, either order.
    x Boolean    asymmetric: every cell errors (SchemaError "failed to
                 determine supertype") except ``Boolean / Decimal``
                 -> Float64.
    x Utf8       ``+`` stringifies the Decimal and concatenates -> Utf8;
                 every other operator errors.
    x temporal / Categorical / Enum / List: every operator errors in
                 both orders -> PLY009.
    Unprobed partners (Unknown, Struct, Binary, unresolved) keep the
    silent fallback.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "d": Decimal(10, 2),
                "d2": Decimal(10, 2),
                "d4": Decimal(12, 4),
                "d38": Decimal(38, 2),
                "i": Int64(),
                "i8": Int8(),
                "u64": UInt64(),
                "f": Float64(),
                "f32": Float32(),
                "b": Boolean(),
                "s": Utf8(),
                "dt": Date(),
                "tm": Time(),
                "du": Duration(),
                "nd": Nullable(Decimal(10, 2)),
                "ni": Nullable(Int64()),
                "u": Unknown(),
            }
        )

    # -- allowed combinations: probed result dtypes ----------------------------

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # Decimal x Decimal: precision saturates to 38, scale is the max
            ("pl.col('d') + pl.col('d2')", Decimal(38, 2)),
            ("pl.col('d') - pl.col('d2')", Decimal(38, 2)),
            ("pl.col('d') * pl.col('d2')", Decimal(38, 2)),
            ("pl.col('d') / pl.col('d2')", Decimal(38, 2)),
            # mixed scales: max(ls, rs) in either order, every operator
            ("pl.col('d') + pl.col('d4')", Decimal(38, 4)),
            ("pl.col('d4') - pl.col('d')", Decimal(38, 4)),
            ("pl.col('d') * pl.col('d4')", Decimal(38, 4)),
            ("pl.col('d4') / pl.col('d')", Decimal(38, 4)),
            # already-saturated precision stays put
            ("pl.col('d38') + pl.col('d2')", Decimal(38, 2)),
            ("pl.col('d38') * pl.col('d38')", Decimal(38, 2)),
            # Decimal x int: precision widens to 38, the Decimal's scale kept
            ("pl.col('d') + pl.col('i')", Decimal(38, 2)),
            ("pl.col('i') + pl.col('d')", Decimal(38, 2)),
            ("pl.col('d') - pl.col('i8')", Decimal(38, 2)),
            ("pl.col('d') * pl.col('u64')", Decimal(38, 2)),
            ("pl.col('d') / pl.col('i')", Decimal(38, 2)),
            ("pl.col('i') / pl.col('d')", Decimal(38, 2)),
            ("pl.col('d4') * pl.col('i')", Decimal(38, 4)),
            # int literals behave like Int64 columns (probed)
            ("pl.col('d') + 1", Decimal(38, 2)),
            ("1 + pl.col('d')", Decimal(38, 2)),
            ("pl.col('d') * 2", Decimal(38, 2)),
            ("pl.col('d') / 2", Decimal(38, 2)),
            # Decimal x float -> Float64 (every operator but **)
            ("pl.col('d') + pl.col('f')", Float64()),
            ("pl.col('f') - pl.col('d')", Float64()),
            ("pl.col('d') * pl.col('f32')", Float64()),
            ("pl.col('d') / pl.col('f')", Float64()),
            ("pl.col('d') // pl.col('f')", Float64()),
            ("pl.col('d') % pl.col('f')", Float64()),
            ("pl.col('d') + 2.5", Float64()),
            # the one asymmetric Boolean cell that works
            ("pl.col('b') / pl.col('d')", Float64()),
            # string concat stringifies the Decimal
            ("pl.col('d') + pl.col('s')", Utf8()),
            ("pl.col('s') + pl.col('d')", Utf8()),
            # Null literal: all-null output, dtype still widens (probed)
            ("pl.col('d') + None", Nullable(Decimal(38, 2))),
            ("pl.col('d') - None", Nullable(Decimal(38, 2))),
            ("pl.col('d') * None", Nullable(Decimal(38, 2))),
            ("pl.col('d') / None", Nullable(Decimal(38, 2))),
            ("pl.col('d4') + None", Nullable(Decimal(38, 4))),
        ],
    )
    def test_allowed_combination_infers_dtype(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    # -- known-invalid combinations -> PLY009 ----------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # Decimal x Decimal: // % ** raise InvalidOperationError
            "pl.col('d') // pl.col('d2')",
            "pl.col('d') % pl.col('d2')",
            "pl.col('d') ** pl.col('d2')",
            "pl.col('d') // pl.col('d4')",
            # Decimal x int: // % ** error in either order
            "pl.col('d') // pl.col('i')",
            "pl.col('i') // pl.col('d')",
            "pl.col('d') % pl.col('i')",
            "pl.col('i') % pl.col('d')",
            "pl.col('d') ** pl.col('i')",
            "pl.col('i') ** pl.col('d')",
            "pl.col('d') // 2",
            "pl.col('d') % 2",
            "pl.col('d') ** 2",
            # ** is the only float partner cell that errors
            "pl.col('d') ** pl.col('f')",
            "pl.col('f') ** pl.col('d')",
            # Boolean: everything except bool / Decimal
            "pl.col('d') + pl.col('b')",
            "pl.col('b') + pl.col('d')",
            "pl.col('d') - pl.col('b')",
            "pl.col('b') * pl.col('d')",
            "pl.col('d') / pl.col('b')",
            "pl.col('b') // pl.col('d')",
            "pl.col('b') % pl.col('d')",
            "pl.col('b') ** pl.col('d')",
            "pl.col('d') ** pl.col('b')",
            # string: only + concatenates
            "pl.col('d') - pl.col('s')",
            "pl.col('s') * pl.col('d')",
            "pl.col('d') / pl.col('s')",
            "pl.col('s') // pl.col('d')",
            "pl.col('d') % pl.col('s')",
            "pl.col('s') ** pl.col('d')",
            # temporals: every operator, both orders
            "pl.col('d') + pl.col('dt')",
            "pl.col('dt') - pl.col('d')",
            "pl.col('d') * pl.col('du')",
            "pl.col('du') / pl.col('d')",
            "pl.col('d') - pl.col('tm')",
            "pl.col('tm') + pl.col('d')",
            # Null literal partner only supports + - * /
            "pl.col('d') // None",
            "pl.col('d') % None",
            "pl.col('d') ** None",
        ],
    )
    def test_invalid_combination_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        # The error is the signal — the output registers as Unknown.
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('d') + pl.col('c')",
            "pl.col('c') - pl.col('d')",
            "pl.col('d') * pl.col('e')",
            "pl.col('e') / pl.col('d')",
            "pl.col('d') + pl.col('xs')",
            "pl.col('xs') % pl.col('d')",
        ],
    )
    def test_categorical_enum_list_partner_flags_ply009(self, expr: str) -> None:
        """Probed: every Decimal x {Categorical, Enum, List} cell errors."""
        frame = FrameType(
            {
                "d": Decimal(10, 2),
                "c": Categorical(),
                "e": Enum(),
                "xs": ListT(Int64()),
            }
        )
        analyzer = _run_body(frame, f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]

    def test_error_message_names_dtypes_and_op(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('d') // pl.col('d2'))")
        assert "Decimal(10, 2) // Decimal(10, 2)" in analyzer.errors[0]

    # -- nullability propagation ----------------------------------------------

    def test_nullable_decimal_plus_int_is_nullable_decimal(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nd') + pl.col('i'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Decimal(38, 2))

    def test_decimal_times_nullable_int_is_nullable_decimal(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('d') * pl.col('ni'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Decimal(38, 2))

    def test_nullable_decimal_div_float_is_nullable_float64(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nd') / pl.col('f'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Float64())

    def test_invalid_pair_detected_under_nullable_wrapper(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nd') ** pl.col('i'))")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]

    # -- silent fallback for unprobed partners ---------------------------------

    # upgrade trigger: Unknown operand becomes inferable upstream
    @pytest.mark.imprecision
    def test_unknown_partner_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('d') + pl.col('u'))")
        assert analyzer.errors == []

    # upgrade trigger: Decimal x Struct gets a probed cell in the arithmetic table
    @pytest.mark.imprecision
    def test_struct_partner_is_silent(self):
        frame = FrameType({"d": Decimal(10, 2), "st": Struct({"a": Int64()})})
        analyzer = _run_body(frame, "out = df.select(r=pl.col('d') + pl.col('st'))")
        assert analyzer.errors == []

    # upgrade trigger: unresolved call expressions gain inference
    @pytest.mark.imprecision
    def test_unresolved_partner_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('d') + helper())")
        assert analyzer.errors == []


class TestNullabilityPropagationBoolean:
    """Issue #18: elementwise boolean ops propagate operand nullability.

    polars semantics: ``null > 0`` is null, ``null & true`` is null,
    ``~null`` is null — so a Nullable operand makes the Boolean result
    Nullable. Non-nullable operands keep the bare Boolean result.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "a": Int64(),
                "n": Nullable(Int64()),
                "flag": Boolean(),
                "nflag": Nullable(Boolean()),
            }
        )

    # -- comparisons ---------------------------------------------------------

    def test_compare_non_nullable_operands_is_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') > 0)")
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_compare_nullable_left_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('n') > 0)")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_compare_nullable_comparator_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('a') == pl.col('n'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    # -- bitwise logical & | ^ -----------------------------------------------

    def test_and_of_non_nullable_predicates_is_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=(pl.col('a') > 0) & pl.col('flag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_and_with_nullable_operand_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('flag') & pl.col('nflag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_or_with_nullable_predicate_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=(pl.col('n') > 0) | pl.col('flag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_xor_with_nullable_operand_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nflag') ^ pl.col('flag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    # -- unary ~ / not ---------------------------------------------------------

    def test_invert_non_nullable_is_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=~pl.col('flag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_invert_nullable_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=~pl.col('nflag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_not_nullable_is_nullable_boolean(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=not pl.col('nflag'))")
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_invert_is_null_predicate_stays_non_nullable(self):
        """is_null() on a nullable column is non-nullable Boolean; so is ~it."""
        analyzer = _run_body(self._frame(), "out = df.select(r=~pl.col('n').is_null())")
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()


class TestSemiAntiJoin:
    """#15: semi/anti joins return the left frame's schema unchanged."""

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_semi_anti_join_keeps_left_schema(self, how: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class A(pa.DataFrameModel):
                id: int
                v: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                w: str

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[A]:
                return a.join(b, on="id", how="{how}")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft == FrameType({"id": Int64(), "v": Float64()})
        assert ft is not None and "w" not in ft.columns

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_how_via_module_constant(self, how: str):
        """`how` is constant-aware: a name bound to 'semi'/'anti' resolves."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            HOW = "{how}"

            class A(pa.DataFrameModel):
                id: int
                v: pl.Float64

            class B(pa.DataFrameModel):
                id: int
                w: str

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[A]:
                return a.join(b, on="id", how=HOW)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"id": Int64(), "v": Float64()})

    def test_semi_join_lazy_receiver_stays_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import LazyFrame

            class A(pa.DataFrameModel):
                id: int
                v: pl.Float64

            class B(pa.DataFrameModel):
                id: int

            def f(a: LazyFrame[A], b: LazyFrame[B]) -> LazyFrame[A]:
                return a.join(b, on="id", how="semi")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.is_lazy is True

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_missing_key_still_errors(self, how: str):
        """Key validation still fires: PLY010 on a missing join key."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                id: int

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[A]:
                return a.join(b, on="missing", how="{how}")
        """
        )
        results = analyze_source(source)
        assert any("PLY010" in e for e in results[0].errors), results[0].errors

    def test_multi_key_semi_join(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                x: int
                y: str
                a: pl.Float64

            class B(pa.DataFrameModel):
                x: int
                y: str

            def f(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[A]:
                return a.join(b, on=["x", "y"], how="semi")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType(
            {"x": Int64(), "y": Utf8(), "a": Float64()}
        )


class TestJoinCoalesceKwarg:
    """#24: the coalesce= kwarg of .join() is parsed and applied."""

    def test_full_join_coalesce_true_key_non_nullable(self):
        """coalesce=True on a full join: single key column, non-nullable."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                x: int

            class B(pa.DataFrameModel):
                id: int
                y: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="id", how="full", coalesce=True)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["x"].dtype == Nullable(Int64())
        assert ft.columns["y"].dtype == Nullable(Int64())
        assert "id_right" not in ft.columns

    def test_full_join_default_keeps_both_keys(self):
        """Without coalesce=, a full join keeps id (nullable) and id_right."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                x: int

            class B(pa.DataFrameModel):
                id: int
                y: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="id", how="full")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Nullable(Int64())
        assert ft.columns["id_right"].dtype == Nullable(Int64())

    def test_left_join_coalesce_false_keeps_right_key(self):
        """coalesce=False on a left join keeps id_right (nullable)."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                x: int

            class B(pa.DataFrameModel):
                id: int
                y: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="id", how="left", coalesce=False)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["id"].dtype == Int64()
        assert ft.columns["id_right"].dtype == Nullable(Int64())

    def test_non_literal_coalesce_falls_back_to_default(self):
        """A non-literal coalesce= is ignored: the how-specific default applies."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                x: int

            class B(pa.DataFrameModel):
                id: int
                y: int

            def f(a: DataFrame[A], b: DataFrame[B], flag: bool):
                return a.join(b, on="id", how="full", coalesce=flag)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        # full-join default: uncoalesced, both keys kept and nullable
        assert ft.columns["id"].dtype == Nullable(Int64())
        assert ft.columns["id_right"].dtype == Nullable(Int64())

    def test_join_asof_left_on_right_on_keeps_both_keys(self):
        """join_asof never coalesces differently-named keys (polars semantics)."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class L(pa.DataFrameModel):
                t: pl.Datetime
                x: int

            class R(pa.DataFrameModel):
                ts: pl.Datetime
                y: int

            def f(left: DataFrame[L], right: DataFrame[R]):
                return left.join_asof(right, left_on="t", right_on="ts")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "t" in ft.columns
        assert "ts" in ft.columns
        assert ft.columns["y"].dtype == Nullable(Int64())


class TestCrossJoinAnalyzer:
    """#26: how='cross' is inferred (no join keys required)."""

    def test_cross_join_infers_merged_schema(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                x: int

            class B(pa.DataFrameModel):
                rid: int
                y: str

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, how="cross")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft == FrameType({"id": Int64(), "x": Int64(), "rid": Int64(), "y": Utf8()})

    def test_cross_join_collision_gets_suffix(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int
                v: int

            class B(pa.DataFrameModel):
                v: str

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, how="cross")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft == FrameType({"id": Int64(), "v": Int64(), "v_right": Utf8()})

    def test_cross_join_how_via_module_constant(self):
        """how= stays constant-aware for 'cross'."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            HOW = "cross"

            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                rid: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, how=HOW)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"id": Int64(), "rid": Int64()})

    def test_cross_join_with_keys_errors(self):
        """Providing on= with how='cross' raises PLY010 like polars raises."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                id: int

            def f(a: DataFrame[A], b: DataFrame[B]):
                return a.join(b, on="id", how="cross")
        """
        )
        results = analyze_source(source)
        assert any(
            "PLY010" in e and "cross join takes no join keys" in e for e in results[0].errors
        ), results[0].errors

    def test_cross_join_lazy_receiver_stays_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import LazyFrame

            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                rid: int

            def f(a: LazyFrame[A], b: LazyFrame[B]):
                return a.join(b, how="cross")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.is_lazy is True


class TestGatherEvery:
    """#15: gather_every(n) is schema-preserving on both frame kinds."""

    def test_gather_every_eager(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(data: DataFrame[S]) -> DataFrame[S]:
                return data.gather_every(2)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft == FrameType({"id": Int64(), "value": Float64()})
        assert ft is not None and ft.is_lazy is False

    def test_gather_every_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import LazyFrame

            class S(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def f(data: LazyFrame[S]) -> LazyFrame[S]:
                return data.gather_every(2, offset=1)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft == FrameType({"id": Int64(), "value": Float64()})
        assert ft is not None and ft.is_lazy is True


class TestSeqVariants:
    """#21: select_seq / with_columns_seq behave like select / with_columns."""

    def test_select_seq_string_args(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                a: int
                b: str
                c: pl.Float64

            def f(df: DataFrame[S]):
                return df.select_seq("a", "b")
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_select_seq_expr_args(self):
        frame = FrameType({"a": Int64(), "b": Int64()})
        analyzer = _run_body(frame, 'out = df.select_seq(pl.col("a"), pl.col("b"))')
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "b": Int64()})

    def test_with_columns_seq_kwarg_expr(self):
        frame = FrameType({"a": Int64(), "b": Int64()})
        analyzer = _run_body(frame, 'out = df.with_columns_seq(c=pl.col("a") + pl.col("b"))')
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "b": Int64(), "c": Int64()})

    def test_seq_chain(self):
        frame = FrameType({"a": Int64(), "b": Int64()})
        analyzer = _run_body(
            frame,
            'out = df.with_columns_seq(c=pl.col("a") + pl.col("b")).select_seq("a", "c")',
        )
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "c": Int64()})

    def test_seq_variants_fire_no_eager_lazy_errors(self):
        """Both methods exist on DataFrame AND LazyFrame — no PLY030/PLY031."""
        eager = FrameType({"a": Int64()})
        lazy = FrameType({"a": Int64()}, is_lazy=True)
        for frame in (eager, lazy):
            analyzer = _run_body(frame, 'out = df.select_seq("a")')
            assert analyzer.errors == []
            analyzer = _run_body(frame, "out = df.with_columns_seq(b=pl.col('a'))")
            assert analyzer.errors == []

    def test_select_seq_preserves_laziness(self):
        frame = FrameType({"a": Int64()}, is_lazy=True)
        analyzer = _run_body(frame, 'out = df.select_seq("a")')
        assert analyzer.var_types["out"].is_lazy is True

    def test_select_seq_missing_column_errors(self):
        frame = FrameType({"a": Int64()})
        analyzer = _run_body(frame, 'out = df.select_seq("ghost")')
        assert any("ghost" in e for e in analyzer.errors)


class TestPlAllExcludeSelectors:
    """#20: pl.all() / pl.exclude(...) resolve like cs.* selectors."""

    @staticmethod
    def _resolve(expr: str, frame: FrameType):
        import ast

        from polypolarism.analyzer import _resolve_selector

        return _resolve_selector(ast.parse(expr, mode="eval").body, frame)

    def _frame(self) -> FrameType:
        return FrameType({"a": Int64(), "b": Int64(), "name": Utf8()})

    def test_pl_all_no_args_resolves_to_all_columns(self):
        assert self._resolve("pl.all()", self._frame()) == ["a", "b", "name"]

    def test_pl_all_with_arg_is_not_a_selector(self):
        # ``pl.all("x")`` is the boolean "all values truthy" aggregation,
        # not a selector — it must fall through to expression analysis.
        assert self._resolve('pl.all("a")', self._frame()) is None

    def test_pl_exclude_string_arg(self):
        assert self._resolve('pl.exclude("name")', self._frame()) == ["a", "b"]

    def test_pl_exclude_varargs(self):
        assert self._resolve('pl.exclude("a", "name")', self._frame()) == ["b"]

    def test_pl_exclude_list_arg(self):
        assert self._resolve('pl.exclude(["a", "b"])', self._frame()) == ["name"]

    def test_pl_exclude_non_literal_arg_is_not_resolved(self):
        assert self._resolve("pl.exclude(some_var)", self._frame()) is None

    # upgrade trigger: selector expansion becomes row-var-aware on open frames
    @pytest.mark.imprecision
    def test_pl_all_on_open_frame_returns_known_columns(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        assert self._resolve("pl.all()", frame) == ["id"]

    def test_select_pl_all(self):
        frame = FrameType({"a": Int64(), "name": Utf8()})
        analyzer = _run_body(frame, "out = df.select(pl.all())")
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "name": Utf8()})

    def test_select_pl_exclude(self):
        frame = FrameType({"a": Int64(), "b": Int64(), "name": Utf8()})
        analyzer = _run_body(frame, "out = df.select(pl.exclude('name'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "b": Int64()})

    def test_with_columns_pl_exclude_keeps_schema(self):
        frame = FrameType({"a": Int64(), "name": Utf8()})
        analyzer = _run_body(frame, "out = df.with_columns(pl.exclude('name'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"a": Int64(), "name": Utf8()})

    def test_drop_pl_exclude(self):
        frame = FrameType({"id": Int64(), "a": Int64(), "b": Int64()})
        analyzer = _run_body(frame, "out = df.drop(pl.exclude('id'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"] == FrameType({"id": Int64()})


class TestSelectConstantResolution:
    """#22: constant-bound column names resolve in select / with_columns."""

    def test_select_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "a"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.select(KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"a": Int64()})

    def test_select_module_constant_list(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            COLS = ["a", "b"]

            class S(pa.DataFrameModel):
                a: int
                b: str
                c: pl.Float64

            def f(df: DataFrame[S]):
                return df.select(COLS)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_select_local_constant_shadows_module_constant(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "ghost"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                KEY = "b"
                return df.select(KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"b": Utf8()})

    def test_select_kwarg_constant_is_renamed_column_reference(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "a"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.select(x=KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"x": Int64()})

    def test_select_missing_column_via_constant_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "ghost"

            class S(pa.DataFrameModel):
                a: int

            def f(df: DataFrame[S]):
                return df.select(KEY)
        """
        )
        results = analyze_source(source)
        assert any("PLY042" in str(e) and "ghost" in str(e) for e in results[0].errors)

    def test_select_unknown_name_still_falls_through(self):
        """A bare Name that is NOT a constant (e.g. a frame variable) keeps
        going to expression analysis — no false PLY042."""
        frame = FrameType({"a": Int64()})
        analyzer = _run_body(frame, "out = df.select(mystery)")
        assert analyzer.errors == []

    def test_with_columns_constant_keeps_schema(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "a"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.with_columns(KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_with_columns_missing_constant_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "ghost"

            class S(pa.DataFrameModel):
                a: int

            def f(df: DataFrame[S]):
                return df.with_columns(KEY)
        """
        )
        results = analyze_source(source)
        assert any("PLY042" in str(e) and "ghost" in str(e) for e in results[0].errors)

    def test_with_columns_kwarg_constant_is_column_reference(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "a"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.with_columns(x=KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["x"].dtype == Int64()
        assert ft.columns["a"].dtype == Int64()

    def test_select_seq_constant_cross_feature(self):
        """#21 x #22: the seq variant resolves constants too."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            KEY = "a"

            class S(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[S]):
                return df.select_seq(KEY)
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        assert results[0].inferred_return_type == FrameType({"a": Int64()})

    def test_real_world_recon_pattern(self):
        """Issue #22's surfacing pattern: a renamed-select via a constant
        feeding a full join keyed on the same constant."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            RECON_KEY = "txn_id"

            class GL(pa.DataFrameModel):
                txn_id: int
                amount: pl.Float64

            class Sub(pa.DataFrameModel):
                txn_id: int
                sub_amount: pl.Float64

            def recon(gl: DataFrame[GL], sub: DataFrame[Sub]):
                return gl.select(RECON_KEY, gl_amount=pl.col("amount")).join(
                    sub, on=RECON_KEY, how="full"
                )
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert "txn_id" in ft.columns
        assert "gl_amount" in ft.columns
        assert "sub_amount" in ft.columns


class TestImplicitListAggregation:
    """Issue #27: a bare column reference in ``agg`` (no reducer) collects
    each group's values into a list — ``List(dtype)``, not the element dtype.

    Ground truth: ``pl.DataFrame({"k": ["a", "a"], "v": [1, 2]})
    .group_by("k").agg(vs=pl.col("v")).schema`` →
    ``{'k': String, 'vs': List(Int64)}``.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                k: str
                v: int
        """
    )

    def test_bare_col_kwarg_form(self):
        """``agg(vs=pl.col("v"))`` infers vs: List(Int64)."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg(vs=pl.col("v"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"k": Utf8(), "vs": ListT(Int64())})

    def test_bare_col_positional_default_name(self):
        """``agg(pl.col("v"))`` keeps the source column name."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg(pl.col("v"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"k": Utf8(), "v": ListT(Int64())})

    def test_bare_col_alias_form(self):
        """``agg(pl.col("v").alias("vs"))`` renames the list column."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg(pl.col("v").alias("vs"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"k": Utf8(), "vs": ListT(Int64())})

    def test_bare_string_positional(self):
        """``agg("v")`` — polars parses the string as a column reference."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg("v")
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"k": Utf8(), "v": ListT(Int64())})

    def test_bare_string_kwarg(self):
        """``agg(vs="v")`` — string column reference renamed to the kwarg."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg(vs="v")
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"k": Utf8(), "vs": ListT(Int64())})

    def test_nullable_element_dtype_is_preserved(self):
        """Implicit list over a nullable column keeps element nullability."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class NIn(pa.DataFrameModel):
                k: str
                v: int = pa.Field(nullable=True)

            def f(df: DataFrame[NIn]):
                return df.group_by("k").agg(vs=pl.col("v"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType(
            {"k": Utf8(), "vs": ListT(Nullable(Int64()))}
        )

    def test_missing_column_raises_ply011(self):
        """A bare reference to a missing column still surfaces PLY011."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.group_by("k").agg(vs=pl.col("missing"))
            """
        )
        results = analyze_source(source)
        assert any("PLY011" in e and "missing" in e for e in results[0].errors)


class TestFrameLiteralInference:
    """Issue #25: infer the schema of ``pl.DataFrame({...})`` /
    ``pl.LazyFrame({...})`` literal constructors from the dict literal
    (names from keys, dtypes from values / explicit ``schema=``)."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                x: int
        """
    )

    def _infer(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        return results[0].inferred_return_type

    def test_int_list_literal(self):
        ft = self._infer('pl.DataFrame({"a": [1, 2, 3]})')
        assert ft is not None
        assert ft == FrameType({"a": Int64()})
        assert ft.is_lazy is False

    def test_multiple_columns(self):
        ft = self._infer('pl.DataFrame({"a": [1], "b": ["x"], "c": [1.5], "d": [True]})')
        assert ft == FrameType({"a": Int64(), "b": Utf8(), "c": Float64(), "d": Boolean()})

    def test_list_with_none_is_nullable(self):
        ft = self._infer('pl.DataFrame({"a": [1, None]})')
        assert ft == FrameType({"a": Nullable(Int64())})

    def test_all_none_list_is_null(self):
        ft = self._infer('pl.DataFrame({"a": [None, None]})')
        assert ft == FrameType({"a": Null()})

    def test_empty_list_is_null(self):
        # polars: pl.DataFrame({"a": []}).schema == {a: Null} (issue #101)
        ft = self._infer('pl.DataFrame({"a": []})')
        assert ft == FrameType({"a": Null()})

    def test_mixed_numeric_list_unifies(self):
        ft = self._infer('pl.DataFrame({"a": [1, 2.5]})')
        assert ft == FrameType({"a": Float64()})

    def test_non_unifiable_list_is_unknown(self):
        ft = self._infer('pl.DataFrame({"a": [1, "x"]})')
        assert ft == FrameType({"a": Unknown()})

    def test_non_constant_element_is_unknown(self):
        ft = self._infer('pl.DataFrame({"a": [some_value, 2]})')
        assert ft == FrameType({"a": Unknown()})

    def test_scalar_broadcast(self):
        ft = self._infer('pl.DataFrame({"a": [1, 2], "b": "x"})')
        assert ft == FrameType({"a": Int64(), "b": Utf8()})

    def test_range_constructors(self):
        ft = self._infer(
            "pl.DataFrame({"
            '"d": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True),'
            '"ts": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2), eager=True),'
            '"t": pl.time_range(eager=True),'
            '"i": pl.int_range(5, eager=True),'
            "})"
        )
        assert ft == FrameType({"d": Date(), "ts": Datetime(), "t": Time(), "i": Int64()})

    def test_schema_dict_wins_over_data(self):
        ft = self._infer('pl.DataFrame({"a": [1]}, schema={"a": pl.Int32})')
        assert ft == FrameType({"a": Int32()})

    def test_schema_dict_defines_names(self):
        ft = self._infer('pl.DataFrame({"a": [1]}, schema={"x": pl.Int32})')
        assert ft == FrameType({"x": Int32()})

    def test_schema_python_builtin_dtypes(self):
        ft = self._infer('pl.DataFrame(schema={"a": int, "b": float, "c": str, "d": bool})')
        assert ft == FrameType({"a": Int64(), "b": Float64(), "c": Utf8(), "d": Boolean()})

    def test_schema_wins_over_opaque_data(self):
        ft = self._infer('pl.DataFrame(some_rows, schema={"a": pl.Int32})')
        assert ft == FrameType({"a": Int32()})

    def test_schema_list_of_names(self):
        ft = self._infer('pl.DataFrame(some_rows, schema=["a", "b"])')
        assert ft == FrameType({"a": Unknown(), "b": Unknown()})

    def test_schema_unresolvable_dtype_is_unknown(self):
        ft = self._infer('pl.DataFrame(schema={"a": SOME_DTYPE})')
        assert ft == FrameType({"a": Unknown()})

    def test_schema_overrides_patch_data_dtypes(self):
        ft = self._infer('pl.DataFrame({"a": [1], "b": [2]}, schema_overrides={"a": pl.Int8})')
        assert ft == FrameType({"a": Int8(), "b": Int64()})

    def test_opaque_data_without_schema_is_open(self):
        # ADR-0006 amendment: ``pl.DataFrame(some_var)`` provably builds
        # SOME frame — an open one, not an untracked None.
        ft = self._infer("pl.DataFrame(some_rows)")
        assert ft is not None
        assert ft.columns == {} and ft.rest is not None

    def test_lazyframe_literal_is_lazy(self):
        ft = self._infer('pl.LazyFrame({"a": [1]})')
        assert ft is not None
        assert ft == FrameType({"a": Int64()})
        assert ft.is_lazy is True

    def test_literal_frame_is_closed_and_non_strict(self):
        ft = self._infer('pl.DataFrame({"a": [1]})')
        assert ft is not None
        assert ft.rest is None
        assert ft.strict is False

    def test_method_chain_on_literal(self):
        ft = self._infer('pl.DataFrame({"a": [1, 2]}).with_columns(b=pl.col("a") * 2)')
        assert ft == FrameType({"a": Int64(), "b": Int64()})

    def test_lazy_literal_collect_is_eager(self):
        ft = self._infer('pl.LazyFrame({"a": [1]}).collect()')
        assert ft is not None
        assert ft == FrameType({"a": Int64()})
        assert ft.is_lazy is False

    def test_select_missing_column_on_literal_errors(self):
        """The literal frame is closed — a missing column is PLY001."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return pl.DataFrame({"a": [1]}).select(pl.col("nope"))
            """
        )
        results = analyze_source(source)
        assert any("PLY001" in e and "nope" in e for e in results[0].errors)


class TestFilterPredicateDtype:
    """Issue #28: ``df.filter(...)`` with a non-boolean predicate is PLY008."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                flag: bool
                v: pl.Float64 = pa.Field(nullable=True)
        """
    )

    def _analyze(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        return analyze_source(source)

    def test_nonbool_column_expr_predicate_flags_ply008(self):
        results = self._analyze('df.filter(pl.col("a"))')
        assert any("PLY008" in e and "Boolean" in e for e in results[0].errors)

    def test_nonbool_bare_string_predicate_flags_ply008(self):
        results = self._analyze('df.filter("a")')
        assert any("PLY008" in e for e in results[0].errors)

    def test_nonbool_const_name_predicate_flags_ply008(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                key = "a"
                return df.filter(key)
            """
        )
        results = analyze_source(source)
        assert any("PLY008" in e for e in results[0].errors)

    def test_bare_string_missing_column_is_ply001_not_ply008(self):
        results = self._analyze('df.filter("ghost")')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)
        assert not any("PLY008" in e for e in results[0].errors)

    def test_boolean_column_expr_predicate_passes(self):
        results = self._analyze('df.filter(pl.col("flag"))')
        assert results[0].errors == []

    def test_boolean_bare_string_predicate_passes(self):
        results = self._analyze('df.filter("flag")')
        assert results[0].errors == []

    def test_comparison_predicate_passes(self):
        results = self._analyze('df.filter(pl.col("a") > 0)')
        assert results[0].errors == []

    def test_nullable_comparison_predicate_passes(self):
        # Since #18 a comparison over a nullable column infers
        # Nullable(Boolean) — the wrapper must be unwrapped, not flagged.
        results = self._analyze('df.filter(pl.col("v") > 0)')
        assert results[0].errors == []

    def test_combined_predicates_pass(self):
        results = self._analyze('df.filter((pl.col("a") > 0) & pl.col("flag"))')
        assert results[0].errors == []

    def test_is_null_predicate_passes(self):
        results = self._analyze('df.filter(pl.col("v").is_null())')
        assert results[0].errors == []

    def test_unknown_dtype_predicate_not_flagged(self):
        # ``interpolate`` is not in the inference tables — the column is
        # registered as Unknown; an Unknown predicate must never be flagged.
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                df2 = df.with_columns(u=pl.col("a").interpolate())
                return df2.filter(pl.col("u"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_open_frame_bare_string_predicate_not_flagged(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.filter("ghost")')
        assert analyzer.errors == []

    def test_each_positional_predicate_is_checked(self):
        results = self._analyze('df.filter(pl.col("flag"), pl.col("a"))')
        ply008 = [e for e in results[0].errors if "PLY008" in e]
        assert len(ply008) == 1

    def test_kwarg_equality_constraint_not_flagged(self):
        # ``filter(a=1)`` is an equality constraint — boolean by construction.
        results = self._analyze("df.filter(a=1)")
        assert not any("PLY008" in e for e in results[0].errors)

    def test_filter_stays_identity_typed(self):
        results = self._analyze('df.filter(pl.col("flag"))')
        assert results[0].inferred_return_type == FrameType(
            {"a": Int64(), "flag": Boolean(), "v": Nullable(Float64())}
        )


class TestExprFilterPredicateDtype:
    """Issue #28: ``Expr.filter(...)`` predicates get the same PLY008 check."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                g: str
                a: int
                flag: bool
                v: pl.Float64 = pa.Field(nullable=True)
        """
    )

    def _analyze(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        return analyze_source(source)

    def test_nonbool_predicate_in_agg_flags_ply008(self):
        results = self._analyze(
            'df.group_by("g").agg(pl.col("v").filter(pl.col("a")).sum().alias("s"))'
        )
        assert any("PLY008" in e for e in results[0].errors)

    def test_nonbool_bare_string_predicate_flags_ply008(self):
        results = self._analyze('df.select(pl.col("v").filter("a").alias("x"))')
        assert any("PLY008" in e for e in results[0].errors)

    def test_boolean_predicate_passes(self):
        results = self._analyze('df.select(pl.col("v").filter(pl.col("flag")).alias("x"))')
        assert results[0].errors == []

    def test_boolean_bare_string_predicate_passes(self):
        results = self._analyze('df.select(pl.col("v").filter("flag").alias("x"))')
        assert results[0].errors == []

    def test_nullable_comparison_predicate_passes(self):
        results = self._analyze('df.select(pl.col("a").filter(pl.col("v") > 0).alias("x"))')
        assert results[0].errors == []

    def test_missing_column_string_predicate_is_ply001_not_ply008(self):
        results = self._analyze('df.select(pl.col("v").filter("ghost").alias("x"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)
        assert not any("PLY008" in e for e in results[0].errors)


class TestSortKeyValidation:
    """Issue #29: ``sort`` validates key columns like drop/rename do (PLY007)."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                b: str
        """
    )

    def _analyze(self, body: str, extra_imports: str = ""):
        source = (
            extra_imports
            + self.HEADER
            + textwrap.dedent(
                f"""
            def f(df: DataFrame[In]):
                return {body}
            """
            )
        )
        return analyze_source(source)

    def test_missing_string_key_flags_ply007(self):
        results = self._analyze('df.sort("ghost")')
        assert any("PLY007" in e and "ghost" in e for e in results[0].errors)

    def test_existing_string_key_passes(self):
        results = self._analyze('df.sort("a")')
        assert results[0].errors == []

    def test_varargs_keys_each_checked(self):
        results = self._analyze('df.sort("a", "ghost")')
        assert any("PLY007" in e and "ghost" in e for e in results[0].errors)

    def test_list_keys_each_checked(self):
        results = self._analyze('df.sort(["a", "ghost"])')
        assert any("PLY007" in e and "ghost" in e for e in results[0].errors)

    def test_by_kwarg_missing_key_flags_ply007(self):
        results = self._analyze('df.sort(by="ghost")')
        assert any("PLY007" in e and "ghost" in e for e in results[0].errors)

    def test_by_kwarg_list_passes(self):
        results = self._analyze('df.sort(by=["a", "b"])')
        assert results[0].errors == []

    def test_const_name_key_resolves(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                key = "ghost"
                return df.sort(key)
            """
        )
        results = analyze_source(source)
        assert any("PLY007" in e and "ghost" in e for e in results[0].errors)

    def test_pl_col_missing_key_flags_ply001(self):
        results = self._analyze('df.sort(pl.col("ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_pl_col_existing_key_passes(self):
        results = self._analyze('df.sort(pl.col("a"))')
        assert results[0].errors == []

    def test_selector_keys_pass(self):
        results = self._analyze(
            "df.sort(cs.numeric())", extra_imports="import polars.selectors as cs\n"
        )
        assert results[0].errors == []

    def test_modifier_kwargs_are_ignored(self):
        results = self._analyze('df.sort("a", descending=True, nulls_last=True)')
        assert results[0].errors == []

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_open_frame_missing_key_not_flagged(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.sort("ghost")')
        assert analyzer.errors == []

    def test_sort_stays_identity_typed(self):
        results = self._analyze('df.sort("a")')
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_lazy_sort_preserves_laziness(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from pandera.typing.polars import LazyFrame

            class In(pa.DataFrameModel):
                a: int
                b: str

            def f(lf: LazyFrame[In]) -> DataFrame[In]:
                return lf.sort("a").collect()
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})


class TestNamespaceReceiverDtype:
    """Issue #31: namespace accessors validate the receiver dtype (PLY012).

    Verified against polars 1.41.2: ``.str`` rejects every non-String
    receiver at runtime (Categorical/Enum included: "expected String type,
    got: cat"), ``.dt`` accepts Date/Datetime/Time/Duration, ``.list``
    requires List, ``.arr`` requires Array (issue #53 — the containers are
    not interchangeable), ``.struct`` requires Struct, and ``.cat``
    requires Categorical or Enum (issue #54).
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            from typing import Annotated

            class In(pa.DataFrameModel):
                i: int
                s: str
                ts: pl.Datetime
                d: pl.Date
                t: pl.Time
                dur: pl.Duration
                xs: Annotated[pl.List, pl.Int64()]
                q: pl.Array(pl.Int64, 3) = pa.Field()
                cat: pl.Categorical
                en: pl.Enum
                st: pl.Struct
        """
    )

    def _analyze(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(({expr}).alias("out"))
            """
        )
        return analyze_source(source)

    @pytest.mark.parametrize(
        "expr",
        [
            'pl.col("i").str.contains("x")',
            'pl.col("ts").str.contains("x")',
            # .str on Categorical/Enum raises at runtime ("expected String
            # type, got: cat/enum") — flagged too.
            'pl.col("cat").str.to_uppercase()',
            'pl.col("en").str.len_chars()',
            'pl.col("i").dt.year()',
            'pl.col("s").dt.year()',
            'pl.col("xs").dt.year()',
            'pl.col("i").list.sum()',
            'pl.col("s").list.len()',
            'pl.col("i").arr.sum()',
            # Issue #53: Array and List are NOT interchangeable — probed:
            # `.arr` on a List column raises "expected Array datatype" and
            # `.list` on an Array column raises "expected List data type".
            'pl.col("xs").arr.sum()',
            'pl.col("q").list.sum()',
            'pl.col("q").list.eval(pl.element() * 2)',
            'pl.col("i").struct.field("a")',
            # Issue #54: `.cat` requires Categorical or Enum — probed:
            # "SchemaError: expected an Enum or Categorical type".
            'pl.col("i").cat.get_categories()',
            'pl.col("s").cat.get_categories()',
            'pl.col("xs").cat.get_categories()',
            'pl.col("d").cat.len_chars()',
            # Backlog C-9: a bare ``pl.Struct`` column is provably a
            # struct (probed: pandera validates any struct; `.str` on a
            # struct is a runtime SchemaError) — wrong-namespace
            # accessors are now proofs, not Unknown-mediated silence.
            'pl.col("st").str.contains("x")',
            'pl.col("st").cat.get_categories()',
            'pl.col("st").arr.sum()',
        ],
    )
    def test_wrong_receiver_dtype_flags_ply012(self, expr: str):
        results = self._analyze(expr)
        assert any("PLY012" in e for e in results[0].errors), (expr, results[0].errors)

    @pytest.mark.parametrize(
        "expr",
        [
            'pl.col("s").str.contains("x")',
            'pl.col("ts").dt.year()',
            'pl.col("d").dt.year()',
            'pl.col("t").dt.hour()',
            'pl.col("dur").dt.total_seconds()',
            'pl.col("xs").list.sum()',
            # Issue #53: `.arr` on a real Array column is valid.
            'pl.col("q").arr.sum()',
            'pl.col("q").arr.len()',
            # Issue #54: `.cat` on Categorical / Enum is valid.
            'pl.col("cat").cat.get_categories()',
            'pl.col("en").cat.get_categories()',
            # Bare ``pl.Struct`` is an OPEN struct (backlog C-9): the
            # struct-ness is provable, so `.struct` is accepted and field
            # lookups get assumption semantics.
            'pl.col("st").struct.field("x")',
        ],
    )
    def test_valid_or_unknown_receiver_passes(self, expr: str):
        results = self._analyze(expr)
        assert results[0].errors == [], (expr, results[0].errors)

    def test_single_error_and_output_degrades_to_unknown(self):
        results = self._analyze('pl.col("i").str.contains("x")')
        assert len(results[0].errors) == 1, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Unknown()

    def test_message_names_namespace_column_and_dtype(self):
        results = self._analyze('pl.col("i").str.contains("x")')
        err = results[0].errors[0]
        assert "PLY012" in err
        assert ".str" in err
        assert "'i'" in err
        assert "Int64" in err

    def test_nullable_valid_receiver_passes(self):
        """Nullable[Utf8] is still a String column — `.str` is fine."""
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                s: str = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("s").str.contains("x").alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_nullable_wrong_receiver_flags_ply012(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                i: int = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("i").str.contains("x").alias("out"))
            """
        )
        results = analyze_source(source)
        assert any("PLY012" in e for e in results[0].errors), results[0].errors

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_open_frame_unknown_receiver_not_flagged(self):
        """A column missing on an open frame resolves to Unknown — exempt."""
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(
            frame,
            'out = df.select(pl.col("extra").str.contains("x").alias("m"))',
        )
        assert analyzer.errors == []

    # upgrade trigger: unrecognised namespace methods gain a table entry or a warning
    @pytest.mark.imprecision
    def test_unknown_method_on_valid_receiver_still_falls_through(self):
        """An unrecognised namespace method on a valid receiver stays silent
        and registers the output as Unknown (pre-existing behaviour)."""
        results = self._analyze('pl.col("s").str.some_future_method()')
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Unknown()

    def test_chained_receiver_dtype_is_validated(self):
        """`.dt` after an aggregation chain sees the chain's result dtype."""
        results = self._analyze('pl.col("i").max().dt.year()')
        assert any("PLY012" in e for e in results[0].errors), results[0].errors

    def test_chained_receiver_valid_dtype_passes(self):
        results = self._analyze('pl.col("ts").max().dt.year()')
        assert results[0].errors == [], results[0].errors

    def test_arr_on_list_message_names_array_requirement(self):
        results = self._analyze('pl.col("xs").arr.sum()')
        err = results[0].errors[0]
        assert "PLY012" in err
        assert "an Array column" in err
        assert "List[Int64]" in err

    def test_list_on_array_message_names_list_requirement(self):
        results = self._analyze('pl.col("q").list.sum()')
        err = results[0].errors[0]
        assert "PLY012" in err
        assert "a List column" in err
        assert "Array[Int64, 3]" in err

    def test_cat_message_names_schema_error(self):
        # Probed: `.cat` on a wrong dtype raises SchemaError, not
        # InvalidOperationError — the message says so.
        results = self._analyze('pl.col("i").cat.get_categories()')
        err = results[0].errors[0]
        assert "PLY012" in err
        assert "a Categorical or Enum column" in err
        assert "SchemaError" in err

    def test_nullable_cat_receiver_passes(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                c: pl.Categorical = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("c").cat.len_chars().alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_nullable_array_receiver_passes_arr(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                q: pl.Array(pl.Int64, 3) = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("q").arr.sum().alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_nullable_list_receiver_flags_arr(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                xs: pl.List(pl.Int64) = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("xs").arr.sum().alias("out"))
            """
        )
        results = analyze_source(source)
        assert any("PLY012" in e for e in results[0].errors), results[0].errors


class TestArrNamespaceReturns:
    """Issue #53: arr-namespace result dtypes on Array receivers.

    Probed on polars 1.41.2 (see compat.polars_api ARR_NAMESPACE_* /
    container_agg_return): element returns, List-returning de-array
    methods, Array-preserving methods, fixed UInt32/Boolean returns and
    the float-aggregation cells.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                q: pl.Array(pl.Int64, 3) = pa.Field()
                f: pl.Array(pl.Float32, 2) = pa.Field()
                w: pl.Array(pl.Int16, 2) = pa.Field()
        """
    )

    def _dtype_of(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(({expr}).alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], (expr, results[0].errors)
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft.columns["out"].dtype

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # Element returns
            ('pl.col("q").arr.sum()', Int64()),
            ('pl.col("q").arr.min()', Int64()),
            ('pl.col("q").arr.max()', Int64()),
            ('pl.col("q").arr.first()', Int64()),
            ('pl.col("q").arr.last()', Int64()),
            ('pl.col("q").arr.get(0)', Int64()),
            ('pl.col("q").arr.explode()', Int64()),
            # Fixed returns
            ('pl.col("q").arr.len()', UInt32()),
            ('pl.col("q").arr.n_unique()', UInt32()),
            ('pl.col("q").arr.arg_min()', UInt32()),
            ('pl.col("q").arr.count_matches(1)', UInt32()),
            ('pl.col("q").arr.contains(1)', Boolean()),
            ('pl.col("q").arr.any()', Boolean()),
            # De-array into List (probed: the fixed width is lost)
            ('pl.col("q").arr.unique()', ListT(Int64())),
            ('pl.col("q").arr.head(2)', ListT(Int64())),
            ('pl.col("q").arr.to_list()', ListT(Int64())),
            # Array-preserving (width kept — backlog C-7)
            ('pl.col("q").arr.sort()', Array(Int64(), 3)),
            ('pl.col("q").arr.reverse()', Array(Int64(), 3)),
            ('pl.col("q").arr.shift()', Array(Int64(), 3)),
            # Float aggregations: Int64 -> Float64, Float32 keeps Float32
            ('pl.col("q").arr.mean()', Float64()),
            ('pl.col("q").arr.median()', Float64()),
            ('pl.col("q").arr.std()', Float64()),
            ('pl.col("q").arr.var()', Float64()),
            ('pl.col("f").arr.mean()', Float32()),
            ('pl.col("f").arr.sum()', Float32()),
            # sum widens narrow ints to Int64 (probed overflow guard)
            ('pl.col("w").arr.sum()', Int64()),
            # arr.eval keeps the Array container (and width — C-7)
            ('pl.col("q").arr.eval(pl.element() * 2)', Array(Int64(), 3)),
            ('pl.col("q").arr.eval(pl.element().cast(pl.Utf8))', Array(Utf8(), 3)),
            # as_list=True de-arrays into List around the body dtype —
            # probed (polars 1.41.2; the arg landed in 1.41, issue #53):
            # List(body dtype) for dtype-changing, aggregating AND
            # length-changing bodies alike. Explicit as_list=False keeps
            # the Array container (same as omitted).
            ('pl.col("q").arr.eval(pl.element() * 2, as_list=True)', ListT(Int64())),
            ('pl.col("q").arr.eval(pl.element().cast(pl.Utf8), as_list=True)', ListT(Utf8())),
            ('pl.col("q").arr.eval(pl.element() * 2, as_list=False)', Array(Int64(), 3)),
            # Unrecognised method falls through to Unknown (silent)
            ('pl.col("q").arr.to_struct()', Unknown()),
        ],
    )
    def test_arr_return_dtypes(self, expr: str, expected):
        assert self._dtype_of(expr) == expected

    def test_nullable_array_receiver_wraps_result(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                q: pl.Array(pl.Int64, 3) = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("q").arr.sum().alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["out"].dtype == Nullable(Int64())

    def test_arr_eval_bad_body_bubbles_error(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.select(pl.col("q").arr.eval(pl.element() + pl.lit("x")).alias("out"))
            """
        )
        results = analyze_source(source)
        assert any("PLY009" in e for e in results[0].errors), results[0].errors

    def test_arr_eval_as_list_bad_body_bubbles_error(self):
        # The body is type-checked for ``as_list=True`` too — the old
        # Unknown fallback skipped body analysis entirely.
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.select(
                    pl.col("q").arr.eval(pl.element() + pl.lit("x"), as_list=True).alias("out")
                )
            """
        )
        results = analyze_source(source)
        assert any("PLY009" in e for e in results[0].errors), results[0].errors

    # upgrade trigger: non-literal as_list values gain constant propagation
    @pytest.mark.imprecision
    def test_arr_eval_non_literal_as_list_degrades_to_unknown(self):
        # A non-literal ``as_list`` leaves the container kind (Array vs
        # List) unknowable — never guess one of the two.
        expr = 'pl.col("q").arr.eval(pl.element() * 2, as_list=flag)'
        assert self._dtype_of(expr) == Unknown()


class TestContainerAggReturns:
    """Probed fix rolled into #53: list-namespace ``sum``/``mean``/``median``
    (and newly ``std``/``var``) are not element-preserving.

    Probed on polars 1.41.2: ``list.mean`` on List(Int64) -> Float64 (the
    old element-return table claimed Int64); ``list.sum`` on Int8/Int16/
    UInt8/UInt16 widens to Int64; Float32 cells keep Float32.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                xs: pl.List(pl.Int64) = pa.Field()
                f: pl.List(pl.Float32) = pa.Field()
                w: pl.List(pl.Int16) = pa.Field()
                s: pl.List(pl.Utf8) = pa.Field()
        """
    )

    def _dtype_of(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(({expr}).alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], (expr, results[0].errors)
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft.columns["out"].dtype

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ('pl.col("xs").list.mean()', Float64()),
            ('pl.col("xs").list.median()', Float64()),
            ('pl.col("xs").list.std()', Float64()),
            ('pl.col("xs").list.var()', Float64()),
            ('pl.col("xs").list.sum()', Int64()),
            ('pl.col("f").list.mean()', Float32()),
            ('pl.col("f").list.sum()', Float32()),
            ('pl.col("w").list.sum()', Int64()),
            ('pl.col("w").list.min()', Int16()),
            # min/max over string elements is valid (lexicographic; probed).
            ('pl.col("s").list.min()', Utf8()),
            ('pl.col("xs").list.explode()', Int64()),
        ],
    )
    def test_list_agg_return_dtypes(self, expr: str, expected):
        assert self._dtype_of(expr) == expected


class TestContainerAggMatrix:
    """Issue #55: probed reducer matrix for ``.list`` / ``.arr``.

    Probed on polars 1.41.2 (see ``compat.polars_api.container_agg_return``):
    valid cells keep the probed result dtype; probed-invalid cells (runtime
    InvalidOperationError / ComputeError / rust panic) flag PLY016 and the
    output degrades to Unknown; unclaimed cells (degenerate all-null
    results, unprobed dtypes) stay silent with an Unknown output.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                l_i64: pl.List(pl.Int64) = pa.Field()
                l_str: pl.List(pl.Utf8) = pa.Field()
                l_bool: pl.List(pl.Boolean) = pa.Field()
                l_date: pl.List(pl.Date) = pa.Field()
                l_dt: pl.List(pl.Datetime) = pa.Field()
                l_dur: pl.List(pl.Duration) = pa.Field()
                l_time: pl.List(pl.Time) = pa.Field()
                l_dec: pl.List(pl.Decimal(10, 2)) = pa.Field()
                l_nest: pl.List(pl.List(pl.Int64)) = pa.Field()
                l_struct: pl.List(pl.Struct({"f": pl.Int64})) = pa.Field()
                a_i64: pl.Array(pl.Int64, 3) = pa.Field()
                a_str: pl.Array(pl.Utf8, 3) = pa.Field()
                a_bool: pl.Array(pl.Boolean, 3) = pa.Field()
                a_date: pl.Array(pl.Date, 3) = pa.Field()
                a_dur: pl.Array(pl.Duration, 3) = pa.Field()
                a_dec: pl.Array(pl.Decimal(10, 2), 3) = pa.Field()
                a_struct: pl.Array(pl.Struct({"f": pl.Int64}), 3) = pa.Field()
        """
    )

    def _analyze(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(({expr}).alias("out"))
            """
        )
        results = analyze_source(source)
        ft = results[0].inferred_return_type
        dtype = ft.columns["out"].dtype if ft is not None and "out" in ft.columns else None
        return results[0].errors, dtype

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # -- list: probed-valid cells beyond the numeric core ----------
            ('pl.col("l_dur").list.sum()', Duration()),
            ('pl.col("l_dec").list.sum()', Decimal(10, 2)),
            ('pl.col("l_date").list.mean()', Datetime()),
            ('pl.col("l_dt").list.median()', Datetime()),
            ('pl.col("l_time").list.mean()', Time()),
            ('pl.col("l_dur").list.mean()', Duration()),
            ('pl.col("l_dur").list.std()', Duration()),
            ('pl.col("l_dec").list.median()', Float64()),
            ('pl.col("l_dec").list.std()', Float64()),
            ('pl.col("l_dec").list.var()', Float64()),
            ('pl.col("l_bool").list.var()', Float64()),
            ('pl.col("l_bool").list.min()', Boolean()),
            ('pl.col("l_str").list.max()', Utf8()),
            ('pl.col("l_date").list.max()', Date()),
            ('pl.col("l_dur").list.min()', Duration()),
            ('pl.col("l_dec").list.min()', Decimal(10, 2)),
            # -- arr: probed-valid cells ------------------------------------
            ('pl.col("a_i64").arr.min()', Int64()),
            ('pl.col("a_i64").arr.max()', Int64()),
            ('pl.col("a_dur").arr.mean()', Duration()),
            ('pl.col("a_dur").arr.std()', Duration()),
            ('pl.col("a_dec").arr.median()', Float64()),
            ('pl.col("a_bool").arr.sum()', UInt32()),
        ],
    )
    def test_probed_valid_cells(self, expr: str, expected):
        errors, dtype = self._analyze(expr)
        assert errors == [], (expr, errors)
        assert dtype == expected

    @pytest.mark.parametrize(
        "expr",
        [
            # -- list: probed runtime InvalidOperationError -----------------
            'pl.col("l_str").list.sum()',
            'pl.col("l_date").list.sum()',
            'pl.col("l_dt").list.sum()',
            'pl.col("l_time").list.sum()',
            'pl.col("l_nest").list.sum()',
            'pl.col("l_struct").list.sum()',
            'pl.col("l_date").list.var()',
            'pl.col("l_dt").list.var()',
            'pl.col("l_dur").list.var()',
            'pl.col("l_time").list.var()',
            'pl.col("l_nest").list.min()',
            'pl.col("l_struct").list.max()',
            # -- arr: probed ComputeError (sum) / rust panic (min/max) ------
            'pl.col("a_str").arr.sum()',
            'pl.col("a_date").arr.sum()',
            'pl.col("a_dur").arr.sum()',
            'pl.col("a_dec").arr.sum()',
            'pl.col("a_struct").arr.sum()',
            'pl.col("a_dur").arr.var()',
            'pl.col("a_bool").arr.min()',
            'pl.col("a_str").arr.max()',
            'pl.col("a_date").arr.min()',
            'pl.col("a_dec").arr.max()',
            'pl.col("a_struct").arr.min()',
        ],
    )
    def test_probed_invalid_cells_flag_ply016(self, expr: str):
        errors, dtype = self._analyze(expr)
        assert any("PLY016" in e for e in errors), (expr, errors)
        # The output column still exists at runtime semantics-wise; it is
        # registered as Unknown so downstream lookups resolve.
        assert dtype == Unknown(), (expr, dtype)

    @pytest.mark.parametrize(
        "expr",
        [
            # Degenerate all-null Float64 results (probed valid at runtime,
            # deliberately unclaimed) and unprobed cells: silent Unknown.
            'pl.col("l_str").list.mean()',
            'pl.col("l_str").list.var()',
            'pl.col("l_nest").list.mean()',
            'pl.col("a_str").arr.mean()',
            'pl.col("a_date").arr.median()',
            'pl.col("a_date").arr.var()',
        ],
    )
    def test_unclaimed_cells_stay_silent(self, expr: str):
        errors, dtype = self._analyze(expr)
        assert errors == [], (expr, errors)
        assert dtype == Unknown(), (expr, dtype)

    def test_nullable_receiver_does_not_mask_invalid_cell(self):
        # A nullable List(Utf8) column is unwrapped before the verdict:
        # sum over string elements is still the probed runtime error.
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class S(pa.DataFrameModel):
                xs: pl.List(pl.Utf8) = pa.Field(nullable=True)

            def f(df: DataFrame[S]):
                return df.select(pl.col("xs").list.sum().alias("out"))
            """
        )
        results = analyze_source(source)
        assert any("PLY016" in e for e in results[0].errors), results[0].errors


class TestCatNamespaceReturns:
    """Issue #54: cat-namespace result dtypes (probed on polars 1.41.2,
    identical for Categorical and Enum receivers)."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                c: pl.Categorical
                e: pl.Enum
                n: pl.Categorical = pa.Field(nullable=True)
        """
    )

    def _dtype_of(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(({expr}).alias("out"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], (expr, results[0].errors)
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft.columns["out"].dtype

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ('pl.col("c").cat.get_categories()', Utf8()),
            ('pl.col("e").cat.get_categories()', Utf8()),
            ('pl.col("c").cat.len_bytes()', UInt32()),
            ('pl.col("c").cat.len_chars()', UInt32()),
            ('pl.col("c").cat.starts_with("a")', Boolean()),
            ('pl.col("c").cat.ends_with("a")', Boolean()),
            ('pl.col("c").cat.slice(0, 2)', Utf8()),
            # Unrecognised method falls through to Unknown (silent)
            ('pl.col("c").cat.some_future_method()', Unknown()),
            # get_categories is length-changing and never null — the
            # receiver's nullability is NOT inherited (probed).
            ('pl.col("n").cat.get_categories()', Utf8()),
            # Per-row methods keep the receiver's nullability.
            ('pl.col("n").cat.len_chars()', Nullable(UInt32())),
        ],
    )
    def test_cat_return_dtypes(self, expr: str, expected):
        assert self._dtype_of(expr) == expected


class TestOverKeyValidation:
    """Issue #32: ``over`` validates partition/order keys exist (PLY001).

    Verified against polars 1.41.2: positional string args, the
    ``partition_by=`` kwarg and the ``order_by=`` kwarg all resolve strings
    (and lists of strings) as column names — a missing one raises
    ColumnNotFoundError at runtime. ``mapping_strategy=`` is a modifier,
    not a column.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                v: float
                s: str
                t: int
        """
    )

    def _analyze(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        return analyze_source(source)

    def test_missing_literal_key_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").sum().over("ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_existing_literal_key_passes(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s"))')
        assert results[0].errors == [], results[0].errors

    def test_varargs_keys_each_checked(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s", "ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_multiple_existing_keys_pass(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s", "t"))')
        assert results[0].errors == [], results[0].errors

    def test_list_keys_each_checked(self):
        results = self._analyze('df.select(pl.col("v").sum().over(["s", "ghost"]))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_expr_key_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").sum().over(pl.col("ghost")))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_expr_key_existing_passes(self):
        results = self._analyze('df.select(pl.col("v").sum().over(pl.col("s")))')
        assert results[0].errors == [], results[0].errors

    def test_order_by_kwarg_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by="ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_order_by_kwarg_existing_passes(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by="t"))')
        assert results[0].errors == [], results[0].errors

    def test_order_by_list_each_checked(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by=["t", "ghost"]))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_partition_by_kwarg_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").sum().over(partition_by="ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)

    def test_mapping_strategy_kwarg_is_not_a_column(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s", mapping_strategy="join"))')
        assert results[0].errors == [], results[0].errors

    def test_non_literal_key_silently_ignored(self):
        """ExpressionAnalyzer has no constant environment — a Name key is
        silently ignored rather than risking a false positive."""
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                key = "ghost"
                return df.select(pl.col("v").sum().over(key))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_open_frame_missing_key_not_flagged(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.select(pl.col("id").sum().over("ghost"))')
        assert analyzer.errors == []

    def test_over_stays_dtype_preserving(self):
        results = self._analyze('df.select(x=pl.col("v").sum().over("s"))')
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["x"].dtype == Float64()

    def test_with_columns_over_missing_flags_ply001(self):
        """Issue #32 repro shape: with_columns(g=...over("ghost"))."""
        results = self._analyze('df.with_columns(g=pl.col("v").sum().over("ghost"))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)


class TestOverMappingStrategy:
    """Issue #45: ``over(mapping_strategy=...)`` result dtype.

    Probed (polars 1.41.2) over the strategy x expression product:

    - ``"join"`` gathers a *length-preserving / multi-valued* expression
      into ``List(<element dtype>)`` per row (inner nullability preserved:
      a nullable receiver yields lists with null elements); but a
      *scalar-per-group* expression (an aggregation, including arithmetic
      on one, e.g. ``sum() + 1``) broadcasts WITHOUT a List wrapper.
    - ``"explode"`` and ``"group_to_rows"`` (the default) preserve the
      dtype for every expression shape.
    - An unknown / non-literal strategy value degrades to Unknown.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                g: str
                n: float = pa.Field(nullable=True)
        """
    )

    def _dtype(self, expr: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return df.select(o={expr})
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft.columns["o"].dtype

    # -- "join" on length-preserving expressions -> List ----------------------

    def test_join_bare_col_is_list(self):
        dtype = self._dtype('pl.col("a").over("g", mapping_strategy="join")')
        assert dtype == ListT(Int64())

    def test_join_nullable_receiver_keeps_inner_nullable(self):
        dtype = self._dtype('pl.col("n").over("g", mapping_strategy="join")')
        assert dtype == ListT(Nullable(Float64()))

    def test_join_shift_receiver_is_list_of_nullable(self):
        dtype = self._dtype('pl.col("a").shift().over("g", mapping_strategy="join")')
        assert dtype == ListT(Nullable(Int64()))

    def test_join_elementwise_arithmetic_is_list(self):
        dtype = self._dtype('(pl.col("a") * 2).over("g", mapping_strategy="join")')
        assert dtype == ListT(Int64())

    def test_join_rank_receiver_is_list(self):
        dtype = self._dtype('pl.col("a").rank().over("g", mapping_strategy="join")')
        assert dtype == ListT(Float64())

    def test_join_boolean_predicate_is_list(self):
        dtype = self._dtype('pl.col("a").is_null().over("g", mapping_strategy="join")')
        assert dtype == ListT(Boolean())

    def test_join_drop_nulls_receiver_is_list(self):
        # drop_nulls is multi-valued per group (length-changing) -> List;
        # the stripped nullability stays stripped inside the list (probed).
        dtype = self._dtype('pl.col("n").drop_nulls().over("g", mapping_strategy="join")')
        assert dtype == ListT(Float64())

    # -- "join" on scalar-per-group expressions -> broadcast (no List) --------

    def test_join_sum_broadcasts_scalar(self):
        dtype = self._dtype('pl.col("a").sum().over("g", mapping_strategy="join")')
        assert dtype == Int64()

    def test_join_mean_broadcasts_scalar(self):
        dtype = self._dtype('pl.col("a").mean().over("g", mapping_strategy="join")')
        assert dtype == Float64()

    def test_join_agg_arithmetic_broadcasts_scalar(self):
        dtype = self._dtype('(pl.col("a").sum() + 1).over("g", mapping_strategy="join")')
        assert dtype == Int64()

    def test_join_entropy_broadcasts_scalar(self):
        # entropy is the one reduction in the Float64-return set.
        dtype = self._dtype('pl.col("a").entropy().over("g", mapping_strategy="join")')
        assert dtype == Float64()

    # -- "explode" / "group_to_rows" preserve the dtype ------------------------

    def test_explode_preserves_dtype(self):
        dtype = self._dtype('pl.col("a").over("g", mapping_strategy="explode")')
        assert dtype == Int64()

    def test_explode_agg_receiver_preserves_dtype(self):
        dtype = self._dtype('pl.col("a").sum().over("g", mapping_strategy="explode")')
        assert dtype == Int64()

    def test_explicit_group_to_rows_preserves_dtype(self):
        dtype = self._dtype('pl.col("a").over("g", mapping_strategy="group_to_rows")')
        assert dtype == Int64()

    def test_default_strategy_preserves_dtype(self):
        dtype = self._dtype('pl.col("a").over("g")')
        assert dtype == Int64()

    # -- degrade to Unknown instead of guessing --------------------------------

    def test_unknown_strategy_literal_degrades_to_unknown(self):
        dtype = self._dtype('pl.col("a").over("g", mapping_strategy="bogus")')
        assert dtype == Unknown()

    def test_non_literal_strategy_degrades_to_unknown(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In], strat: str):
                return df.select(o=pl.col("a").over("g", mapping_strategy=strat))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["o"].dtype == Unknown()

    def test_join_unknown_cardinality_receiver_degrades_to_unknown(self):
        # when/then/otherwise cardinality is decided by the branch values,
        # which the classifier does not inspect — never guess.
        dtype = self._dtype(
            'pl.when(pl.col("a") > 0).then(pl.col("a")).otherwise(None)'
            '.over("g", mapping_strategy="join")'
        )
        assert dtype == Unknown()

    # -- key validation is unchanged -------------------------------------------

    def test_join_keys_still_validated(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                return df.select(o=pl.col("a").over("ghost", mapping_strategy="join"))
            """
        )
        results = analyze_source(source)
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)


class TestComparisonIncompatibleDtypes:
    """Issue #33: comparisons between incompatible dtypes flag PLY009.

    Ground truth verified against polars 1.41.2 by driving the full
    10-dtype x {==, !=, <, <=, >, >=} product through ``df.select``
    (probe output in the implementing commit message). All six
    comparison operators share one validity table:

    ========  ==== ==== ==== ==== ==== ==== ==== ==== ==== ====
    left\\right  i    f    s    b    d    dt   t    du   cat  en
    i           ok   ok   ERR  ok   ok   ok   ok   ok   ERR  ERR
    f           ok   ok   ERR  ok   ok   ok   ok   ok   ERR  ERR
    s           ERR  ERR  ok   ok   ERR  ERR  ok*  ERR  ok   ok
    b           ok   ok   ok   ok   ERR  ERR  ERR  ERR  ERR  ERR
    d           ok   ok   ERR  ERR  ok   ok   ERR  ERR  ERR  ERR
    dt          ok   ok   ERR  ERR  ok   ok   ERR  ERR  ERR  ERR
    t           ok   ok   ERR* ERR  ERR  ERR  ok   ERR  ERR  ERR
    du          ok   ok   ERR  ERR  ERR  ERR  ERR  ok   ERR  ERR
    cat         ERR  ERR  ok   ERR  ERR  ERR  ERR  ERR  ok   ERR
    en          ERR  ERR  ok   ERR  ERR  ERR  ERR  ERR  ERR  ok
    ========  ==== ==== ==== ==== ==== ==== ==== ==== ==== ====

    Notable: numeric vs temporal comparisons are *allowed* (physical
    repr); str vs time is asymmetric (``s == t`` ok, ``t == s`` ERR) so
    that pair stays silent. Only pairs that error in both directions
    are flagged. Operands outside the closed set (Unknown, Decimal,
    Null literals, unresolved) keep the silent fallback. The result
    stays Boolean — the dtype is not in question, the error is the
    signal.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "i": Int64(),
                "f": Float64(),
                "s": Utf8(),
                "b": Boolean(),
                "d": Date(),
                "dt": Datetime(),
                "t": Time(),
                "du": Duration(),
                "cat": Categorical(),
                "en": Enum(),
                "ni": Nullable(Int64()),
                "ns": Nullable(Utf8()),
                "u": Unknown(),
                "dec": Decimal(10, 2),
            }
        )

    # -- allowed combinations -------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('i') == pl.col('i')",
            "pl.col('i') < pl.col('f')",
            "pl.col('f') > 0",
            "pl.col('i') == 1.5",
            "pl.col('i') != pl.col('b')",
            "pl.col('b') <= pl.col('f')",
            "pl.col('b') == pl.col('b')",
            # numeric vs temporal is allowed (probed)
            "pl.col('i') == pl.col('d')",
            "pl.col('i') < pl.col('dt')",
            "pl.col('f') == pl.col('t')",
            "pl.col('i') >= pl.col('du')",
            "pl.col('d') == 1",
            # string-likes
            "pl.col('s') == pl.col('s')",
            "pl.col('s') != 'x'",
            "pl.col('s') == pl.col('b')",
            "pl.col('s') < pl.col('cat')",
            "pl.col('s') == pl.col('en')",
            "pl.col('cat') == 'x'",
            "pl.col('en') == pl.col('s')",
            "pl.col('cat') == pl.col('cat')",
            "pl.col('en') != pl.col('en')",
            # temporal same-family
            "pl.col('d') == pl.col('dt')",
            "pl.col('dt') < pl.col('d')",
            "pl.col('t') == pl.col('t')",
            "pl.col('du') >= pl.col('du')",
            # asymmetric str/time quirk stays silent (both directions)
            "pl.col('s') == pl.col('t')",
            "pl.col('t') == pl.col('s')",
        ],
    )
    def test_allowed_combination_no_error(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert analyzer.errors == [], analyzer.errors
        dtype = analyzer.var_types["out"].columns["m"].dtype
        base = dtype.inner if isinstance(dtype, Nullable) else dtype
        assert base == Boolean()

    # -- known-invalid combinations -------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # string vs numeric / temporal
            "pl.col('s') == pl.col('i')",
            "pl.col('i') == pl.col('s')",
            "pl.col('s') < pl.col('f')",
            "pl.col('s') != pl.col('i')",
            "pl.col('s') == 1",
            "pl.col('i') == 'x'",
            "pl.col('s') <= pl.col('d')",
            "pl.col('s') == pl.col('dt')",
            "pl.col('s') > pl.col('du')",
            # boolean vs temporal / categorical
            "pl.col('b') == pl.col('d')",
            "pl.col('b') < pl.col('dt')",
            "pl.col('b') == pl.col('t')",
            "pl.col('b') != pl.col('du')",
            "pl.col('b') == pl.col('cat')",
            "pl.col('b') == pl.col('en')",
            # temporal cross-family
            "pl.col('d') == pl.col('t')",
            "pl.col('d') < pl.col('du')",
            "pl.col('dt') == pl.col('t')",
            "pl.col('dt') != pl.col('du')",
            "pl.col('t') == pl.col('du')",
            # categorical / enum
            "pl.col('cat') == pl.col('i')",
            "pl.col('cat') < pl.col('f')",
            "pl.col('cat') == pl.col('d')",
            "pl.col('cat') == pl.col('en')",
            "pl.col('en') == pl.col('f')",
            "pl.col('en') != pl.col('du')",
            "pl.col('cat') == 1",
        ],
    )
    def test_invalid_combination_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        # Comparisons stay Boolean-typed — the error is the signal.
        assert analyzer.var_types["out"].columns["m"].dtype == Boolean()

    def test_error_message_names_dtypes_and_op(self):
        analyzer = _run_body(self._frame(), "out = df.select(m=pl.col('s') == pl.col('i'))")
        assert "comparison 'Utf8 == Int64'" in analyzer.errors[0]

    def test_chained_comparison_checks_each_adjacent_pair(self):
        # cat < i is invalid AND i < s is invalid -> two errors.
        analyzer = _run_body(
            self._frame(), "out = df.select(m=pl.col('cat') < pl.col('i') < pl.col('s'))"
        )
        assert len(analyzer.errors) == 2, analyzer.errors
        assert all("PLY009" in e for e in analyzer.errors)

    def test_parenthesized_comparison_result_is_boolean_operand(self):
        # (i < f) < s is NOT a chained compare: the left operand is the
        # Boolean comparison result, and bool vs str is probed-valid.
        analyzer = _run_body(
            self._frame(), "out = df.select(m=(pl.col('i') < pl.col('f')) < pl.col('s'))"
        )
        assert analyzer.errors == [], analyzer.errors

    def test_chained_comparison_native_syntax(self):
        analyzer = _run_body(self._frame(), "out = df.select(m=pl.col('s') < pl.col('i') < 3)")
        # s < i is invalid; i < 3 is fine.
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["m"].dtype == Boolean()

    # -- Nullable unwrap / nullability propagation -----------------------------

    def test_invalid_pair_detected_under_nullable_wrappers(self):
        analyzer = _run_body(self._frame(), "out = df.select(m=pl.col('ns') == pl.col('ni'))")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        assert "Utf8 == Int64" in analyzer.errors[0]

    def test_nullable_operand_keeps_nullable_boolean_result(self):
        analyzer = _run_body(self._frame(), "out = df.select(m=pl.col('ni') > 0)")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["m"].dtype == Nullable(Boolean())

    # -- silent fallback --------------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('u') == pl.col('s')",  # Unknown operand
            "pl.col('dec') == pl.col('s')",  # Decimal outside the closed set
            "pl.col('s') == helper()",  # unresolved operand
            "pl.col('i') == None",  # Null literal: null-compare, not a dtype error
            "pl.col('s') == None",
        ],
    )
    # upgrade trigger: these operand kinds get probed cells in the comparison table
    @pytest.mark.imprecision
    def test_not_fully_understood_pair_is_silent(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert analyzer.errors == [], analyzer.errors


class TestIsInIncompatibleDtypes:
    """Issue #33: ``is_in`` element dtype incompatible with receiver flags PLY009.

    Ground truth verified against polars 1.41.2 (probe output in the
    implementing commit message). ``is_in`` is stricter than comparison:
    valid pairs are exactly num x num, str x {str, cat, enum}, cat x cat,
    enum x enum, bool x bool, date x date, datetime x datetime,
    time x time, dur x dur. Everything else inside the closed set raises
    InvalidOperationError — including int x bool, date x datetime and
    cat x enum. Enum x str failures are value-dependent (out-of-category
    values) so that pair stays valid/silent. A non-List expression arg of
    dtype T is imploded by polars and acts as element dtype T.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "i": Int64(),
                "f": Float64(),
                "s": Utf8(),
                "b": Boolean(),
                "d": Date(),
                "du": Duration(),
                "cat": Categorical(),
                "en": Enum(),
                "ni": Nullable(Int64()),
                "u": Unknown(),
                "dec": Decimal(10, 2),
                "li": ListT(Int64()),
                "lf": ListT(Float64()),
                "ls": ListT(Utf8()),
                "ldt": ListT(Datetime()),
                "len_": ListT(Enum()),
                "nls": Nullable(ListT(Utf8())),
            }
        )

    # -- allowed combinations ---------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # literal lists / tuples
            "pl.col('i').is_in([1, 2, 3])",
            "pl.col('i').is_in((1, 2))",
            "pl.col('i').is_in([1.5, 2.5])",  # int/float interchangeable
            "pl.col('f').is_in([1, 2])",
            "pl.col('s').is_in(['x', 'y'])",
            "pl.col('b').is_in([True])",
            "pl.col('cat').is_in(['x', 'y'])",
            "pl.col('en').is_in(['x', 'y'])",  # value-dependent only -> silent
            # Null elements are unwrapped before the check
            "pl.col('i').is_in([1, None])",
            # Nullable receiver unwraps
            "pl.col('ni').is_in([1, 2])",
            # expression args: List(T) contributes T
            "pl.col('i').is_in(pl.col('li'))",
            "pl.col('i').is_in(pl.col('lf'))",
            "pl.col('s').is_in(pl.col('ls'))",
            "pl.col('cat').is_in(pl.col('ls'))",
            # non-List expr arg of dtype T acts as element dtype T
            "pl.col('i').is_in(pl.col('f'))",
            # Nullable List expr arg unwraps at both levels
            "pl.col('s').is_in(pl.col('nls'))",
        ],
    )
    def test_allowed_combination_no_error(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["m"].dtype == Boolean()

    # -- known-invalid combinations ----------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # the issue #33 repro
            "pl.col('i').is_in(['x', 'y'])",
            "pl.col('s').is_in([1, 2])",
            # is_in is stricter than comparison: bool/int don't mix
            "pl.col('i').is_in([True])",
            "pl.col('b').is_in([1])",
            "pl.col('du').is_in([1])",
            "pl.col('cat').is_in([1, 2])",
            # expression args
            "pl.col('s').is_in(pl.col('li'))",
            "pl.col('i').is_in(pl.col('ls'))",
            "pl.col('d').is_in(pl.col('ldt'))",  # date vs datetime errors
            "pl.col('cat').is_in(pl.col('len_'))",  # cat vs enum errors
            # non-List expr arg with incompatible dtype
            "pl.col('i').is_in(pl.col('s'))",
            # Nullable receiver still checked after unwrap
            "pl.col('ni').is_in(['x'])",
        ],
    )
    def test_invalid_combination_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        # Result stays Boolean — the error is the signal.
        assert analyzer.var_types["out"].columns["m"].dtype == Boolean()

    def test_error_message_mirrors_polars_wording(self):
        analyzer = _run_body(self._frame(), "out = df.select(m=pl.col('i').is_in(['x', 'y']))")
        assert "is_in" in analyzer.errors[0]
        assert "Utf8" in analyzer.errors[0]
        assert "Int64" in analyzer.errors[0]

    # -- silent fallback ----------------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('i').is_in([])",  # empty list: polars accepts
            "pl.col('i').is_in([1, 'x'])",  # non-unifiable literals
            "pl.col('i').is_in([x, y])",  # non-constant elements
            "pl.col('u').is_in(['x'])",  # Unknown receiver
            "pl.col('dec').is_in([1])",  # Decimal outside the closed set
            "pl.col('i').is_in(helper())",  # unresolvable arg
            "pl.col('i').is_in()",  # no args
        ],
    )
    # upgrade trigger: these argument kinds get probed cells in the is_in table
    @pytest.mark.imprecision
    def test_not_fully_understood_is_silent(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(m={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["m"].dtype == Boolean()


class TestCastImpossibleDtypes:
    """Issue #34: ``cast()`` to a structurally impossible target flags PLY013.

    Ground truth verified against polars 1.41.2 with BOTH ``strict=True``
    and ``strict=False`` (probe output in the implementing commit
    message) — only combinations that fail in both modes are flagged;
    value-dependent failures (``Utf8 -> Int64``, ``num -> Categorical``,
    ``str/int -> Enum``, ``str -> Struct``) stay silent. Flagged source ->
    target category pairs:

    - str -> bool;  bool -> cat/enum
    - date -> bool/time/dur/cat/enum/list/struct
    - datetime -> bool/dur/cat/enum/list/struct
    - time -> bool/date/datetime/cat/enum/list/struct
    - dur -> str/bool/date/datetime/time/cat/enum/list/struct
    - cat/enum -> float/bool/date/datetime/time/dur/list/struct
    - list -> any non-list; list -> list recurses on element dtypes
      (List(Date) -> List(Duration) errors in both modes)

    Notable probed-OK cells: Struct -> anything (polars casts the fields;
    Struct -> Utf8 yields String), num -> List/Struct (wraps), cat/enum
    -> int (physical), num <-> temporal. On a flagged expression cast the
    output degrades to Unknown — fabricating the target dtype would hide
    declared-type mismatches (the issue #34 repro).
    """

    def _frame(self) -> FrameType:
        from polypolarism.types import Struct

        return FrameType(
            {
                "i": Int64(),
                "f": Float64(),
                "s": Utf8(),
                "b": Boolean(),
                "d": Date(),
                "dt": Datetime(),
                "t": Time(),
                "du": Duration(),
                "cat": Categorical(),
                "en": Enum(),
                "li": ListT(Int64()),
                "ld": ListT(Date()),
                "st": Struct({"x": Int64()}),
                "nli": Nullable(ListT(Int64())),
                "ni": Nullable(Int64()),
                "u": Unknown(),
                "dec": Decimal(10, 2),
            }
        )

    # -- known-impossible casts -------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            # the issue #34 repro and other List -> scalar casts
            "pl.col('li').cast(pl.Int64)",
            "pl.col('li').cast(pl.Utf8)",
            "pl.col('li').cast(pl.Boolean)",
            "pl.col('ld').cast(pl.Date)",
            # strict=False exempts nothing structural (probed)
            "pl.col('li').cast(pl.Int64, strict=False)",
            # temporal cross-family
            "pl.col('d').cast(pl.Time)",
            "pl.col('d').cast(pl.Boolean)",
            "pl.col('dt').cast(pl.Duration)",
            "pl.col('t').cast(pl.Date)",
            "pl.col('t').cast(pl.Datetime)",
            "pl.col('du').cast(pl.Utf8)",
            "pl.col('du').cast(pl.Date)",
            # string / bool / categorical
            "pl.col('s').cast(pl.Boolean)",
            "pl.col('b').cast(pl.Categorical)",
            "pl.col('b').cast(pl.Enum)",
            "pl.col('cat').cast(pl.Float64)",
            "pl.col('en').cast(pl.Float64)",
            "pl.col('cat').cast(pl.Duration)",
            "pl.col('d').cast(pl.Enum)",
            # scalar -> List only errors for temporal/categorical sources
            "pl.col('d').cast(pl.List(pl.Int64))",
            # list -> list recurses on the element pair
            "pl.col('ld').cast(pl.List(pl.Duration))",
            # Nullable receiver unwraps
            "pl.col('nli').cast(pl.Int64)",
        ],
    )
    def test_impossible_cast_flags_ply013_and_degrades_to_unknown(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY013" in analyzer.errors[0]
        # The error is the signal — don't fabricate the target dtype, or a
        # declared-type mismatch downstream would be hidden (issue #34).
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_error_message_names_source_and_target(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('li').cast(pl.Int64))")
        assert "List[Int64]" in analyzer.errors[0]
        assert "Int64" in analyzer.errors[0]
        assert "strict=False" in analyzer.errors[0]

    # -- allowed / value-dependent casts stay silent ------------------------------

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("pl.col('i').cast(pl.Utf8)", Utf8()),
            ("pl.col('i').cast(pl.Float64)", Float64()),
            # value-dependent failure -> allowed
            ("pl.col('s').cast(pl.Int64)", Int64()),
            ("pl.col('i').cast(pl.Date)", Date()),
            ("pl.col('b').cast(pl.Int64)", Int64()),
            ("pl.col('cat').cast(pl.Int64)", Int64()),  # physical repr
            ("pl.col('cat').cast(pl.Utf8)", Utf8()),
            ("pl.col('en').cast(pl.Categorical)", Categorical()),
            ("pl.col('i').cast(pl.Categorical)", Categorical()),  # strict=False ok
            ("pl.col('s').cast(pl.Enum)", Enum()),  # value-dependent
            ("pl.col('dt').cast(pl.Time)", Time()),  # probed ok
            ("pl.col('d').cast(pl.Datetime)", Datetime()),
            # list -> list with a castable element pair
            ("pl.col('li').cast(pl.List(pl.Float64))", ListT(Float64())),
            ("pl.col('li').cast(pl.List(pl.Utf8))", ListT(Utf8())),
            # num -> List wraps each value (probed ok)
            ("pl.col('i').cast(pl.List(pl.Int64))", ListT(Int64())),
        ],
    )
    def test_allowed_cast_infers_target_dtype(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    def test_struct_source_casts_are_silent(self):
        # Struct -> scalar is probed-OK (polars casts the fields).
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('st').cast(pl.Int64))")
        assert analyzer.errors == [], analyzer.errors

    def test_nullable_receiver_keeps_nullable_target(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('ni').cast(pl.Float64))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Float64())

    # -- Unknown / outside-the-set exemptions --------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('u').cast(pl.Int64)",  # Unknown source
            "pl.col('dec').cast(pl.Time)",  # Decimal outside the closed set
            "pl.col('v').interpolate().cast(pl.Int64)",  # un-inferable receiver pins
            "pl.col('li').cast(unknown_target)",  # unresolvable target
        ],
    )
    # upgrade trigger: these source/target kinds get probed cells in the cast table
    @pytest.mark.imprecision
    def test_unknown_source_or_target_is_silent(self, expr: str) -> None:
        from polypolarism.types import ColumnSpec

        frame = self._frame()
        frame.columns["v"] = ColumnSpec(dtype=Int64())
        analyzer = _run_body(frame, f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors


class TestCastImpossibleFrameLevel:
    """Issue #34: frame-level ``df.cast({...})`` applies the same per-column check."""

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "v": ListT(Int64()),
                "w": Int64(),
                "u": Unknown(),
            }
        )

    def test_frame_cast_list_to_int_flags_ply013(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'v': pl.Int64})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY013" in analyzer.errors[0]
        assert "'v'" in analyzer.errors[0]

    def test_frame_cast_keeps_source_spec_for_flagged_column(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'v': pl.Int64, 'w': pl.Int32})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY013" in analyzer.errors[0]
        # The flagged column keeps its source dtype; the valid one is cast.
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Int64())
        assert analyzer.var_types["out"].columns["w"].dtype == Int32()

    def test_frame_cast_strict_false_still_flags(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'v': pl.Int64}, strict=False)")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY013" in analyzer.errors[0]

    def test_frame_cast_valid_column_no_error(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'w': pl.Utf8})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["w"].dtype == Utf8()

    # upgrade trigger: Unknown source becomes inferable upstream
    @pytest.mark.imprecision
    def test_frame_cast_unknown_source_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'u': pl.Int64})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["u"].dtype == Int64()


class TestArrayCastDtypes:
    """Issue #53: Array cells in the PLY013 cast layer.

    Probed (polars 1.41.2, both strict modes):

    - Array -> any scalar/categorical/struct target: InvalidOperationError
      ("cannot cast Array type") — flagged.
    - scalar/temporal/categorical/struct -> Array: InvalidOperationError /
      ComputeError — flagged.
    - Array -> Array recurses on the element pair (Array(Int64) ->
      Array(Date) is probed-OK via the int->date element cast; Array(Date)
      -> Array(Duration) fails in both modes).
    - Array -> List: probed-OK for EVERY probed element pair (even
      Array(Date) -> List(Duration)) — never flagged.
    - List -> Array: ComputeError only when the list lengths don't match
      the width (value-dependent; equal-width probe succeeds) — silent.
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "q": Array(Int64()),
                "qd": Array(Date()),
                "li": ListT(Int64()),
                "i": Int64(),
                "s": Utf8(),
                "d": Date(),
                "qu": Array(Unknown()),
                "nq": Nullable(Array(Int64())),
            }
        )

    @pytest.mark.parametrize(
        "expr",
        [
            # Array -> scalar targets
            "pl.col('q').cast(pl.Int64)",
            "pl.col('q').cast(pl.Utf8)",
            "pl.col('q').cast(pl.Boolean)",
            "pl.col('q').cast(pl.Float64)",
            "pl.col('q').cast(pl.Date)",
            "pl.col('q').cast(pl.Categorical)",
            "pl.col('q').cast(pl.Int64, strict=False)",
            # sources -> Array targets
            "pl.col('i').cast(pl.Array(pl.Int64, 1))",
            "pl.col('s').cast(pl.Array(pl.Utf8, 1))",
            "pl.col('d').cast(pl.Array(pl.Date, 1))",
            # Array -> Array element recursion
            "pl.col('qd').cast(pl.Array(pl.Duration, 1))",
            # Nullable receiver unwraps
            "pl.col('nq').cast(pl.Int64)",
        ],
    )
    def test_impossible_array_cast_flags_ply013(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, (expr, analyzer.errors)
        assert "PLY013" in analyzer.errors[0]

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # Array -> List always allowed (probed)
            ("pl.col('q').cast(pl.List(pl.Int64))", ListT(Int64())),
            ("pl.col('q').cast(pl.List(pl.Utf8))", ListT(Utf8())),
            ("pl.col('qd').cast(pl.List(pl.Duration))", ListT(Duration())),
            # List -> Array is width/value-dependent — silent
            ("pl.col('li').cast(pl.Array(pl.Int64, 3))", Array(Int64(), 3)),
            # Array -> Array with a castable element pair
            ("pl.col('q').cast(pl.Array(pl.Float64, 3))", Array(Float64(), 3)),
            ("pl.col('q').cast(pl.Array(pl.Date, 3))", Array(Date(), 3)),
            # Unknown element on either side stays silent
            ("pl.col('qu').cast(pl.Array(pl.Int64, 3))", Array(Int64(), 3)),
        ],
    )
    def test_allowed_array_cast_infers_target(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], (expr, analyzer.errors)
        assert analyzer.var_types["out"].columns["r"].dtype == expected


class TestArrayOperationLayers:
    """Issue #53: remaining Array decisions in the operation layers."""

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "q": Array(Int64()),
                "nq": Nullable(Array(Int64())),
                "li": ListT(Int64()),
                "i": Int64(),
                "s": Utf8(),
            }
        )

    # -- explode (probed: Array explodes to its element dtype) ------------------

    def test_explode_array_column_yields_element(self):
        analyzer = _run_body(self._frame(), 'out = df.explode("q")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["q"].dtype == Int64()

    def test_explode_nullable_array_column_yields_nullable_element(self):
        analyzer = _run_body(self._frame(), 'out = df.explode("nq")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["nq"].dtype == Nullable(Int64())

    def test_explode_scalar_column_message_mentions_both_containers(self):
        analyzer = _run_body(self._frame(), 'out = df.explode("i")')
        assert len(analyzer.errors) == 1
        assert "PLY021" in analyzer.errors[0]
        assert "List/Array" in analyzer.errors[0]

    # -- is_in (probed: an Array(T) expression argument contributes T) ----------

    def test_is_in_array_expr_arg_matching_element_passes(self):
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("i").is_in(pl.col("q")))')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_is_in_array_expr_arg_mismatched_element_flags_ply009(self):
        # Probed: Utf8.is_in(Array(Int64)) raises InvalidOperationError.
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("s").is_in(pl.col("q")))')
        assert any("PLY009" in e for e in analyzer.errors), analyzer.errors

    def test_is_in_array_receiver_stays_silent(self):
        # An Array receiver is outside the probed is_in category set.
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("q").is_in([1, 2]))')
        assert analyzer.errors == [], analyzer.errors

    # -- cum_* (probed: cum_sum/cum_prod/cum_min/cum_max reject Array) ----------

    @pytest.mark.parametrize("method", ["cum_sum", "cum_prod", "cum_min", "cum_max"])
    def test_cum_methods_on_array_flag_ply016(self, method: str):
        analyzer = _run_body(self._frame(), f'out = df.select(r=pl.col("q").{method}())')
        assert any("PLY016" in e for e in analyzer.errors), (method, analyzer.errors)

    def test_cum_count_on_array_is_fine(self):
        # Probed: cum_count returns UInt32 for every receiver, Array included.
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("q").cum_count())')
        assert analyzer.errors == [], analyzer.errors

    # -- arithmetic / comparison stay silent (probed: Array * 2 and == work) ----

    def test_array_arithmetic_stays_silent(self):
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("q") * 2)')
        assert analyzer.errors == [], analyzer.errors

    def test_array_comparison_stays_silent(self):
        analyzer = _run_body(self._frame(), 'out = df.select(r=pl.col("q") == pl.col("q"))')
        assert analyzer.errors == [], analyzer.errors

    # -- concat (unify_types path: equal Array unifies, mismatch flags) ---------

    def test_concat_same_array_columns_passes(self):
        analyzer = _run_body(
            FrameType({"q": Array(Int64())}),
            "out = pl.concat([df, df])",
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["q"].dtype == Array(Int64())


class TestWhenConditionDtype:
    """Issue #37: ``pl.when(<cond>)`` with a non-Boolean condition is PLY008.

    Probed (polars 1.41.2): ``pl.when(pl.col("a"))`` with ``a: Int64`` raises
    ``SchemaError: invalid series dtype: expected `Boolean`, got `i64```;
    the same holds for a chained ``.when(...)`` condition and for a bare
    non-bool constant (``pl.when(1)``).
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                flag: bool
                v: pl.Float64 = pa.Field(nullable=True)
        """
    )

    def _analyze(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        return analyze_source(source)

    def test_nonbool_column_condition_flags_ply008(self):
        results = self._analyze('df.select(x=pl.when(pl.col("a")).then(1).otherwise(0))')
        assert any("PLY008" in e and "when" in e and "Int64" in e for e in results[0].errors), (
            results[0].errors
        )

    def test_chained_when_nonbool_condition_flags_ply008(self):
        results = self._analyze(
            'df.select(x=pl.when(pl.col("flag")).then(1).when(pl.col("a")).then(2).otherwise(0))'
        )
        assert any("PLY008" in e and "when" in e for e in results[0].errors)

    def test_nonbool_constant_condition_flags_ply008(self):
        # Probed: ``pl.when(1)`` raises SchemaError (expected Boolean, got i32).
        results = self._analyze("df.select(x=pl.when(1).then(1).otherwise(0))")
        assert any("PLY008" in e for e in results[0].errors)

    def test_bare_string_condition_is_column_reference(self):
        # Probed: ``pl.when("flag")`` resolves the string as a column name.
        results = self._analyze('df.select(x=pl.when("a").then(1).otherwise(0))')
        assert any("PLY008" in e for e in results[0].errors)

    def test_boolean_bare_string_condition_passes(self):
        results = self._analyze('df.select(x=pl.when("flag").then(1).otherwise(0))')
        assert results[0].errors == []

    def test_missing_column_condition_is_ply001_not_ply008(self):
        results = self._analyze('df.select(x=pl.when("ghost").then(1).otherwise(0))')
        assert any("PLY042" in e and "ghost" in e for e in results[0].errors)
        assert not any("PLY008" in e for e in results[0].errors)

    def test_boolean_condition_passes(self):
        results = self._analyze('df.select(x=pl.when(pl.col("flag")).then(1).otherwise(0))')
        assert results[0].errors == []

    def test_comparison_condition_passes(self):
        results = self._analyze('df.select(x=pl.when(pl.col("a") > 0).then(1).otherwise(0))')
        assert results[0].errors == []

    def test_nullable_boolean_condition_passes(self):
        # ``Nullable[Boolean]`` is a valid condition — probed: a null
        # condition row simply takes the otherwise branch.
        results = self._analyze('df.select(x=pl.when(pl.col("v") > 0).then(1).otherwise(0))')
        assert results[0].errors == []

    def test_each_positional_condition_is_checked(self):
        # ``when(p1, p2)`` ANDs its positional predicates (probed).
        results = self._analyze(
            'df.select(x=pl.when(pl.col("flag"), pl.col("a")).then(1).otherwise(0))'
        )
        ply008 = [e for e in results[0].errors if "PLY008" in e]
        assert len(ply008) == 1

    def test_kwarg_equality_constraint_not_flagged(self):
        # ``when(a=1)`` is an equality constraint — boolean by construction.
        results = self._analyze("df.select(x=pl.when(a=1).then(1).otherwise(0))")
        assert not any("PLY008" in e for e in results[0].errors)

    def test_unknown_dtype_condition_not_flagged(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                df2 = df.with_columns(u=pl.col("a").interpolate())
                return df2.select(x=pl.when(pl.col("u")).then(1).otherwise(0))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []


class TestWhenThenOtherwiseInference:
    """Issue #40: when/then/otherwise infers the common supertype of its
    branches instead of Unknown.

    Probed (polars 1.41.2):
      - ``when(c).then(pl.lit(1)).otherwise(pl.lit("x"))`` -> String
      - a null condition row takes the *otherwise* branch
        (``[True, None, False]`` -> ``[10, 20, 20]``, null_count 0), so a
        Nullable condition does NOT make the result nullable
      - ``when(c).then(10)`` without otherwise pads unmatched rows with
        null (``[10, None, None]``) -> Nullable result
      - strings in then/otherwise are *column references*
        (``then("s")`` selects column s — schema shows its dtype)
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                f: pl.Float64
                s: str
                flag: bool
                v: pl.Float64 = pa.Field(nullable=True)
                xs: pl.List(pl.Int64) = pa.Field()
        """
    )

    def _infer(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_same_type_branches_infer_precisely(self):
        ft = self._infer('df.select(x=pl.when(pl.col("a") > 0).then(1).otherwise(0))')
        assert ft.columns["x"].dtype == Int64()

    def test_mixed_int_str_branches_infer_string(self):
        # The #40 repro: polars Schema({'literal': String}).
        ft = self._infer(
            'df.select(x=pl.when(pl.col("a") > 0).then(pl.lit(1)).otherwise(pl.lit("x")))'
        )
        assert ft.columns["x"].dtype == Utf8()

    def test_numeric_branches_promote(self):
        ft = self._infer('df.select(x=pl.when(pl.col("a") > 0).then(2.5).otherwise(pl.col("a")))')
        assert ft.columns["x"].dtype == Float64()

    def test_column_branches_infer_supertype(self):
        ft = self._infer(
            'df.select(x=pl.when(pl.col("flag")).then(pl.col("a")).otherwise(pl.col("s")))'
        )
        assert ft.columns["x"].dtype == Utf8()

    def test_string_branch_is_column_reference(self):
        # ``then("s")`` selects column s (Utf8), it is not a literal.
        ft = self._infer('df.select(x=pl.when(pl.col("flag")).then("f").otherwise(0))')
        assert ft.columns["x"].dtype == Float64()

    def test_no_otherwise_is_nullable(self):
        # Probed: unmatched rows become null.
        ft = self._infer('df.select(x=pl.when(pl.col("a") > 0).then(1))')
        assert ft.columns["x"].dtype == Nullable(Int64())

    def test_null_branch_makes_result_nullable(self):
        ft = self._infer('df.select(x=pl.when(pl.col("a") > 0).then(None).otherwise(0))')
        assert ft.columns["x"].dtype == Nullable(Int64())

    def test_nullable_branch_makes_result_nullable(self):
        ft = self._infer('df.select(x=pl.when(pl.col("a") > 0).then(pl.col("v")).otherwise(0.0))')
        assert ft.columns["x"].dtype == Nullable(Float64())

    def test_nullable_condition_does_not_make_result_nullable(self):
        # Probed: a null condition row takes the otherwise branch — the
        # result has null_count 0 when both branches are non-null.
        ft = self._infer('df.select(x=pl.when(pl.col("v") > 0).then(1).otherwise(0))')
        assert ft.columns["x"].dtype == Int64()

    def test_multi_when_folds_all_branches(self):
        ft = self._infer(
            'df.select(x=pl.when(pl.col("a") > 1).then(1)'
            '.when(pl.col("a") > 0).then(2.5).otherwise(pl.lit("x")))'
        )
        assert ft.columns["x"].dtype == Utf8()

    def test_unknown_branch_stays_unknown(self):
        ft = self._infer(
            'df.select(x=pl.when(pl.col("flag")).then(pl.col("a").interpolate()).otherwise(0))'
        )
        assert ft.columns["x"].dtype == Unknown()

    def test_no_supertype_branches_stay_unknown(self):
        # List + Int64 has no polars supertype; inference stays silent
        # (the runtime error is out of scope for #40).
        ft = self._infer(
            'df.select(x=pl.when(pl.col("flag")).then(pl.col("xs")).otherwise(pl.col("a")))'
        )
        assert ft.columns["x"].dtype == Unknown()

    def test_alias_names_the_output(self):
        ft = self._infer('df.select(pl.when(pl.col("a") > 0).then(1).otherwise(0).alias("y"))')
        assert ft.columns["y"].dtype == Int64()

    def test_when_chain_result_usable_downstream(self):
        ft = self._infer(
            'df.with_columns(x=pl.when(pl.col("a") > 0).then(1).otherwise(0))'
            '.with_columns(y=pl.col("x") + 1)'
        )
        assert ft.columns["x"].dtype == Int64()
        assert ft.columns["y"].dtype == Int64()

    def test_method_chain_after_otherwise(self):
        ft = self._infer(
            'df.select(x=pl.when(pl.col("a") > 0).then(1).otherwise(0).cast(pl.Int32))'
        )
        assert ft.columns["x"].dtype == Int32()

    def test_when_chain_in_agg(self):
        ft = self._infer(
            'df.group_by("s").agg(x=pl.when(pl.col("a") > 0).then(1).otherwise(0).sum())'
        )
        assert ft.columns["x"].dtype == Int64()


class TestShiftFillValue:
    """Issue #43: ``shift(n, fill_value=...)`` plugs the shifted-in slots, so
    the result is NOT wrapped Nullable; the dtype follows the probed
    fill rules (see TestInferShiftFill in test_expr_infer for the matrix).

    Probed (polars 1.41.2):
      - ``[1, 2, 3].shift(1, fill_value=0)`` -> ``[0, 1, 2]``, null_count 0,
        dtype Int64
      - ``[1, None, 3].shift(1, fill_value=0)`` -> ``[0, 1, None]`` (original
        nulls keep flowing)
      - ``shift(1, fill_value="x")`` on Int64 -> String
      - ``fill_value`` is keyword-only (``shift(1, 0)`` is a TypeError), and
        ``fill_value=None`` behaves like no fill
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                s: str
                d: pl.Date
                v: pl.Float64 = pa.Field(nullable=True)
                xs: pl.List(pl.Int64) = pa.Field()
        """
    )

    def _infer(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_same_dtype_fill_is_non_nullable(self):
        # The #43 repro: pl.col("a").shift(1, fill_value=0) is non-null Int64.
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=0))')
        assert ft.columns["a"].dtype == Int64()

    def test_cross_dtype_string_fill_supertypes(self):
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value="x"))')
        assert ft.columns["a"].dtype == Utf8()

    def test_cross_dtype_float_fill_supertypes(self):
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=0.5))')
        assert ft.columns["a"].dtype == Float64()

    def test_int_literal_fill_on_date_keeps_date(self):
        # Probed leniency: shift(Date, fill_value=5) -> Date (1970-01-06).
        ft = self._infer('df.select(pl.col("d").shift(1, fill_value=5))')
        assert ft.columns["d"].dtype == Date()

    def test_pl_lit_fill_counts_as_literal(self):
        # Probed: shift(Date, fill_value=pl.lit(5)) -> Date as well.
        ft = self._infer('df.select(pl.col("d").shift(1, fill_value=pl.lit(5)))')
        assert ft.columns["d"].dtype == Date()

    def test_nullable_receiver_stays_nullable(self):
        # Original nulls keep flowing; only shifted-in slots are filled.
        ft = self._infer('df.select(pl.col("v").shift(1, fill_value=0.0))')
        assert ft.columns["v"].dtype == Nullable(Float64())

    def test_fill_value_none_behaves_like_no_fill(self):
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=None))')
        assert ft.columns["a"].dtype == Nullable(Int64())

    def test_no_fill_value_stays_nullable_regression(self):
        ft = self._infer('df.select(pl.col("a").shift(1))')
        assert ft.columns["a"].dtype == Nullable(Int64())

    def test_expression_fill_follows_supertype(self):
        # Probed via fill_value=pl.col(...).first(): expression fills follow
        # the supertype matrix (Int64 + String -> String).
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=pl.col("s").first()))')
        assert ft.columns["a"].dtype == Utf8()

    def test_expression_fill_without_supertype_keeps_receiver(self):
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=pl.col("xs").first()))')
        assert ft.columns["a"].dtype == Int64()

    # upgrade trigger: the fill expression kind gains inference (e.g. interpolate)
    @pytest.mark.imprecision
    def test_unresolved_fill_keeps_receiver_dtype(self):
        # The fill expression is not modelled — fall back to the receiver
        # dtype (slots are filled with *something*, so no Nullable wrap).
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=pl.col("a").interpolate()))')
        assert ft.columns["a"].dtype == Int64()


class TestUniqueSubsetValidation:
    """Issue #35: ``unique`` validates subset columns like drop_nulls does (PLY014)."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                b: str
        """
    )

    def _analyze(self, body: str, extra_imports: str = ""):
        source = (
            extra_imports
            + self.HEADER
            + textwrap.dedent(
                f"""
            def f(df: DataFrame[In]):
                return {body}
            """
            )
        )
        return analyze_source(source)

    def test_missing_subset_list_kwarg_flags_ply014(self):
        results = self._analyze('df.unique(subset=["ghost"])')
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_missing_subset_string_kwarg_flags_ply014(self):
        results = self._analyze('df.unique(subset="ghost")')
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_missing_subset_positional_list_flags_ply014(self):
        results = self._analyze('df.unique(["a", "ghost"])')
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_missing_subset_positional_string_flags_ply014(self):
        results = self._analyze('df.unique("ghost")')
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_existing_subset_passes(self):
        results = self._analyze('df.unique(subset=["a"])')
        assert results[0].errors == []

    def test_bare_unique_passes(self):
        results = self._analyze("df.unique()")
        assert results[0].errors == []

    def test_local_const_name_subset_resolves(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                cols = ["ghost"]
                return df.unique(subset=cols)
            """
        )
        results = analyze_source(source)
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_module_const_name_subset_resolves(self):
        source = (
            self.HEADER
            + 'COLS = ["ghost"]\n'
            + textwrap.dedent(
                """
            def f(df: DataFrame[In]):
                return df.unique(subset=COLS)
            """
            )
        )
        results = analyze_source(source)
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_local_const_shadows_module_const(self):
        source = (
            self.HEADER
            + 'COLS = ["a"]\n'
            + textwrap.dedent(
                """
            def f(df: DataFrame[In]):
                COLS = ["ghost"]
                return df.unique(subset=COLS)
            """
            )
        )
        results = analyze_source(source)
        assert any("PLY014" in e and "ghost" in e for e in results[0].errors)

    def test_keep_and_maintain_order_kwargs_are_ignored(self):
        results = self._analyze('df.unique(subset=["a"], keep="first", maintain_order=True)')
        assert results[0].errors == []

    def test_selector_subset_passes(self):
        results = self._analyze(
            "df.unique(subset=cs.numeric())", extra_imports="import polars.selectors as cs\n"
        )
        assert results[0].errors == []

    # upgrade trigger: open frames gain row-var bounds (provably-absent columns)
    @pytest.mark.imprecision
    def test_open_frame_missing_subset_not_flagged(self):
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(frame, 'out = df.unique(subset=["ghost"])')
        assert analyzer.errors == []

    def test_unique_stays_identity_typed(self):
        results = self._analyze('df.unique(subset=["a"])')
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})

    def test_lazy_unique_preserves_laziness(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            from pandera.typing.polars import LazyFrame

            class In(pa.DataFrameModel):
                a: int
                b: str

            def f(lf: LazyFrame[In]) -> DataFrame[In]:
                return lf.unique(subset=["a"]).collect()
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "b": Utf8()})


class TestDecimalCastPrecisionScale:
    """Issue #38: ``cast(pl.Decimal(p, s))`` preserves precision/scale.

    Ground truth (polars 1.41.2): ``pl.col("x").cast(pl.Decimal(10, 2))``
    produces ``Decimal(precision=10, scale=2)``; omitted args take polars'
    defaults (precision=38, scale=0).
    """

    def _frame(self) -> FrameType:
        return FrameType({"x": Int64(), "n": Nullable(Int64())})

    def _cast_dtype(self, target: str):
        analyzer = _run_body(self._frame(), f"out = df.select(d=pl.col('x').cast({target}))")
        assert analyzer.errors == [], analyzer.errors
        return analyzer.var_types["out"].columns["d"].dtype

    def test_positional_args_preserved(self):
        assert self._cast_dtype("pl.Decimal(10, 2)") == Decimal(10, 2)

    def test_keyword_args_preserved(self):
        assert self._cast_dtype("pl.Decimal(precision=10, scale=2)") == Decimal(10, 2)

    def test_precision_only_defaults_scale(self):
        assert self._cast_dtype("pl.Decimal(10)") == Decimal(10, 0)

    def test_scale_only_defaults_precision(self):
        assert self._cast_dtype("pl.Decimal(scale=2)") == Decimal(38, 2)

    def test_bare_call_uses_polars_defaults(self):
        assert self._cast_dtype("pl.Decimal()") == Decimal(38, 0)

    def test_bare_attribute_uses_polars_defaults(self):
        assert self._cast_dtype("pl.Decimal") == Decimal(38, 0)

    def test_non_literal_args_degrade_to_unknown(self):
        # ``pl.Decimal(p, s)`` with variable args: claiming the bare default
        # would be a false-positive trap — the cast target is unresolved and
        # the column degrades to Unknown (still registered, never an error).
        analyzer = _run_body(self._frame(), "out = df.select(d=pl.col('x').cast(pl.Decimal(p, s)))")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Unknown()

    def test_nullable_receiver_wrapper_preserved(self):
        analyzer = _run_body(
            self._frame(), "out = df.select(d=pl.col('n').cast(pl.Decimal(10, 2)))"
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Decimal(10, 2))

    def test_frame_level_dict_cast_preserved(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'x': pl.Decimal(10, 2)})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["x"].dtype == Decimal(10, 2)


class TestFrameLiteralConstantValues:
    """Issue #39a: ``pl.DataFrame({"col": VAR})`` resolves constant bindings
    used as column values (a ``str`` / ``list[str]`` constant -> Utf8),
    consistent with the literal-list case (#25)."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                x: int
        """
    )

    def test_module_const_str_list_value_is_utf8(self):
        source = (
            self.HEADER
            + 'NAMES = ["x", "y", "z"]\n'
            + textwrap.dedent(
                """
            def f(df: DataFrame[In]):
                return pl.DataFrame({"step": [1, 2, 3], "name": NAMES})
            """
            )
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"step": Int64(), "name": Utf8()})

    def test_local_const_str_list_value_is_utf8(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                names = ["x", "y"]
                return pl.DataFrame({"name": names})
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"name": Utf8()})

    def test_const_str_value_broadcasts_to_utf8(self):
        source = (
            self.HEADER
            + 'SOURCE = "manual"\n'
            + textwrap.dedent(
                """
            def f(df: DataFrame[In]):
                return pl.DataFrame({"a": [1, 2], "src": SOURCE})
            """
            )
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"a": Int64(), "src": Utf8()})

    def test_local_const_shadows_module_const(self):
        source = (
            self.HEADER
            + 'NAMES = "scalar"\n'
            + textwrap.dedent(
                """
            def f(df: DataFrame[In]):
                NAMES = ["x", "y"]
                return pl.DataFrame({"name": NAMES})
            """
            )
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"name": Utf8()})

    def test_unresolvable_name_value_stays_unknown(self):
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                names = load_names()
                return pl.DataFrame({"name": names})
            """
        )
        results = analyze_source(source)
        assert results[0].errors == []
        assert results[0].inferred_return_type == FrameType({"name": Unknown()})


class TestDuplicateOutputColumns:
    """Issue #36: duplicate output names within one select/with_columns call.

    polars raises DuplicateError (select) / ComputeError (with_columns) at
    runtime when two expressions of the same call produce the same output
    name. Overwriting a pre-existing input column in ``with_columns`` is
    legal (that's its whole point) — only intra-call repeats are flagged.
    Probed against polars 1.41.2.
    """

    def _frame(self) -> FrameType:
        return FrameType({"a": Int64(), "b": Float64()})

    def test_select_alias_collides_with_plain_col(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a"), pl.col("b").alias("a"))')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]
        assert "'a'" in analyzer.errors[0]
        assert "select" in analyzer.errors[0]

    def test_select_duplicate_keeps_last_dtype(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a"), pl.col("b").alias("a"))')
        # Registration keeps the last dtype so downstream stays sane.
        assert analyzer.var_types["out"].columns["a"].dtype == Float64()

    def test_select_duplicate_bare_strings(self):
        analyzer = _run_body(self._frame(), 'out = df.select("a", "a")')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_select_selector_collides_with_string(self):
        analyzer = _run_body(self._frame(), 'out = df.select(cs.all(), "a")')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_select_kwarg_collides_with_positional(self):
        analyzer = _run_body(self._frame(), "out = df.select(pl.col('a'), a=pl.lit(1))")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_select_kwarg_string_rename_collides(self):
        analyzer = _run_body(self._frame(), "out = df.select('a', a='b')")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_select_plural_col_overlaps_single(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "b"), pl.col("a"))')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_select_distinct_outputs_no_error(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a"), pl.col("b").alias("c"))')
        assert analyzer.errors == [], analyzer.errors

    def test_with_columns_intra_call_duplicate(self):
        analyzer = _run_body(
            self._frame(), 'out = df.with_columns(pl.col("a"), pl.col("b").alias("a"))'
        )
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]
        assert "with_columns" in analyzer.errors[0]

    def test_with_columns_overwrite_existing_is_legal(self):
        analyzer = _run_body(self._frame(), "out = df.with_columns(a=pl.lit(1))")
        assert analyzer.errors == [], analyzer.errors

    def test_with_columns_alias_overwrite_existing_is_legal(self):
        analyzer = _run_body(self._frame(), 'out = df.with_columns(pl.col("b").alias("a"))')
        assert analyzer.errors == [], analyzer.errors

    def test_with_columns_selector_collides_with_expr(self):
        analyzer = _run_body(self._frame(), 'out = df.with_columns(cs.all(), pl.col("a"))')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_with_columns_string_collides_with_kwarg(self):
        analyzer = _run_body(self._frame(), "out = df.with_columns('a', a=pl.lit(1))")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]


class TestPluralColExpansion:
    """Issue #42: a plural ``pl.col("a", "b")`` nested inside an expression
    expands the whole expression per column, matching polars semantics
    (probed on 1.41.2: ``select(pl.col("a","b") * 10)`` → columns a, b).
    """

    def _frame(self) -> FrameType:
        return FrameType({"s": Utf8(), "a": Int64(), "b": Float64()})

    def test_select_plural_arithmetic_expands(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "b") * 10)')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        assert list(out.columns.keys()) == ["a", "b"]
        assert out.columns["a"].dtype == Int64()
        assert out.columns["b"].dtype == Float64()

    def test_select_plural_list_form_expands(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col(["a", "b"]) * 10)')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        assert out.columns["a"].dtype == Int64()
        assert out.columns["b"].dtype == Float64()

    def test_select_plural_cast_expands(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "b").cast(pl.Float64))')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        assert out.columns["a"].dtype == Float64()
        assert out.columns["b"].dtype == Float64()

    def test_with_columns_plural_arithmetic_expands(self):
        analyzer = _run_body(self._frame(), 'out = df.with_columns(pl.col("a", "b") / 2)')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        # True division retypes both columns to Float64; "s" survives.
        assert out.columns["a"].dtype == Float64()
        assert out.columns["b"].dtype == Float64()
        assert out.columns["s"].dtype == Utf8()

    def test_agg_plural_sum_expands(self):
        analyzer = _run_body(self._frame(), 'out = df.group_by("s").agg(pl.col("a", "b").sum())')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        assert list(out.columns.keys()) == ["s", "a", "b"]
        assert out.columns["a"].dtype == Int64()
        assert out.columns["b"].dtype == Float64()

    def test_agg_bare_plural_implicit_list(self):
        analyzer = _run_body(self._frame(), 'out = df.group_by("s").agg(pl.col("a", "b"))')
        assert analyzer.errors == [], analyzer.errors
        out = analyzer.var_types["out"]
        # Probed: agg(pl.col("a","b")) collects each column into a list.
        assert out.columns["a"].dtype == ListT(Int64())
        assert out.columns["b"].dtype == ListT(Float64())

    def test_aliased_plural_flags_duplicate_output(self):
        # Probed: select(pl.col("a","b").alias("x")) raises DuplicateError —
        # the expansion produces "x" twice and PLY015 catches it (issue #36).
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "b").alias("x"))')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY015" in analyzer.errors[0]

    def test_missing_column_in_plural_expression_errors(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "missing") * 10)')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY001" in analyzer.errors[0]
        assert "missing" in analyzer.errors[0]

    def test_two_plural_nodes_stay_silent(self):
        # Pairwise plural-x-plural semantics are out of scope — no errors,
        # unchanged (single-name) inference path.
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a", "b") + pl.col("a", "b"))')
        assert analyzer.errors == [], analyzer.errors

    def test_single_col_expression_path_unchanged(self):
        analyzer = _run_body(self._frame(), 'out = df.select(pl.col("a") * 10)')
        assert analyzer.errors == [], analyzer.errors
        assert list(analyzer.var_types["out"].columns.keys()) == ["a"]


class TestListEvalBody:
    """Issue #44: type-check the ``list.eval(...)`` body with ``pl.element()``
    bound to the list's inner dtype. Probed on polars 1.41.2:
    ``eval(pl.element() * 2)`` on List(Int64) → List(Int64);
    ``eval(pl.element() + pl.lit("x"))`` raises InvalidOperationError.
    """

    def _frame(self) -> FrameType:
        return FrameType({"v": ListT(Int64()), "s": ListT(Utf8())})

    def test_valid_eval_arithmetic_dtype(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("v").list.eval(pl.element() * 2))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Int64())

    def test_invalid_eval_body_flags_ply009(self):
        analyzer = _run_body(
            self._frame(),
            'out = df.select(pl.col("v").list.eval(pl.element() + pl.lit("x")))',
        )
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]

    def test_invalid_eval_body_degrades_to_list_unknown(self):
        analyzer = _run_body(
            self._frame(),
            'out = df.select(pl.col("v").list.eval(pl.element() + pl.lit("x")))',
        )
        # The error is the signal; the output column stays registered.
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Unknown())

    def test_eval_cast_changes_element_dtype(self):
        analyzer = _run_body(
            self._frame(),
            'out = df.select(pl.col("v").list.eval(pl.element().cast(pl.Utf8)))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Utf8())

    def test_eval_comparison_yields_list_boolean(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(pl.col("v").list.eval(pl.element() > 1))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Boolean())

    def test_eval_string_concat_on_str_list(self):
        analyzer = _run_body(
            self._frame(),
            'out = df.select(pl.col("s").list.eval(pl.element() + pl.lit("!")))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["s"].dtype == ListT(Utf8())

    def test_element_outside_eval_stays_silent(self):
        # ``pl.element()`` is invalid outside eval at runtime, but flagging
        # it is out of scope — stay silent (no element binding, no errors).
        analyzer = _run_body(self._frame(), "out = df.select(x=pl.element() + 1)")
        assert analyzer.errors == [], analyzer.errors

    def test_outer_nullable_receiver_preserved(self):
        frame = FrameType({"v": Nullable(ListT(Int64()))})
        analyzer = _run_body(frame, 'out = df.select(pl.col("v").list.eval(pl.element() * 2))')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == Nullable(ListT(Int64()))

    def test_nullable_inner_element_propagates(self):
        frame = FrameType({"v": ListT(Nullable(Int64()))})
        analyzer = _run_body(frame, 'out = df.select(pl.col("v").list.eval(pl.element() * 2))')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == ListT(Nullable(Int64()))


class TestResolvePlDtypeDatetime:
    """``_resolve_pl_dtype`` Datetime call forms route through the shared
    ``compat.parse_datetime_call`` (issue #50) — same forms, same results,
    on the analyzer side (cast targets, ``schema=`` dicts)."""

    @staticmethod
    def _resolve(src: str):
        import ast as _ast

        from polypolarism.analyzer import _resolve_pl_dtype

        return _resolve_pl_dtype(_ast.parse(src, mode="eval").body)

    def test_bare_attribute_is_naive(self):
        assert self._resolve("pl.Datetime") == Datetime()

    def test_call_no_args_is_naive(self):
        assert self._resolve("pl.Datetime()") == Datetime()

    def test_positional_tz(self):
        assert self._resolve('pl.Datetime("us", "UTC")') == Datetime(tz="UTC")

    def test_keyword_tz(self):
        assert self._resolve('pl.Datetime(time_zone="Asia/Tokyo")') == Datetime(tz="Asia/Tokyo")

    def test_explicit_none_tz_is_naive(self):
        assert self._resolve('pl.Datetime("us", None)') == Datetime()

    def test_time_unit_is_tracked(self):
        # Issue #66: the unit participates in dtype identity.
        assert self._resolve('pl.Datetime("ms")') == Datetime(unit="ms")
        assert self._resolve('pl.Datetime(time_unit="ns")') == Datetime(unit="ns")

    def test_non_literal_tz_is_unresolved(self):
        # A variable time zone is unknowable; claiming naive would be a
        # false-positive trap now that tz mismatches are flagged.
        assert self._resolve("pl.Datetime(time_zone=tz)") is None

    def test_non_literal_time_unit_is_unresolved(self):
        # Same rule for the unit (issue #66).
        assert self._resolve("pl.Datetime(unit_var)") is None

    def test_duration_call_carries_unit(self):
        # Issue #66: ``pl.Duration("ms")`` call form.
        assert self._resolve('pl.Duration("ms")') == Duration(unit="ms")
        assert self._resolve("pl.Duration()") == Duration()
        assert self._resolve("pl.Duration(unit_var)") is None

    def test_cast_target_carries_tz(self):
        # End-to-end: ``cast(pl.Datetime("us", "UTC"))`` pins the tz-aware
        # dtype on the output column (probed: any tz x tz cast is valid).
        frame = FrameType({"t": Datetime()})
        analyzer = _run_body(
            frame, 'out = df.with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["t"].dtype == Datetime(tz="UTC")


class TestResolvePlDtypeArray:
    """``_resolve_pl_dtype`` resolves ``pl.Array(X, n)`` call forms to the
    Array dtype (issue #53) — cast targets and ``schema=`` dicts."""

    @staticmethod
    def _resolve(src: str):
        import ast as _ast

        from polypolarism.analyzer import _resolve_pl_dtype

        return _resolve_pl_dtype(_ast.parse(src, mode="eval").body)

    def test_call_form_with_width(self):
        assert self._resolve("pl.Array(pl.Int64, 3)") == Array(Int64(), 3)

    def test_call_form_nested_list_element(self):
        assert self._resolve("pl.Array(pl.List(pl.Utf8), 2)") == Array(ListT(Utf8()), 2)

    def test_unresolvable_element_degrades_to_array_unknown(self):
        assert self._resolve("pl.Array(some_var, 3)") == Array(Unknown(), 3)

    def test_bare_attribute_is_unresolved(self):
        # Consistent with bare ``pl.List``: a bare container reference in a
        # cast / schema position stays unresolved (silent).
        assert self._resolve("pl.Array") is None

    def test_list_of_array(self):
        assert self._resolve("pl.List(pl.Array(pl.Int64, 3))") == ListT(Array(Int64(), 3))

    def test_cast_target_array(self):
        # End-to-end: casting a List column to ``pl.Array(...)`` (valid at
        # runtime when the widths match — value-dependent, never flagged).
        frame = FrameType({"v": ListT(Int64())})
        analyzer = _run_body(
            frame, 'out = df.with_columns(pl.col("v").cast(pl.Array(pl.Int64, 3)))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["v"].dtype == Array(Int64(), 3)

    def test_frame_literal_schema_array(self):
        frame = FrameType({"id": Int64()})
        analyzer = _run_body(
            frame,
            'out = pl.DataFrame({"q": [[1, 2]]}, schema={"q": pl.Array(pl.Int64, 2)})',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["q"].dtype == Array(Int64(), 2)


class TestTzMismatchedDatetimeOps:
    """Issue #50: mixing tz-aware and tz-naive Datetimes (or two different
    time zones) in binary operations.

    Ground truth (polars 1.41.2 probe):

    ==========================================  =========================
    operation                                    naive/UTC    UTC/Tokyo
    ==========================================  =========================
    ``concat`` vertical / diagonal               SchemaError  SchemaError
    ``-`` (Datetime - Datetime)                  SchemaError  SchemaError
    ``==`` ``<`` (all six comparison ops)        SchemaError  SchemaError
    join on Datetime keys (incl. ``join_asof``)  SchemaError  SchemaError
    when/then branches (supertype)               SchemaError  SchemaError
    ``is_in``                                    OK           OK
    ``±  Duration``                              OK (tz kept) OK (tz kept)
    Date vs tz-aware Datetime (cmp / - / sup.)   OK           OK
    ==========================================  =========================

    Same-tz pairs keep working everywhere (``tz - tz`` -> Duration etc.,
    pinned in TestArithmeticIncompatibleDtypes).
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "n": Datetime(),
                "utc": Datetime(tz="UTC"),
                "utc2": Datetime(tz="UTC"),
                "tokyo": Datetime(tz="Asia/Tokyo"),
                "d": Date(),
                "du": Duration(),
            }
        )

    # -- arithmetic (PLY009) ---------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('utc') - pl.col('n')",
            "pl.col('n') - pl.col('utc')",
            "pl.col('utc') - pl.col('tokyo')",
        ],
    )
    def test_tz_mismatched_subtraction_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("pl.col('utc') - pl.col('utc2')", Duration()),
            ("pl.col('tokyo') - pl.col('tokyo')", Duration()),
            ("pl.col('n') - pl.col('n')", Duration()),
            # Date vs tz-aware Datetime is probed-valid in either order.
            ("pl.col('d') - pl.col('utc')", Duration()),
            ("pl.col('utc') - pl.col('d')", Duration()),
            # Duration shifts keep the operand's tz.
            ("pl.col('utc') + pl.col('du')", Datetime(tz="UTC")),
            ("pl.col('du') + pl.col('tokyo')", Datetime(tz="Asia/Tokyo")),
            ("pl.col('utc') - pl.col('du')", Datetime(tz="UTC")),
        ],
    )
    def test_tz_compatible_arithmetic_keeps_working(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    # -- comparisons (PLY009) ---------------------------------------------------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('utc') == pl.col('n')",
            "pl.col('n') < pl.col('utc')",
            "pl.col('utc') >= pl.col('tokyo')",
            "pl.col('tokyo') != pl.col('utc')",
        ],
    )
    def test_tz_mismatched_comparison_flags_ply009(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY009" in analyzer.errors[0]

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('utc') == pl.col('utc2')",
            "pl.col('utc') < pl.col('utc2')",
            "pl.col('n') == pl.col('n')",
            # Date vs tz-aware Datetime comparison is probed-valid.
            "pl.col('d') == pl.col('utc')",
            "pl.col('utc') > pl.col('d')",
        ],
    )
    def test_tz_compatible_comparison_keeps_working(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    # -- is_in stays valid across tz (probed OK — do NOT "fix" this) -----------

    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('utc').is_in(pl.col('n'))",
            "pl.col('utc').is_in(pl.col('tokyo'))",
            "pl.col('n').is_in(pl.col('utc'))",
        ],
    )
    def test_is_in_across_tz_stays_silent(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    # -- join keys (PLY010 via JoinError) ---------------------------------------

    def _run_two(self, left: FrameType, right: FrameType, body: str):
        import ast as _ast

        errors: list[str] = []
        analyzer = FunctionBodyAnalyzer({"a": left, "b": right}, errors)
        tree = _ast.parse(textwrap.dedent(body))
        for stmt in tree.body:
            analyzer.visit(stmt)
        return analyzer

    def test_join_on_tz_mismatched_keys_flags_ply010(self) -> None:
        left = FrameType({"t": Datetime(), "x": Int64()})
        right = FrameType({"t": Datetime(tz="UTC"), "y": Int64()})
        analyzer = self._run_two(left, right, "out = a.join(b, on='t', how='inner')")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY010" in analyzer.errors[0]

    def test_join_on_same_tz_keys_passes(self) -> None:
        left = FrameType({"t": Datetime(tz="UTC"), "x": Int64()})
        right = FrameType({"t": Datetime(tz="UTC"), "y": Int64()})
        analyzer = self._run_two(left, right, "out = a.join(b, on='t', how='inner')")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["t"].dtype == Datetime(tz="UTC")

    # -- concat (PLY020 via unify_types) ----------------------------------------

    def test_concat_vertical_tz_mismatch_flags_ply020(self) -> None:
        naive = FrameType({"t": Datetime()})
        utc = FrameType({"t": Datetime(tz="UTC")})
        analyzer = self._run_two(naive, utc, "out = pl.concat([a, b], how='vertical')")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY020" in analyzer.errors[0]

    def test_concat_vertical_same_tz_passes(self) -> None:
        utc = FrameType({"t": Datetime(tz="UTC")})
        utc2 = FrameType({"t": Datetime(tz="UTC")})
        analyzer = self._run_two(utc, utc2, "out = pl.concat([a, b], how='vertical')")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["t"].dtype == Datetime(tz="UTC")


class TestTzAwareDtypeOutputs:
    """Collateral of issue #50: methods that SET a time zone must not keep
    claiming the receiver's (or a naive) Datetime now that tz mismatches
    are flagged — that would manufacture false positives against tz-aware
    declared schemas.

    Probed (polars 1.41.2):
    - ``dt.replace_time_zone("X")`` / ``dt.convert_time_zone("X")`` ->
      ``Datetime[X]`` for naive AND aware receivers;
      ``replace_time_zone(None)`` -> naive.
    - ``str.to_datetime()`` -> naive; ``time_zone="UTC"`` -> ``Datetime[UTC]``;
      a format literal containing ``%z`` -> ``Datetime[UTC]``.
    - ``pl.datetime_range(..., time_zone="UTC", eager=True)`` -> ``Datetime[UTC]``.

    Non-literal time zones / formats are unknowable -> Unknown (silent).
    """

    def _frame(self) -> FrameType:
        return FrameType(
            {
                "n": Datetime(),
                "utc": Datetime(tz="UTC"),
                "nn": Nullable(Datetime()),
                "s": Utf8(),
                "d": Date(),
            }
        )

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("pl.col('n').dt.replace_time_zone('UTC')", Datetime(tz="UTC")),
            ("pl.col('utc').dt.replace_time_zone('Asia/Tokyo')", Datetime(tz="Asia/Tokyo")),
            ("pl.col('utc').dt.replace_time_zone(None)", Datetime()),
            ("pl.col('utc').dt.convert_time_zone('Asia/Tokyo')", Datetime(tz="Asia/Tokyo")),
            ("pl.col('n').dt.convert_time_zone('UTC')", Datetime(tz="UTC")),
            # str.to_datetime tz forms
            ("pl.col('s').str.to_datetime()", Datetime()),
            ("pl.col('s').str.to_datetime('%Y-%m-%d %H:%M:%S')", Datetime()),
            ("pl.col('s').str.to_datetime(time_zone='UTC')", Datetime(tz="UTC")),
            ("pl.col('s').str.to_datetime('%Y-%m-%dT%H:%M:%S%z')", Datetime(tz="UTC")),
            ("pl.col('s').str.to_datetime(format='%Y-%m-%dT%H:%M:%S%z')", Datetime(tz="UTC")),
            # Every chrono offset-directive variant resolves the dtype
            # to Datetime[UTC] — probed (polars 1.41.2), including %#z
            # and the colon forms.
            ("pl.col('s').str.to_datetime('%Y-%m-%dT%H:%M:%S%:z')", Datetime(tz="UTC")),
            ("pl.col('s').str.to_datetime('%Y-%m-%dT%H:%M:%S%::z')", Datetime(tz="UTC")),
            ("pl.col('s').str.to_datetime('%Y-%m-%dT%H:%M:%S%:::z')", Datetime(tz="UTC")),
            ("pl.col('s').str.to_datetime('%Y-%m-%dT%H:%M:%S%#z')", Datetime(tz="UTC")),
        ],
    )
    def test_tz_setting_method_output(self, expr: str, expected) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    # upgrade trigger: types.py Datetime gains a tz wildcard ("aware,
    # tz statically unknown") — a non-literal format/time_zone always
    # yields a Datetime, but with which tz cannot be known statically
    # and Datetime equality/subtyping is exact on tz, so claiming any
    # concrete tz (or naive) would be wrong.
    @pytest.mark.imprecision
    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('s').str.to_datetime(fmtvar)",
            "pl.col('s').str.to_datetime(time_zone=tzvar)",
        ],
    )
    def test_to_datetime_non_literal_args_degrade_to_unknown(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    # upgrade trigger: types.py Datetime gains a tz wildcard ("aware,
    # tz statically unknown") — these always yield SOME Datetime
    # (replace_time_zone(var) may even strip to naive if the variable
    # is None), but Datetime equality/subtyping is exact on tz, so any
    # concrete claim would be a guess.
    @pytest.mark.imprecision
    @pytest.mark.parametrize(
        "expr",
        [
            "pl.col('n').dt.replace_time_zone(tzvar)",
            "pl.col('utc').dt.convert_time_zone(tzvar)",
        ],
    )
    def test_time_zone_non_literal_arg_degrades_to_unknown(self, expr: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    # upgrade trigger: convert_time_zone(None) gains a guaranteed-crash
    # diagnostic — probed (polars 1.41.2): it raises TypeError at
    # expression-construction time ("argument 'time_zone': 'None' is
    # not an instance of 'str'"), so any dtype claim is moot and the
    # silent Unknown is sound but undiagnosed.
    @pytest.mark.imprecision
    def test_convert_time_zone_none_degrades_to_unknown(self) -> None:
        analyzer = _run_body(
            self._frame(), "out = df.select(r=pl.col('utc').dt.convert_time_zone(None))"
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_nullable_receiver_keeps_wrapper(self) -> None:
        analyzer = _run_body(
            self._frame(), "out = df.select(r=pl.col('nn').dt.replace_time_zone('UTC'))"
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Datetime(tz="UTC"))

    # upgrade trigger: .dt.replace_time_zone on non-Datetime receivers gets probed
    @pytest.mark.imprecision
    def test_non_datetime_receiver_keeps_legacy_preserving(self) -> None:
        # ``.dt.replace_time_zone`` on a Date column is outside the probed
        # surface — the legacy dtype-preserving fallback stays (silent).
        analyzer = _run_body(
            self._frame(), "out = df.select(r=pl.col('d').dt.replace_time_zone('UTC'))"
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Date()

    def test_replace_then_subtract_same_tz_passes(self) -> None:
        # End-to-end shape of the realistic false-positive scenario: make
        # both sides UTC, subtract -> Duration, no errors.
        analyzer = _run_body(
            self._frame(),
            "out = df.select(r=pl.col('n').dt.replace_time_zone('UTC') - pl.col('utc'))",
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Duration()

    def test_datetime_range_with_literal_tz(self) -> None:
        analyzer = _run_body(
            FrameType({}),
            "out = pl.DataFrame({'ts': pl.datetime_range(a, b, time_zone='UTC', eager=True)})",
        )
        assert analyzer.var_types["out"].columns["ts"].dtype == Datetime(tz="UTC")

    def test_datetime_range_with_non_literal_tz_is_unknown(self) -> None:
        analyzer = _run_body(
            FrameType({}),
            "out = pl.DataFrame({'ts': pl.datetime_range(a, b, time_zone=tzv, eager=True)})",
        )
        assert analyzer.var_types["out"].columns["ts"].dtype == Unknown()

    def test_datetime_range_without_tz_stays_naive(self) -> None:
        analyzer = _run_body(
            FrameType({}),
            "out = pl.DataFrame({'ts': pl.datetime_range(a, b, eager=True)})",
        )
        assert analyzer.var_types["out"].columns["ts"].dtype == Datetime()


class TestBinNamespace:
    """Issue #51: the ``.bin`` expression namespace.

    Probed (polars 1.41.2):
    - ``.bin`` on a non-Binary receiver raises SchemaError ("expected
      `Binary`") for Int64 AND String alike -> PLY012; only Binary passes.
    - Return dtypes: ``encode("hex"/"base64")`` -> String, ``decode`` ->
      Binary, ``size()`` -> UInt32, ``contains`` / ``starts_with`` /
      ``ends_with`` -> Boolean.
    """

    def _frame(self) -> FrameType:
        from polypolarism.types import Binary

        return FrameType(
            {
                "b": Binary(),
                "nb": Nullable(Binary()),
                "i": Int64(),
                "s": Utf8(),
                "u": Unknown(),
            }
        )

    @pytest.mark.parametrize("col", ["i", "s"])
    def test_bin_on_non_binary_flags_ply012(self, col: str) -> None:
        analyzer = _run_body(self._frame(), f"out = df.select(r=pl.col('{col}').bin.size())")
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY012" in analyzer.errors[0]
        assert "Binary" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_bin_on_unknown_receiver_stays_silent(self) -> None:
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('u').bin.size())")
        assert analyzer.errors == [], analyzer.errors

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("pl.col('b').bin.encode('hex')", Utf8()),
            ("pl.col('b').bin.encode('base64')", Utf8()),
            ("pl.col('b').bin.decode('hex')", None),  # placeholder, fixed below
            ("pl.col('b').bin.size()", UInt32()),
            ("pl.col('b').bin.contains(b'a')", Boolean()),
            ("pl.col('b').bin.starts_with(b'a')", Boolean()),
            ("pl.col('b').bin.ends_with(b'a')", Boolean()),
        ],
    )
    def test_bin_return_dtypes(self, expr: str, expected) -> None:
        from polypolarism.types import Binary

        if expected is None:
            expected = Binary()
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected

    def test_nullable_receiver_keeps_wrapper(self) -> None:
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('nb').bin.size())")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(UInt32())

    def test_unlisted_method_degrades_to_unknown(self) -> None:
        # ``bin.reinterpret`` returns an argument-dependent dtype — not
        # modeled, falls through to Unknown without errors.
        analyzer = _run_body(
            self._frame(), "out = df.select(r=pl.col('b').bin.reinterpret(dtype=pl.Int64))"
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    # -- PLY013 cast layer: Binary is outside the understood set — silent ------

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            # Probed-valid casts resolve precisely.
            ("pl.col('s').cast(pl.Binary)", None),  # Utf8 -> Binary, fixed below
            ("pl.col('i').cast(pl.Binary)", None),
            ("pl.col('b').cast(pl.String)", Utf8()),
            # Binary -> Int64 is a probed runtime error in both strict
            # modes, but Binary is outside the PLY013 category set — it
            # must stay SILENT (false positives are worse), and the cast
            # pins the target dtype as usual.
            ("pl.col('b').cast(pl.Int64)", Int64()),
            ("pl.col('b').cast(pl.Boolean)", Boolean()),
        ],
    )
    def test_binary_casts_stay_silent(self, expr: str, expected) -> None:
        from polypolarism.types import Binary

        if expected is None:
            expected = Binary()
        analyzer = _run_body(self._frame(), f"out = df.select(r={expr})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == expected


class TestBytesLiterals:
    """bytes literals infer Binary (probed: ``pl.lit(b"x")``, frame-literal
    dict values and ``select`` kwarg broadcasts are all Binary on polars
    1.41.2)."""

    def test_select_kwarg_bytes_constant(self) -> None:
        from polypolarism.types import Binary

        analyzer = _run_body(FrameType({"i": Int64()}), "out = df.select(x=b'abc')")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["x"].dtype == Binary()

    def test_pl_lit_bytes(self) -> None:
        from polypolarism.types import Binary

        analyzer = _run_body(FrameType({"i": Int64()}), "out = df.select(x=pl.lit(b'abc'))")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["x"].dtype == Binary()

    def test_frame_literal_bytes_list(self) -> None:
        from polypolarism.types import Binary

        analyzer = _run_body(FrameType({}), "out = pl.DataFrame({'b': [b'x', b'y']})")
        assert analyzer.var_types["out"].columns["b"].dtype == Binary()

    def test_frame_literal_bytes_with_none_is_nullable(self) -> None:
        from polypolarism.types import Binary

        analyzer = _run_body(FrameType({}), "out = pl.DataFrame({'b': [b'x', None]})")
        assert analyzer.var_types["out"].columns["b"].dtype == Nullable(Binary())

    def test_frame_literal_bytes_mixed_with_str_is_unknown(self) -> None:
        analyzer = _run_body(FrameType({}), "out = pl.DataFrame({'b': [b'x', 'y']})")
        assert analyzer.var_types["out"].columns["b"].dtype == Unknown()


class TestNameNamespace:
    """Issue #56: ``.name.*`` manipulates OUTPUT column names (and struct
    FIELD names) — the dtype is never changed.

    Probed on polars 1.41.2: ``prefix`` / ``suffix`` / ``to_uppercase`` /
    ``to_lowercase`` apply to the expression's CURRENT output name (an
    earlier ``.alias`` included); ``keep`` restores the chain's ROOT
    column name, overriding any earlier ``.alias``; a trailing ``.alias``
    after a name transform wins; ``prefix_fields`` / ``suffix_fields``
    rename Struct FIELD names (output name unchanged).
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            import polars.selectors as cs

            class In(pa.DataFrameModel):
                a: int
                b: str
                n: float = pa.Field(nullable=True)
        """
    )

    def _frame(self, body: str, *, expect_errors: bool = False):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        if not expect_errors:
            assert results[0].errors == [], (body, results[0].errors)
        return results[0]

    def _columns(self, body: str):
        ft = self._frame(body).inferred_return_type
        assert ft is not None, body
        return {name: spec.dtype for name, spec in ft.columns.items()}

    # -- single-expression name transforms ------------------------------

    def test_prefix_literal(self):
        cols = self._columns('df.select(pl.col("a").name.prefix("p_"))')
        assert cols == {"p_a": Int64()}

    def test_suffix_on_arithmetic_chain(self):
        cols = self._columns('df.select((pl.col("a") * 2).name.suffix("_x2"))')
        assert cols == {"a_x2": Int64()}

    def test_prefix_applies_to_current_alias(self):
        # Probed: the rename applies to the current name, alias included.
        cols = self._columns('df.select(pl.col("a").alias("zz").name.prefix("p_"))')
        assert cols == {"p_zz": Int64()}

    def test_trailing_alias_wins(self):
        cols = self._columns('df.select(pl.col("a").name.prefix("p_").alias("q"))')
        assert cols == {"q": Int64()}

    def test_keep_restores_root_overriding_alias(self):
        # Probed: keep returns the ROOT column name, not the latest alias.
        cols = self._columns('df.select((pl.col("a") * 2).alias("z").name.keep())')
        assert cols == {"a": Int64()}

    def test_keep_after_prefix(self):
        cols = self._columns('df.select(pl.col("a").name.prefix("p_").name.keep())')
        assert cols == {"a": Int64()}

    def test_to_uppercase(self):
        cols = self._columns('df.select(pl.col("a").name.to_uppercase())')
        assert cols == {"A": Int64()}

    def test_to_lowercase_applies_to_alias(self):
        cols = self._columns('df.select(pl.col("a").alias("ZZ").name.to_lowercase())')
        assert cols == {"zz": Int64()}

    def test_dtype_and_nullability_preserved(self):
        cols = self._columns('df.select(pl.col("n").name.prefix("p_"))')
        assert cols == {"p_n": Nullable(Float64())}

    def test_name_transform_on_namespace_chain(self):
        cols = self._columns('df.select(pl.col("b").str.to_uppercase().name.suffix("_u"))')
        assert cols == {"b_u": Utf8()}

    # -- graceful degradation --------------------------------------------

    def test_non_literal_prefix_opens_the_frame(self):
        # The output name is unknowable — the column cannot be registered,
        # so the result frame opens (rest set) instead of losing it.
        source = self.HEADER + textwrap.dedent(
            """
            P = get_prefix()

            def f(df: DataFrame[In]):
                return df.select("b", pl.col("a").name.prefix(P))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert set(ft.columns) == {"b"}
        assert ft.rest is not None

    def test_name_map_warns_plw004_and_opens_the_frame(self):
        result = self._frame('df.select(pl.col("a").name.map(lambda c: c.upper()))')
        assert any("PLW004" in w for w in result.warnings), result.warnings
        ft = result.inferred_return_type
        assert ft is not None
        assert ft.columns == {}
        assert ft.rest is not None

    def test_map_fields_warns_plw004_dtype_unknown_name_kept(self):
        result = self._frame('df.select(pl.col("a").name.map_fields(lambda c: c.upper()))')
        assert any("PLW004" in w for w in result.warnings), result.warnings
        ft = result.inferred_return_type
        assert ft is not None
        assert {name: spec.dtype for name, spec in ft.columns.items()} == {"a": Unknown()}

    # -- struct field renames --------------------------------------------

    STRUCT_HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class WithStruct(pa.DataFrameModel):
                s: pl.Struct({"x": pl.Int64, "y": pl.Utf8}) = pa.Field()
        """
    )

    def _struct_columns(self, body: str):
        source = self.STRUCT_HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[WithStruct]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], (body, results[0].errors)
        ft = results[0].inferred_return_type
        assert ft is not None, body
        return {name: spec.dtype for name, spec in ft.columns.items()}

    def test_prefix_fields_renames_struct_fields(self):
        cols = self._struct_columns('df.select(pl.col("s").name.prefix_fields("p_"))')
        assert cols == {"s": Struct({"p_x": Int64(), "p_y": Utf8()})}

    def test_suffix_fields_renames_struct_fields(self):
        cols = self._struct_columns('df.select(pl.col("s").name.suffix_fields("_f"))')
        assert cols == {"s": Struct({"x_f": Int64(), "y_f": Utf8()})}

    def test_prefix_fields_unnest_round_trip(self):
        cols = self._struct_columns('df.select(pl.col("s").name.prefix_fields("p_")).unnest("s")')
        assert cols == {"p_x": Int64(), "p_y": Utf8()}

    def test_prefix_fields_non_literal_degrades_to_unknown(self):
        source = self.STRUCT_HEADER + textwrap.dedent(
            """
            P = get_prefix()

            def f(df: DataFrame[WithStruct]):
                return df.select(pl.col("s").name.prefix_fields(P))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Unknown()

    def test_prefix_fields_on_non_struct_degrades_to_unknown(self):
        # Runtime InvalidOperationError, but flagging is out of scope —
        # silent Unknown (false negatives are acceptable).
        cols = self._columns('df.select(pl.col("a").name.prefix_fields("p_"))')
        assert cols == {"a": Unknown()}

    # -- multi-column expansion (selector / plural roots) -----------------

    def test_pl_all_prefix_expands_per_column(self):
        # The issue #56 repro shape.
        cols = self._columns('df.select(pl.all().name.prefix("pre_"))')
        assert cols == {"pre_a": Int64(), "pre_b": Utf8(), "pre_n": Nullable(Float64())}

    def test_cs_selector_suffix_expands_per_column(self):
        cols = self._columns('df.select(cs.numeric().name.suffix("_n"))')
        assert cols == {"a_n": Int64(), "n_n": Nullable(Float64())}

    def test_plural_col_suffix_expands_per_column(self):
        cols = self._columns('df.select(pl.col("a", "b").name.suffix("_s"))')
        assert cols == {"a_s": Int64(), "b_s": Utf8()}

    def test_with_columns_selector_prefix_adds_renamed_copies(self):
        cols = self._columns('df.with_columns(pl.all().name.prefix("c_"))')
        assert cols == {
            "a": Int64(),
            "b": Utf8(),
            "n": Nullable(Float64()),
            "c_a": Int64(),
            "c_b": Utf8(),
            "c_n": Nullable(Float64()),
        }

    def test_with_columns_non_literal_prefix_opens_the_frame(self):
        # Mirrors the select degradation: the added column's name is
        # unknowable, so the result frame opens; existing columns are kept.
        source = self.HEADER + textwrap.dedent(
            """
            P = get_prefix()

            def f(df: DataFrame[In]):
                return df.with_columns(pl.col("a").name.prefix(P))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert set(ft.columns) == {"a", "b", "n"}
        assert ft.rest is not None

    def test_selector_chain_without_name_also_expands(self):
        # The expansion keys on the selector being the chain ROOT, not on
        # ``.name`` specifically — ``pl.all().sum()`` aggregates per column.
        cols = self._columns("df.select(cs.integer().sum())")
        assert cols == {"a": Int64()}

    def test_selector_in_over_key_is_not_expanded(self):
        # A selector in ARGUMENT position must not expand the surrounding
        # expression — ``over(cs.by_name(...))`` is a partition key.
        cols = self._columns('df.select(pl.col("a").sum().over(cs.by_name("b")))')
        assert cols == {"a": Int64()}


class TestUnmodeledMethodWarning:
    """Backlog B-4: an unmodeled method on a precisely-resolved receiver
    silently degrades the dtype to Unknown — PLW007 makes the degradation
    visible so drift against new polars releases stops being silent.

    Probed (polars 1.41.2): ``peak_max`` returns Boolean — a real polars
    method the analyzer does not model, used here as the canonical
    unmodeled call.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                b: str
        """
    )

    def _frame(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        return results[0]

    def test_unmodeled_chain_method_warns_plw007(self):
        result = self._frame('df.select(pl.col("a").peak_max())')
        assert any("PLW007" in w for w in result.warnings), result.warnings

    def test_unmodeled_namespace_method_warns_plw007(self):
        result = self._frame('df.select(pl.col("b").str.not_a_real_method(), pl.col("a"))')
        assert any("PLW007" in w for w in result.warnings), result.warnings

    def test_modeled_methods_do_not_warn(self):
        result = self._frame('df.select(pl.col("a").sum(), pl.col("b").str.to_uppercase())')
        assert not any("PLW007" in w for w in result.warnings), result.warnings

    def test_degraded_receiver_does_not_cascade(self):
        # Only the FIRST unmodeled call warns; the second sees an
        # already-degraded receiver and stays silent (no warning pile-up).
        result = self._frame('df.select(pl.col("a").peak_max().also_not_real())')
        plw007 = [w for w in result.warnings if "PLW007" in w]
        assert len(plw007) == 1, result.warnings

    def test_unknown_receiver_does_not_warn(self):
        # A column read off an OPEN frame is already Unknown — warning on
        # every later call would blame the wrong place. The non-literal
        # name.prefix opens the frame (output name unknowable), so "mystery"
        # resolves to Unknown rather than erroring.
        source = self.HEADER + textwrap.dedent(
            """
            P = get_prefix()

            def f(df: DataFrame[In]):
                return df.select(pl.col("a").name.prefix(P)).select(pl.col("mystery").peak_max())
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        assert not any("PLW007" in w for w in results[0].warnings), results[0].warnings

    def test_cast_right_after_unmodeled_call_retracts_the_warning(self):
        # The explicit cast is exactly the repair PLW007 recommends, so the
        # warning emitted for the receiver chain is retracted.
        result = self._frame('df.select(pl.col("a").peak_max().cast(pl.Int64))')
        assert not any("PLW007" in w for w in result.warnings), result.warnings


class TestAnnotatedAssignmentChecking:
    """ADR-0005 two-direction rule: ``x: DataFrame[A] = expr`` is checked
    against the inferred RHS. Pure narrowing assertions (declared <:
    inferred) warn PLW008 with a ``Schema.validate`` remedy; contradictions
    where neither subtype direction holds are PLY033 errors. The annotation
    still wins for the variable's downstream type in both cases."""

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                v: int
                s: str

            class Halved(pa.DataFrameModel):
                v: int
                half: float

            class WrongDtype(pa.DataFrameModel):
                v: int
                half: str

            class NarrowNonNull(pa.DataFrameModel):
                v: int
                m: int
        """
    )

    def _result(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]) -> DataFrame[In]:
{textwrap.indent(textwrap.dedent(body), "                ")}
            """
        )
        results = analyze_source(source)
        return results[0]

    def test_matching_annotation_is_silent(self):
        result = self._result(
            """
            x: DataFrame[Halved] = df.select("v", half=pl.col("v") / 2)
            return df
            """
        )
        assert not any("PLW008" in w for w in result.warnings), result.warnings
        assert result.errors == [], result.errors

    def test_unrelated_dtype_contradiction_is_error(self):
        # half infers Float64; WrongDtype declares str — neither subtype
        # direction holds, so the re-interpretation is a PLY033 error.
        result = self._result(
            """
            x: DataFrame[WrongDtype] = df.select("v", half=pl.col("v") / 2)
            return df
            """
        )
        ply = [e for e in result.errors if "PLY033" in e]
        assert ply, result.errors
        assert "half" in ply[0]
        assert not any("PLW008" in w for w in result.warnings), result.warnings

    def test_narrowing_assertion_warns(self):
        # Left-join shape: m infers Int64? (nullable); the annotation
        # asserts non-null Int64 — declared <: inferred, a pure narrowing
        # assertion: PLW008, never an error.
        source = self.HEADER + textwrap.dedent(
            """
            class Other(pa.DataFrameModel):
                v: int
                m: int

            def f(df: DataFrame[In], other: DataFrame[Other]) -> DataFrame[In]:
                x: DataFrame[NarrowNonNull] = df.join(other, on="v", how="left").select("v", "m")
                return df
            """
        )
        results = analyze_source(source)
        assert any("PLW008" in w for w in results[0].warnings), results[0].warnings
        assert results[0].errors == [], results[0].errors

    def test_unknown_inference_stays_silent(self):
        # peak_max degrades to Unknown — Unknown-leniency must keep the
        # annotation non-contradictory (the PLW005 pivot workflow shape).
        result = self._result(
            """
            x: DataFrame[NarrowNonNull] = df.select("v", m=pl.col("v").peak_max())
            return df
            """
        )
        assert not any("PLW008" in w for w in result.warnings), result.warnings
        assert not any("PLY033" in e for e in result.errors), result.errors

    def test_annotation_still_wins_downstream(self):
        # After the PLY033, downstream typing follows the annotation:
        # selecting 'half' off x must not raise a column error on top.
        result = self._result(
            """
            x: DataFrame[WrongDtype] = df.select("v", half=pl.col("v") / 2)
            y = x.select("half")
            return df
            """
        )
        non_ply033 = [e for e in result.errors if "PLY033" not in e]
        assert non_ply033 == [], result.errors

    def test_missing_column_over_nonstrict_select_is_narrowing(self):
        # Halved declares 'half' and the select result lacks it, but the
        # result frame is non-strict — absence is not PROVABLE (issue #63),
        # so this is the narrowing class. The strict-frame error case is
        # pinned in TestAnnotationCheckIssues63And64.
        result = self._result(
            """
            x: DataFrame[Halved] = df.select("v")
            return df
            """
        )
        assert not any("PLY033" in e for e in result.errors), result.errors
        assert any("PLW008" in w for w in result.warnings), result.warnings

    def test_laziness_mismatch_is_error(self):
        result = self._result(
            """
            x: DataFrame[Halved] = df.lazy().select("v", half=pl.col("v") / 2)
            return df
            """
        )
        ply = [e for e in result.errors if "PLY033" in e]
        assert ply, result.errors
        assert "Lazy" in ply[0]

    def test_strict_extra_column_is_error(self):
        # The strict annotation says nothing else is present, but 's' is
        # provably there — not narrowable.
        source = self.HEADER + textwrap.dedent(
            """
            class StrictV(pa.DataFrameModel):
                v: int

                class Config:
                    strict = True

            def f(df: DataFrame[In]) -> DataFrame[In]:
                x: DataFrame[StrictV] = df.select("v", "s")
                return df
            """
        )
        results = analyze_source(source)
        assert any("PLY033" in e for e in results[0].errors), results[0].errors

    def test_mixed_narrowing_and_contradiction_reports_both(self):
        # 'extra_typed' is an unrelated re-interpretation (PLY033) while
        # 'm' is a pure narrowing (PLW008) — both surface, separately.
        source = self.HEADER + textwrap.dedent(
            """
            class Other(pa.DataFrameModel):
                v: int
                m: int

            class MixedTarget(pa.DataFrameModel):
                v: int
                m: int
                extra_typed: str

            def f(df: DataFrame[In], other: DataFrame[Other]) -> DataFrame[In]:
                x: DataFrame[MixedTarget] = df.join(other, on="v", how="left").select(
                    "v", "m", extra_typed=pl.col("v") * 2
                )
                return df
            """
        )
        results = analyze_source(source)
        assert any("PLY033" in e and "extra_typed" in e for e in results[0].errors), results[
            0
        ].errors
        assert any("PLW008" in w and "'m'" in w for w in results[0].warnings), results[0].warnings


class TestFloat32ReductionWidth:
    """Select- and agg-context float reductions keep Float32 (backlog N-2).

    Probed (polars 1.41.2): ``mean``/``std``/``var``/``median``/``quantile``
    on a Float32 column return **Float32** in both ``select`` and
    ``group_by().agg()`` contexts. Nullability rules (issue #60) are
    unchanged — only the width follows the receiver.
    """

    def _select(self, schema_field: str, expr: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class In(pa.DataFrameModel):
                g: str
                {schema_field}

            def f(data: DataFrame[In]):
                return data.select({expr})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    def test_select_mean_float32(self):
        ft = self._select("v: pl.Float32", 'pl.col("v").mean().alias("m")')
        assert ft.columns["m"].dtype == Float32()

    def test_select_mean_nullable_float32(self):
        ft = self._select(
            "v: pl.Float32 = pa.Field(nullable=True)",
            'pl.col("v").mean().alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float32())

    def test_select_std_float32_nullable_float32(self):
        ft = self._select("v: pl.Float32", 'pl.col("v").std().alias("s")')
        assert ft.columns["s"].dtype == Nullable(Float32())

    def test_select_std_float32_ddof_zero_non_nullable_float32(self):
        # The ddof=0 refinement (issue #60) must unwrap Float32 results too.
        ft = self._select("v: pl.Float32", 'pl.col("v").std(ddof=0).alias("s")')
        assert ft.columns["s"].dtype == Float32()

    def test_select_var_float32_ddof_zero_positional_non_nullable_float32(self):
        ft = self._select("v: pl.Float32", 'pl.col("v").var(0).alias("s")')
        assert ft.columns["s"].dtype == Float32()

    def test_select_median_float32(self):
        ft = self._select("v: pl.Float32", 'pl.col("v").median().alias("m")')
        assert ft.columns["m"].dtype == Float32()

    def test_select_quantile_float32(self):
        ft = self._select("v: pl.Float32", 'pl.col("v").quantile(0.5).alias("q")')
        assert ft.columns["q"].dtype == Float32()

    def test_groupby_agg_mean_float32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: pl.Float32

            def f(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("v").mean().alias("m"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["m"].dtype == Float32()

    def test_groupby_agg_std_ddof_zero_float32(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                g: str
                v: pl.Float32

            def f(data: DataFrame[In]):
                return data.group_by("g").agg(pl.col("v").std(ddof=0).alias("s"))
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["s"].dtype == Float32()


class TestRollingFloat32Width:
    """The rolling float family keeps the Float32 width (backlog N-2).

    Probed (polars 1.41.2): ``rolling_mean``/``rolling_std``/``rolling_var``/
    ``rolling_median``/``rolling_quantile`` on a Float32 receiver return
    **Float32**. Every other accepted receiver yields Float64 — including
    Float16, which the rolling family widens to Float64 (unlike the
    select-context reductions, which keep Float16).
    """

    def _select(self, schema_field: str, expr: str):
        source = textwrap.dedent(
            PANDERA_HEADER
            + f"""
            class S(pa.DataFrameModel):
                {schema_field}

            def f(data: DataFrame[S]):
                return data.select({expr})
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        return ft

    @pytest.mark.parametrize(
        "call",
        [
            "rolling_mean(window_size=3)",
            "rolling_std(window_size=3)",
            "rolling_var(window_size=3)",
            "rolling_median(window_size=3)",
            "rolling_quantile(quantile=0.5, window_size=3)",
        ],
    )
    def test_rolling_float32_receiver_keeps_float32(self, call: str):
        ft = self._select("v: pl.Float32", f'pl.col("v").{call}.alias("r")')
        assert ft.columns["r"].dtype == Nullable(Float32())

    def test_rolling_mean_float32_total_window_non_nullable_float32(self):
        ft = self._select(
            "v: pl.Float32",
            'pl.col("v").rolling_mean(window_size=3, min_samples=1).alias("m")',
        )
        assert ft.columns["m"].dtype == Float32()

    def test_rolling_std_float32_ddof_zero_total_non_nullable_float32(self):
        ft = self._select(
            "v: pl.Float32",
            'pl.col("v").rolling_std(window_size=3, min_samples=1, ddof=0).alias("s")',
        )
        assert ft.columns["s"].dtype == Float32()

    def test_rolling_mean_nullable_float32_receiver_stays_nullable_float32(self):
        ft = self._select(
            "v: pl.Float32 = pa.Field(nullable=True)",
            'pl.col("v").rolling_mean(window_size=3, min_samples=1).alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float32())

    def test_rolling_mean_float16_receiver_widens_to_float64(self):
        # Probed: Float16 is NOT width-preserved by the rolling family.
        ft = self._select(
            "v: pl.Float16",
            'pl.col("v").rolling_mean(window_size=3).alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float64())

    def test_rolling_mean_float64_receiver_still_float64(self):
        ft = self._select(
            "v: pl.Float64",
            'pl.col("v").rolling_mean(window_size=3).alias("m")',
        )
        assert ft.columns["m"].dtype == Nullable(Float64())


class TestArrayWidthCast:
    """Casting an Array to a different width is PLY013 (backlog C-7).

    Probed (polars 1.41.2): "cannot cast Array to a different width" raises
    in both strict modes; the element-only cast at the SAME width works.
    """

    def _frame(self) -> FrameType:
        return FrameType({"q": Array(Int64(), 3)})

    def test_width_change_cast_flags_ply013(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(r=pl.col("q").cast(pl.Array(pl.Int64, 5)))'
        )
        assert any("PLY013" in e for e in analyzer.errors), analyzer.errors

    def test_same_width_element_cast_passes(self):
        analyzer = _run_body(
            self._frame(), 'out = df.select(r=pl.col("q").cast(pl.Array(pl.Float64, 3)))'
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Array(Float64(), 3)

    def test_unknown_width_cast_stays_silent(self):
        # A width the parser could not resolve is a wildcard — no error.
        analyzer = _run_body(
            FrameType({"q": Array(Int64())}),
            'out = df.select(r=pl.col("q").cast(pl.Array(pl.Int64, 5)))',
        )
        assert analyzer.errors == [], analyzer.errors


class TestUnmodeledFrameMethodWarning:
    """Backlog N-3: an unmodeled FRAME method on a tracked receiver
    silently untracks the variable — every downstream check dies quietly.
    PLW007 fires only for methods probed (polars 1.41.2) to return a
    DataFrame/LazyFrame, because only then does schema tracking silently
    die; terminal methods (``to_dicts``, ``write_*``, ``height``, ...)
    legitimately return non-frames and stay silent, as do unknown names
    (typos, plugin namespaces — conservative).

    Probed (polars 1.41.2): ``interpolate`` exists on both DataFrame and
    LazyFrame, returns a frame, and polypolarism does not model it — the
    canonical unmodeled frame-returning call here.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class In(pa.DataFrameModel):
                a: int
                b: str
        """
    )

    def _returning(self, body: str):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[In]):
                return {body}
            """
        )
        return analyze_source(source)[0]

    def _plw007(self, result):
        return [w for w in result.warnings if "PLW007" in w]

    def test_unmodeled_frame_returning_method_warns_plw007(self):
        result = self._returning("df.interpolate()")
        assert result.errors == [], result.errors
        warnings = self._plw007(result)
        assert len(warnings) == 1, result.warnings
        assert ".interpolate()" in warnings[0], warnings

    def test_terminal_method_does_not_warn(self):
        # ``to_dicts`` returns list[dict] — tracking a frame schema past it
        # is meaningless, so there is nothing to warn about.
        result = self._returning("df.to_dicts()")
        assert result.errors == [], result.errors
        assert self._plw007(result) == [], result.warnings

    def test_unknown_method_name_stays_silent(self):
        # Typos / plugin namespaces are unknowable — stay conservative.
        result = self._returning("df.not_a_real_frame_method()")
        assert self._plw007(result) == [], result.warnings

    def test_modeled_methods_do_not_warn(self):
        result = self._returning('df.select(pl.col("a")).filter(pl.col("a") > 0).head(5)')
        assert result.errors == [], result.errors
        assert self._plw007(result) == [], result.warnings

    def test_untracked_receiver_does_not_warn(self):
        # The degradation happened upstream (unknown variable) — warning on
        # the method call would blame the wrong place.
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]):
                mystery = load_other()
                return mystery.interpolate()
            """
        )
        result = analyze_source(source)[0]
        assert self._plw007(result) == [], result.warnings

    def test_lazy_receiver_uses_the_lazyframe_probe_set(self):
        result = self._returning("df.lazy().interpolate()")
        assert result.errors == [], result.errors
        assert len(self._plw007(result)) == 1, result.warnings

    def test_wrong_side_method_gets_the_eager_lazy_error_only(self):
        # ``transpose`` is eager-only: the lazy receiver already gets a
        # precise PLY030 — piling a "not modeled" warning on top of the
        # error would be noise (the lazy probe set does not contain it).
        result = self._returning("df.lazy().transpose()")
        assert any("PLY030" in e for e in result.errors), result.errors
        assert self._plw007(result) == [], result.warnings

    def test_one_warning_per_chain(self):
        # After the first unmodeled call the variable is untracked; the
        # second call sees no FrameType receiver and stays silent.
        result = self._returning("df.interpolate().fill_null(0)")
        assert len(self._plw007(result)) == 1, result.warnings

    def test_agg_chain_does_not_double_fire(self):
        # ``.group_by(...).agg(...)`` analyzes the grouped receiver twice
        # (once for laziness, once inside _infer_agg_call) — the warning
        # must still fire once per source call.
        result = self._returning('df.interpolate().group_by("a").agg(pl.col("a").sum())')
        assert len(self._plw007(result)) == 1, result.warnings

    def test_validate_wrapping_the_call_retracts_the_warning(self):
        # ``Schema.validate(...)`` immediately retypes the result — exactly
        # the repair PLW007 recommends — so wrapping the unmodeled call
        # retracts the warning (frame-level analog of the expression-level
        # cast retraction).
        result = self._returning("In.validate(df.interpolate())")
        assert result.errors == [], result.errors
        assert self._plw007(result) == [], result.warnings

    def test_validate_on_a_later_statement_keeps_the_warning(self):
        # Between the unmodeled call and the validate the variable really
        # was untracked — the warning at the degradation point stands.
        source = self.HEADER + textwrap.dedent(
            """
            def f(df: DataFrame[In]) -> DataFrame[In]:
                out = df.interpolate()
                return In.validate(out)
            """
        )
        result = analyze_source(source)[0]
        assert result.errors == [], result.errors
        assert len(self._plw007(result)) == 1, result.warnings


class TestAnnotationCheckIssues63And64:
    """Issues #63 / #64: classification refinements at annotation sites.

    #63 — a column missing from a NON-STRICT inferred frame is not
    provably absent (the runtime frame may carry extras the schema
    tolerates): narrowing class, PLW008, never PLY033. Provable absence
    requires a strict inferred frame.

    #64 — coerce leniency is sound at return positions (check_types
    really coerces) but annotations are runtime-inert: a coercible
    declared/inferred difference at an annotation site is an unbacked
    re-type — PLW008 naming coerce, not silence.
    """

    HEADER = textwrap.dedent(
        PANDERA_HEADER
        + """
            class SrcOpen(pa.DataFrameModel):
                a: int

                class Config:
                    strict = False
                    coerce = True

            class WithB(pa.DataFrameModel):
                a: int
                b: str

                class Config:
                    strict = False
                    coerce = True

            class SrcStrict(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True

            class StrOut(pa.DataFrameModel):
                a: str

                class Config:
                    strict = True
                    coerce = True
        """
    )

    def _result(self, body: str, *, param: str = "SrcOpen"):
        source = self.HEADER + textwrap.dedent(
            f"""
            def f(df: DataFrame[{param}]) -> DataFrame[{param}]:
{textwrap.indent(textwrap.dedent(body), "                ")}
            """
        )
        results = analyze_source(source)
        return results[0]

    # -- issue #63 --------------------------------------------------------

    def test_missing_column_over_nonstrict_frame_is_narrowing(self):
        result = self._result(
            """
            x: DataFrame[WithB] = df.filter(pl.col("a") > 0)
            return df
            """
        )
        assert not any("PLY033" in e for e in result.errors), result.errors
        plw = [w for w in result.warnings if "PLW008" in w]
        assert plw, result.warnings
        assert "'b'" in plw[0]

    def test_missing_column_over_strict_frame_is_error(self):
        result = self._result(
            """
            x: DataFrame[WithB] = df
            return df
            """,
            param="SrcStrict",
        )
        assert any("PLY033" in e for e in result.errors), result.errors

    # -- issue #64 --------------------------------------------------------

    def test_coercible_difference_at_annotation_site_warns(self):
        result = self._result(
            """
            y: DataFrame[StrOut] = df.select(a=pl.col("a"))
            return df
            """
        )
        assert not any("PLY033" in e for e in result.errors), result.errors
        plw = [w for w in result.warnings if "PLW008" in w]
        assert plw, result.warnings
        assert "coerce" in plw[0]

    def test_validate_rhs_still_silent(self):
        # Schema.validate(...) retypes WITH runtime backing — no warning.
        result = self._result(
            """
            y = StrOut.validate(df.select(a=pl.col("a")))
            return df
            """
        )
        assert not any("PLW008" in w for w in result.warnings), result.warnings


class TestSmallIntLandmarkReductionContexts:
    """Backlog N-5: select vs grouped context for the widened agg matrix.

    Probed (polars 1.41.2): sub-32-bit ints aggregate fine in both contexts
    (sum/product upcast to Int64, float reductions to Float64); Float16
    keeps its width through every select-context reduction; but
    mean/median/quantile on Float16 and product on UInt128 PANIC in rust
    in every grouped evaluation — ``group_by().agg()`` and ``Expr.over``
    windows alike.
    """

    def _frame(self, dtype) -> FrameType:
        return FrameType({"g": Utf8(), "v": dtype})

    # -- select context now accepts the widened receivers -----------------
    def test_select_mean_int8_is_float64(self):
        analyzer = _run_body(self._frame(Int8()), 'out = df.select(a=pl.col("v").mean())')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == Float64()

    def test_select_sum_uint16_is_int64(self):
        analyzer = _run_body(self._frame(UInt16()), 'out = df.select(a=pl.col("v").sum())')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == Int64()

    def test_select_mean_float16_keeps_width(self):
        analyzer = _run_body(self._frame(Float16()), 'out = df.select(a=pl.col("v").mean())')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == Float16()

    def test_select_product_uint128_keeps_width(self):
        analyzer = _run_body(self._frame(UInt128()), 'out = df.select(a=pl.col("v").product())')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == UInt128()

    def test_pl_shorthand_mean_float16_keeps_width(self):
        # ``pl.mean("v")`` shorthand goes through the same select context.
        analyzer = _run_body(self._frame(Float16()), 'out = df.select(a=pl.mean("v"))')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == Float16()

    # -- group_by().agg() accepts the non-panicking widened cells ---------
    def test_agg_sum_int8_is_int64(self):
        analyzer = _run_body(
            self._frame(Int8()),
            'out = df.group_by("g").agg(pl.col("v").sum().alias("total"))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["total"].dtype == Int64()

    def test_agg_sum_float16_keeps_width(self):
        analyzer = _run_body(
            self._frame(Float16()),
            'out = df.group_by("g").agg(pl.col("v").sum().alias("total"))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["total"].dtype == Float16()

    # -- grouped panic cells are errors ------------------------------------
    def test_agg_mean_float16_errors(self):
        analyzer = _run_body(
            self._frame(Float16()),
            'out = df.group_by("g").agg(pl.col("v").mean().alias("avg"))',
        )
        assert len(analyzer.errors) == 1, analyzer.errors
        err = analyzer.errors[0]
        assert "PLY011" in err and "Float16" in err, err
        assert "panic" in err.lower(), err

    def test_agg_product_uint128_errors(self):
        analyzer = _run_body(
            self._frame(UInt128()),
            'out = df.group_by("g").agg(pl.col("v").product().alias("p"))',
        )
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY011" in analyzer.errors[0], analyzer.errors

    def test_agg_chain_fallback_mean_float16_errors(self):
        # ``mean().cast(...)`` routes through the chain fallback (the
        # expression analyser); the agg context must survive the re-entry.
        analyzer = _run_body(
            self._frame(Float16()),
            'out = df.group_by("g").agg(pl.col("v").mean().cast(pl.Float64).alias("avg"))',
        )
        assert any("panic" in e.lower() for e in analyzer.errors), analyzer.errors

    def test_over_mean_float16_errors(self):
        # Probed (polars 1.41.2): ``mean().over(...)`` panics on Float16
        # exactly like group_by().agg() — windows are grouped evaluation.
        analyzer = _run_body(
            self._frame(Float16()),
            'out = df.select(a=pl.col("v").mean().over("g"))',
        )
        assert any("PLY011" in e and "panic" in e.lower() for e in analyzer.errors), analyzer.errors

    def test_over_product_uint128_errors(self):
        analyzer = _run_body(
            self._frame(UInt128()),
            'out = df.select(a=pl.col("v").product().over("g"))',
        )
        assert any("PLY011" in e and "panic" in e.lower() for e in analyzer.errors), analyzer.errors

    def test_over_mean_float64_stays_clean(self):
        analyzer = _run_body(
            self._frame(Float64()),
            'out = df.select(a=pl.col("v").mean().over("g"))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["a"].dtype == Float64()


class TestOpenFrameSources:
    """ADR-0006: bare frame annotations and ``pl.read_*`` / ``pl.scan_*``
    produce empty OPEN frames — the function is checked for what its body
    itself determines, under the assumption that runtime column lookups
    succeeded (their absence is never provable)."""

    def test_bare_dataframe_param_binds_open_frame(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def helper(df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(doubled=pl.col("x") * 2)
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        analysis = results[0]
        assert analysis.errors == [], analysis.errors
        param = analysis.input_types["df"]
        # Backward narrowing (ADR-0006 amendment): the body's assumed
        # lookup of 'x' pins it into the open param frame.
        assert param.rest is not None
        assert set(param.columns) <= {"x"}
        assert param.is_lazy is False
        inferred = analysis.inferred_return_type
        assert inferred is not None and inferred.rest is not None
        assert "doubled" in inferred.columns

    def test_bare_lazyframe_param_is_lazy(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def helper(lf: pl.LazyFrame) -> pl.DataFrame:
                return lf.collect()
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert results[0].errors == []
        inferred = results[0].inferred_return_type
        assert inferred is not None and inferred.is_lazy is False

    def test_eager_only_method_on_bare_lazyframe_flagged(self):
        # Assignment form — bare expression statements are not inferred
        # (pre-existing behavior for all frames, open or closed).
        source = textwrap.dedent(
            """
            import polars as pl

            def helper(lf: pl.LazyFrame) -> None:
                x = lf.to_pandas()
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert any("PLY030" in str(e) for e in results[0].errors), results[0].errors

    def test_non_polars_prefix_is_not_claimed(self):
        # ``pd.DataFrame`` may be pandas — the function stays unanalyzed.
        source = textwrap.dedent(
            """
            def helper(df: pd.DataFrame):
                return df.select("a")
            """
        )
        assert analyze_source(source) == []

    def test_read_parquet_is_open_eager_frame(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def load() -> pl.DataFrame:
                df = pl.read_parquet("data.parquet")
                return df.with_columns(total=pl.col("a") + pl.col("b"))
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None and inferred.rest is not None
        assert inferred.is_lazy is False
        assert "total" in inferred.columns

    def test_scan_parquet_is_open_lazy_frame(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def load() -> pl.DataFrame:
                return pl.scan_parquet("data.parquet").select("a", "b").collect()
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.is_lazy is False
        assert set(inferred.columns) == {"a", "b"}

    def test_select_closes_open_frame_making_later_miss_provable(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def helper(df: pl.DataFrame) -> pl.DataFrame:
                picked = df.select("a", "b")
                return picked.select(pl.col("c"))
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert any("PLY001" in str(e) and "'c'" in str(e) for e in results[0].errors), results[
            0
        ].errors

    def test_pinned_dtype_contradiction_is_provable(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def helper(df: pl.DataFrame) -> pl.DataFrame:
                tagged = df.with_columns(label=pl.lit("x"))
                return tagged.select(out=pl.col("label") - 1)
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert any("PLY009" in str(e) for e in results[0].errors), results[0].errors


class TestOpenFrameOperations:
    """Operations on open frames must not manufacture proofs (ADR-0006)."""

    def _open(self, **cols) -> FrameType:
        return FrameType(dict(cols), rest=RowVar("src"))

    def test_rename_of_unpinned_column_pins_target(self):
        analyzer = _run_body(self._open(), 'out = df.rename({"a": "b"})')
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert "b" in out.columns and out.rest is not None

    def test_cast_of_unpinned_column_pins_target_dtype(self):
        analyzer = _run_body(self._open(), 'out = df.cast({"a": pl.Int64})')
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.columns["a"].dtype == Int64()

    def test_drop_nulls_unknown_subset_is_silent(self):
        analyzer = _run_body(self._open(), 'out = df.drop_nulls(subset=["a"])')
        assert analyzer.errors == []

    def test_join_key_on_open_side_is_assumed(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Right(pa.DataFrameModel):
                k: int
                v: str

            def helper(df: pl.DataFrame, other: DataFrame[Right]) -> pl.DataFrame:
                return df.join(other, on="k", how="left")
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None and inferred.rest is not None
        # The right column's NAME is pinned, but with an open left frame
        # its dtype is conditional on a left-rest collision (polars would
        # suffix the right column away) — Unknown, not the right dtype
        # (issue #79).
        v_dtype = inferred.columns["v"].dtype
        v_base = v_dtype.inner if isinstance(v_dtype, Nullable) else v_dtype
        assert isinstance(v_base, Unknown), v_dtype

    def test_selector_keeps_select_result_open(self):
        analyzer = _run_body(
            self._open(num=Int64()),
            "out = df.select(cs.numeric())",
        )
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.rest is not None
        assert out.columns["num"].dtype == Int64()

    def test_concat_vertical_open_with_closed_keeps_pins(self):
        from polypolarism.ops.reshape import concat_vertical

        closed = FrameType({"a": Int64(), "b": Utf8()})
        result = concat_vertical([self._open(a=Int64()), closed])
        assert result.rest is not None
        assert result.columns["a"].dtype == Int64()
        # ``b`` is unpinned on the open side — its dtype there is unknowable.
        assert result.columns["b"].dtype == Unknown()

    def test_concat_vertical_pinned_conflict_still_provable(self):
        from polypolarism.ops.reshape import ReshapeError, concat_vertical

        closed = FrameType({"a": Date()})
        with pytest.raises(ReshapeError):
            concat_vertical([self._open(a=Int64()), closed])

    def test_unpivot_on_open_frame_value_is_unknown(self):
        analyzer = _run_body(
            self._open(),
            'out = df.unpivot(index="id", on=["x", "y"])',
        )
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert set(out.columns) == {"id", "variable", "value"}
        assert out.rest is None
        assert out.columns["value"].dtype == Unknown()


class TestPctChangeDtype:
    """Issue #71: ``pct_change()`` divides — it is NOT dtype-preserving.

    Probed (polars 1.41.2):
    - every int width (Int8..Int128, UInt8..UInt128), Boolean, the
      temporals (Date / Datetime any unit or tz / Time / Duration),
      Decimal and Null -> Float64
    - Float16 -> Float16, Float32 -> Float32 (float width preserved)
    - Utf8 is accepted (non-strict cast; all-null Float64)
    - Binary / Categorical / Enum / List / Array raise at runtime
      (InvalidOperationError or ComputeError); Struct ABORTS the process
      (rust crash) — all flagged PLY016
    - the head slot is always null -> the result is always Nullable
    """

    def _run(self, dtype, expr: str = 'pl.col("v").pct_change()'):
        return _run_body(FrameType({"v": dtype}), f"out = df.select(d={expr})")

    @pytest.mark.parametrize(
        "receiver",
        [
            Int8(),
            Int16(),
            Int32(),
            Int64(),
            Int128(),
            UInt8(),
            UInt16(),
            UInt32(),
            UInt64(),
            UInt128(),
        ],
        ids=["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128"],
    )
    def test_int_receiver_is_nullable_float64(self, receiver):
        analyzer = self._run(receiver)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Float64())

    @pytest.mark.parametrize(
        "receiver",
        [Float16(), Float32(), Float64()],
        ids=["f16", "f32", "f64"],
    )
    def test_float_receiver_keeps_width(self, receiver):
        analyzer = self._run(receiver)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(receiver)

    @pytest.mark.parametrize(
        "receiver",
        [
            Boolean(),
            Utf8(),
            Date(),
            Datetime(),
            Datetime(tz="UTC"),
            Datetime(unit="ms"),
            Time(),
            Duration(unit="ms"),
            Decimal(10, 2),
            Null(),
        ],
        ids=["bool", "utf8", "date", "dt", "dt_tz", "dt_ms", "time", "dur_ms", "dec", "null"],
    )
    def test_castable_receiver_is_nullable_float64(self, receiver):
        # polars casts these to Float64 non-strictly and divides (probed;
        # Utf8 yields an all-null Float64 column).
        analyzer = self._run(receiver)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Float64())

    def test_nullable_int_receiver_is_nullable_float64(self):
        analyzer = self._run(Nullable(Int64()))
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Float64())

    @pytest.mark.parametrize(
        "receiver",
        [
            Binary(),
            Categorical(),
            Enum(("a", "b")),
            ListT(Int64()),
            Array(Int64(), 2),
            Struct({"f": Int64()}),
        ],
        ids=["bin", "cat", "enum", "list", "array", "struct"],
    )
    def test_invalid_receiver_flags_ply016(self, receiver):
        analyzer = self._run(receiver)
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY016" in analyzer.errors[0]
        assert "pct_change" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["d"].dtype == Unknown()

    def test_unknown_receiver_stays_silent(self):
        analyzer = self._run(Unknown())
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Unknown())

    def test_shift_regression_keeps_receiver_dtype(self):
        # Regression guard: only pct_change leaves the shift-like family —
        # ``shift`` stays dtype-preserving.
        analyzer = self._run(Int32(), 'pl.col("v").shift(1)')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["d"].dtype == Nullable(Int32())


class TestNotBitwiseDtype:
    """Issue #72: ``not_()`` / ``~`` is bitwise NOT on integers.

    Probed (polars 1.41.2) — documented in the ``Expr.not_`` docstring
    ("operates bitwise on integers"):
    - Boolean -> Boolean (null-preserving: ~null is null)
    - every int width (Int8..Int128, UInt8..UInt128) -> same dtype
    - everything else (floats incl. Float16, Utf8, Binary, temporals,
      Decimal, Categorical, Enum, List, Array, Struct, Null) raises
      InvalidOperationError "dtype X not supported in 'not' operation"
    """

    _INT_RECEIVERS = [
        Int8(),
        Int16(),
        Int32(),
        Int64(),
        Int128(),
        UInt8(),
        UInt16(),
        UInt32(),
        UInt64(),
        UInt128(),
    ]
    _INT_IDS = ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128"]

    def _run(self, dtype, expr: str):
        return _run_body(FrameType({"v": dtype}), f"out = df.select(r={expr})")

    @pytest.mark.parametrize("receiver", _INT_RECEIVERS, ids=_INT_IDS)
    def test_not_method_on_int_preserves_dtype(self, receiver):
        analyzer = self._run(receiver, 'pl.col("v").not_()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == receiver

    @pytest.mark.parametrize("receiver", _INT_RECEIVERS, ids=_INT_IDS)
    def test_invert_operator_on_int_preserves_dtype(self, receiver):
        analyzer = self._run(receiver, '~pl.col("v")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == receiver

    def test_not_method_on_boolean_is_boolean(self):
        analyzer = self._run(Boolean(), 'pl.col("v").not_()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_not_method_on_nullable_boolean_keeps_nullable(self):
        # ~null is null (probed) — the Nullable wrapper carries through.
        analyzer = self._run(Nullable(Boolean()), 'pl.col("v").not_()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Boolean())

    def test_invert_on_nullable_int_keeps_nullable(self):
        analyzer = self._run(Nullable(Int64()), '~pl.col("v")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Nullable(Int64())

    @pytest.mark.parametrize(
        "receiver",
        [
            Float16(),
            Float32(),
            Float64(),
            Utf8(),
            Binary(),
            Date(),
            Datetime(),
            Time(),
            Duration(),
            Decimal(10, 2),
            Categorical(),
            Enum(("a", "b")),
            ListT(Int64()),
            ListT(Boolean()),
            Array(Int64(), 2),
            Struct({"f": Int64()}),
            Null(),
        ],
        ids=[
            "f16",
            "f32",
            "f64",
            "utf8",
            "bin",
            "date",
            "dt",
            "time",
            "dur",
            "dec",
            "cat",
            "enum",
            "list_int",
            "list_bool",
            "array",
            "struct",
            "null",
        ],
    )
    @pytest.mark.parametrize("expr", ['pl.col("v").not_()', '~pl.col("v")'], ids=["not_", "~"])
    def test_invalid_receiver_flags_ply016(self, receiver, expr):
        analyzer = self._run(receiver, expr)
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY016" in analyzer.errors[0]
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_nullable_invalid_receiver_still_flags(self):
        analyzer = self._run(Nullable(Float64()), 'pl.col("v").not_()')
        assert len(analyzer.errors) == 1, analyzer.errors
        assert "PLY016" in analyzer.errors[0]

    @pytest.mark.parametrize("expr", ['pl.col("v").not_()', '~pl.col("v")'], ids=["not_", "~"])
    def test_unknown_receiver_stays_silent(self, expr):
        analyzer = self._run(Unknown(), expr)
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_invert_of_comparison_is_boolean(self):
        # Regression guard: ~(a > 0) stays Boolean.
        analyzer = self._run(Int64(), '~(pl.col("v") > 0)')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()

    def test_other_boolean_predicates_unaffected(self):
        # Regression guard: is_null on a non-Boolean column is still Boolean.
        analyzer = self._run(Int64(), 'pl.col("v").is_null()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["r"].dtype == Boolean()


class TestDtEpochTimeUnit:
    """Issue #73: ``dt.epoch``'s return dtype depends on its ``time_unit``.

    Probed (polars 1.41.2): "ns"/"us"/"ms"/"s" -> Int64; "d" -> Int32; the
    no-arg default is "us" -> Int64. An invalid literal raises ValueError at
    expression-construction time and a non-literal argument is unknowable —
    both degrade to Unknown.
    """

    def _run(self, expr: str, dtype=None):
        return _run_body(
            FrameType({"t": dtype if dtype is not None else Datetime()}),
            f"out = df.select(e={expr})",
        )

    def test_epoch_default_is_int64(self):
        analyzer = self._run('pl.col("t").dt.epoch()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int64()

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_epoch_subsecond_units_are_int64(self, unit):
        analyzer = self._run(f'pl.col("t").dt.epoch("{unit}")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int64()

    def test_epoch_day_is_int32(self):
        analyzer = self._run('pl.col("t").dt.epoch("d")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int32()

    def test_epoch_day_keyword_is_int32(self):
        analyzer = self._run('pl.col("t").dt.epoch(time_unit="d")')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int32()

    def test_epoch_day_on_date_receiver_is_int32(self):
        analyzer = self._run('pl.col("t").dt.epoch("d")', dtype=Date())
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int32()

    def test_epoch_nullable_receiver_wraps_nullable(self):
        analyzer = self._run('pl.col("t").dt.epoch("d")', dtype=Nullable(Datetime()))
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Nullable(Int32())

    def test_epoch_non_literal_unit_degrades_to_unknown(self):
        analyzer = _run_body(
            FrameType({"t": Datetime()}),
            'unit = some_unit\nout = df.select(e=pl.col("t").dt.epoch(unit))',
        )
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Unknown()

    def test_epoch_invalid_literal_degrades_to_unknown(self):
        # epoch("x") raises ValueError before any frame exists — never
        # claim a dtype for it.
        analyzer = self._run('pl.col("t").dt.epoch("x")')
        assert analyzer.var_types["out"].columns["e"].dtype == Unknown()

    def test_epoch_timestamp_regression_is_int64(self):
        # Regression guard: ``dt.timestamp`` keeps its fixed Int64 entry.
        analyzer = self._run('pl.col("t").dt.timestamp()')
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["e"].dtype == Int64()


class TestSchemaDefinitionErrors:
    """Issue #69 (PLY041): a function referencing a schema whose ``Annotated``
    field arity provably crashes pandera is dead on arrival — the deferred
    TypeError fires the first time the schema is used (to_schema / validate /
    @pa.check_types), so the static verdict must be FAIL."""

    HEADER = textwrap.dedent(
        """
        import typing
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Src(pa.DataFrameModel):
            a: int

        class Broken(pa.DataFrameModel):
            v: typing.Annotated[pl.Array, pl.Int64(), 2]
        """
    )

    def _analyze(self, func_src: str):
        source = self.HEADER + textwrap.dedent(func_src)
        results = analyze_source(source)
        assert len(results) == 1, results
        return results[0]

    def test_broken_return_schema_is_flagged(self):
        result = self._analyze(
            """
            def f(df: DataFrame[Src]) -> DataFrame[Broken]:
                return df.select(v=pl.concat_list(pl.col("a")).cast(pl.Array(pl.Int64, 2)))
            """
        )
        ply = [e for e in result.errors if "PLY041" in e]
        assert ply, result.errors
        assert "Broken" in ply[0]
        assert "v" in ply[0]

    def test_broken_param_schema_is_flagged(self):
        result = self._analyze(
            """
            def f(df: DataFrame[Broken]) -> DataFrame[Src]:
                return df.select(a=pl.lit(1))
            """
        )
        assert any("PLY041" in e for e in result.errors), result.errors

    def test_same_broken_schema_reported_once_per_function(self):
        result = self._analyze(
            """
            def f(df: DataFrame[Broken]) -> DataFrame[Broken]:
                x: DataFrame[Broken] = df.select(pl.col("v"))
                return x
            """
        )
        ply = [e for e in result.errors if "PLY041" in e]
        assert len(ply) == 1, result.errors

    def test_body_annotation_only_is_flagged(self):
        result = self._analyze(
            """
            def f(df: DataFrame[Src]) -> DataFrame[Src]:
                x: DataFrame[Broken] = df.select(v=pl.concat_list(pl.col("a")).cast(pl.Array(pl.Int64, 2)))
                return df
            """
        )
        assert any("PLY041" in e for e in result.errors), result.errors

    def test_body_validate_call_is_flagged(self):
        result = self._analyze(
            """
            def f(df: DataFrame[Src]) -> DataFrame[Src]:
                checked = Broken.validate(df)
                return df
            """
        )
        assert any("PLY041" in e for e in result.errors), result.errors

    def test_legal_schema_is_silent(self):
        source = textwrap.dedent(
            """
            import typing
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Src(pa.DataFrameModel):
                a: int

            class Legal(pa.DataFrameModel):
                v: typing.Annotated[pl.Array, pl.Int64(), 2, None]

            def f(df: DataFrame[Src]) -> DataFrame[Legal]:
                return df.select(v=pl.concat_list(pl.col("a")).cast(pl.Array(pl.Int64, 2)))
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert not any("PLY041" in e for e in results[0].errors), results[0].errors


class TestOpenFrameNegativeKnowledge:
    """Issue #78: ``drop`` / ``rename`` create PROVABLE absence on open
    frames — conditional on reaching the next line, exactly the ADR-0006
    conditionality. A later reference to the removed/old name is a
    guaranteed ColumnNotFoundError, so it must flag; reintroducing the
    name clears the mark."""

    def test_use_after_drop_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def use_after_drop(df: pl.DataFrame) -> pl.DataFrame:
                return df.drop("a").select(pl.col("a"))
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert any("'a'" in str(e) for e in results[0].errors), results[0].errors

    def test_use_old_name_after_rename_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def use_old(df: pl.DataFrame) -> pl.DataFrame:
                return df.rename({"a": "b"}).select(pl.col("a"))
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        assert any("'a'" in str(e) for e in results[0].errors), results[0].errors

    def test_new_name_after_rename_is_fine(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def use_new(df: pl.DataFrame) -> pl.DataFrame:
                return df.rename({"a": "b"}).select(pl.col("b"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_with_columns_reintroduction_clears_absence(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def reintroduce(df: pl.DataFrame) -> pl.DataFrame:
                return df.drop("a").with_columns(a=pl.lit(1)).select(pl.col("a") + 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_rename_swap_does_not_mark_reused_target(self):
        # rename({"a": "b", "c": "a"}) — 'a' is renamed away AND reused as
        # a target: it still exists afterwards.
        source = textwrap.dedent(
            """
            import polars as pl

            def swap(df: pl.DataFrame) -> pl.DataFrame:
                return df.rename({"a": "b", "c": "a"}).select(pl.col("a"))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_double_drop_flags_second(self):
        # The second drop("a") is a guaranteed ColumnNotFoundError
        # (polars drop is strict by default).
        source = textwrap.dedent(
            """
            import polars as pl

            def double(df: pl.DataFrame) -> pl.DataFrame:
                return df.drop("a").drop("a")
            """
        )
        results = analyze_source(source)
        assert any("PLY002" in str(e) for e in results[0].errors), results[0].errors

    def test_cast_of_absent_column_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def cast_gone(df: pl.DataFrame) -> pl.DataFrame:
                return df.drop("a").cast({"a": pl.Int64})
            """
        )
        results = analyze_source(source)
        assert any("PLY004" in str(e) for e in results[0].errors), results[0].errors

    def test_declared_return_missing_absent_column_fails(self):
        # Checker side: a declared column that is provably absent from the
        # open inferred frame is a real MissingColumn, not a leniency.
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Out(pa.DataFrameModel):
                a: int

            def gone(df: pl.DataFrame) -> DataFrame[Out]:
                return df.drop("a")
            """
        )
        from polypolarism.checker import check_source

        results = check_source(source)
        assert len(results) == 1
        assert not results[0].passed
        assert any("a" in str(e) and "Missing" in str(e) for e in results[0].errors), results[
            0
        ].errors

    def test_join_key_provably_absent_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Right(pa.DataFrameModel):
                k: int

            def bad_key(df: pl.DataFrame, other: DataFrame[Right]) -> pl.DataFrame:
                return df.drop("k").join(other, on="k")
            """
        )
        results = analyze_source(source)
        assert any("'k'" in str(e) for e in results[0].errors), results[0].errors


class TestOpenLeftJoinCollision:
    """Issue #79: joining a closed right frame onto an OPEN left frame —
    a right-side pin is conditional on no collision in the left rest
    (polars suffixes the RIGHT column away), so its dtype degrades to
    Unknown; collisions with PINNED left columns stay deterministic."""

    def _analyze(self, body: str):
        source = textwrap.dedent(
            f"""
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class KZ(pa.DataFrameModel):
                k: pl.Int64
                z: pl.Int64

            def f(df: pl.DataFrame, g: DataFrame[KZ]) -> pl.DataFrame:
                return {body}
            """
        )
        results = analyze_source(source)
        assert len(results) == 1
        return results[0]

    def test_right_pin_on_open_left_does_not_manufacture_proof(self):
        # The issue's repro: if df happens to carry z: String, pl.col("z")
        # is the LEFT column and .str succeeds — no proof, no error.
        result = self._analyze('df.join(g, on="k").select(pl.col("z").str.to_uppercase())')
        assert result.errors == [], result.errors

    def test_right_column_name_still_pinned_as_unknown(self):
        result = self._analyze('df.join(g, on="k")')
        inferred = result.inferred_return_type
        assert inferred is not None and inferred.rest is not None
        assert "z" in inferred.columns
        z_dtype = inferred.columns["z"].dtype
        base = z_dtype.inner if isinstance(z_dtype, Nullable) else z_dtype
        assert isinstance(base, Unknown), z_dtype

    def test_closed_left_keeps_the_genuine_proof(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class KOnly(pa.DataFrameModel):
                k: pl.Int64

                class Config:
                    strict = True

            class KZ(pa.DataFrameModel):
                k: pl.Int64
                z: pl.Int64

            def f(a: DataFrame[KOnly], g: DataFrame[KZ]) -> pl.DataFrame:
                return a.join(g, on="k").select(pl.col("z").str.to_uppercase())
            """
        )
        results = analyze_source(source)
        assert any("PLY012" in str(e) for e in results[0].errors), results[0].errors

    def test_pinned_left_collision_stays_deterministic(self):
        # The left frame PINS z (via with_columns), so the right z is
        # deterministically suffixed to z_right with its precise dtype.
        result = self._analyze('df.with_columns(z=pl.lit("s")).join(g, on="k")')
        inferred = result.inferred_return_type
        assert inferred is not None
        assert inferred.columns["z"].dtype == Utf8()
        assert inferred.columns["z_right"].dtype == Int64()


class TestOpenReturnSchemaCallSites:
    """Issue #81: a function whose declared return schema is strict=False
    is pandera's "at least these columns" — check_types passes the
    caller's extra columns through (row polymorphism), so the call-site
    result binds as an OPEN frame. strict=True returns stay closed."""

    _HELPER = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class HasPrice(pa.DataFrameModel):
            price: float

            class Config:
                strict = False
                coerce = True

        class HasPriceTotal(pa.DataFrameModel):
            price: float
            total: float

            class Config:
                strict = False
                coerce = True

        class WideSales(pa.DataFrameModel):
            sku: str
            price: float

            class Config:
                strict = True
                coerce = True

        def add_total(df: DataFrame[HasPrice]) -> DataFrame[HasPriceTotal]:
            return df.with_columns(total=pl.col("price") * 1.1)
    """

    def test_nonstrict_call_result_is_open(self):
        source = textwrap.dedent(
            self._HELPER
            + """
        def pipeline(df: DataFrame[WideSales]) -> pl.DataFrame:
            out = add_total(df)
            return out.select(pl.col("sku"), pl.col("price"), pl.col("total"))
        """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "pipeline"]
        assert len(target) == 1
        # The issue's repro: selecting the caller-preserved 'sku' through
        # the helper call must not be a provable miss.
        assert target[0].errors == [], target[0].errors
        inferred = target[0].inferred_return_type
        assert inferred is not None
        assert set(inferred.columns) == {"sku", "price", "total"}

    def test_declared_columns_keep_their_dtypes(self):
        source = textwrap.dedent(
            self._HELPER
            + """
        def declared_precise(df: DataFrame[WideSales]) -> pl.DataFrame:
            out = add_total(df)
            return out.select(pl.col("total") - 1)
        """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "declared_precise"]
        assert target[0].errors == [], target[0].errors

    def test_strict_return_stays_closed(self):
        source = textwrap.dedent(
            """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class StrictOut(pa.DataFrameModel):
            price: float

            class Config:
                strict = True

        class Src(pa.DataFrameModel):
            sku: str
            price: float

        def helper(df: DataFrame[Src]) -> DataFrame[StrictOut]:
            return df.select(pl.col("price"))

        def caller(df: DataFrame[Src]) -> pl.DataFrame:
            return helper(df).select(pl.col("sku"))
        """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "caller"]
        # strict=True return is genuinely closed — 'sku' is a provable miss.
        assert any("PLY001" in str(e) and "'sku'" in str(e) for e in target[0].errors), target[
            0
        ].errors


class TestStrictParamExtraColumns:
    """Issue #82: passing a frame with provable extra columns into a
    strict=True parameter — check_types validates the argument at runtime
    and rejects it, so the call site must flag it. Open-frame extras
    (unprovable) stay lenient."""

    def test_wide_closed_frame_into_strict_param_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class StrictPrice(pa.DataFrameModel):
                price: float

                class Config:
                    strict = True
                    coerce = True

            class Wide(pa.DataFrameModel):
                sku: str
                price: float

                class Config:
                    strict = True
                    coerce = True

            def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
                return df.filter(pl.col("price") > 0)

            def wide_into_strict_param(df: DataFrame[Wide]) -> DataFrame[StrictPrice]:
                return strict_helper(df)
            """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "wide_into_strict_param"]
        assert len(target) == 1
        assert any(
            "extra column" in str(e) and "sku" in str(e) and "strict" in str(e)
            for e in target[0].errors
        ), target[0].errors

    def test_open_frame_unknown_extras_stay_lenient(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class StrictPrice(pa.DataFrameModel):
                price: float

                class Config:
                    strict = True
                    coerce = True

            def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
                return df.filter(pl.col("price") > 0)

            def open_into_strict(df: pl.DataFrame) -> DataFrame[StrictPrice]:
                return strict_helper(df)
            """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "open_into_strict"]
        # The open frame pins nothing — its extras aren't provable.
        assert target[0].errors == [], target[0].errors

    def test_pinned_extra_on_open_frame_is_provable(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class StrictPrice(pa.DataFrameModel):
                price: float

                class Config:
                    strict = True
                    coerce = True

            def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
                return df.filter(pl.col("price") > 0)

            def pinned_extra(df: pl.DataFrame) -> DataFrame[StrictPrice]:
                tagged = df.with_columns(label=pl.lit("x"))
                return strict_helper(tagged)
            """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "pinned_extra"]
        assert any("extra column" in str(e) and "label" in str(e) for e in target[0].errors), (
            target[0].errors
        )


class TestCheckedIslandNonStrictSchemas:
    """Issue #83 (checked-island design): a strict=False declared schema
    is the function's interface — referencing an undeclared column flags
    PLY042 with honest wording (the schema admits caller extras at
    runtime, so it is an undeclared dependency, NOT a provable runtime
    failure like PLY001 on exact frames)."""

    _SCHEMA = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class HasPrice(pa.DataFrameModel):
            price: float

            class Config:
                strict = False
                coerce = True
    """

    def test_undeclared_reference_flags_ply042(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def filter_by_region(df: DataFrame[HasPrice]) -> DataFrame[HasPrice]:
            return df.filter(pl.col("region") != "test")
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "region" in e and "HasPrice" in e for e in errors), errors
        assert not any("PLY001" in e for e in errors), errors

    def test_string_select_flags_ply042(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def pick(df: DataFrame[HasPrice]) -> pl.DataFrame:
            return df.select("region")
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "region" in e for e in errors), errors

    def test_strict_schema_keeps_ply001_proof(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class StrictPrice(pa.DataFrameModel):
                price: float

                class Config:
                    strict = True

            def f(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
                return df.filter(pl.col("region") != "test")
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY001" in e for e in errors), errors

    def test_select_output_is_exact_again(self):
        # Shape-determining calls re-anchor the island: the select output
        # is exact, so a later miss is a genuine PLY001 proof.
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def narrowed(df: DataFrame[HasPrice]) -> pl.DataFrame:
            picked = df.select(pl.col("price"))
            return picked.select(pl.col("region"))
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY001" in e and "region" in e for e in errors), errors

    def test_provenance_survives_with_columns(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def chained(df: DataFrame[HasPrice]) -> pl.DataFrame:
            tagged = df.with_columns(tax=pl.col("price") * 0.1)
            return tagged.select("region")
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "region" in e for e in errors), errors

    def test_validate_narrowing_carries_provenance(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def narrowed(df: pl.DataFrame) -> pl.DataFrame:
            out = HasPrice.validate(df)
            return out.select("region")
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "region" in e for e in errors), errors


class TestOptionalColumnsAtStrictBoundaries:
    """Issue #84 (boundary of #82): a column declared Optional[T]
    (required=False — MAY be absent) is not a provable extra: there are
    runtime inputs without it on which the call succeeds. Only
    required=True pins prove strict-extra violations."""

    _COMMON = """
        import typing
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class StrictPrice(pa.DataFrameModel):
            price: float

            class Config:
                strict = True
                coerce = True

        class OptionalSku(pa.DataFrameModel):
            sku: typing.Optional[str]
            price: float

            class Config:
                strict = True
                coerce = True

        def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
            return df.filter(pl.col("price") > 0)
    """

    def test_optional_extra_into_strict_param_is_lenient(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        def pipeline(df: DataFrame[OptionalSku]) -> DataFrame[StrictPrice]:
            return strict_helper(df)
        """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "pipeline"]
        assert target[0].errors == [], target[0].errors

    def test_required_extra_still_provable(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        class RequiredSku(pa.DataFrameModel):
            sku: str
            price: float

            class Config:
                strict = True

        def pipeline(df: DataFrame[RequiredSku]) -> DataFrame[StrictPrice]:
            return strict_helper(df)
        """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "pipeline"]
        assert any("extra column" in str(e) and "sku" in str(e) for e in target[0].errors), target[
            0
        ].errors

    def test_optional_extra_against_strict_declared_return_is_lenient(self):
        # The checker-side twin: an optional inferred column against a
        # strict declared return is value-dependent, not provable.
        from polypolarism.checker import check_source

        source = textwrap.dedent(
            self._COMMON
            + """
        def passthrough(df: DataFrame[OptionalSku]) -> DataFrame[StrictPrice]:
            return df
        """
        )
        results = check_source(source)
        target = [r for r in results if r.function_name == "passthrough"]
        assert target[0].passed, target[0].errors
        assert any("sku" in note for note in target[0].leniency), target[0].leniency


class TestObjectApiSchemaNarrowing:
    """Backlog C-11: object-API schemas participate in the full pipeline —
    ``schema.validate(df)`` narrowing, checked-island provenance, and
    PLW011 surfacing — exactly like class schemas."""

    def test_validate_narrowing_with_object_schema(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            order_schema = pa.DataFrameSchema(
                {"order_id": pa.Column(int), "amount": pa.Column(float)},
                strict=True,
            )

            def load(df: pl.DataFrame) -> pl.DataFrame:
                out = order_schema.validate(df)
                return out.select(pl.col("amount") - 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_strict_object_schema_makes_misses_provable(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            order_schema = pa.DataFrameSchema({"order_id": pa.Column(int)}, strict=True)

            def load(df: pl.DataFrame) -> pl.DataFrame:
                out = order_schema.validate(df)
                return out.select(pl.col("missing"))
            """
        )
        results = analyze_source(source)
        assert any("PLY001" in str(e) for e in results[0].errors), results[0].errors

    def test_nonstrict_object_schema_is_checked_island(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            order_schema = pa.DataFrameSchema({"order_id": pa.Column(int)})

            def load(df: pl.DataFrame) -> pl.DataFrame:
                out = order_schema.validate(df)
                return out.select(pl.col("undeclared"))
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "order_schema" in e for e in errors), errors

    def test_pipe_validate_with_object_schema(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int)}, strict=True)

            def f(df: pl.DataFrame) -> pl.DataFrame:
                return df.pipe(s.validate).select(pl.col("a") + 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_dtype_violation_against_object_schema(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            s = pa.DataFrameSchema({"a": pa.Column(int)}, strict=True)

            def f(df: pl.DataFrame) -> pl.DataFrame:
                out = s.validate(df)
                return out.select(pl.col("a").str.to_uppercase())
            """
        )
        results = analyze_source(source)
        assert any("PLY012" in str(e) for e in results[0].errors), results[0].errors


class TestOpenStruct:
    """Backlog C-9 (open-struct design): a bare ``pl.Struct`` annotation
    is "some struct, fields unknown" — ``Struct({}, open=True)`` instead
    of ``Unknown``. The struct-ness is provable (probed: pandera's bare
    ``pl.Struct`` validates any struct and rejects non-structs; ``.str``
    on a struct column is a runtime SchemaError), so receiver checks
    fire; field lookups get assumption semantics."""

    _SCHEMA = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class WithMeta(pa.DataFrameModel):
            id: int
            meta: pl.Struct

            class Config:
                strict = True
    """

    def test_str_on_bare_struct_column_is_provable(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[WithMeta]) -> pl.DataFrame:
            return df.select(pl.col("meta").str.to_uppercase())
        """
        )
        results = analyze_source(source)
        assert any("PLY012" in str(e) for e in results[0].errors), results[0].errors

    def test_struct_field_on_open_struct_is_assumed(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[WithMeta]) -> pl.DataFrame:
            return df.select(pl.col("meta").struct.field("city"))
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_struct_field_typo_on_closed_struct_still_provable(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class WithAddr(pa.DataFrameModel):
                addr: pl.Struct({"city": pl.Utf8})

                class Config:
                    strict = True

            def f(df: DataFrame[WithAddr]) -> pl.DataFrame:
                return df.select(pl.col("addr").struct.field("ctiy"))
            """
        )
        results = analyze_source(source)
        assert any("PLY001" in str(e) and "ctiy" in str(e) for e in results[0].errors), results[
            0
        ].errors

    def test_unnest_of_open_struct_opens_the_frame(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[WithMeta]) -> pl.DataFrame:
            return df.unnest("meta").select(pl.col("id"), pl.col("anything"))
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert "anything" in inferred.columns

    def test_rename_fields_on_open_struct_degrades(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[WithMeta]) -> pl.DataFrame:
            return df.select(pl.col("meta").struct.rename_fields(["a"]))
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors


class TestOpenStructVerdict:
    """Checker-side open-struct comparisons (mirrors Array-width / Enum-
    category wildcards): overlapping pins compared, a pin missing from a
    CLOSED other side is a provable mismatch, otherwise leniency."""

    def test_open_inferred_vs_closed_declared_is_lenient(self):
        from polypolarism.checker import _subtype_verdict
        from polypolarism.types import Struct, Utf8

        verdict = _subtype_verdict(Struct({}, open=True), Struct({"city": Utf8()}))
        assert verdict.ok
        assert verdict.reason is not None and "Struct" in verdict.reason

    def test_overlapping_pin_conflict_is_provable(self):
        from polypolarism.checker import _subtype_verdict
        from polypolarism.types import Int64, Struct, Utf8

        verdict = _subtype_verdict(Struct({"city": Int64()}, open=True), Struct({"city": Utf8()}))
        assert not verdict.ok

    def test_extra_pin_vs_closed_declared_is_provable(self):
        # Struct dtypes are exact at runtime: an inferred struct provably
        # carrying a field the closed declared struct lacks cannot match.
        from polypolarism.checker import _subtype_verdict
        from polypolarism.types import Int64, Struct, Utf8

        verdict = _subtype_verdict(
            Struct({"city": Utf8(), "zip": Int64()}, open=True), Struct({"city": Utf8()})
        )
        assert not verdict.ok

    def test_closed_vs_closed_stays_exact(self):
        from polypolarism.checker import _subtype_verdict
        from polypolarism.types import Struct, Utf8

        assert _subtype_verdict(Struct({"city": Utf8()}), Struct({"city": Utf8()})).ok
        assert not _subtype_verdict(Struct({"city": Utf8()}), Struct({"town": Utf8()})).ok


class TestValidateResultBinding:
    """Issue #88: validate RESULTS follow pandera's three strict modes —
    strict=False binds as an open island (extras provably flow through;
    undeclared lookups keep the PLY042 lint), strict='filter' and
    strict=True bind closed (filter's removed-column lookups are PLY001
    proofs)."""

    def test_nonstrict_validate_result_extras_survive(self):
        # The issue's FP repro: validate(strict=False) passes 'b' through,
        # so the declared return is satisfiable — leniency, not Missing.
        from polypolarism.checker import check_source

        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            open_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)})

            class Src(pa.DataFrameModel):
                a: int
                b: str

            class NeedsB(pa.DataFrameModel):
                a: int
                b: str

                class Config:
                    strict = False
                    coerce = True

            def open_obj_extras_survive(df: DataFrame[Src]) -> DataFrame[NeedsB]:
                out = open_schema.validate(df.select(pl.col("a"), pl.col("b")))
                return out
            """
        )
        results = check_source(source)
        target = [r for r in results if r.function_name == "open_obj_extras_survive"]
        assert target[0].passed, target[0].errors

    def test_nonstrict_validate_result_keeps_island_lint(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            open_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)})

            def f(df: pl.DataFrame) -> pl.DataFrame:
                out = open_schema.validate(df)
                return out.select(pl.col("undeclared"))
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY042" in e and "open_schema" in e for e in errors), errors

    def test_filter_removed_column_lookup_is_ply001_proof(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa

            filter_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)}, strict="filter")

            def f(df: pl.DataFrame) -> pl.DataFrame:
                out = filter_schema.validate(df)
                return out.select(pl.col("b"))
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY001" in e and "'b'" in e for e in errors), errors
        assert not any("PLY042" in e for e in errors), errors

    def test_filter_class_config_supported_too(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class FilterSchema(pa.DataFrameModel):
                a: int

                class Config:
                    strict = "filter"

            def f(df: pl.DataFrame) -> pl.DataFrame:
                out = FilterSchema.validate(df)
                return out.select(pl.col("b"))
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("PLY001" in e and "'b'" in e for e in errors), errors


class TestValidateInputChecking:
    """Issue #89: ``Schema.validate(arg)`` checks its INPUT for provable
    incompatibilities — a required column missing from a genuinely exact
    frame, a pinned dtype that coerce cannot repair, a required pinned
    extra against strict=True. Island/open frames stay lenient: upgrading
    a weaker frame is what validate-narrowing is for."""

    _COMMON = """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Src(pa.DataFrameModel):
            a: int

            class Config:
                strict = True
                coerce = True

        class StrictA(pa.DataFrameModel):
            a: str

            class Config:
                strict = True
    """

    def test_provable_dtype_conflict_flags(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        def f(df: DataFrame[Src]) -> pl.DataFrame:
            out = StrictA.validate(df.select(pl.col("a")))
            return out
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("validate" in e and "'a'" in e and "Utf8" in e for e in errors), errors

    def test_coerce_repairable_difference_passes(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Src(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True

            class CoercedStr(pa.DataFrameModel):
                a: str

                class Config:
                    strict = True
                    coerce = True

            def f(df: DataFrame[Src]) -> pl.DataFrame:
                return CoercedStr.validate(df.select(pl.col("a")))
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_provable_missing_column_on_exact_frame_flags(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        class NeedsB(pa.DataFrameModel):
            b: int

        def f(df: DataFrame[Src]) -> pl.DataFrame:
            picked = df.select(pl.col("a"))
            return NeedsB.validate(picked)
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("missing column" in e and "'b'" in e for e in errors), errors

    def test_island_frame_upgrade_stays_lenient(self):
        # Upgrading a non-strict-sourced frame is the validate-narrowing
        # use case — its runtime extras may satisfy the target schema.
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Loose(pa.DataFrameModel):
                a: int

            class Wide(pa.DataFrameModel):
                a: int
                b: str

            def f(df: DataFrame[Loose]) -> pl.DataFrame:
                return Wide.validate(df)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_object_schema_input_checked_too(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        strict_s = pa.DataFrameSchema({"a": pa.Column(str)}, strict=True)

        def f(df: DataFrame[Src]) -> pl.DataFrame:
            return strict_s.validate(df.select(pl.col("a")))
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("validate" in e and "Utf8" in e for e in errors), errors

    def test_pipe_validate_input_checked(self):
        source = textwrap.dedent(
            self._COMMON
            + """
        def f(df: DataFrame[Src]) -> pl.DataFrame:
            return df.select(pl.col("a")).pipe(StrictA.validate)
        """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("validate" in e and "Utf8" in e for e in errors), errors


class TestGroupedAllNullTemporalAggs:
    """Issue #91 (boundary of #85): std/var/sum on Date/Datetime/Time
    SUCCEED in grouped contexts (all-null, receiver dtype — probed on
    1.37.0 through 1.41.2) while raising as whole-frame reductions. The
    grouped flag was a false positive; it is now an accepted
    Nullable(receiver) with a PLW012 "provably all-null" advisory.
    Select-context keeps the PLY011 proof."""

    _SCHEMA = """
        import typing
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame

        class Src(pa.DataFrameModel):
            g: str
            t: typing.Annotated[pl.Datetime, "us", None]

            class Config:
                strict = True
    """

    def test_grouped_std_datetime_accepted_with_warning(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[Src]) -> pl.DataFrame:
            return df.group_by("g").agg(x=pl.col("t").std())
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        assert any("PLW012" in w for w in results[0].warnings), results[0].warnings
        inferred = results[0].inferred_return_type
        assert inferred is not None
        assert inferred.columns["x"].dtype == Nullable(Datetime())

    def test_select_std_datetime_keeps_ply011_proof(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[Src]) -> pl.DataFrame:
            return df.select(pl.col("t").std())
        """
        )
        results = analyze_source(source)
        assert any("PLY011" in str(e) for e in results[0].errors), results[0].errors

    def test_grouped_sum_datetime_direct_form_warns(self):
        source = textwrap.dedent(
            self._SCHEMA
            + """
        def f(df: DataFrame[Src]) -> pl.DataFrame:
            return df.group_by("g").agg(pl.col("t").sum())
        """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        assert any("PLW012" in w for w in results[0].warnings), results[0].warnings

    def test_var_on_duration_still_rejected_both_contexts(self):
        source = textwrap.dedent(
            """
            import typing
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Src(pa.DataFrameModel):
                g: str
                d: typing.Annotated[pl.Duration, "us"]

                class Config:
                    strict = True

            def f(df: DataFrame[Src]) -> pl.DataFrame:
                return df.group_by("g").agg(x=pl.col("d").var())
            """
        )
        results = analyze_source(source)
        assert any("PLY011" in str(e) for e in results[0].errors), results[0].errors


class TestValidateNullabilityNotProof:  # issue #92
    """Issue #92: pandera's nullable check is VALUE-based — a
    Nullable-typed column with no actual nulls passes a non-nullable
    schema. Validating a post-join nullable into a non-null schema is
    exactly the PLW008-prescribed narrowing assertion, so the #89 input
    proof must not claim 'SchemaError on every call' for it. Base-dtype
    conflicts stay proofs."""

    def test_nullable_into_nonnull_validate_passes(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Left(pa.DataFrameModel):
                k: int

                class Config:
                    strict = True

            class Right(pa.DataFrameModel):
                k: int
                v: int

                class Config:
                    strict = True

            class Joined(pa.DataFrameModel):
                k: int
                v: int  # non-nullable: the validate IS the assertion

                class Config:
                    strict = True

            def f(a: DataFrame[Left], b: DataFrame[Right]) -> DataFrame[Joined]:
                out = a.join(b, on="k", how="left")  # v becomes Int64?
                return Joined.validate(out)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_base_dtype_conflict_still_a_proof(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Src(pa.DataFrameModel):
                a: int

                class Config:
                    strict = True

            class WantsStr(pa.DataFrameModel):
                a: str

                class Config:
                    strict = True

            def f(df: DataFrame[Src]) -> pl.DataFrame:
                # Nullable wrapping must not LAUNDER a base conflict.
                shifted = df.select(pl.col("a").shift(1))
                return WantsStr.validate(shifted)
            """
        )
        results = analyze_source(source)
        errors = [str(e) for e in results[0].errors]
        assert any("validate" in e and "Utf8" in e for e in errors), errors


class TestBackwardNarrowing:
    """ADR-0006 future-work: an assumption lookup on an open frame pins
    the column INTO the frame (object identity carries it forward) — if
    line N's ``select("a")`` succeeded, ``df`` provably has ``a`` for
    every later statement, making downstream strict-extra and
    extra-column proofs real."""

    def test_assumed_lookup_pins_for_strict_param_proof(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class StrictPrice(pa.DataFrameModel):
                price: float

                class Config:
                    strict = True
                    coerce = True

            def helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
                return df.filter(pl.col("price") > 0)

            def f(df: pl.DataFrame) -> pl.DataFrame:
                checked = df.filter(pl.col("region") != "x")  # region assumed -> pinned
                return helper(checked)  # region provably present -> strict extra
            """
        )
        results = analyze_source(source)
        target = [r for r in results if r.name == "f"]
        assert any("extra column" in str(e) and "region" in str(e) for e in target[0].errors), (
            target[0].errors
        )

    def test_pin_survives_via_object_identity_alias(self):
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Out(pa.DataFrameModel):
                total: float

                class Config:
                    strict = True
                    coerce = True

            def f(df: pl.DataFrame) -> DataFrame[Out]:
                df2 = df.filter(pl.col("extra").is_not_null())  # pins 'extra'
                return df2.select(total=pl.col("amount").cast(pl.Float64))
            """
        )
        # select closes the frame to {total} — the pin doesn't leak into
        # the shape-determined output.
        results = analyze_source(source)
        target = [r for r in results if r.name == "f"]
        assert target[0].errors == [], target[0].errors

    def test_no_pinning_on_island_frames(self):
        # Island lookups error (PLY042) — nothing is assumed, so nothing
        # pins; the frame's declared shape stays authoritative.
        source = textwrap.dedent(
            """
            import polars as pl
            import pandera.polars as pa
            from pandera.typing.polars import DataFrame

            class Loose(pa.DataFrameModel):
                a: int

            def f(df: DataFrame[Loose]) -> pl.DataFrame:
                return df.filter(pl.col("ghost") > 0)
            """
        )
        results = analyze_source(source)
        assert any("PLY042" in str(e) for e in results[0].errors), results[0].errors


class TestFrameConstructorOpenFallback:
    """ADR-0006 future-work: ``pl.DataFrame(some_var)`` provably builds a
    frame — an OPEN one (untracked lost all downstream checking), while
    the no-args constructor is the provably EMPTY frame."""

    def test_non_literal_data_is_open_frame(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f(rows) -> pl.DataFrame:
                df = pl.DataFrame(rows)
                return df.with_columns(x=pl.col("a") + 1)
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None and inferred.rest is not None
        assert "x" in inferred.columns

    def test_lazyframe_constructor_keeps_laziness(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f(rows) -> pl.DataFrame:
                lf = pl.LazyFrame(rows)
                return lf.collect()
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors
        inferred = results[0].inferred_return_type
        assert inferred is not None and inferred.is_lazy is False

    def test_no_args_constructor_is_provably_empty(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f() -> pl.DataFrame:
                return pl.DataFrame().select(pl.col("a"))
            """
        )
        results = analyze_source(source)
        assert any("PLY001" in str(e) for e in results[0].errors), results[0].errors


class TestBareReturnLaziness:
    """ADR-0006 future-work: a bare ``-> pl.DataFrame`` / ``pl.LazyFrame``
    return annotation makes no schema claim, but the eager/lazy bit is a
    real contract — returning the wrong side is PLY032."""

    def test_returning_lazy_against_bare_dataframe_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f(df: pl.DataFrame) -> pl.DataFrame:
                return df.lazy()
            """
        )
        results = analyze_source(source)
        assert any("PLY032" in str(e) for e in results[0].errors), results[0].errors

    def test_returning_eager_against_bare_lazyframe_flags(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.collect()
            """
        )
        results = analyze_source(source)
        assert any("PLY032" in str(e) for e in results[0].errors), results[0].errors

    def test_matching_sides_pass(self):
        source = textwrap.dedent(
            """
            import polars as pl

            def f(df: pl.DataFrame) -> pl.LazyFrame:
                return df.lazy()
            """
        )
        results = analyze_source(source)
        assert results[0].errors == [], results[0].errors

    def test_uninferable_return_stays_silent(self):
        # The bare annotation makes no schema claim — an uninferable body
        # must NOT produce a could-not-infer error.
        source = textwrap.dedent(
            """
            import polars as pl

            def f(df: pl.DataFrame) -> pl.DataFrame:
                return some_external_thing(df)
            """
        )
        from polypolarism.checker import check_source

        results = check_source(source)
        assert results[0].passed, results[0].errors


class TestMethodQualifiedNames:
    """Methods report class-qualified names (user report 2026-06-12):
    bare names made `Pipeline.process` and `Other.process` print as the
    same `process`, and the flat name->line table misattributed the
    failing one's line number to the other class's method."""

    SOURCE = textwrap.dedent(
        """
        import pandera.polars as pa
        import polars as pl
        from pandera.typing.polars import DataFrame


        class S(pa.DataFrameModel):
            id: int


        class Pipeline:
            def process(self, df: DataFrame[S]) -> DataFrame[S]:
                return df.with_columns(pl.col("missing") * 2)


        class Other:
            def process(self, df: DataFrame[S]) -> DataFrame[S]:
                return df


        def process(df: DataFrame[S]) -> DataFrame[S]:
            return df
        """
    )

    def test_methods_get_class_qualified_names(self):
        analyses = analyze_source(self.SOURCE)
        names = {a.name for a in analyses}
        assert names == {"Pipeline.process", "Other.process", "process"}

    def test_qualified_names_keep_distinct_line_numbers(self):
        analyses = analyze_source(self.SOURCE)
        by_name = {a.name: a for a in analyses}
        assert by_name["Pipeline.process"].lineno < by_name["Other.process"].lineno
        assert by_name["Pipeline.process"].has_errors
        assert not by_name["Other.process"].has_errors

    def test_module_level_function_name_stays_bare(self):
        analyses = analyze_source(self.SOURCE)
        assert any(a.name == "process" and not a.has_errors for a in analyses)


class TestSchemaContextInMessages:
    """Column-not-found diagnostics name the schema the frame came from
    (user request 2026-06-12): with several schemas in play, a bare
    "Column 'x' not found" does not say which contract was violated."""

    def _analyze(self, body: str):
        source = textwrap.dedent(
            """
            import pandera.polars as pa
            import polars as pl
            from pandera.typing.polars import DataFrame


            class Sales(pa.DataFrameModel):
                sku: str
                qty: int

                class Config:
                    strict = True
            """
        ) + textwrap.dedent(body)
        return analyze_source(source)

    def test_ply001_names_schema_on_parameter_frame(self):
        analyses = self._analyze(
            """
            def f(df: DataFrame[Sales]) -> DataFrame[Sales]:
                return df.select(pl.col("missing"))
            """
        )
        (a,) = analyses
        assert any("PLY001" in e and "in frame from schema 'Sales'" in e for e in a.errors), (
            a.errors
        )

    def test_context_survives_column_ops(self):
        analyses = self._analyze(
            """
            def f(df: DataFrame[Sales]) -> pl.DataFrame:
                return df.drop("qty").select(pl.col("zzz"))
            """
        )
        (a,) = analyses
        assert any("in frame from schema 'Sales'" in e for e in a.errors), a.errors

    def test_groupby_key_error_names_schema(self):
        analyses = self._analyze(
            """
            def f(df: DataFrame[Sales]) -> pl.DataFrame:
                return df.group_by("region").agg(pl.col("qty").sum())
            """
        )
        (a,) = analyses
        assert any("region" in e and "schema 'Sales'" in e for e in a.errors), a.errors

    def test_unlabeled_frame_keeps_plain_message(self):
        analyses = self._analyze(
            """
            def f() -> pl.DataFrame:
                return pl.DataFrame({"a": [1]}).select(pl.col("missing"))
            """
        )
        (a,) = analyses
        assert any("Column 'missing' not found" in e and "schema" not in e for e in a.errors), (
            a.errors
        )


class TestInferenceTrace:
    """--verbose inference trace (user request 2026-06-12): analyses can
    record the step-by-step frame transformations so the CLI can show
    HOW polypolarism arrived at its verdict."""

    SOURCE = textwrap.dedent(
        """
        import pandera.polars as pa
        import polars as pl
        from pandera.typing.polars import DataFrame


        class Sales(pa.DataFrameModel):
            sku: str
            qty: int

            class Config:
                strict = True


        class Out(pa.DataFrameModel):
            sku: str
            total: pl.Int64

            class Config:
                strict = True


        def f(df: DataFrame[Sales]) -> DataFrame[Out]:
            widened = df.with_columns(doubled=pl.col("qty") * 2)
            return widened.group_by("sku").agg(total=pl.col("qty").sum())
        """
    )

    def test_trace_disabled_by_default(self):
        (a,) = analyze_source(self.SOURCE)
        assert a.trace == []

    def test_trace_records_param_chain_and_return(self):
        (a,) = analyze_source(self.SOURCE, collect_trace=True)
        labels = [e.label for e in a.trace]
        assert any("param df" in label for label in labels)
        assert any("with_columns" in label for label in labels)
        assert any("agg" in label for label in labels)
        assert any(label == "return" for label in labels)

    def test_trace_renders_frames_compactly(self):
        (a,) = analyze_source(self.SOURCE, collect_trace=True)
        by_label = {e.label: e.result for e in a.trace}
        assert by_label["param df"] == "DataFrame{sku: Utf8, qty: Int64} (strict, schema=Sales)"
        assert "doubled: Int64" in next(e.result for e in a.trace if "with_columns" in e.label)
        assert by_label["return"] == "DataFrame{sku: Utf8, total: Int64}"

    def test_trace_events_carry_line_numbers(self):
        (a,) = analyze_source(self.SOURCE, collect_trace=True)
        chain = next(e for e in a.trace if "with_columns" in e.label)
        assert chain.lineno > 0


class TestOutOfFunctionScopes:
    """Issue #110: provable missing-column / dtype errors are flagged outside a
    frame-typed function signature when the receiver schema is statically known
    (module top level, ``if __name__`` guard, frame-untyped functions), while
    open / unknown receivers stay silent."""

    HEADER = textwrap.dedent(
        """
        import polars as pl
        import pandera.polars as pa
        from pandera.typing.polars import DataFrame


        class KV(pa.DataFrameModel):
            k: str
            v: float

            class Config:
                strict = True


        class Loose(pa.DataFrameModel):
            k: str


        def get_kv() -> DataFrame[KV]:
            return pl.DataFrame({"k": ["a"], "v": [1.0]})


        def get_loose() -> DataFrame[Loose]:
            return pl.DataFrame({"k": ["a"]})
        """
    )

    def _named(self, results):
        return {r.name: r for r in results}

    def test_module_top_level_missing_column_flagged(self):
        source = self.HEADER + textwrap.dedent(
            """
            src = get_kv()
            boom = src.select(pl.col("nope_module"))
            """
        )
        by_name = self._named(analyze_source(source))
        assert "<module>" in by_name
        assert any("nope_module" in e for e in by_name["<module>"].errors)

    def test_if_name_guard_missing_column_flagged(self):
        source = self.HEADER + textwrap.dedent(
            """
            src = get_kv()

            if __name__ == "__main__":
                boom = src.select(pl.col("nope_guard"))
            """
        )
        by_name = self._named(analyze_source(source))
        assert "<module>" in by_name
        assert any("nope_guard" in e for e in by_name["<module>"].errors)

    def test_frame_untyped_function_missing_column_flagged(self):
        source = self.HEADER + textwrap.dedent(
            """
            def main() -> None:
                df = get_kv()
                boom = df.select(pl.col("nope_main"))
            """
        )
        by_name = self._named(analyze_source(source))
        assert "main" in by_name
        assert any("nope_main" in e for e in by_name["main"].errors)

    def test_valid_reference_on_known_frame_is_silent(self):
        source = self.HEADER + textwrap.dedent(
            """
            src = get_kv()
            fine = src.select(pl.col("k"), pl.col("v"))
            """
        )
        by_name = self._named(analyze_source(source))
        # No <module> entry is produced when nothing flags.
        assert "<module>" not in by_name

    def test_open_frame_missing_column_stays_silent(self):
        """A non-strict schema's call result is an OPEN frame: a 'missing'
        reference is not provable, so it must NOT be flagged (soundness)."""
        source = self.HEADER + textwrap.dedent(
            """
            loose = get_loose()
            maybe = loose.select(pl.col("not_declared"))
            """
        )
        by_name = self._named(analyze_source(source))
        assert "<module>" not in by_name

    def test_unpinnable_local_stays_silent(self):
        """A local bound from an unknown call has no static schema — silent."""
        source = self.HEADER + textwrap.dedent(
            """
            mystery = some_unknown_loader()
            anything = mystery.select(pl.col("whatever"))
            """
        )
        by_name = self._named(analyze_source(source))
        assert "<module>" not in by_name

    def test_typed_function_not_double_reported(self):
        """The module pass must not re-analyze typed function bodies (no
        spilling into nested defs)."""
        source = self.HEADER + textwrap.dedent(
            """
            def transform(df: DataFrame[KV]) -> DataFrame[KV]:
                return df.select(pl.col("nope_typed"), pl.col("k"), pl.col("v"))

            src = get_kv()
            ok = src.select(pl.col("k"), pl.col("v"))
            """
        )
        results = analyze_source(source)
        nope_typed_hits = [r.name for r in results if any("nope_typed" in e for e in r.errors)]
        # Reported exactly once, under the typed function — not also <module>.
        assert nope_typed_hits == ["transform"]

    def test_mutually_recursive_untyped_helpers_do_not_recurse_forever(self):
        """The out-of-function passes inline untyped calls to infer return
        types; a mutually recursive untyped pair must not blow the stack
        (recursion guard on FunctionRegistry)."""
        source = textwrap.dedent(
            """
            import polars as pl


            def ping(x):
                return pong(x)


            def pong(x):
                return ping(x)


            def main() -> None:
                y = ping(1)
            """
        )
        # Must simply return without raising RecursionError.
        results = analyze_source(source)
        assert isinstance(results, list)
