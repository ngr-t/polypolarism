"""Tests for AST analyzer."""

import pytest
import textwrap

from polypolarism.types import (
    FrameType,
    Int64,
    Float64,
    Utf8,
    UInt32,
    Nullable,
)
from polypolarism.analyzer import (
    analyze_source,
    FunctionAnalysis,
    AnalysisError,
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
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"

    def test_extracts_input_frame_types(self):
        """Extract input FrameType from parameter annotations."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        assert "data" in results[0].input_types
        input_type = results[0].input_types["data"]
        assert input_type == FrameType({"id": Int64(), "name": Utf8()})

    def test_extracts_return_frame_type(self):
        """Extract return FrameType from annotation."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int
                name: str

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        expected = FrameType({"id": Int64(), "name": Utf8()})
        assert results[0].declared_return_type == expected

    def test_multiple_input_parameters(self):
        """Handle multiple DataFrame parameters."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        assert len(results[0].input_types) == 2
        assert "left" in results[0].input_types
        assert "right" in results[0].input_types

    def test_ignores_functions_without_df_annotations(self):
        """Ignore functions that don't use DataFrame types."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int

            def helper(x: int) -> int:
                return x + 1

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"


class TestAnalyzeJoinOperations:
    """Test join operation analysis."""

    def test_infers_inner_join_result(self):
        """Infer result type of inner join."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "user_id": Int64(),
            "name": Utf8(),
            "order_id": Int64(),
            "amount": Float64(),
        })
        assert results[0].inferred_return_type == expected

    def test_infers_left_join_with_nullable(self):
        """Left join makes right columns nullable."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "user_id": Int64(),
            "name": Utf8(),
            "amount": Nullable(Float64()),
        })
        assert results[0].inferred_return_type == expected

    def test_detects_join_key_missing_error(self):
        """Detect when join key is missing from right frame."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("user_id" in str(e) for e in results[0].errors)


class TestAnalyzeGroupByOperations:
    """Test group_by().agg() operation analysis."""

    def test_infers_groupby_agg_result(self):
        """Infer result type of group_by().agg()."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "category": Utf8(),
            "total": Float64(),
        })
        assert results[0].inferred_return_type == expected

    def test_infers_count_returns_uint32(self):
        """count() aggregation returns UInt32."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "category": Utf8(),
            "count": UInt32(),
        })
        assert results[0].inferred_return_type == expected

    def test_detects_groupby_nonexistent_column(self):
        """Detect when group_by uses non-existent column."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("category" in str(e) for e in results[0].errors)


class TestAnalyzeChainedOperations:
    """Test chained DataFrame operations."""

    def test_infers_join_then_groupby(self):
        """Infer result of join followed by group_by."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "region": Utf8(),
            "total_sales": Float64(),
        })
        assert results[0].inferred_return_type == expected


class TestAnalyzeSelectOperations:
    """Test select operation analysis."""

    def test_infers_select_with_col(self):
        """Infer result of select with pl.col()."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "id": Int64(),
            "value": Float64(),
        })
        assert results[0].inferred_return_type == expected

    def test_detects_select_nonexistent_column(self):
        """Detect when select references non-existent column."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("amount" in str(e) for e in results[0].errors)


class TestAnalyzeWithColumn:
    """Test with_columns operation analysis."""

    def test_infers_with_columns_adds_column(self):
        """with_columns adds new column to existing ones."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "id": Int64(),
            "value": Float64(),
            "doubled": Float64(),
        })
        assert results[0].inferred_return_type == expected


class TestFunctionAnalysisDataClass:
    """Test FunctionAnalysis data class."""

    def test_has_errors_property(self):
        """has_errors returns True when errors exist."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class In(pa.DataFrameModel):
                id: int

            class Out(pa.DataFrameModel):
                id: int
                value: pl.Float64

            def bad_col(
                data: DataFrame[In],
            ) -> DataFrame[Out]:
                return data.select(pl.col("missing"))
        ''')

        results = analyze_source(source)

        assert results[0].has_errors is True

    def test_has_errors_false_when_valid(self):
        """has_errors returns False when no errors."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int

            def identity(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        assert results[0].has_errors is False


class TestAnalyzeIntermediateVariables:
    """Test analysis with intermediate variable assignments."""

    def test_tracks_variable_assignment(self):
        """Track DataFrame type through variable assignment."""
        source = textwrap.dedent(PANDERA_HEADER + '''
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
        ''')

        results = analyze_source(source)

        expected = FrameType({
            "user_id": Int64(),
            "name": Utf8(),
            "order_id": Int64(),
            "amount": Float64(),
        })
        assert results[0].inferred_return_type == expected


class TestSourceLocation:
    """Test source location information in analysis results."""

    def test_function_analysis_has_lineno(self):
        """FunctionAnalysis should include function definition line number."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class S(pa.DataFrameModel):
                id: int

            def process(
                data: DataFrame[S],
            ) -> DataFrame[S]:
                return data
        ''')

        results = analyze_source(source)

        assert len(results) == 1
        # Function def line is determined by the schema preamble.
        assert results[0].lineno > 0

    def test_multiple_functions_have_correct_linenos(self):
        """Multiple functions should each have their correct line numbers."""
        source = textwrap.dedent(PANDERA_HEADER + '''
            class A(pa.DataFrameModel):
                id: int

            class B(pa.DataFrameModel):
                name: str

            def first(df: DataFrame[A]) -> DataFrame[A]:
                return df

            def second(df: DataFrame[B]) -> DataFrame[B]:
                return df
        ''')

        results = analyze_source(source)

        assert len(results) == 2
        results.sort(key=lambda r: r.lineno)
        assert results[0].name == "first"
        assert results[1].name == "second"
        # Second function comes after the first
        assert results[1].lineno > results[0].lineno
