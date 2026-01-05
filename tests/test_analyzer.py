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


class TestAnalyzeSourceBasic:
    """Test basic source code analysis."""

    def test_finds_function_with_df_annotation(self):
        """Find function with DF type annotations."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def process(
                data: DF["{id: Int64, name: Utf8}"],
            ) -> DF["{id: Int64, name: Utf8}"]:
                return data
        ''')

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"

    def test_extracts_input_frame_types(self):
        """Extract input FrameType from parameter annotations."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def process(
                data: DF["{id: Int64, name: Utf8}"],
            ) -> DF["{id: Int64, name: Utf8}"]:
                return data
        ''')

        results = analyze_source(source)

        assert "data" in results[0].input_types
        input_type = results[0].input_types["data"]
        assert input_type == FrameType({"id": Int64(), "name": Utf8()})

    def test_extracts_return_frame_type(self):
        """Extract return FrameType from annotation."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def process(
                data: DF["{id: Int64, name: Utf8}"],
            ) -> DF["{id: Int64, name: Utf8}"]:
                return data
        ''')

        results = analyze_source(source)

        expected = FrameType({"id": Int64(), "name": Utf8()})
        assert results[0].declared_return_type == expected

    def test_multiple_input_parameters(self):
        """Handle multiple DF parameters."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def merge(
                left: DF["{id: Int64, value: Float64}"],
                right: DF["{id: Int64, name: Utf8}"],
            ) -> DF["{id: Int64, value: Float64, name: Utf8}"]:
                return left.join(right, on="id")
        ''')

        results = analyze_source(source)

        assert len(results[0].input_types) == 2
        assert "left" in results[0].input_types
        assert "right" in results[0].input_types

    def test_ignores_functions_without_df_annotations(self):
        """Ignore functions that don't use DF types."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def helper(x: int) -> int:
                return x + 1

            def process(
                data: DF["{id: Int64}"],
            ) -> DF["{id: Int64}"]:
                return data
        ''')

        results = analyze_source(source)

        assert len(results) == 1
        assert results[0].name == "process"


class TestAnalyzeJoinOperations:
    """Test join operation analysis."""

    def test_infers_inner_join_result(self):
        """Infer result type of inner join."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def merge(
                users: DF["{user_id: Int64, name: Utf8}"],
                orders: DF["{order_id: Int64, user_id: Int64, amount: Float64}"],
            ) -> DF["{user_id: Int64, name: Utf8, order_id: Int64, amount: Float64}"]:
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def merge(
                users: DF["{user_id: Int64, name: Utf8}"],
                orders: DF["{user_id: Int64, amount: Float64}"],
            ) -> DF["{user_id: Int64, name: Utf8, amount: Float64?}"]:
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
        source = textwrap.dedent('''
            from polypolarism import DF

            def bad_join(
                users: DF["{user_id: Int64, name: Utf8}"],
                orders: DF["{order_id: Int64, amount: Float64}"],
            ) -> DF["{user_id: Int64, name: Utf8, order_id: Int64, amount: Float64}"]:
                return users.join(orders, on="user_id", how="inner")
        ''')

        results = analyze_source(source)

        assert len(results[0].errors) > 0
        assert any("user_id" in str(e) for e in results[0].errors)


class TestAnalyzeGroupByOperations:
    """Test group_by().agg() operation analysis."""

    def test_infers_groupby_agg_result(self):
        """Infer result type of group_by().agg()."""
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def summarize(
                data: DF["{category: Utf8, amount: Float64}"],
            ) -> DF["{category: Utf8, total: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def count_by_category(
                data: DF["{category: Utf8, value: Int64}"],
            ) -> DF["{category: Utf8, count: UInt32}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def bad_groupby(
                data: DF["{id: Int64, value: Float64}"],
            ) -> DF["{category: Utf8, total: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def sales_summary(
                orders: DF["{order_id: Int64, customer_id: Int64, amount: Float64}"],
                customers: DF["{customer_id: Int64, region: Utf8}"],
            ) -> DF["{region: Utf8, total_sales: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def select_columns(
                data: DF["{id: Int64, name: Utf8, value: Float64}"],
            ) -> DF["{id: Int64, value: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def bad_select(
                data: DF["{id: Int64, value: Float64}"],
            ) -> DF["{id: Int64, amount: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def add_doubled(
                data: DF["{id: Int64, value: Float64}"],
            ) -> DF["{id: Int64, value: Float64, doubled: Float64}"]:
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
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def bad_col(
                data: DF["{id: Int64}"],
            ) -> DF["{id: Int64, value: Float64}"]:
                return data.select(pl.col("missing"))
        ''')

        results = analyze_source(source)

        assert results[0].has_errors is True

    def test_has_errors_false_when_valid(self):
        """has_errors returns False when no errors."""
        source = textwrap.dedent('''
            from polypolarism import DF

            def identity(
                data: DF["{id: Int64}"],
            ) -> DF["{id: Int64}"]:
                return data
        ''')

        results = analyze_source(source)

        assert results[0].has_errors is False


class TestAnalyzeIntermediateVariables:
    """Test analysis with intermediate variable assignments."""

    def test_tracks_variable_assignment(self):
        """Track DataFrame type through variable assignment."""
        source = textwrap.dedent('''
            import polars as pl
            from polypolarism import DF

            def with_intermediate(
                users: DF["{user_id: Int64, name: Utf8}"],
                orders: DF["{order_id: Int64, user_id: Int64, amount: Float64}"],
            ) -> DF["{user_id: Int64, name: Utf8, order_id: Int64, amount: Float64}"]:
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
