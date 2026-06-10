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
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Null,
    Nullable,
    RowVar,
    Time,
    UInt32,
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

        assert any("PLY001" in e for e in results[0].errors)
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

        assert any("PLY001" in e for e in results[0].errors)


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
            ("std", Float64()),
            ("var", Float64()),
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
            ('pl.col("name").str.to_decimal()', Decimal(38, 0)),
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
        assert ft.columns["v"].dtype == Float64()

    @pytest.mark.parametrize(
        "method",
        ["rolling_mean", "rolling_std", "rolling_var", "rolling_median"],
    )
    def test_rolling_float_methods(self, method: str):
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
        assert ft.columns["v"].dtype == Float64()


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

    def test_struct_mixed_varargs_and_list(self):
        from polypolarism.types import Struct

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
        assert results[0].has_errors is False, results[0].errors
        ft = results[0].inferred_return_type
        assert ft is not None
        assert ft.columns["ab"].dtype == Struct({"a": Int64(), "b": Float64()})

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
        assert any("PLY001" in e and "missing" in e for e in results[0].errors)

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
        assert any("PLY001" in e and "nope" in e for e in results[0].errors)


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
        assert any("PLY001" in e and "nope" in e for e in results[0].errors)


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
        assert any("[PLY001]" in e for e in results[0].errors)

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
        assert any("PLY001" in e for e in results[0].errors)
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

    def test_drop_missing_column_on_open_frame_no_error(self):
        analyzer = _run_body(self._open_frame(), "out = df.drop('ghost')")
        assert analyzer.errors == []
        assert "ghost" not in analyzer.var_types["out"].columns

    def test_drop_missing_column_on_closed_frame_still_errors(self):
        analyzer = _run_body(FrameType({"id": Int64()}), "out = df.drop('ghost')")
        assert any("ghost" in e for e in analyzer.errors)

    def test_explode_missing_column_on_open_frame_becomes_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.explode('items')")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["items"].dtype == Unknown()

    def test_unnest_missing_column_on_open_frame_no_error(self):
        analyzer = _run_body(self._open_frame(), "out = df.unnest('s')")
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.rest is not None
        assert "s" not in out.columns

    def test_select_col_expr_missing_on_open_frame_registers_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.select(pl.col('ghost'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["ghost"].dtype == Unknown()

    def test_select_plural_col_missing_on_open_frame_registers_unknown(self):
        analyzer = _run_body(self._open_frame(), "out = df.select(pl.col('a', 'b'))")
        assert analyzer.errors == []
        out = analyzer.var_types["out"]
        assert out.columns["a"].dtype == Unknown()
        assert out.columns["b"].dtype == Unknown()

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

    Operands outside the closed set (Unknown, unresolved, Decimal, List,
    ...) and Null literals keep the legacy silent fallback — false
    positives are worse than false negatives here.
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
                "dec": Decimal(10, 2),
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
            # temporal differences -> Duration
            ("pl.col('d') - pl.col('d')", Duration()),
            ("pl.col('d') - pl.col('dt')", Duration()),
            ("pl.col('dt') - pl.col('d')", Duration()),
            ("pl.col('dt') - pl.col('dt')", Duration()),
            ("pl.col('tz') - pl.col('tz')", Duration()),
            ("pl.col('t') - pl.col('t')", Duration()),
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

    def test_unknown_operand_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('u') + pl.col('s'))")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Unknown()

    def test_unresolved_operand_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('s') + helper())")
        assert analyzer.errors == []
        assert analyzer.var_types["out"].columns["r"].dtype == Utf8()

    def test_decimal_operand_is_silent(self):
        """Decimal is outside the fully-understood set — no PLY009."""
        analyzer = _run_body(self._frame(), "out = df.select(r=pl.col('dec') + pl.col('i'))")
        assert analyzer.errors == []

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
        assert any("PLY001" in str(e) and "ghost" in str(e) for e in results[0].errors)

    def test_select_unknown_name_still_falls_through(self):
        """A bare Name that is NOT a constant (e.g. a frame variable) keeps
        going to expression analysis — no false PLY001."""
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
        assert any("PLY001" in str(e) and "ghost" in str(e) for e in results[0].errors)

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

    def test_empty_list_is_unknown(self):
        ft = self._infer('pl.DataFrame({"a": []})')
        assert ft == FrameType({"a": Unknown()})

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

    def test_opaque_data_without_schema_is_uninferable(self):
        ft = self._infer("pl.DataFrame(some_rows)")
        assert ft is None

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
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)
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
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

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
    got: cat"), ``.dt`` accepts Date/Datetime/Time/Duration, ``.list`` /
    ``.arr`` require List/Array, ``.struct`` requires Struct.
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
            'pl.col("i").struct.field("a")',
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
            # polypolarism models polars Array as List, so `.arr` on a List
            # dtype passes too — the two are statically indistinguishable.
            'pl.col("xs").arr.sum()',
            # Bare ``pl.Struct`` parses to Unknown — exempt from validation.
            'pl.col("st").struct.field("x")',
            'pl.col("st").str.contains("x")',
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

    def test_open_frame_unknown_receiver_not_flagged(self):
        """A column missing on an open frame resolves to Unknown — exempt."""
        frame = FrameType({"id": Int64()}, rest=RowVar("r"))
        analyzer = _run_body(
            frame,
            'out = df.select(pl.col("extra").str.contains("x").alias("m"))',
        )
        assert analyzer.errors == []

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
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_existing_literal_key_passes(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s"))')
        assert results[0].errors == [], results[0].errors

    def test_varargs_keys_each_checked(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s", "ghost"))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_multiple_existing_keys_pass(self):
        results = self._analyze('df.select(pl.col("v").sum().over("s", "t"))')
        assert results[0].errors == [], results[0].errors

    def test_list_keys_each_checked(self):
        results = self._analyze('df.select(pl.col("v").sum().over(["s", "ghost"]))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_expr_key_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").sum().over(pl.col("ghost")))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_expr_key_existing_passes(self):
        results = self._analyze('df.select(pl.col("v").sum().over(pl.col("s")))')
        assert results[0].errors == [], results[0].errors

    def test_order_by_kwarg_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by="ghost"))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_order_by_kwarg_existing_passes(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by="t"))')
        assert results[0].errors == [], results[0].errors

    def test_order_by_list_each_checked(self):
        results = self._analyze('df.select(pl.col("v").first().over("s", order_by=["t", "ghost"]))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

    def test_partition_by_kwarg_missing_flags_ply001(self):
        results = self._analyze('df.select(pl.col("v").sum().over(partition_by="ghost"))')
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)

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
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)


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

    def test_frame_cast_unknown_source_is_silent(self):
        analyzer = _run_body(self._frame(), "out = df.cast({'u': pl.Int64})")
        assert analyzer.errors == [], analyzer.errors
        assert analyzer.var_types["out"].columns["u"].dtype == Int64()


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
        assert any("PLY001" in e and "ghost" in e for e in results[0].errors)
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

    def test_unresolved_fill_keeps_receiver_dtype(self):
        # The fill expression is not modelled — fall back to the receiver
        # dtype (slots are filled with *something*, so no Nullable wrap).
        ft = self._infer('df.select(pl.col("a").shift(1, fill_value=pl.col("a").interpolate()))')
        assert ft.columns["a"].dtype == Int64()
