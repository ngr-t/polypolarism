"""Tests for AST analyzer."""

import textwrap

import pytest

from polypolarism.analyzer import (
    analyze_source,
)
from polypolarism.types import (
    Boolean,
    Date,
    Datetime,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Nullable,
    UInt32,
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
    """Schema-preserving (identity-typed) DataFrame methods added in M1."""

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

    def test_unpivot_value_columns_not_unifiable_errors(self):
        source = textwrap.dedent(
            PANDERA_HEADER
            + """
            class In(pa.DataFrameModel):
                id: int
                a: pl.Float64
                b: str

            def f(data: DataFrame[In]):
                return data.unpivot(index=["id"], on=["a", "b"])
        """
        )
        results = analyze_source(source)
        assert results[0].has_errors is True


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
