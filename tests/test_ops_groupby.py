"""Tests for groupby operation type inference."""

import pytest

from polypolarism.types import (
    DataType,
    Int64,
    Float64,
    Utf8,
    UInt32,
    Nullable,
    List,
    FrameType,
)
from polypolarism.ops.groupby import (
    infer_agg_result_type,
    infer_groupby_result,
    AggExpr,
    AggFunction,
    GroupByTypeError,
)


class TestAggFunctionSignatures:
    """Test aggregation function type signatures."""

    def test_sum_int64_returns_int64(self):
        """sum(Int64) -> Int64"""
        result = infer_agg_result_type(AggFunction.SUM, Int64())
        assert result == Int64()

    def test_sum_float64_returns_float64(self):
        """sum(Float64) -> Float64"""
        result = infer_agg_result_type(AggFunction.SUM, Float64())
        assert result == Float64()

    def test_mean_int64_returns_float64(self):
        """mean(Int64) -> Float64"""
        result = infer_agg_result_type(AggFunction.MEAN, Int64())
        assert result == Float64()

    def test_mean_float64_returns_float64(self):
        """mean(Float64) -> Float64"""
        result = infer_agg_result_type(AggFunction.MEAN, Float64())
        assert result == Float64()

    def test_count_returns_uint32(self):
        """count(*) -> UInt32"""
        result = infer_agg_result_type(AggFunction.COUNT, Int64())
        assert result == UInt32()

    def test_count_any_type_returns_uint32(self):
        """count works on any type"""
        assert infer_agg_result_type(AggFunction.COUNT, Utf8()) == UInt32()
        assert infer_agg_result_type(AggFunction.COUNT, Float64()) == UInt32()

    def test_n_unique_returns_uint32(self):
        """n_unique(T) -> UInt32"""
        result = infer_agg_result_type(AggFunction.N_UNIQUE, Utf8())
        assert result == UInt32()

    def test_list_returns_list_of_inner_type(self):
        """list(T) -> List[T]"""
        result = infer_agg_result_type(AggFunction.LIST, Int64())
        assert result == List(Int64())

    def test_list_utf8_returns_list_utf8(self):
        """list(Utf8) -> List[Utf8]"""
        result = infer_agg_result_type(AggFunction.LIST, Utf8())
        assert result == List(Utf8())

    def test_first_returns_same_type(self):
        """first(T) -> T"""
        result = infer_agg_result_type(AggFunction.FIRST, Int64())
        assert result == Int64()

    def test_first_utf8_returns_utf8(self):
        """first(Utf8) -> Utf8"""
        result = infer_agg_result_type(AggFunction.FIRST, Utf8())
        assert result == Utf8()

    def test_last_returns_same_type(self):
        """last(T) -> T"""
        result = infer_agg_result_type(AggFunction.LAST, Int64())
        assert result == Int64()

    def test_min_returns_same_type(self):
        """min(T) -> T"""
        result = infer_agg_result_type(AggFunction.MIN, Int64())
        assert result == Int64()

    def test_max_returns_same_type(self):
        """max(T) -> T"""
        result = infer_agg_result_type(AggFunction.MAX, Float64())
        assert result == Float64()


class TestAggFunctionTypeErrors:
    """Test that invalid aggregation function applications raise errors."""

    def test_sum_on_utf8_raises_error(self):
        """sum cannot be applied to Utf8."""
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.SUM, Utf8())
        assert "sum" in str(exc_info.value).lower()
        assert "Utf8" in str(exc_info.value)

    def test_mean_on_utf8_raises_error(self):
        """mean cannot be applied to Utf8."""
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_agg_result_type(AggFunction.MEAN, Utf8())
        assert "mean" in str(exc_info.value).lower()


class TestNullableHandling:
    """Test nullable type handling in aggregations."""

    def test_sum_nullable_int64_returns_nullable_int64(self):
        """sum(Nullable[Int64]) -> Nullable[Int64]"""
        result = infer_agg_result_type(AggFunction.SUM, Nullable(Int64()))
        assert result == Nullable(Int64())

    def test_mean_nullable_int64_returns_nullable_float64(self):
        """mean(Nullable[Int64]) -> Nullable[Float64]"""
        result = infer_agg_result_type(AggFunction.MEAN, Nullable(Int64()))
        assert result == Nullable(Float64())

    def test_count_ignores_nullability(self):
        """count always returns UInt32, regardless of nullability."""
        result = infer_agg_result_type(AggFunction.COUNT, Nullable(Int64()))
        assert result == UInt32()

    def test_n_unique_ignores_nullability_for_return(self):
        """n_unique always returns UInt32."""
        result = infer_agg_result_type(AggFunction.N_UNIQUE, Nullable(Utf8()))
        assert result == UInt32()

    def test_list_preserves_nullability(self):
        """list(Nullable[T]) -> List[Nullable[T]]"""
        result = infer_agg_result_type(AggFunction.LIST, Nullable(Int64()))
        assert result == List(Nullable(Int64()))

    def test_first_preserves_nullability(self):
        """first(Nullable[T]) -> Nullable[T]"""
        result = infer_agg_result_type(AggFunction.FIRST, Nullable(Int64()))
        assert result == Nullable(Int64())


class TestInferGroupByResult:
    """Test infer_groupby_result function."""

    def test_single_key_single_agg(self):
        """Basic group_by with one key and one aggregation."""
        input_frame = FrameType({
            "country": Utf8(),
            "amount": Float64(),
        })
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias="total_amount"),
        ]
        result = infer_groupby_result(input_frame, ["country"], agg_exprs)

        expected = FrameType({
            "country": Utf8(),
            "total_amount": Float64(),
        })
        assert result == expected

    def test_single_key_multiple_agg(self):
        """Group by with multiple aggregations."""
        input_frame = FrameType({
            "product_id": Int64(),
            "quantity": Int64(),
            "price": Float64(),
        })
        agg_exprs = [
            AggExpr(column="quantity", function=AggFunction.SUM, alias="total_qty"),
            AggExpr(column="price", function=AggFunction.MEAN, alias="avg_price"),
            AggExpr(column="price", function=AggFunction.COUNT, alias="order_count"),
        ]
        result = infer_groupby_result(input_frame, ["product_id"], agg_exprs)

        expected = FrameType({
            "product_id": Int64(),
            "total_qty": Int64(),
            "avg_price": Float64(),
            "order_count": UInt32(),
        })
        assert result == expected

    def test_multiple_keys(self):
        """Group by with multiple keys."""
        input_frame = FrameType({
            "country": Utf8(),
            "year": Int64(),
            "sales": Float64(),
        })
        agg_exprs = [
            AggExpr(column="sales", function=AggFunction.SUM, alias="total_sales"),
        ]
        result = infer_groupby_result(input_frame, ["country", "year"], agg_exprs)

        expected = FrameType({
            "country": Utf8(),
            "year": Int64(),
            "total_sales": Float64(),
        })
        assert result == expected

    def test_key_column_not_found_raises_error(self):
        """Raises error when group by key column doesn't exist."""
        input_frame = FrameType({
            "product": Utf8(),
            "amount": Float64(),
        })
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias="total"),
        ]
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_groupby_result(input_frame, ["category"], agg_exprs)
        assert "category" in str(exc_info.value)

    def test_agg_column_not_found_raises_error(self):
        """Raises error when aggregation column doesn't exist."""
        input_frame = FrameType({
            "category": Utf8(),
            "amount": Float64(),
        })
        agg_exprs = [
            AggExpr(column="price", function=AggFunction.SUM, alias="total"),
        ]
        with pytest.raises(GroupByTypeError) as exc_info:
            infer_groupby_result(input_frame, ["category"], agg_exprs)
        assert "price" in str(exc_info.value)


class TestDefaultAliasNaming:
    """Test Polars default naming conventions when alias is not provided."""

    def test_sum_default_alias(self):
        """Default alias for sum is column name."""
        input_frame = FrameType({
            "category": Utf8(),
            "amount": Float64(),
        })
        agg_exprs = [
            AggExpr(column="amount", function=AggFunction.SUM, alias=None),
        ]
        result = infer_groupby_result(input_frame, ["category"], agg_exprs)

        # Polars default: column name is used as-is for most aggregations
        assert "amount" in result.columns
        assert result.columns["amount"] == Float64()

    def test_count_default_alias(self):
        """Default alias for count is column name."""
        input_frame = FrameType({
            "category": Utf8(),
            "value": Int64(),
        })
        agg_exprs = [
            AggExpr(column="value", function=AggFunction.COUNT, alias=None),
        ]
        result = infer_groupby_result(input_frame, ["category"], agg_exprs)

        assert "value" in result.columns
        assert result.columns["value"] == UInt32()


class TestAggExprDataClass:
    """Test AggExpr data class."""

    def test_agg_expr_with_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM, alias="total")
        assert expr.column == "amount"
        assert expr.function == AggFunction.SUM
        assert expr.alias == "total"

    def test_agg_expr_without_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM)
        assert expr.alias is None

    def test_agg_expr_output_name_with_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM, alias="total")
        assert expr.output_name == "total"

    def test_agg_expr_output_name_without_alias(self):
        expr = AggExpr(column="amount", function=AggFunction.SUM)
        assert expr.output_name == "amount"
