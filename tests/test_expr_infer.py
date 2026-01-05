"""Tests for Expr type inference."""

import pytest

from polypolarism.types import (
    Boolean,
    DataType,
    Float32,
    Float64,
    FrameType,
    Int32,
    Int64,
    Null,
    Nullable,
    Utf8,
)
from polypolarism.expr_infer import (
    infer_col,
    infer_lit,
    infer_cast,
    infer_when_then_otherwise,
    promote_types,
    unify_types,
    ColumnNotFoundError,
    TypePromotionError,
    TypeUnificationError,
)


class TestInferCol:
    """Tests for pl.col() type inference."""

    def test_col_returns_column_type_from_frame(self) -> None:
        """pl.col("x") should return the dtype of column x from FrameType."""
        frame = FrameType(columns={"x": Int64(), "y": Utf8()})
        result = infer_col("x", frame)
        assert result == Int64()

    def test_col_returns_different_column_type(self) -> None:
        """pl.col("y") should return Utf8 when y is Utf8."""
        frame = FrameType(columns={"x": Int64(), "y": Utf8()})
        result = infer_col("y", frame)
        assert result == Utf8()

    def test_col_raises_error_for_nonexistent_column(self) -> None:
        """pl.col("z") should raise error when z doesn't exist."""
        frame = FrameType(columns={"x": Int64()})
        with pytest.raises(ColumnNotFoundError) as exc_info:
            infer_col("z", frame)
        assert "z" in str(exc_info.value)

    def test_col_returns_nullable_type(self) -> None:
        """pl.col("x") should return Nullable type when column is nullable."""
        frame = FrameType(columns={"x": Nullable(Int64())})
        result = infer_col("x", frame)
        assert result == Nullable(Int64())

    def test_col_error_message_includes_available_columns(self) -> None:
        """Error message should include available columns for debugging."""
        frame = FrameType(columns={"a": Int64(), "b": Utf8()})
        with pytest.raises(ColumnNotFoundError) as exc_info:
            infer_col("c", frame)
        error_msg = str(exc_info.value)
        assert "a" in error_msg or "b" in error_msg


class TestInferLit:
    """Tests for pl.lit() type inference."""

    def test_lit_int_returns_int64(self) -> None:
        """pl.lit(42) should infer Int64."""
        result = infer_lit(42)
        assert result == Int64()

    def test_lit_float_returns_float64(self) -> None:
        """pl.lit(3.14) should infer Float64."""
        result = infer_lit(3.14)
        assert result == Float64()

    def test_lit_str_returns_utf8(self) -> None:
        """pl.lit("hello") should infer Utf8."""
        result = infer_lit("hello")
        assert result == Utf8()

    def test_lit_bool_returns_boolean(self) -> None:
        """pl.lit(True) should infer Boolean."""
        result = infer_lit(True)
        assert result == Boolean()

    def test_lit_none_returns_nullable(self) -> None:
        """pl.lit(None) should infer Nullable type (null literal)."""
        # None is a special case - Polars infers it as Null type
        # We represent this as Nullable with a placeholder inner type
        result = infer_lit(None)
        assert result == Null()


class TestPromoteTypes:
    """Tests for arithmetic operators type promotion."""

    def test_int64_plus_int64_returns_int64(self) -> None:
        """Int64 + Int64 should return Int64."""
        result = promote_types(Int64(), Int64())
        assert result == Int64()

    def test_int64_plus_float64_returns_float64(self) -> None:
        """Int64 + Float64 should promote to Float64."""
        result = promote_types(Int64(), Float64())
        assert result == Float64()

    def test_float64_plus_int64_returns_float64(self) -> None:
        """Float64 + Int64 should promote to Float64 (order doesn't matter)."""
        result = promote_types(Float64(), Int64())
        assert result == Float64()

    def test_int32_plus_int64_returns_int64(self) -> None:
        """Int32 + Int64 should promote to Int64 (larger type wins)."""
        result = promote_types(Int32(), Int64())
        assert result == Int64()

    def test_float32_plus_float64_returns_float64(self) -> None:
        """Float32 + Float64 should promote to Float64."""
        result = promote_types(Float32(), Float64())
        assert result == Float64()

    def test_int32_plus_float32_returns_float64(self) -> None:
        """Int32 + Float32 should promote to Float64 (Polars behavior)."""
        result = promote_types(Int32(), Float32())
        assert result == Float64()

    def test_nullable_int64_plus_int64_returns_nullable_int64(self) -> None:
        """Nullable[Int64] + Int64 should return Nullable[Int64]."""
        result = promote_types(Nullable(Int64()), Int64())
        assert result == Nullable(Int64())

    def test_int64_plus_nullable_int64_returns_nullable_int64(self) -> None:
        """Int64 + Nullable[Int64] should return Nullable[Int64]."""
        result = promote_types(Int64(), Nullable(Int64()))
        assert result == Nullable(Int64())

    def test_nullable_int64_plus_float64_returns_nullable_float64(self) -> None:
        """Nullable[Int64] + Float64 should return Nullable[Float64]."""
        result = promote_types(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())

    def test_null_plus_int64_returns_nullable_int64(self) -> None:
        """Null + Int64 should return Nullable[Int64]."""
        result = promote_types(Null(), Int64())
        assert result == Nullable(Int64())

    def test_int64_plus_utf8_raises_error(self) -> None:
        """Int64 + Utf8 should raise TypePromotionError."""
        with pytest.raises(TypePromotionError) as exc_info:
            promote_types(Int64(), Utf8())
        assert "Int64" in str(exc_info.value)
        assert "Utf8" in str(exc_info.value)

    def test_boolean_plus_int64_raises_error(self) -> None:
        """Boolean + Int64 should raise TypePromotionError."""
        with pytest.raises(TypePromotionError):
            promote_types(Boolean(), Int64())


class TestInferCast:
    """Tests for cast() type inference."""

    def test_cast_int64_to_float64(self) -> None:
        """cast(Int64, Float64) should return Float64."""
        result = infer_cast(Int64(), Float64())
        assert result == Float64()

    def test_cast_float64_to_int64(self) -> None:
        """cast(Float64, Int64) should return Int64."""
        result = infer_cast(Float64(), Int64())
        assert result == Int64()

    def test_cast_int64_to_utf8(self) -> None:
        """cast(Int64, Utf8) should return Utf8."""
        result = infer_cast(Int64(), Utf8())
        assert result == Utf8()

    def test_cast_nullable_int64_to_float64(self) -> None:
        """cast(Nullable[Int64], Float64) should return Nullable[Float64].

        Nullability is preserved through cast.
        """
        result = infer_cast(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())

    def test_cast_int64_to_nullable_float64(self) -> None:
        """cast(Int64, Nullable[Float64]) should return Nullable[Float64].

        If target type is nullable, result is nullable.
        """
        result = infer_cast(Int64(), Nullable(Float64()))
        assert result == Nullable(Float64())

    def test_cast_nullable_to_non_nullable_preserves_nullability(self) -> None:
        """cast(Nullable[Int64], Float64) should still be Nullable[Float64].

        Cast cannot remove nullability - data may still contain nulls.
        """
        result = infer_cast(Nullable(Int64()), Float64())
        assert result == Nullable(Float64())


class TestUnifyTypes:
    """Tests for type unification (used by when/then/otherwise)."""

    def test_unify_same_types_returns_same_type(self) -> None:
        """Unifying Int64 with Int64 should return Int64."""
        result = unify_types(Int64(), Int64())
        assert result == Int64()

    def test_unify_int64_and_float64_returns_float64(self) -> None:
        """Unifying Int64 with Float64 should return Float64."""
        result = unify_types(Int64(), Float64())
        assert result == Float64()

    def test_unify_nullable_and_non_nullable_returns_nullable(self) -> None:
        """Unifying Nullable[Int64] with Int64 should return Nullable[Int64]."""
        result = unify_types(Nullable(Int64()), Int64())
        assert result == Nullable(Int64())

    def test_unify_null_and_int64_returns_nullable_int64(self) -> None:
        """Unifying Null with Int64 should return Nullable[Int64]."""
        result = unify_types(Null(), Int64())
        assert result == Nullable(Int64())

    def test_unify_int64_and_utf8_raises_error(self) -> None:
        """Unifying Int64 with Utf8 should raise TypeUnificationError."""
        with pytest.raises(TypeUnificationError):
            unify_types(Int64(), Utf8())

    def test_unify_utf8_and_utf8_returns_utf8(self) -> None:
        """Unifying Utf8 with Utf8 should return Utf8."""
        result = unify_types(Utf8(), Utf8())
        assert result == Utf8()

    def test_unify_boolean_and_boolean_returns_boolean(self) -> None:
        """Unifying Boolean with Boolean should return Boolean."""
        result = unify_types(Boolean(), Boolean())
        assert result == Boolean()


class TestInferWhenThenOtherwise:
    """Tests for when/then/otherwise type inference."""

    def test_when_then_otherwise_same_types(self) -> None:
        """when/then/otherwise with same types should return that type."""
        result = infer_when_then_otherwise(
            condition=Boolean(),
            then_type=Int64(),
            otherwise_type=Int64(),
        )
        assert result == Int64()

    def test_when_then_otherwise_promotes_numeric_types(self) -> None:
        """when/then/otherwise should promote Int64 and Float64 to Float64."""
        result = infer_when_then_otherwise(
            condition=Boolean(),
            then_type=Int64(),
            otherwise_type=Float64(),
        )
        assert result == Float64()

    def test_when_then_otherwise_nullable_increases(self) -> None:
        """when/then/otherwise should be nullable if either branch is nullable."""
        result = infer_when_then_otherwise(
            condition=Boolean(),
            then_type=Nullable(Int64()),
            otherwise_type=Int64(),
        )
        assert result == Nullable(Int64())

    def test_when_then_otherwise_nullable_condition_makes_result_nullable(self) -> None:
        """when/then/otherwise with nullable condition makes result nullable.

        This is because when condition is null, the result could be null.
        """
        result = infer_when_then_otherwise(
            condition=Nullable(Boolean()),
            then_type=Int64(),
            otherwise_type=Int64(),
        )
        assert result == Nullable(Int64())

    def test_when_then_otherwise_with_null_then(self) -> None:
        """when/then/otherwise with Null in then branch."""
        result = infer_when_then_otherwise(
            condition=Boolean(),
            then_type=Null(),
            otherwise_type=Int64(),
        )
        assert result == Nullable(Int64())

    def test_when_then_otherwise_with_null_otherwise(self) -> None:
        """when/then/otherwise with Null in otherwise branch."""
        result = infer_when_then_otherwise(
            condition=Boolean(),
            then_type=Int64(),
            otherwise_type=Null(),
        )
        assert result == Nullable(Int64())

    def test_when_then_otherwise_incompatible_types_raises_error(self) -> None:
        """when/then/otherwise with incompatible types should raise error."""
        with pytest.raises(TypeUnificationError):
            infer_when_then_otherwise(
                condition=Boolean(),
                then_type=Int64(),
                otherwise_type=Utf8(),
            )

    def test_when_then_otherwise_condition_must_be_boolean(self) -> None:
        """when/then/otherwise condition must be Boolean or Nullable[Boolean]."""
        with pytest.raises(TypeError) as exc_info:
            infer_when_then_otherwise(
                condition=Int64(),
                then_type=Int64(),
                otherwise_type=Int64(),
            )
        assert "Boolean" in str(exc_info.value)
