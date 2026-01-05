"""Expr type inference for Polars expressions."""

from typing import Any

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


class ColumnNotFoundError(Exception):
    """Raised when a column is not found in the FrameType."""

    pass


class TypePromotionError(Exception):
    """Raised when types cannot be promoted for an operation."""

    pass


class TypeUnificationError(Exception):
    """Raised when types cannot be unified."""

    pass


# Type hierarchy for numeric type promotion
# Higher number = wider type
_NUMERIC_TYPE_ORDER: dict[type, int] = {
    Int32: 1,
    Int64: 2,
    Float32: 3,
    Float64: 4,
}

# Set of numeric types that can be promoted
_NUMERIC_TYPES = frozenset(_NUMERIC_TYPE_ORDER.keys())


def infer_col(column_name: str, frame: FrameType) -> DataType:
    """Infer the type of pl.col(column_name) from FrameType.

    Args:
        column_name: The name of the column to look up.
        frame: The FrameType containing column definitions.

    Returns:
        The DataType of the column.

    Raises:
        ColumnNotFoundError: If the column does not exist in the frame.
    """
    dtype = frame.get_column_type(column_name)
    if dtype is None:
        available = list(frame.columns.keys())
        raise ColumnNotFoundError(
            f"Column '{column_name}' not found. Available columns: {available}"
        )
    return dtype


def infer_lit(value: Any) -> DataType:
    """Infer the type of pl.lit(value).

    Args:
        value: The literal value.

    Returns:
        The inferred DataType.
    """
    if value is None:
        return Null()
    # Note: bool check must come before int because bool is a subclass of int
    if isinstance(value, bool):
        return Boolean()
    if isinstance(value, int):
        return Int64()
    if isinstance(value, float):
        return Float64()
    if isinstance(value, str):
        return Utf8()
    # For unsupported types, we could raise an error or return a generic type
    raise TypeError(f"Unsupported literal type: {type(value).__name__}")


def _unwrap_nullable(dtype: DataType) -> tuple[DataType, bool]:
    """Unwrap a Nullable type and return (inner_type, is_nullable)."""
    if isinstance(dtype, Nullable):
        return dtype.inner, True
    return dtype, False


def _is_numeric(dtype: DataType) -> bool:
    """Check if a type is numeric."""
    return type(dtype) in _NUMERIC_TYPES


def _promote_numeric(left: DataType, right: DataType) -> DataType:
    """Promote two numeric types to a common type.

    Polars rules:
    - Int + Float -> Float64 (always promotes to Float64)
    - Same category: larger type wins
    """
    left_order = _NUMERIC_TYPE_ORDER.get(type(left), 0)
    right_order = _NUMERIC_TYPE_ORDER.get(type(right), 0)

    left_is_float = isinstance(left, (Float32, Float64))
    right_is_float = isinstance(right, (Float32, Float64))

    # If mixing int and float, always promote to Float64
    if left_is_float != right_is_float:
        return Float64()

    # Otherwise, take the wider type
    if left_order >= right_order:
        return left
    return right


def promote_types(left: DataType, right: DataType) -> DataType:
    """Promote two types to a common type for arithmetic operations.

    Args:
        left: The left operand type.
        right: The right operand type.

    Returns:
        The promoted type.

    Raises:
        TypePromotionError: If types cannot be promoted.
    """
    # Handle Null type
    if isinstance(left, Null):
        if isinstance(right, Null):
            return Null()
        # Null + T -> Nullable[T]
        inner, is_nullable = _unwrap_nullable(right)
        return Nullable(inner)
    if isinstance(right, Null):
        inner, _ = _unwrap_nullable(left)
        return Nullable(inner)

    # Unwrap Nullable types and track nullability
    left_inner, left_nullable = _unwrap_nullable(left)
    right_inner, right_nullable = _unwrap_nullable(right)
    result_nullable = left_nullable or right_nullable

    # Check if both are numeric
    if not _is_numeric(left_inner) or not _is_numeric(right_inner):
        raise TypePromotionError(
            f"Cannot promote types {left} and {right} for arithmetic operation"
        )

    # Promote numeric types
    promoted = _promote_numeric(left_inner, right_inner)

    # Wrap in Nullable if needed
    if result_nullable:
        return Nullable(promoted)
    return promoted


def infer_cast(source: DataType, target: DataType) -> DataType:
    """Infer the result type of a cast operation.

    The result type is the target type, but nullability is preserved
    from the source type (nullability can only increase, not decrease).

    Args:
        source: The source type being cast.
        target: The target type to cast to.

    Returns:
        The result type of the cast.
    """
    # Check if source is nullable
    source_inner, source_nullable = _unwrap_nullable(source)

    # Check if target is nullable
    target_inner, target_nullable = _unwrap_nullable(target)

    # Result is nullable if either source or target is nullable
    result_nullable = source_nullable or target_nullable

    if result_nullable:
        return Nullable(target_inner)
    return target_inner


def unify_types(left: DataType, right: DataType) -> DataType:
    """Unify two types to find a common type.

    Used for when/then/otherwise expressions where both branches
    must produce compatible types.

    Args:
        left: The first type.
        right: The second type.

    Returns:
        The unified type.

    Raises:
        TypeUnificationError: If types cannot be unified.
    """
    # Handle Null type
    if isinstance(left, Null):
        if isinstance(right, Null):
            return Null()
        inner, _ = _unwrap_nullable(right)
        return Nullable(inner)
    if isinstance(right, Null):
        inner, _ = _unwrap_nullable(left)
        return Nullable(inner)

    # Unwrap Nullable types and track nullability
    left_inner, left_nullable = _unwrap_nullable(left)
    right_inner, right_nullable = _unwrap_nullable(right)
    result_nullable = left_nullable or right_nullable

    # Check if types are exactly the same
    if left_inner == right_inner:
        if result_nullable:
            return Nullable(left_inner)
        return left_inner

    # Try numeric promotion
    if _is_numeric(left_inner) and _is_numeric(right_inner):
        promoted = _promote_numeric(left_inner, right_inner)
        if result_nullable:
            return Nullable(promoted)
        return promoted

    # Types cannot be unified
    raise TypeUnificationError(
        f"Cannot unify types {left} and {right}"
    )


def infer_when_then_otherwise(
    condition: DataType,
    then_type: DataType,
    otherwise_type: DataType,
) -> DataType:
    """Infer the result type of a when/then/otherwise expression.

    Args:
        condition: The type of the condition expression (should be Boolean).
        then_type: The type of the then branch.
        otherwise_type: The type of the otherwise branch.

    Returns:
        The unified type of then and otherwise branches.

    Raises:
        TypeError: If condition is not Boolean.
        TypeUnificationError: If then and otherwise types cannot be unified.
    """
    # Validate condition type
    condition_inner, condition_nullable = _unwrap_nullable(condition)
    if not isinstance(condition_inner, Boolean):
        raise TypeError(
            f"Condition must be Boolean, got {condition}"
        )

    # Unify then and otherwise types
    unified = unify_types(then_type, otherwise_type)

    # If condition is nullable, result is also nullable
    # (because when condition is null, the result could be null)
    if condition_nullable:
        inner, _ = _unwrap_nullable(unified)
        return Nullable(inner)

    return unified
