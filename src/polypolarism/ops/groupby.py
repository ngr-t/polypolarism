"""GroupBy operation type inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

from polypolarism.types import (
    DataType,
    Float64,
    Int64,
    Int32,
    UInt32,
    UInt64,
    Float32,
    Nullable,
    List,
    Utf8,
    Boolean,
    FrameType,
)


class GroupByTypeError(Exception):
    """Exception raised for type errors in group_by operations."""

    pass


class AggFunction(Enum):
    """Aggregation function types."""

    SUM = auto()
    MEAN = auto()
    COUNT = auto()
    N_UNIQUE = auto()
    LIST = auto()
    FIRST = auto()
    LAST = auto()
    MIN = auto()
    MAX = auto()


@dataclass
class AggExpr:
    """Represents an aggregation expression."""

    column: str
    function: AggFunction
    alias: Optional[str] = None

    @property
    def output_name(self) -> str:
        """Return the output column name (alias or original column name)."""
        return self.alias if self.alias is not None else self.column


# Type sets for validation
NUMERIC_TYPES = (Int64, Int32, UInt32, UInt64, Float64, Float32)


def _is_numeric(dtype: DataType) -> bool:
    """Check if a type is numeric (considering Nullable wrapper)."""
    inner = dtype.inner if isinstance(dtype, Nullable) else dtype
    return isinstance(inner, NUMERIC_TYPES)


def _unwrap_nullable(dtype: DataType) -> tuple[DataType, bool]:
    """Unwrap Nullable and return (inner_type, is_nullable)."""
    if isinstance(dtype, Nullable):
        return dtype.inner, True
    return dtype, False


def _wrap_nullable(dtype: DataType, is_nullable: bool) -> DataType:
    """Wrap type in Nullable if needed."""
    if is_nullable:
        return Nullable(dtype)
    return dtype


# Aggregation function type signatures
# Each function returns (result_type, preserves_nullability)
# where preserves_nullability means the result should be Nullable if input is Nullable

def _infer_sum(dtype: DataType) -> DataType:
    """sum(T) -> T for numeric types."""
    inner, is_nullable = _unwrap_nullable(dtype)

    if not isinstance(inner, NUMERIC_TYPES):
        raise GroupByTypeError(
            f"Cannot apply sum to type {dtype}: sum requires numeric type"
        )

    # sum preserves the type (Int64 -> Int64, Float64 -> Float64)
    return _wrap_nullable(inner, is_nullable)


def _infer_mean(dtype: DataType) -> DataType:
    """mean(T) -> Float64 for numeric types."""
    inner, is_nullable = _unwrap_nullable(dtype)

    if not isinstance(inner, NUMERIC_TYPES):
        raise GroupByTypeError(
            f"Cannot apply mean to type {dtype}: mean requires numeric type"
        )

    # mean always returns Float64
    return _wrap_nullable(Float64(), is_nullable)


def _infer_count(dtype: DataType) -> DataType:
    """count(*) -> UInt32 for any type."""
    # count always returns UInt32, never nullable
    return UInt32()


def _infer_n_unique(dtype: DataType) -> DataType:
    """n_unique(T) -> UInt32 for any type."""
    # n_unique always returns UInt32, never nullable
    return UInt32()


def _infer_list(dtype: DataType) -> DataType:
    """list(T) -> List[T]."""
    # list wraps the element type (including Nullable if present)
    return List(dtype)


def _infer_first(dtype: DataType) -> DataType:
    """first(T) -> T."""
    # first preserves the exact type
    return dtype


def _infer_last(dtype: DataType) -> DataType:
    """last(T) -> T."""
    # last preserves the exact type
    return dtype


def _infer_min(dtype: DataType) -> DataType:
    """min(T) -> T."""
    # min preserves the exact type
    return dtype


def _infer_max(dtype: DataType) -> DataType:
    """max(T) -> T."""
    # max preserves the exact type
    return dtype


# Mapping from AggFunction to inference function
_AGG_INFER_MAP: dict[AggFunction, Callable[[DataType], DataType]] = {
    AggFunction.SUM: _infer_sum,
    AggFunction.MEAN: _infer_mean,
    AggFunction.COUNT: _infer_count,
    AggFunction.N_UNIQUE: _infer_n_unique,
    AggFunction.LIST: _infer_list,
    AggFunction.FIRST: _infer_first,
    AggFunction.LAST: _infer_last,
    AggFunction.MIN: _infer_min,
    AggFunction.MAX: _infer_max,
}


def infer_agg_result_type(func: AggFunction, input_type: DataType) -> DataType:
    """
    Infer the result type of an aggregation function applied to a column.

    Args:
        func: The aggregation function
        input_type: The type of the input column

    Returns:
        The result type of the aggregation

    Raises:
        GroupByTypeError: If the aggregation function cannot be applied to the type
    """
    infer_fn = _AGG_INFER_MAP.get(func)
    if infer_fn is None:
        raise GroupByTypeError(f"Unknown aggregation function: {func}")
    return infer_fn(input_type)


def infer_groupby_result(
    input_frame: FrameType,
    keys: list[str],
    agg_exprs: list[AggExpr],
) -> FrameType:
    """
    Infer the result FrameType of a group_by().agg() operation.

    Args:
        input_frame: The input DataFrame's type
        keys: Column names to group by
        agg_exprs: List of aggregation expressions

    Returns:
        The result FrameType

    Raises:
        GroupByTypeError: If key or aggregation columns don't exist,
                         or if aggregation functions cannot be applied
    """
    result_columns: dict[str, DataType] = {}

    # 1. Add group key columns (preserve their types)
    for key in keys:
        if not input_frame.has_column(key):
            raise GroupByTypeError(
                f"Group by key column '{key}' not found in DataFrame"
            )
        key_type = input_frame.get_column_type(key)
        assert key_type is not None  # We just checked has_column
        result_columns[key] = key_type

    # 2. Add aggregation result columns
    for agg_expr in agg_exprs:
        col_name = agg_expr.column
        if not input_frame.has_column(col_name):
            raise GroupByTypeError(
                f"Aggregation column '{col_name}' not found in DataFrame"
            )

        col_type = input_frame.get_column_type(col_name)
        assert col_type is not None  # We just checked has_column

        result_type = infer_agg_result_type(agg_expr.function, col_type)
        output_name = agg_expr.output_name
        result_columns[output_name] = result_type

    return FrameType(columns=result_columns)
