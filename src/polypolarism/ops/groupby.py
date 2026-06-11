"""GroupBy operation type inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from polypolarism.types import (
    DataType,
    Float32,
    Float64,
    FrameType,
    Int32,
    Int64,
    List,
    Nullable,
    UInt32,
    UInt64,
    Unknown,
)
from polypolarism.types import (
    unwrap_nullable as _unwrap_nullable,
)
from polypolarism.types import (
    wrap_nullable as _wrap_nullable,
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
    STD = auto()
    VAR = auto()
    MEDIAN = auto()
    QUANTILE = auto()
    PRODUCT = auto()


@dataclass
class AggExpr:
    """Represents an aggregation expression.

    Two construction shapes are supported:

    - Direct form: ``column`` + ``function`` (and optional ``alias``). Used by
      the simple ``pl.col("X").<agg>().alias("Y")`` pattern; ``infer_groupby_result``
      validates the column exists and computes the result dtype from
      ``infer_agg_result_type(function, col_dtype)``.
    - Pre-resolved form: ``column`` + ``dtype`` (and optional ``alias``). Used by
      the chain fallback (``pl.col("X").max().dt.year().alias("Y")``). The
      caller has already validated the chain via the expression analyser, so
      ``infer_groupby_result`` skips the column-existence check and uses
      ``dtype`` directly.
    """

    column: str
    function: AggFunction | None = None
    alias: str | None = None
    dtype: DataType | None = None

    @property
    def output_name(self) -> str:
        """Return the output column name (alias or original column name)."""
        return self.alias if self.alias is not None else self.column


# Receiver dtypes accepted by numeric aggregations (sum/mean/std/...).
# Deliberately a SUBSET of ``types.NUMERIC_DTYPES``: only these widths have
# probed aggregation signatures (e.g. polars upcasts ``sum(Int8)`` to Int64,
# which ``_infer_sum``'s type-preserving rule would get wrong). Widening this
# tuple requires probing the small-int/Float16/128-bit signatures first.
NUMERIC_TYPES = (Int64, Int32, UInt32, UInt64, Float64, Float32)


def _is_numeric(dtype: DataType) -> bool:
    """Check if a type is numeric (considering Nullable wrapper)."""
    inner = dtype.inner if isinstance(dtype, Nullable) else dtype
    return isinstance(inner, NUMERIC_TYPES)


# Aggregation function type signatures
# Each function returns (result_type, preserves_nullability)
# where preserves_nullability means the result should be Nullable if input is Nullable


def _float_reduction_width(inner: DataType) -> DataType:
    """Result width of a float-returning reduction (mean/std/var/median/quantile).

    Probed (polars 1.41.2; backlog N-2): a Float32 receiver returns
    Float32; every other numeric receiver returns Float64.
    """
    return Float32() if isinstance(inner, Float32) else Float64()


def _infer_sum(dtype: DataType) -> DataType:
    """sum(T) -> T for numeric types."""
    inner, is_nullable = _unwrap_nullable(dtype)

    if not isinstance(inner, NUMERIC_TYPES):
        raise GroupByTypeError(f"Cannot apply sum to type {dtype}: sum requires numeric type")

    # sum preserves the type (Int64 -> Int64, Float64 -> Float64)
    return _wrap_nullable(inner, is_nullable)


def _infer_mean(dtype: DataType) -> DataType:
    """mean(T) -> Float64 for numeric types; mean(Float32) -> Float32."""
    inner, is_nullable = _unwrap_nullable(dtype)

    if not isinstance(inner, NUMERIC_TYPES):
        raise GroupByTypeError(f"Cannot apply mean to type {dtype}: mean requires numeric type")

    # Probed (polars 1.41.2; backlog N-2): a Float32 receiver keeps its
    # width in both group_by().agg() and select contexts; every other
    # numeric receiver yields Float64.
    return _wrap_nullable(_float_reduction_width(inner), is_nullable)


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


def _infer_float_reduction(name: str, *, always_nullable: bool = False):
    """Build an inference fn for numeric -> Float64 reductions (std/var/median/quantile).

    A Float32 receiver keeps Float32 (probed, polars 1.41.2; backlog N-2).

    ``always_nullable`` is set for std/var (issue #60): with the default
    ``ddof=1`` they are null whenever only one sample is available (probed,
    polars 1.41.2: any singleton group), so the result is Nullable even on
    non-nullable input. median/quantile are total on non-empty non-null
    input (probed) and only propagate the input's nullability.
    """

    def _infer(dtype: DataType) -> DataType:
        inner, is_nullable = _unwrap_nullable(dtype)
        if not isinstance(inner, NUMERIC_TYPES):
            raise GroupByTypeError(
                f"Cannot apply {name} to type {dtype}: {name} requires numeric type"
            )
        return _wrap_nullable(_float_reduction_width(inner), is_nullable or always_nullable)

    return _infer


def _infer_product(dtype: DataType) -> DataType:
    """product(T) -> T for numeric types."""
    inner, is_nullable = _unwrap_nullable(dtype)
    if not isinstance(inner, NUMERIC_TYPES):
        raise GroupByTypeError(
            f"Cannot apply product to type {dtype}: product requires numeric type"
        )
    return _wrap_nullable(inner, is_nullable)


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
    AggFunction.STD: _infer_float_reduction("std", always_nullable=True),
    AggFunction.VAR: _infer_float_reduction("var", always_nullable=True),
    AggFunction.MEDIAN: _infer_float_reduction("median"),
    AggFunction.QUANTILE: _infer_float_reduction("quantile"),
    AggFunction.PRODUCT: _infer_product,
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
    # Unknown input never raises — the gradual-typing escape hatch must not
    # produce false positives. Count-like aggregations still have a precise
    # result; everything else stays Unknown.
    inner, _ = _unwrap_nullable(input_type)
    if isinstance(inner, Unknown):
        if func in (AggFunction.COUNT, AggFunction.N_UNIQUE):
            return UInt32()
        if func is AggFunction.LIST:
            return List(Unknown())
        return Unknown()
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
    # An open input frame may contain extra unknown columns; references to
    # columns we can't see resolve to Unknown instead of raising.
    is_open = input_frame.rest is not None

    # 1. Add group key columns (preserve their types)
    for key in keys:
        if not input_frame.has_column(key):
            if is_open:
                result_columns[key] = Unknown()
                continue
            raise GroupByTypeError(f"Group by key column '{key}' not found in DataFrame")
        key_type = input_frame.get_column_type(key)
        assert key_type is not None  # We just checked has_column
        result_columns[key] = key_type

    # 2. Add aggregation result columns
    for agg_expr in agg_exprs:
        # Pre-resolved chain form: dtype was computed upstream by the
        # expression analyser, which already validated column references.
        if agg_expr.dtype is not None:
            result_columns[agg_expr.output_name] = agg_expr.dtype
            continue

        col_name = agg_expr.column
        col_type: DataType | None
        if not input_frame.has_column(col_name):
            if not is_open:
                raise GroupByTypeError(f"Aggregation column '{col_name}' not found in DataFrame")
            col_type = Unknown()
        else:
            col_type = input_frame.get_column_type(col_name)
        assert col_type is not None  # We just checked has_column
        assert agg_expr.function is not None  # Direct form must carry a function

        result_type = infer_agg_result_type(agg_expr.function, col_type)
        output_name = agg_expr.output_name
        result_columns[output_name] = result_type

    return FrameType(columns=result_columns)
