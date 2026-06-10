"""Expr type inference for Polars expressions."""

from typing import Any

from polypolarism.types import (
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
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
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Unknown,
    Utf8,
)
from polypolarism.types import (
    List as ListT,
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
        The DataType of the column. On an open frame (``frame.rest`` is not
        ``None``) a missing column resolves to ``Unknown()`` — it may be one
        of the frame's unknown extra columns.

    Raises:
        ColumnNotFoundError: If the column does not exist in a closed frame.
    """
    dtype = frame.get_column_type(column_name)
    if dtype is None:
        if frame.rest is not None:
            return Unknown()
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

    # Unknown absorbs everything — uncertainty propagates, never errors.
    # Mirrors ``unify_types``.
    if isinstance(left_inner, Unknown) or isinstance(right_inner, Unknown):
        return Unknown()

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

    # Unknown absorbs everything — uncertainty propagates, never errors.
    if isinstance(left_inner, Unknown) or isinstance(right_inner, Unknown):
        return Unknown()

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
    raise TypeUnificationError(f"Cannot unify types {left} and {right}")


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
        raise TypeError(f"Condition must be Boolean, got {condition}")

    # Unify then and otherwise types
    unified = unify_types(then_type, otherwise_type)

    # If condition is nullable, result is also nullable
    # (because when condition is null, the result could be null)
    if condition_nullable:
        inner, _ = _unwrap_nullable(unified)
        return Nullable(inner)

    return unified


# ---------------------------------------------------------------------------
# polars common-supertype relation (issues #37/#40/#41/#43)
#
# Ground truth: polars 1.41.2. The full pair matrix over every dtype
# polypolarism models was driven through
# ``pl.when(c).then(pl.col("l")).otherwise(pl.col("r"))`` and cross-checked
# against ``DataFrame.unpivot`` and ``Expr.shift(fill_value=pl.col(...))`` —
# all three operations agree on every probed cell. Representative output::
#
#     Int64 + String   -> String       Boolean + Int64   -> Int64
#     Int32 + Float32  -> Float64      Int8 + Float32    -> Float32
#     Int64 + UInt64   -> Float64      Int8 + UInt8      -> Int16
#     Date + Datetime  -> Datetime     Date + Int64      -> Int64
#     Duration + Int64 -> Int64        Null + Int64      -> Int64 (rows null)
#     List(i64) + List(str) -> List(String)
#     List(i64) + i64  -> SchemaError "failed to determine supertype"
#     Boolean + Date   -> SchemaError  String + Duration -> InvalidOperationError
# ---------------------------------------------------------------------------

_SIGNED_INT_WIDTH: dict[type, int] = {Int8: 8, Int16: 16, Int32: 32, Int64: 64, Int128: 128}
_UNSIGNED_INT_WIDTH: dict[type, int] = {UInt8: 8, UInt16: 16, UInt32: 32, UInt64: 64}
_FLOAT_WIDTH: dict[type, int] = {Float32: 32, Float64: 64}
_SIGNED_BY_WIDTH: dict[int, type] = {8: Int8, 16: Int16, 32: Int32, 64: Int64, 128: Int128}
_UNSIGNED_BY_WIDTH: dict[int, type] = {8: UInt8, 16: UInt16, 32: UInt32, 64: UInt64}

# Temporal x numeric supertypes follow the temporal dtype's *physical*
# representation, with probed quirks that defeat any clean formula
# (Date + UInt64 -> Int64 although Int32 + UInt64 -> Float64; Time + UInt32
# errors although Datetime + UInt32 -> Int64). Widths absent from a table
# are probed SchemaErrors (Int8/Int16/Int128/UInt8/UInt16 everywhere).
_DATE_NUMERIC_SUPERTYPE: dict[type, DataType] = {
    Int32: Int32(),
    Int64: Int64(),
    UInt32: Int64(),
    UInt64: Int64(),
    Float32: Float32(),
    Float64: Float64(),
}
_DATETIME_DURATION_NUMERIC_SUPERTYPE: dict[type, DataType] = {
    Int32: Int64(),
    Int64: Int64(),
    UInt32: Int64(),
    UInt64: Int64(),
    Float32: Float64(),
    Float64: Float64(),
}
_TIME_NUMERIC_SUPERTYPE: dict[type, DataType] = {
    Int32: Int64(),
    Int64: Int64(),
    Float32: Float64(),
    Float64: Float64(),
}


def _is_probed_numeric(dtype: DataType) -> bool:
    """Numeric dtype whose supertype lattice was probed (excludes the exotic
    ``Float16`` / ``UInt128`` widths, which stay :class:`Unknown`)."""
    t = type(dtype)
    return t in _SIGNED_INT_WIDTH or t in _UNSIGNED_INT_WIDTH or t in _FLOAT_WIDTH


def _numeric_supertype(left: DataType, right: DataType) -> DataType:
    """Probed numeric lattice; both operands must satisfy _is_probed_numeric."""
    lt, rt = type(left), type(right)
    l_float, r_float = lt in _FLOAT_WIDTH, rt in _FLOAT_WIDTH
    if l_float and r_float:
        return Float64() if 64 in (_FLOAT_WIDTH[lt], _FLOAT_WIDTH[rt]) else Float32()
    if l_float or r_float:
        float_t, int_t = (lt, rt) if l_float else (rt, lt)
        if _FLOAT_WIDTH[float_t] == 64:
            return Float64()
        # Float32 keeps its width only against ints it can represent
        # exactly (probed: Int8/Int16/UInt8/UInt16 -> Float32, wider -> Float64).
        int_width = _SIGNED_INT_WIDTH.get(int_t) or _UNSIGNED_INT_WIDTH[int_t]
        return Float32() if int_width <= 16 else Float64()
    l_signed, r_signed = lt in _SIGNED_INT_WIDTH, rt in _SIGNED_INT_WIDTH
    if l_signed and r_signed:
        return _SIGNED_BY_WIDTH[max(_SIGNED_INT_WIDTH[lt], _SIGNED_INT_WIDTH[rt])]()
    if not l_signed and not r_signed:
        return _UNSIGNED_BY_WIDTH[max(_UNSIGNED_INT_WIDTH[lt], _UNSIGNED_INT_WIDTH[rt])]()
    signed_t, unsigned_t = (lt, rt) if l_signed else (rt, lt)
    if signed_t is Int128:
        return Int128()  # probed: Int128 absorbs every unsigned width
    needed = max(_SIGNED_INT_WIDTH[signed_t], 2 * _UNSIGNED_INT_WIDTH[unsigned_t])
    if needed <= 64:
        return _SIGNED_BY_WIDTH[needed]()
    return Float64()  # probed: signed(<=64) + UInt64 -> Float64


def _supertype_base(left: DataType, right: DataType) -> DataType | None:
    """Supertype of two bare (non-Nullable, non-Null, non-Unknown) dtypes."""
    if left == right:
        return left

    # Struct combinations are not a real supertype lattice: when/then
    # broadcasts scalars *into* the struct's fields (Struct{f: i64} + Int64
    # -> Struct{f: i64}) and stringifies against Utf8 — both probed, both
    # too field/data-dependent to claim. Stay silent.
    if isinstance(left, Struct) or isinstance(right, Struct):
        return Unknown()

    # List recursion (probed: List(i64) + List(str) -> List(String), nested
    # lists recurse, and an inner pair without a supertype fails the outer
    # pair with the same SchemaError).
    if isinstance(left, ListT) and isinstance(right, ListT):
        inner = supertype(left.inner, right.inner)
        if inner is None:
            return None
        return ListT(inner)
    # List vs any scalar: probed SchemaError / InvalidOperationError for
    # every scalar dtype in the matrix.
    if isinstance(left, ListT) or isinstance(right, ListT):
        return None

    # Utf8 absorbs most scalars (probed; values are stringified).
    if isinstance(left, Utf8) or isinstance(right, Utf8):
        other = right if isinstance(left, Utf8) else left
        if _is_probed_numeric(other) or isinstance(
            other, (Boolean, Date, Time, Datetime, Decimal, Categorical, Enum)
        ):
            return Utf8()
        if isinstance(other, Duration):
            return None  # probed InvalidOperationError, unlike every other temporal
        return Unknown()

    if _is_probed_numeric(left) and _is_probed_numeric(right):
        return _numeric_supertype(left, right)

    # Boolean widens into any probed numeric; everything else errors.
    if isinstance(left, Boolean) or isinstance(right, Boolean):
        other = right if isinstance(left, Boolean) else left
        if _is_probed_numeric(other):
            return other
        if isinstance(other, (Date, Time, Datetime, Duration, Decimal, Categorical, Enum)):
            return None  # probed SchemaError
        return Unknown()

    temporal = (Date, Time, Datetime, Duration)
    l_temporal, r_temporal = isinstance(left, temporal), isinstance(right, temporal)
    if l_temporal and r_temporal:
        if isinstance(left, Date) and isinstance(right, Datetime):
            return right
        if isinstance(left, Datetime) and isinstance(right, Date):
            return left
        # Equal pairs were handled above; what remains is probed to fail:
        # Datetime tz mismatch, Date/Time, Time/Datetime and every
        # Duration/other-temporal pairing.
        return None
    if l_temporal or r_temporal:
        temp, other = (left, right) if l_temporal else (right, left)
        if _is_probed_numeric(other):
            if isinstance(temp, Date):
                return _DATE_NUMERIC_SUPERTYPE.get(type(other))
            if isinstance(temp, Time):
                return _TIME_NUMERIC_SUPERTYPE.get(type(other))
            return _DATETIME_DURATION_NUMERIC_SUPERTYPE.get(type(other))
        if isinstance(other, (Decimal, Categorical, Enum)):
            return None  # probed SchemaError
        return Unknown()

    if isinstance(left, Decimal) or isinstance(right, Decimal):
        other = right if isinstance(left, Decimal) else left
        if type(other) in _FLOAT_WIDTH:
            return Float64()  # probed for both float widths
        if isinstance(other, (Categorical, Enum)):
            return None  # probed SchemaError
        # Decimal + int succeeds with data-dependent precision arithmetic
        # (Decimal(10,2)+Int8 -> Decimal(10,2) but +Int32 -> Decimal(38,2));
        # Decimal + Decimal with differing precision/scale is unprobed.
        return Unknown()

    if isinstance(left, (Categorical, Enum)) or isinstance(right, (Categorical, Enum)):
        # Utf8 / Null / equal-dtype cases were handled above. Every other
        # combination — numerics, Categorical x Enum — is a probed SchemaError.
        return None

    return Unknown()


def supertype(left: DataType, right: DataType) -> DataType | None:
    """Polars' common-supertype relation (probed on polars 1.41.2).

    Used to type ``when/then/otherwise`` branches (#40), ``unpivot`` value
    columns (#41) and ``shift(fill_value=...)`` results (#43).

    Returns:
        - a precise ``DataType`` for probed combinations (a ``Nullable``
          operand makes the result ``Nullable``);
        - ``Unknown()`` when either side is Unknown, or when the probed
          behaviour is too quirky / data-dependent to claim — callers must
          stay silent;
        - ``None`` when polars provably fails at runtime ("failed to
          determine supertype" / InvalidOperationError) — callers may treat
          this as a guaranteed error.
    """
    left_base = left.inner if isinstance(left, Nullable) else left
    right_base = right.inner if isinstance(right, Nullable) else right
    # Unknown absorbs everything — checked before Null so that
    # supertype(Null, Unknown) stays a bare Unknown.
    if isinstance(left_base, Unknown) or isinstance(right_base, Unknown):
        return Unknown()
    # Null + T -> T with the null rows preserved, which polypolarism models
    # as Nullable[T] (probed; consistent with ``unify_types``).
    if isinstance(left, Null) and isinstance(right, Null):
        return Null()
    if isinstance(left, Null):
        return Nullable(right_base)
    if isinstance(right, Null):
        return Nullable(left_base)

    left_inner, left_nullable = _unwrap_nullable(left)
    right_inner, right_nullable = _unwrap_nullable(right)
    result = _supertype_base(left_inner, right_inner)
    if result is None or isinstance(result, Unknown):
        return result
    if left_nullable or right_nullable:
        return result if isinstance(result, Nullable) else Nullable(result)
    return result
