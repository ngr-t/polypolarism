"""Join operation type inference."""

from __future__ import annotations

from typing import Literal, Optional

from polypolarism.types import DataType, FrameType, Nullable


class JoinError(Exception):
    """Error raised when join operation fails type checking."""

    pass


JoinHow = Literal["inner", "left", "right", "full"]


def _get_base_type(dtype: DataType) -> DataType:
    """Get the base type, unwrapping Nullable if present."""
    if isinstance(dtype, Nullable):
        return dtype.inner
    return dtype


def _types_compatible(left_type: DataType, right_type: DataType) -> bool:
    """Check if two types are compatible for joining (ignoring nullability)."""
    return _get_base_type(left_type) == _get_base_type(right_type)


def _make_nullable(dtype: DataType) -> DataType:
    """Make a type nullable, avoiding double-wrapping."""
    if isinstance(dtype, Nullable):
        return dtype
    return Nullable(dtype)


def infer_join(
    left: FrameType,
    right: FrameType,
    *,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: JoinHow = "inner",
) -> FrameType:
    """
    Infer the result FrameType of a join operation.

    Args:
        left: Left FrameType
        right: Right FrameType
        on: Column name to join on (same name in both frames)
        left_on: Column name in left frame to join on
        right_on: Column name in right frame to join on
        how: Join type ('inner', 'left', 'right', 'full')

    Returns:
        FrameType representing the result of the join

    Raises:
        JoinError: If join key is missing or types don't match
    """
    # Determine join keys
    if on is not None:
        left_key = on
        right_key = on
    elif left_on is not None and right_on is not None:
        left_key = left_on
        right_key = right_on
    else:
        raise JoinError("Must specify either 'on' or both 'left_on' and 'right_on'")

    # Validate left key exists
    left_key_type = left.get_column_type(left_key)
    if left_key_type is None:
        raise JoinError(f"Column '{left_key}' not found in left frame")

    # Validate right key exists
    right_key_type = right.get_column_type(right_key)
    if right_key_type is None:
        raise JoinError(f"Column '{right_key}' not found in right frame")

    # Validate types are compatible
    if not _types_compatible(left_key_type, right_key_type):
        raise JoinError(
            f"Join key dtype mismatch: left '{left_key}' is {left_key_type}, "
            f"right '{right_key}' is {right_key_type}"
        )

    # Build result columns
    result_columns: dict[str, DataType] = {}

    # Determine which columns from each side need to be made nullable
    left_nullable = how in ("right", "full")
    right_nullable = how in ("left", "full")

    # Determine key columns to skip from right side
    # When using 'on', the key column from right is not added (uses left's key)
    # When using left_on/right_on, both key columns are preserved
    skip_right_key = on is not None

    # For 'on' join, determine which side's key type to use:
    # - inner/left: use left key type (non-nullable unless originally nullable)
    # - right: use right key type (non-nullable unless originally nullable)
    # - full: key becomes nullable
    key_from_right = how == "right"

    # Add left columns
    for col_name, col_type in left.columns.items():
        # Special handling for key column when using 'on'
        if skip_right_key and col_name == left_key:
            if how == "full":
                result_columns[col_name] = _make_nullable(col_type)
            elif key_from_right:
                # For right join, key column uses right side's type (not nullable)
                result_columns[col_name] = right_key_type
            else:
                # For inner/left, key uses left side's type
                result_columns[col_name] = col_type
        elif left_nullable:
            result_columns[col_name] = _make_nullable(col_type)
        else:
            result_columns[col_name] = col_type

    # Add right columns
    for col_name, col_type in right.columns.items():
        # Skip the key column if using 'on'
        if skip_right_key and col_name == right_key:
            continue

        # Handle column name conflicts
        final_name = col_name
        if col_name in result_columns:
            final_name = f"{col_name}_right"

        # Apply nullability
        if right_nullable:
            result_columns[final_name] = _make_nullable(col_type)
        else:
            result_columns[final_name] = col_type

    return FrameType(columns=result_columns)
