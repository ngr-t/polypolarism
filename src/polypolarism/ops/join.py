"""Join operation type inference."""

from __future__ import annotations

from typing import Literal

from polypolarism.compat.polars_api import join_left_nullable, join_right_nullable
from polypolarism.types import DataType, FrameType, Nullable, Unknown


class JoinError(Exception):
    """Error raised when join operation fails type checking."""

    pass


JoinHow = Literal["inner", "left", "right", "full", "cross", "semi", "anti"]


def _get_base_type(dtype: DataType) -> DataType:
    """Get the base type, unwrapping Nullable if present."""
    if isinstance(dtype, Nullable):
        return dtype.inner
    return dtype


def _types_compatible(left_type: DataType, right_type: DataType) -> bool:
    """Check if two types are compatible for joining (ignoring nullability).

    An ``Unknown`` base on either side is compatible with anything —
    gradual typing: uncertainty must not error (issue #39), consistent
    with ``checker._is_subtype``.
    """
    left_base = _get_base_type(left_type)
    right_base = _get_base_type(right_type)
    if isinstance(left_base, Unknown) or isinstance(right_base, Unknown):
        return True
    return left_base == right_base


def _make_nullable(dtype: DataType) -> DataType:
    """Make a type nullable, avoiding double-wrapping."""
    if isinstance(dtype, Nullable):
        return dtype
    return Nullable(dtype)


def _as_key_list(value: str | list[str]) -> list[str]:
    """Normalize a single key or a list of keys to a list."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _right_pin_dtype(dtype: DataType, left: FrameType, *, collides_with_pin: bool) -> DataType:
    """Dtype claimable for a right-side column pinned into the join result.

    Issue #79: with an OPEN left frame, a right column that does NOT
    collide with a pinned left name may still collide with one of the
    left ``rest``'s unknown extras — polars would then suffix the RIGHT
    column away and the unsuffixed name would be the LEFT column, dtype
    unknown. The name itself provably exists in the output either way,
    but its dtype is conditional — claim ``Unknown`` instead of
    manufacturing a proof. Collisions with PINNED left names are
    deterministic (the suffix is decided), so those keep the precise
    dtype; a closed left frame keeps every pin precise.
    """
    if left.rest is not None and not collides_with_pin:
        return Unknown()
    return dtype


def infer_join(
    left: FrameType,
    right: FrameType,
    *,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: JoinHow = "inner",
    suffix: str = "_right",
    coalesce: bool | None = None,
) -> FrameType:
    """
    Infer the result FrameType of a join operation.

    Args:
        left: Left FrameType
        right: Right FrameType
        on: Column name(s) to join on (same name in both frames)
        left_on: Column name(s) in left frame to join on
        right_on: Column name(s) in right frame to join on
        how: Join type ('inner', 'left', 'right', 'full', 'cross',
            'semi', 'anti')
        suffix: Suffix appended to right-side columns whose name collides
            with a left-side column (polars default: '_right'; irrelevant
            for 'semi'/'anti', which add no right-side columns)
        coalesce: Whether the key columns of the two sides are merged into
            one. ``None`` (the polars default) coalesces for equi-joins
            except 'full'; ignored for 'cross'/'semi'/'anti'

    Returns:
        FrameType representing the result of the join

    Raises:
        JoinError: If join key is missing or types don't match
    """
    # An open input side (ADR-0006) makes the result open too: unknown
    # extra columns survive every join kind, and their collisions with
    # the other side's names (suffixing) cannot be tracked.
    result_rest = left.rest or right.rest

    # Negative knowledge survives a join (issue #78): a name provably
    # lacking on BOTH sides cannot appear in the output (suffixed names
    # are new spellings; reintroductions are pinned and subtracted by the
    # FrameType constructor).
    result_absent = {
        name for name in (left.absent | right.absent) if left.lacks(name) and right.lacks(name)
    }

    # Cross joins take no keys: the result is simply all left columns plus
    # all right columns (suffixed on collision), with dtypes unchanged and
    # no nullability introduced.
    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            raise JoinError("cross join takes no join keys")
        result_columns: dict[str, DataType] = {
            col_name: col_spec.dtype for col_name, col_spec in left.columns.items()
        }
        for col_name, col_spec in right.columns.items():
            final_name = f"{col_name}{suffix}" if col_name in result_columns else col_name
            result_columns[final_name] = _right_pin_dtype(
                col_spec.dtype, left, collides_with_pin=final_name != col_name
            )
        return FrameType(columns=result_columns, rest=result_rest, absent=result_absent)

    # Determine join keys (each key pair is validated independently).
    if on is not None:
        left_keys = _as_key_list(on)
        right_keys = list(left_keys)
    elif left_on is not None and right_on is not None:
        left_keys = _as_key_list(left_on)
        right_keys = _as_key_list(right_on)
        if len(left_keys) != len(right_keys):
            raise JoinError(
                f"'left_on' and 'right_on' must have the same number of columns "
                f"({len(left_keys)} vs {len(right_keys)})"
            )
    else:
        raise JoinError("Must specify either 'on' or both 'left_on' and 'right_on'")

    if not left_keys:
        raise JoinError("join: at least one join key required")

    # Validate every key pair: existence on both sides + dtype compatibility.
    # On an OPEN side a missing key may exist among the unknown extras
    # (ADR-0006) — assume the join succeeded and treat it as Unknown.
    # Remember each pair's right-side type so a coalesced full join can
    # check whether nullability leaks in from the right key.
    right_type_for_left_key: dict[str, DataType] = {}
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        left_key_type = left.get_column_type(left_key)
        if left_key_type is None:
            if left.lacks(left_key):
                # Closed frame, or provably removed on an open one (#78).
                raise JoinError(f"Column '{left_key}' not found in left frame")
            left_key_type = Unknown()

        right_key_type = right.get_column_type(right_key)
        if right_key_type is None:
            if right.lacks(right_key):
                raise JoinError(f"Column '{right_key}' not found in right frame")
            right_key_type = Unknown()

        if not _types_compatible(left_key_type, right_key_type):
            raise JoinError(
                f"Join key dtype mismatch: left '{left_key}' is {left_key_type}, "
                f"right '{right_key}' is {right_key_type}"
            )
        right_type_for_left_key[left_key] = right_key_type

    # Semi/anti joins only filter rows: the result is exactly the left
    # frame's schema — no right columns, no nullability changes.
    if how in ("semi", "anti"):
        return FrameType(
            columns=dict(left.columns), strict=left.strict, rest=left.rest, absent=left.absent
        )

    # polars coalesces the key columns by default for inner/left/right
    # equi-joins, but NOT for full joins (both keys are kept, the right
    # one suffixed). An explicit ``coalesce=`` overrides the default.
    effective_coalesce = coalesce if coalesce is not None else how != "full"

    # Determine which columns from each side need to be made nullable
    left_nullable = join_left_nullable(how)
    right_nullable = join_right_nullable(how)

    # Coalescing drops one side's key columns: a right join keeps the
    # right key column (under its own name) and drops the left one;
    # inner/left/full keep the left key and drop the right one. Without
    # coalescing, the keys behave like ordinary columns from each side.
    left_key_set = set(left_keys)
    if effective_coalesce:
        drop_left_keys = left_key_set if how == "right" else set()
        drop_right_keys = set() if how == "right" else set(right_keys)
    else:
        drop_left_keys = set()
        drop_right_keys = set()

    # Build result columns
    result_columns = {}

    # Add left columns
    for col_name, col_spec in left.columns.items():
        if col_name in drop_left_keys:
            continue
        col_type = col_spec.dtype
        if effective_coalesce and col_name in left_key_set:
            # Surviving coalesced key (inner/left/full — a right join drops
            # the left key instead, so it never reaches this branch).
            if how == "full" and isinstance(right_type_for_left_key[col_name], Nullable):
                # A coalesced full-join key is null only where BOTH input
                # keys are null, so it is nullable only if either input
                # key is nullable (#24).
                result_columns[col_name] = _make_nullable(col_type)
            else:
                result_columns[col_name] = col_type
        elif left_nullable:
            result_columns[col_name] = _make_nullable(col_type)
        else:
            result_columns[col_name] = col_type

    # Add right columns. With an open left frame an unsuffixed pin's
    # dtype is conditional on no left-rest collision (issue #79) — the
    # NAME provably exists in the output either way (as the right column,
    # or as the left's colliding column), but the dtype degrades to
    # Unknown; the join-kind nullability wrap is skipped for degraded
    # pins (it belongs to the right-column world only).
    for col_name, col_spec in right.columns.items():
        if col_name in drop_right_keys:
            continue
        col_type = col_spec.dtype

        # Handle column name conflicts
        final_name = col_name
        if col_name in result_columns:
            final_name = f"{col_name}{suffix}"

        pinned = _right_pin_dtype(col_type, left, collides_with_pin=final_name != col_name)
        if right_nullable and not isinstance(pinned, Unknown):
            pinned = _make_nullable(pinned)
        result_columns[final_name] = pinned

    return FrameType(columns=result_columns, rest=result_rest, absent=result_absent)
