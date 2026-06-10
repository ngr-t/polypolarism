"""Frame restructuring (concat / unpivot) type inference."""

from __future__ import annotations

from polypolarism.expr_infer import TypeUnificationError, supertype, unify_types
from polypolarism.types import (
    ColumnSpec,
    DataType,
    FrameType,
    Nullable,
    Utf8,
)


class ReshapeError(Exception):
    """Raised when a restructuring op cannot be type-inferred."""


def _make_nullable(dtype: DataType) -> DataType:
    return dtype if isinstance(dtype, Nullable) else Nullable(dtype)


def concat_vertical(frames: list[FrameType]) -> FrameType:
    """Vertical concat: every input must have the same column set.

    Per-column dtypes are unified. ``required`` is preserved if all inputs
    agree; otherwise the column becomes optional. Nullability is widened
    if any input is Nullable.
    """
    if not frames:
        raise ReshapeError("concat: at least one frame required")
    base = frames[0]
    for other in frames[1:]:
        if set(base.columns.keys()) != set(other.columns.keys()):
            extra_left = set(base.columns) - set(other.columns)
            extra_right = set(other.columns) - set(base.columns)
            parts = []
            if extra_left:
                parts.append(f"first frame has {sorted(extra_left)} not in another")
            if extra_right:
                parts.append(f"another frame has {sorted(extra_right)} not in first")
            raise ReshapeError("concat(how='vertical'): column sets differ — " + "; ".join(parts))

    merged: dict[str, ColumnSpec] = {}
    for col_name, base_spec in base.columns.items():
        cur_dtype: DataType = base_spec.dtype
        cur_required = base_spec.required
        for other in frames[1:]:
            spec = other.columns[col_name]
            try:
                cur_dtype = unify_types(cur_dtype, spec.dtype)
            except TypeUnificationError as e:
                raise ReshapeError(f"concat: column '{col_name}' — {e}") from e
            cur_required = cur_required and spec.required
        merged[col_name] = ColumnSpec(dtype=cur_dtype, required=cur_required)
    return FrameType(columns=merged)


def concat_horizontal(frames: list[FrameType]) -> FrameType:
    """Horizontal concat: column sets must be disjoint."""
    merged: dict[str, ColumnSpec] = {}
    for ft in frames:
        for col_name, spec in ft.columns.items():
            if col_name in merged:
                raise ReshapeError(
                    f"concat(how='horizontal'): column '{col_name}' appears in multiple frames"
                )
            merged[col_name] = spec
    return FrameType(columns=merged)


def concat_diagonal(frames: list[FrameType]) -> FrameType:
    """Diagonal concat: union of columns; columns absent in any input are Nullable."""
    all_names: list[str] = []
    seen: set[str] = set()
    for ft in frames:
        for name in ft.columns:
            if name not in seen:
                seen.add(name)
                all_names.append(name)

    merged: dict[str, ColumnSpec] = {}
    for name in all_names:
        present_in_all = True
        cur_dtype: DataType | None = None
        cur_required = True
        for ft in frames:
            spec = ft.columns.get(name)
            if spec is None:
                present_in_all = False
                continue
            if cur_dtype is None:
                cur_dtype = spec.dtype
            else:
                try:
                    cur_dtype = unify_types(cur_dtype, spec.dtype)
                except TypeUnificationError as e:
                    raise ReshapeError(f"concat(how='diagonal'): column '{name}' — {e}") from e
            cur_required = cur_required and spec.required
        assert cur_dtype is not None
        if not present_in_all:
            cur_dtype = _make_nullable(cur_dtype)
        merged[name] = ColumnSpec(dtype=cur_dtype, required=cur_required)
    return FrameType(columns=merged)


def unpivot(
    input_frame: FrameType,
    index: list[str],
    on: list[str],
    variable_name: str = "variable",
    value_name: str = "value",
) -> FrameType:
    """``unpivot(index=..., on=..., variable_name=..., value_name=...)``.

    Output schema: ``{index..., variable_name: Utf8, value_name: T}`` where T
    is the polars common supertype of the ``on`` columns' dtypes (issue #41).
    Probed (polars 1.41.2): ``unpivot(index="id", on=["a", "s"])`` with
    ``a: Int64``, ``s: String`` yields ``value: String``; only combinations
    without a polars supertype (e.g. ``List`` + scalar) raise — quirky
    combinations ``supertype`` does not model degrade to ``Unknown``.
    """
    for col in index:
        if col not in input_frame.columns:
            raise ReshapeError(f"unpivot: index column '{col}' not found")
    for col in on:
        if col not in input_frame.columns:
            raise ReshapeError(f"unpivot: value column '{col}' not found")

    if not on:
        raise ReshapeError("unpivot: at least one `on` column required")

    value_dtype: DataType = input_frame.columns[on[0]].dtype
    for col in on[1:]:
        spec = input_frame.columns[col]
        merged = supertype(value_dtype, spec.dtype)
        if merged is None:
            raise ReshapeError(
                f"unpivot: value columns {on} have incompatible dtypes — "
                f"polars finds no common supertype of {value_dtype} and {spec.dtype}"
            )
        value_dtype = merged

    out: dict[str, ColumnSpec] = {}
    for col in index:
        out[col] = input_frame.columns[col]
    out[variable_name] = ColumnSpec(dtype=Utf8())
    out[value_name] = ColumnSpec(dtype=value_dtype)
    return FrameType(columns=out)
