"""Frame restructuring (concat / unpivot) type inference."""

from __future__ import annotations

from polypolarism.expr_infer import TypeUnificationError, supertype, unify_types
from polypolarism.types import (
    ColumnSpec,
    DataType,
    FrameType,
    Nullable,
    RowVar,
    Unknown,
    Utf8,
)


class ReshapeError(Exception):
    """Raised when a restructuring op cannot be type-inferred."""


def _make_nullable(dtype: DataType) -> DataType:
    return dtype if isinstance(dtype, Nullable) else Nullable(dtype)


def _concat_rest(frames: list[FrameType]) -> RowVar | None:
    """Result openness of a concat: open iff any input is open (ADR-0006)."""
    if any(f.rest is not None for f in frames):
        return RowVar("concat")
    return None


def _pinned_union(frames: list[FrameType]) -> list[str]:
    """Union of pinned column names across inputs, first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for ft in frames:
        for name in ft.columns:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def concat_vertical(frames: list[FrameType]) -> FrameType:
    """Vertical concat: every input must have the same column set.

    Per-column dtypes are unified. ``required`` is preserved if all inputs
    agree; otherwise the column becomes optional. Nullability is widened
    if any input is Nullable.

    With OPEN inputs (ADR-0006) the per-column contribution rule applies:
    a frame contributes its pinned dtype; an open frame that does not pin
    the column may hold it among its unknown extras and contributes
    ``Unknown``; a CLOSED frame that lacks a column pinned elsewhere is a
    provable runtime mismatch. Pinned-vs-pinned conflicts are unified
    FIRST so a provable dtype error is not masked by an Unknown
    contribution.
    """
    if not frames:
        raise ReshapeError("concat: at least one frame required")
    if all(f.rest is None for f in frames):
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
                raise ReshapeError(
                    "concat(how='vertical'): column sets differ — " + "; ".join(parts)
                )

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

    merged_open: dict[str, ColumnSpec] = {}
    for name in _pinned_union(frames):
        pinned_specs = [f.columns[name] for f in frames if name in f.columns]
        for f in frames:
            if name not in f.columns and f.rest is None:
                raise ReshapeError(
                    f"concat(how='vertical'): column '{name}' is missing from a frame "
                    f"with a fully known column set"
                )
        cur = pinned_specs[0].dtype
        for spec in pinned_specs[1:]:
            try:
                cur = unify_types(cur, spec.dtype)
            except TypeUnificationError as e:
                raise ReshapeError(f"concat: column '{name}' — {e}") from e
        if any(name not in f.columns for f in frames):
            # Some open frame may hold the column with an unknowable dtype.
            cur = Unknown()
        merged_open[name] = ColumnSpec(
            dtype=cur, required=all(spec.required for spec in pinned_specs)
        )
    return FrameType(columns=merged_open, rest=_concat_rest(frames))


def concat_horizontal(frames: list[FrameType]) -> FrameType:
    """Horizontal concat: column sets must be disjoint.

    Disjointness is provable only among pinned columns; collisions with an
    open frame's unknown extras stay silent, and any open input makes the
    result open (ADR-0006).
    """
    merged: dict[str, ColumnSpec] = {}
    for ft in frames:
        for col_name, spec in ft.columns.items():
            if col_name in merged:
                raise ReshapeError(
                    f"concat(how='horizontal'): column '{col_name}' appears in multiple frames"
                )
            merged[col_name] = spec
    return FrameType(columns=merged, rest=_concat_rest(frames))


def concat_diagonal(frames: list[FrameType]) -> FrameType:
    """Diagonal concat: union of columns; columns absent in any input are Nullable.

    With OPEN inputs (ADR-0006): a column's presence in an open frame that
    does not pin it is unknowable, so both the null-fill wrap and the dtype
    are unclaimable — the column degrades to ``Unknown`` (absence from a
    CLOSED frame still proves the Nullable wrap). Any open input makes the
    result open.
    """
    merged: dict[str, ColumnSpec] = {}
    for name in _pinned_union(frames):
        absent_from_closed = False
        unknowable_presence = False
        cur_dtype: DataType | None = None
        cur_required = True
        for ft in frames:
            spec = ft.columns.get(name)
            if spec is None:
                if ft.rest is None:
                    absent_from_closed = True
                else:
                    unknowable_presence = True
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
        if unknowable_presence:
            cur_dtype = Unknown()
        elif absent_from_closed:
            cur_dtype = _make_nullable(cur_dtype)
        merged[name] = ColumnSpec(dtype=cur_dtype, required=cur_required)
    return FrameType(columns=merged, rest=_concat_rest(frames))


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
    is_open = input_frame.rest is not None
    if not is_open:
        for col in index:
            if col not in input_frame.columns:
                raise ReshapeError(f"unpivot: index column '{col}' not found")
        for col in on:
            if col not in input_frame.columns:
                raise ReshapeError(f"unpivot: value column '{col}' not found")

    if not on:
        raise ReshapeError("unpivot: at least one `on` column required")

    # Fold the supertype over the PINNED `on` columns first so a provable
    # incompatibility is reported even on an open frame; unpinned columns
    # (possible among an open frame's extras — ADR-0006) then absorb the
    # claim into Unknown.
    pinned_on = [col for col in on if col in input_frame.columns]
    value_dtype: DataType = input_frame.columns[pinned_on[0]].dtype if pinned_on else Unknown()
    for col in pinned_on[1:]:
        spec = input_frame.columns[col]
        merged = supertype(value_dtype, spec.dtype)
        if merged is None:
            raise ReshapeError(
                f"unpivot: value columns {on} have incompatible dtypes — "
                f"polars finds no common supertype of {value_dtype} and {spec.dtype}"
            )
        value_dtype = merged
    if len(pinned_on) < len(on):
        value_dtype = Unknown()

    # The output shape is fully determined by the call (index + variable +
    # value) — the result is CLOSED even when the input was open.
    out: dict[str, ColumnSpec] = {}
    for col in index:
        spec_or_none = input_frame.columns.get(col)
        out[col] = spec_or_none if spec_or_none is not None else ColumnSpec(dtype=Unknown())
    out[variable_name] = ColumnSpec(dtype=Utf8())
    out[value_name] = ColumnSpec(dtype=value_dtype)
    return FrameType(columns=out)
