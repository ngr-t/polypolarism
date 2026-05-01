"""Translate Pandera DataFrameModel field annotations to ColumnSpec.

Recognised annotation shapes:
- Python builtins: ``int``, ``str``, ``float``, ``bool``
- Polars dtype classes: ``pl.Int64``, ``pl.Utf8``, ``pl.Float64`` etc. (with or without ``()``)
- ``Series[T]`` (the canonical pandera class-based form) -> equivalent to bare ``T``.
  Both bare and qualified heads (``Series``, ``pa.typing.Series``,
  ``pandera.typing.polars.Series``) are accepted.
- ``Optional[T]`` and ``T | None`` -> ``required=False``
- ``Annotated[pl.List, pl.Int64()]`` -> ``List(Int64())``
- ``Annotated[pl.Array, pl.Int64(), 3]`` -> ``List(Int64())`` (width ignored)
- ``Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]`` -> ``Struct({...})``

Field RHS:
- ``pa.Field(nullable=True)`` wraps the parsed dtype in ``Nullable(...)``.
- Other Field options (``coerce``, ``unique``, value constraints) are runtime-only and ignored.
"""

from __future__ import annotations

import ast

from polypolarism.types import (
    Boolean,
    Categorical,
    ColumnSpec,
    DataType,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    Nullable,
    Struct,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

_BUILTIN_MAP: dict[str, DataType] = {
    "int": Int64(),
    "str": Utf8(),
    "float": Float64(),
    "bool": Boolean(),
    "bytes": Utf8(),
    # Python stdlib temporal types — accepted as bare names when the
    # user wrote ``from datetime import date, datetime`` (the canonical
    # pandera form). The ``<module>.date`` qualified form is handled in
    # ``_parse_plain_dtype`` separately.
    "date": Date(),
    "datetime": Datetime(),
    "timedelta": Duration(),
}


# Stdlib temporal type attribute names: ``datetime.date`` / ``dt.datetime`` /
# etc. — accepted under any prefix that isn't ``pl`` (which has its own
# uppercase-attribute table). Names are deliberately lowercase to avoid
# colliding with ``pl.Date`` / ``pl.Datetime`` / ``pl.Duration``.
_STDLIB_TEMPORAL_ATTR_MAP: dict[str, DataType] = {
    "date": Date(),
    "datetime": Datetime(),
    "timedelta": Duration(),
}


_PL_DTYPE_MAP: dict[str, DataType] = {
    "Int8": Int8(),
    "Int16": Int16(),
    "Int32": Int32(),
    "Int64": Int64(),
    "UInt8": UInt8(),
    "UInt16": UInt16(),
    "UInt32": UInt32(),
    "UInt64": UInt64(),
    "Float32": Float32(),
    "Float64": Float64(),
    "Utf8": Utf8(),
    "String": Utf8(),
    "Boolean": Boolean(),
    "Date": Date(),
    "Datetime": Datetime(),
    "Duration": Duration(),
    "Categorical": Categorical(),
    "Null": Null(),
}


def parse_field_annotation(
    annotation: ast.expr,
    value: ast.expr | None = None,
) -> ColumnSpec | None:
    """Parse a Pandera class-body field annotation + optional ``pa.Field(...)`` value.

    Returns ``None`` if the annotation cannot be translated.
    """
    parsed = _parse_dtype_expr(annotation)
    if parsed is None:
        return None
    dtype, required = parsed

    if value is not None and _is_field_with_nullable(value):
        dtype = _make_nullable(dtype)

    return ColumnSpec(dtype=dtype, required=required)


def _parse_dtype_expr(node: ast.expr) -> tuple[DataType, bool] | None:
    """Parse a type expression. Returns (dtype, required)."""
    if _is_optional(node):
        inner = _optional_inner(node)
        if inner is None:
            return None
        result = _parse_dtype_expr(inner)
        if result is None:
            return None
        dtype, _ = result
        return dtype, False

    if _is_series(node):
        # ``Series[T]`` is pandera's canonical class-based form; the
        # wrapper carries no extra info we use, so unwrap to ``T`` and
        # re-enter the parser. Recursing handles ``Series[Optional[T]]``
        # and other nested shapes for free.
        assert isinstance(node, ast.Subscript)
        return _parse_dtype_expr(node.slice)

    if _is_annotated(node):
        return _parse_annotated(node)

    dtype = _parse_plain_dtype(node)
    if dtype is None:
        return None
    return dtype, True


def _is_series(node: ast.expr) -> bool:
    """``Series[T]`` (bare or qualified, e.g. ``pa.typing.Series[T]``)."""
    return isinstance(node, ast.Subscript) and _name_matches(node.value, "Series")


def _parse_plain_dtype(node: ast.expr) -> DataType | None:
    """Parse a non-Optional, non-Annotated dtype expression."""
    if isinstance(node, ast.Name):
        return _BUILTIN_MAP.get(node.id)

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == "pl":
                return _PL_DTYPE_MAP.get(node.attr)
            # ``datetime.date`` / ``dt.datetime`` / ``datetime.timedelta``
            # — any non-``pl`` prefix is treated as the Python stdlib
            # ``datetime`` module (or an alias of it). Lowercase attr
            # names disambiguate from polars' uppercase dtype names.
            if node.attr in _STDLIB_TEMPORAL_ATTR_MAP:
                return _STDLIB_TEMPORAL_ATTR_MAP[node.attr]
        return None

    if isinstance(node, ast.Call):
        return _parse_plain_dtype(node.func)

    return None


def _is_optional(node: ast.expr) -> bool:
    if isinstance(node, ast.Subscript) and _name_matches(node.value, "Optional"):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_none_constant(node.left) or _is_none_constant(node.right)
    return False


def _optional_inner(node: ast.expr) -> ast.expr | None:
    if isinstance(node, ast.Subscript):
        return node.slice
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if _is_none_constant(node.right):
            return node.left
        if _is_none_constant(node.left):
            return node.right
    return None


def _is_none_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_annotated(node: ast.expr) -> bool:
    return isinstance(node, ast.Subscript) and _name_matches(node.value, "Annotated")


def _parse_annotated(node: ast.expr) -> tuple[DataType, bool] | None:
    assert isinstance(node, ast.Subscript)
    slice_ = node.slice
    elts = slice_.elts if isinstance(slice_, ast.Tuple) else [slice_]
    if not elts:
        return None
    head = elts[0]
    meta = elts[1:]

    if _is_pl_attr(head, "List") or _is_pl_attr(head, "Array"):
        if not meta:
            return None
        inner_dtype = _parse_plain_dtype(meta[0])
        if inner_dtype is None:
            return None
        return List(inner_dtype), True

    if _is_pl_attr(head, "Struct"):
        if not meta or not isinstance(meta[0], ast.Dict):
            return None
        fields: dict[str, DataType] = {}
        for k, v in zip(meta[0].keys, meta[0].values, strict=True):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            inner = _parse_plain_dtype(v)
            if inner is None:
                return None
            fields[k.value] = inner
        return Struct(fields), True

    return _parse_dtype_expr(head)


def _is_pl_attr(node: ast.expr, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pl"
        and node.attr == attr
    )


def _name_matches(node: ast.expr, name: str) -> bool:
    """Check if node matches a bare ``Name(name)`` or any qualified ``X.name``."""
    if isinstance(node, ast.Name) and node.id == name:
        return True
    return isinstance(node, ast.Attribute) and node.attr == name


def _is_field_with_nullable(node: ast.expr) -> bool:
    """Check if value expression is ``Field(nullable=True, ...)`` or ``pa.Field(nullable=True, ...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        if func.id != "Field":
            return False
    elif isinstance(func, ast.Attribute):
        if func.attr != "Field":
            return False
    else:
        return False
    return any(kw.arg == "nullable" and _is_true_constant(kw.value) for kw in node.keywords)


def _is_true_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _make_nullable(dtype: DataType) -> DataType:
    if isinstance(dtype, Nullable):
        return dtype
    return Nullable(dtype)
