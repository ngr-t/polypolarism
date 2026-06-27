"""Translate Patito ``Model`` field annotations to ``ColumnSpec`` (ADR-0010).

Patito fields are plain Python type hints whose Patito semantics differ from
Pandera's in three ways (all probed against patito 0.8.6 / polars 1.42 via
``Model.dtypes`` / ``Model.valid_dtypes`` / ``Model.nullable_columns``):

- ``Optional[T]`` / ``T | None`` makes the VALUE nullable (the column is
  still required), the inverse of Pandera's ``required=False``.
- ``int`` accepts ANY integer width and ``float`` ANY float width — modeled
  as a :class:`~polypolarism.types.DataTypeGroup` so a non-canonical width is
  not falsely rejected (ADR-0009). ``Literal["a", "b"]`` accepts ``String``
  or its ``Enum``.
- ``pt.Field(dtype=pl.UInt16)`` forces an exact polars dtype, overriding the
  annotation's default mapping.

A nested ``Inner(pt.Model)`` field becomes a ``Struct`` of the inner model's
columns; resolution is two-phase (the collector fills it in once every model
is parsed), so this module only flags the reference via the returned
``nested_model`` name.

The dialect-neutral leaf parsing (``pl.Int64``, ``pl.List(...)``,
``datetime.date``, ``decimal.Decimal`` …) is reused from
:mod:`polypolarism.pandera_dtype`.
"""

from __future__ import annotations

import ast

from polypolarism.compat.patito_api import PATITO_FIELD_CALLABLE, PATITO_FIELD_DTYPE_KW
from polypolarism.pandera_dtype import _is_optional, _optional_inner, _parse_plain_dtype
from polypolarism.types import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    Binary,
    Boolean,
    ColumnSpec,
    DataType,
    DataTypeGroup,
    Datetime,
    Duration,
    Enum,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    Unknown,
    Utf8,
)


def integer_group() -> DataTypeGroup:
    """Patito ``int`` — any integer width satisfies the slot (canonical Int64)."""
    return DataTypeGroup(
        frozenset(cls() for cls in INTEGER_DTYPES), label="integer", canonical=Int64()
    )


def float_group() -> DataTypeGroup:
    """Patito ``float`` — any float width satisfies the slot (canonical Float64)."""
    return DataTypeGroup(
        frozenset(cls() for cls in FLOAT_DTYPES), label="float", canonical=Float64()
    )


def datetime_group() -> DataTypeGroup:
    """Patito ``datetime.datetime`` — any Datetime (any time unit / time zone)
    satisfies the slot (#119, probed). The single canonical member acts as a
    Datetime type-class wildcard in the checker's group membership; a ``Date``
    column is still rejected. Canonical ``Datetime()`` (us) for inference math.
    """
    return DataTypeGroup(frozenset({Datetime()}), label="datetime", canonical=Datetime())


def duration_group() -> DataTypeGroup:
    """Patito ``datetime.timedelta`` — any Duration (any time unit) satisfies
    the slot (#119, probed). Canonical ``Duration()`` (us) for inference math.
    """
    return DataTypeGroup(frozenset({Duration()}), label="timedelta", canonical=Duration())


def parse_patito_field(
    annotation: ast.expr,
    value: ast.expr | None = None,
    model_names: frozenset[str] = frozenset(),
) -> tuple[ColumnSpec, str | None]:
    """Parse one Patito field into ``(ColumnSpec, nested_model_name | None)``.

    ``model_names`` is the set of known Patito model class names so a field
    annotated with another model is recognised as a nested struct; the name is
    returned for the collector's second resolution pass (the spec carries an
    open ``Struct`` placeholder until then). Returns ``ColumnSpec(Unknown())``
    for annotations this translator does not recognise (sound — ``Unknown``
    accepts everything; the column still exists).
    """
    inner, nullable = _strip_optional(annotation)

    override = _field_dtype_override(value)
    if override is not None:
        dtype: DataType = override
        nested: str | None = None
    else:
        dtype, nested = _patito_dtype(inner, model_names)

    if nullable:
        dtype = Nullable(dtype)

    # All Patito columns are required (a missing declared column raises at
    # validate time — probed). ``Optional`` only toggles value nullability.
    return ColumnSpec(dtype=dtype, required=True), nested


def _strip_optional(node: ast.expr) -> tuple[ast.expr, bool]:
    """Unwrap one ``Optional[T]`` / ``T | None`` layer; return ``(inner, nullable)``."""
    if _is_optional(node):
        inner = _optional_inner(node)
        if inner is not None:
            return inner, True
    return node, False


def _patito_dtype(node: ast.expr, model_names: frozenset[str]) -> tuple[DataType, str | None]:
    """Map a (already Optional-stripped) annotation to ``(dtype, nested_model)``."""
    # Stdlib temporal types that carry a unit map to acceptance groups (#119):
    # ``datetime.datetime`` accepts any Datetime (unit/tz), ``timedelta`` any
    # Duration. Matched on the trailing name so the bare (``datetime`` /
    # ``timedelta``) and qualified (``dt.datetime`` / ``datetime.timedelta``)
    # forms both resolve. ``date`` / ``time`` carry no unit and stay exact
    # (handled by the leaf parser below); ``pl.Datetime`` is capitalised, so
    # this lowercase match never claims it. A ``Field(dtype=pl.Datetime("ms"))``
    # override is exact and handled before this function runs.
    tail = _tail_name(node)
    if tail == "datetime":
        return datetime_group(), None
    if tail == "timedelta":
        return duration_group(), None

    # Bare builtin names with Patito-specific group semantics.
    if isinstance(node, ast.Name):
        if node.id == "int":
            return integer_group(), None
        if node.id == "float":
            return float_group(), None
        if node.id == "str":
            return Utf8(), None
        if node.id == "bool":
            return Boolean(), None
        if node.id == "bytes":
            return Binary(), None
        if node.id in model_names:
            # Nested model -> Struct; resolved in the collector's second pass.
            return Struct(open=True), node.id

    # ``Literal[...]`` value sets.
    if _is_literal(node):
        return _parse_literal(node), None

    # ``list[T]`` / ``List[T]`` (PEP 585 + typing) -> List(inner).
    list_inner = _list_inner(node)
    if list_inner is not None:
        inner_dtype, _ = _patito_dtype(list_inner, model_names)
        return List(inner_dtype), None

    # Everything else (``pl.Int64``, ``pl.List(...)``, ``datetime.date``,
    # ``decimal.Decimal``, parametrized scalars) reuses the dialect-neutral
    # leaf parser. An explicit polars dtype is exact (no group) — like a
    # ``Field(dtype=)`` override.
    leaf = _parse_plain_dtype(node)
    if leaf is not None:
        return leaf, None

    return Unknown(), None


def _field_dtype_override(value: ast.expr | None) -> DataType | None:
    """Read ``pt.Field(dtype=pl.X)`` and return the forced dtype, else ``None``."""
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Name):
        if func.id != PATITO_FIELD_CALLABLE:
            return None
    elif isinstance(func, ast.Attribute):
        if func.attr != PATITO_FIELD_CALLABLE:
            return None
    else:
        return None
    for kw in value.keywords:
        if kw.arg == PATITO_FIELD_DTYPE_KW:
            return _parse_plain_dtype(kw.value)
    return None


def _is_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Subscript) and _tail_name(node.value) == "Literal"


def _parse_literal(node: ast.expr) -> DataType:
    """``Literal[...]`` -> the dtype group of its value space.

    All-string literals accept ``String`` or the corresponding ``Enum``
    (probed: ``valid_dtypes == {String, Enum(categories=[...])}``); all-int
    accept any integer width. Mixed / unsupported value kinds degrade to
    ``Unknown`` (sound).
    """
    assert isinstance(node, ast.Subscript)
    slice_ = node.slice
    elts = slice_.elts if isinstance(slice_, ast.Tuple) else [slice_]
    consts = [e.value for e in elts if isinstance(e, ast.Constant)]
    if len(consts) != len(elts) or not consts:
        return Unknown()
    if all(isinstance(c, str) for c in consts):
        str_consts = tuple(c for c in consts if isinstance(c, str))
        return DataTypeGroup(
            frozenset({Utf8(), Enum(categories=str_consts)}),
            label=f"Literal{list(str_consts)}",
            canonical=Utf8(),
        )
    if all(isinstance(c, int) and not isinstance(c, bool) for c in consts):
        return integer_group()
    if all(isinstance(c, bool) for c in consts):
        return Boolean()
    return Unknown()


def _list_inner(node: ast.expr) -> ast.expr | None:
    """The element annotation of ``list[T]`` / ``List[T]``, else ``None``.

    A bare ``list`` / ``List`` (no subscript) or a multi-arg subscript is not
    a single-element list shape and returns ``None``.
    """
    if not isinstance(node, ast.Subscript):
        return None
    if _tail_name(node.value) not in ("list", "List"):
        return None
    if isinstance(node.slice, ast.Tuple):
        return None
    return node.slice


def _tail_name(node: ast.expr) -> str | None:
    """Trailing name of a bare ``Name`` or qualified ``a.b.Name``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
