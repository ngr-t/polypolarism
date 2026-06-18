"""Translate Pandera DataFrameModel field annotations to ColumnSpec.

Recognised annotation shapes:
- Python builtins: ``int``, ``str``, ``float``, ``bool``
- Python stdlib classes (issue #77): bare ``date`` / ``datetime`` /
  ``timedelta`` / ``time`` and the qualified ``datetime.date`` etc. forms;
  ``decimal.Decimal`` / bare ``Decimal`` -> pandera's ``Decimal(28, 0)``
  (same engine default as bare ``pl.Decimal``, issue #75; nested positions
  are runtime wildcards -> ``Unknown``)
- Polars dtype classes: ``pl.Int64``, ``pl.Utf8``, ``pl.Float64`` etc. (with or without ``()``)
- ``Series[T]`` (the canonical pandera class-based form) -> equivalent to bare ``T``.
  Both bare and qualified heads (``Series``, ``pa.typing.Series``,
  ``pandera.typing.polars.Series``) are accepted.
- ``Optional[T]`` and ``T | None`` -> ``required=False``
- ``Annotated[pl.List, pl.Int64()]`` -> ``List(Int64())``
- ``Annotated[pl.Array, pl.Int64(), 3, None]`` -> ``Array(Int64(), 3)`` (width
  tracked; C-7. pandera requires all three of inner/shape/width)
- ``Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]`` -> ``Struct({...})``
- Parametrized scalars, Annotated form (pandera passes the metadata as the
  dtype's positional arguments and requires all of them):
  ``Annotated[pl.Datetime, "ns", None]`` -> ``Datetime(unit="ns")`` (#66),
  ``Annotated[pl.Duration, "ms"]`` -> ``Duration("ms")`` (#66),
  ``Annotated[pl.Decimal, 12, 4]`` -> ``Decimal(12, 4)`` (#65),
  ``Annotated[pl.Enum, ["a", "b"]]`` -> ``Enum(("a", "b"))`` (#67).
- Call-form containers: ``pl.List(pl.Int64)`` -> ``List(Int64())``,
  ``pl.Array(pl.Int64, 4)`` -> ``Array(Int64(), 4)`` (width tracked; C-7),
  ``pl.Struct({"a": pl.Utf8})`` -> ``Struct({...})``. Unparseable element /
  field dtypes fall back to ``Unknown()``.
- Call-form parametrized scalars: ``pl.Decimal(12, 4)``,
  ``pl.Datetime("ns")``, ``pl.Duration("ms")``, ``pl.Enum(["a", "b"])``.
- Bare containers: ``pl.List`` -> ``List(Unknown())``; ``pl.Array`` ->
  ``Array(Unknown())``; ``pl.Struct`` -> ``Unknown()`` (a struct without
  field info has no usable shape — ``Struct({})`` would wrongly mean
  "empty struct").

Field RHS:
- ``pa.Field(nullable=True)`` wraps the parsed dtype in ``Nullable(...)``.
- Other Field options (``coerce``, ``unique``, value constraints) are runtime-only and ignored.

``Annotated`` forms whose metadata arity provably crashes pandera at runtime
(issue #69) are detected separately by :func:`annotated_arity_error`; the
parse itself stays lax (partial forms still degrade to a best-effort dtype /
``Unknown``) so the diagnostic, not the parse, carries the verdict.
"""

from __future__ import annotations

import ast

from polypolarism.compat.pandera_api import FIELD_CALLABLE_NAME, PANDERA_BARE_DECIMAL
from polypolarism.compat.polars_api import (
    DTYPE_NAME_MAP,
    parse_array_shape,
    parse_datetime_call,
    parse_decimal_call,
    parse_duration_call,
    parse_enum_call,
    parse_int_shape,
    parse_time_unit,
)
from polypolarism.types import (
    Array,
    Binary,
    Boolean,
    ColumnSpec,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    Int64,
    List,
    Nullable,
    Struct,
    Time,
    Unknown,
    Utf8,
)

_BUILTIN_MAP: dict[str, DataType] = {
    "int": Int64(),
    "str": Utf8(),
    "float": Float64(),
    "bool": Boolean(),
    # Probed (pandera + polars 1.41.2): ``x: bytes`` validates a Binary
    # column and rejects a String one — pandera maps bytes to pl.Binary.
    "bytes": Binary(),
    # Python stdlib temporal types — accepted as bare names when the
    # user wrote ``from datetime import date, datetime`` (the canonical
    # pandera form). The ``<module>.date`` qualified form is handled in
    # ``_parse_plain_dtype`` separately.
    "date": Date(),
    "datetime": Datetime(),
    "timedelta": Duration(),
    # Issue #77 — read as ``from datetime import time`` (pandera resolves
    # ``datetime.time`` to Time; probed 0.31.1). Ambiguity note: the bare
    # NAME could also be the stdlib ``time`` MODULE (``import time``),
    # which pandera rejects with a TypeError at first use — but a module
    # annotation is never meaningful, so the datetime.time reading is the
    # only useful one.
    "time": Time(),
}


# Stdlib temporal type attribute names: ``datetime.date`` / ``dt.datetime`` /
# etc. — accepted under any prefix that isn't ``pl`` (which has its own
# uppercase-attribute table). Names are deliberately lowercase to avoid
# colliding with ``pl.Date`` / ``pl.Datetime`` / ``pl.Duration``.
_STDLIB_TEMPORAL_ATTR_MAP: dict[str, DataType] = {
    "date": Date(),
    "datetime": Datetime(),
    "timedelta": Duration(),
    "time": Time(),
}


_PL_DTYPE_MAP: dict[str, DataType] = DTYPE_NAME_MAP


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

    # A TOP-LEVEL bare Decimal-class annotation resolves through pandera's
    # engine default — (28, 0), not polars' materialized (38, 0) (issue
    # #75; probed: ``to_schema()`` reports 28 and ``validate`` rejects 38).
    # This covers ``pl.Decimal``, the stdlib ``decimal.Decimal`` qualified
    # form, and the bare ``Decimal`` name (issue #77; probed: pandera
    # resolves the stdlib class to the same Decimal(28, 0)). Call forms
    # (``pl.Decimal()``, omitted/None args) carry a polars instance and
    # keep polars' 38; nested bare forms are runtime wildcards and parse
    # to Unknown in ``_parse_plain_dtype``.
    if _is_bare_decimal_class(node):
        return PANDERA_BARE_DECIMAL, True

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
        # A bare ``Decimal`` name (``from decimal import Decimal``) in a
        # NESTED position is a class-level wildcard at runtime, exactly
        # like nested bare ``pl.Decimal`` (issues #75/#77 — probed:
        # ``pl.List(decimal.Decimal)`` builds polars' ``List(Decimal)``
        # and validates List(Decimal(28,0)) AND List(Decimal(38,0))).
        # The TOP-LEVEL bare form is special-cased in ``_parse_dtype_expr``.
        if node.id == "Decimal":
            return Unknown()
        return _BUILTIN_MAP.get(node.id)

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id == "pl":
                # A bare ``pl.Decimal`` in a NESTED position (container
                # element / struct field / unreadable-call fallback) is a
                # class-level wildcard at runtime — probed (issue #75):
                # ``pl.List(pl.Decimal)`` validates List(Decimal(38,0))
                # AND List(Decimal(28,0)) — so claiming any precision
                # would be a false-positive trap. The TOP-LEVEL bare form
                # enforces pandera's (28, 0) instead and is special-cased
                # in ``_parse_dtype_expr``.
                if node.attr == "Decimal":
                    return Unknown()
                resolved = _PL_DTYPE_MAP.get(node.attr)
                if resolved is not None:
                    return resolved
                # Bare container forms carry no element/field information:
                # ``pl.List`` / ``pl.Array`` hold elements of an unknown
                # dtype; bare ``pl.Struct`` is an OPEN struct (backlog C-9
                # — "some struct, fields unknown"; probed: pandera
                # validates any struct and rejects non-structs, so the
                # struct-ness is claimable, while closed ``Struct({})``
                # would wrongly mean "empty struct").
                if node.attr == "List":
                    return List(Unknown())
                if node.attr == "Array":
                    return Array(Unknown())
                if node.attr == "Struct":
                    return Struct(open=True)
                return None
            # ``datetime.date`` / ``dt.datetime`` / ``datetime.timedelta``
            # — any non-``pl`` prefix is treated as the Python stdlib
            # ``datetime`` module (or an alias of it). Lowercase attr
            # names disambiguate from polars' uppercase dtype names.
            if node.attr in _STDLIB_TEMPORAL_ATTR_MAP:
                return _STDLIB_TEMPORAL_ATTR_MAP[node.attr]
            # ``decimal.Decimal`` / ``dec.Decimal`` under a non-``pl``
            # prefix in a NESTED position — the same class-level runtime
            # wildcard as the bare-name form above (issue #77).
            if node.attr == "Decimal":
                return Unknown()
        return None

    if isinstance(node, ast.Call):
        # Call-form containers: ``pl.List(pl.Int64)``, ``pl.Array(pl.Int64, 4)``
        # (width tracked — backlog C-7; consistent with the Annotated
        # handling), and ``pl.Struct({"a": pl.Utf8, ...})``.
        if _is_pl_attr(node.func, "List"):
            inner = _parse_plain_dtype(node.args[0]) if node.args else None
            return List(inner if inner is not None else Unknown())
        if _is_pl_attr(node.func, "Array"):
            inner = _parse_plain_dtype(node.args[0]) if node.args else None
            return Array(inner if inner is not None else Unknown(), parse_array_shape(node))
        if _is_pl_attr(node.func, "Struct"):
            return _parse_struct_call(node)
        # ``pl.Decimal(precision, scale)`` preserves its arguments (shared
        # parser in compat; omitted args take polars' defaults — probed:
        # pandera keeps a call-form instance's 38, issue #75). Non-literal
        # args fall through to the bare-attribute branch above — Unknown,
        # the honest claim for unreadable parameters.
        if _is_pl_attr(node.func, "Decimal"):
            decimal_dt = parse_decimal_call(node)
            if decimal_dt is not None:
                return decimal_dt
        # ``pl.Datetime("us", "UTC")`` / ``pl.Datetime(time_zone=...)``
        # preserves the time unit and zone (shared parser in compat;
        # issues #50/#66). A non-literal argument is unknowable — degrade
        # to ``Unknown`` rather than claiming the defaults.
        if _is_pl_attr(node.func, "Datetime"):
            datetime_dt = parse_datetime_call(node)
            return datetime_dt if datetime_dt is not None else Unknown()
        # ``pl.Duration("ms")`` preserves the time unit (issue #66).
        if _is_pl_attr(node.func, "Duration"):
            duration_dt = parse_duration_call(node)
            return duration_dt if duration_dt is not None else Unknown()
        # ``pl.Enum(["a", "b"])`` preserves the ordered category tuple
        # (issue #67); a non-literal category list keeps the Enum with
        # unknown categories (the checker treats that as a wildcard).
        if _is_pl_attr(node.func, "Enum"):
            return parse_enum_call(node)
        return _parse_plain_dtype(node.func)

    return None


def _parse_struct_call(node: ast.Call) -> DataType:
    """Parse ``pl.Struct({...})``.

    A dict literal yields a closed ``Struct`` whose unparseable field
    values fall back to ``Unknown``. Any other argument shape
    (``pl.Struct([pl.Field(...)])``, a variable, no args, ``**spread`` /
    non-string keys) still provably constructs SOME struct — an OPEN
    ``Struct`` with unknown fields (backlog C-9).
    """
    if not (node.args and isinstance(node.args[0], ast.Dict)):
        return Struct(open=True)
    mapping = node.args[0]
    fields: dict[str, DataType] = {}
    for k, v in zip(mapping.keys, mapping.values, strict=True):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            # ``**spread`` / non-string keys — no readable shape.
            return Struct(open=True)
        inner = _parse_plain_dtype(v)
        fields[k.value] = inner if inner is not None else Unknown()
    return Struct(fields)


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
        if _is_pl_attr(head, "Array"):
            width = parse_int_shape(meta[1]) if len(meta) > 1 else None
            return Array(inner_dtype, width), True
        return List(inner_dtype), True

    if _is_pl_attr(head, "Struct"):
        if not meta:
            return None
        if len(meta) == 1 and not isinstance(meta[0], ast.Dict):
            # Correct arity but an unreadable field mapping (a variable,
            # a list of pl.Field objects) — still provably SOME struct:
            # an OPEN Struct (backlog C-9).
            return Struct(open=True), True
        if not isinstance(meta[0], ast.Dict):
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

    # ``Annotated[pl.Datetime, "us", "UTC"]`` — pandera passes the metadata
    # as the dtype's positional arguments and requires ALL of them (probed:
    # the 1-arg form is a pandera TypeError "requires all positional
    # arguments"). meta[0] is the time unit (issue #66), meta[1] the time
    # zone (issue #50); a ``None`` literal takes the polars default.
    # Anything else (wrong arity, non-literal arguments) is either that
    # runtime TypeError or unknowable — ``Unknown``.
    if _is_pl_attr(head, "Datetime") and meta:
        if len(meta) == 2 and isinstance(meta[1], ast.Constant):
            unit = parse_time_unit(meta[0])
            if unit is not None:
                if meta[1].value is None:
                    return Datetime(unit=unit), True
                if isinstance(meta[1].value, str):
                    return Datetime(tz=meta[1].value, unit=unit), True
        return Unknown(), True

    # ``Annotated[pl.Duration, "ms"]`` — the single positional argument is
    # the time unit (issue #66; probed against pandera 0.31.1).
    if _is_pl_attr(head, "Duration") and meta:
        if len(meta) == 1:
            unit = parse_time_unit(meta[0])
            if unit is not None:
                return Duration(unit=unit), True
        return Unknown(), True

    # ``Annotated[pl.Decimal, 12, 4]`` — precision and scale, both required
    # by pandera (issue #65; probed: the 1-arg form is a TypeError, a
    # ``None`` literal takes the polars default — ``Annotated[pl.Decimal,
    # None, 4]`` -> Decimal(38, 4)).
    if _is_pl_attr(head, "Decimal") and meta:
        if len(meta) == 2:
            precision = _decimal_annotated_arg(meta[0], default=38)
            scale = _decimal_annotated_arg(meta[1], default=0)
            if precision is not None and scale is not None:
                return Decimal(precision, scale), True
        return Unknown(), True

    # ``Annotated[pl.Enum, ["a", "b"]]`` — the single positional argument
    # is the category list (issue #67; probed: pandera builds
    # ``Enum(categories=['a', 'b'])``). An unreadable list keeps the Enum
    # with unknown categories — the declared slot is provably SOME Enum.
    if _is_pl_attr(head, "Enum") and meta:
        if len(meta) == 1 and isinstance(meta[0], (ast.List, ast.Tuple)):
            cats: list[str] = []
            for elt in meta[0].elts:
                if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                    return Enum(), True
                cats.append(elt.value)
            return Enum(categories=tuple(cats)), True
        return Enum(), True

    return _parse_dtype_expr(head)


# pandera's ``get_dtype_kwargs`` maps the ``Annotated[pl.<Dtype>, ...]``
# metadata 1:1 onto the dtype class's ``__init__`` parameters and requires
# EXACTLY all of them — ``len(metadata) != len(signature(cls).parameters)``
# is a TypeError ("Annotation 'Array' requires all positional arguments
# ['inner', 'shape', 'width']") raised the first time the schema is used
# (``to_schema`` / ``validate`` / ``@pa.check_types``); the class statement
# itself imports cleanly (probed: pandera 0.31.1). Parameter names per
# head, probed via ``inspect.signature`` on polars 1.41.2 (keyword-only
# params like Array's ``width`` count too):
_ANNOTATED_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "List": ("inner",),
    "Array": ("inner", "shape", "width"),
    "Struct": ("fields",),
    "Datetime": ("time_unit", "time_zone"),
    "Duration": ("time_unit",),
    "Decimal": ("precision", "scale"),
    "Enum": ("categories",),
    "Categorical": ("categories", "ordering"),
}


def annotated_arity_error(annotation: ast.expr) -> str | None:
    """Detect ``Annotated[pl.<Dtype>, ...]`` forms that provably crash at runtime.

    Issue #69: polypolarism's parser is laxer than pandera — partial-argument
    forms used to degrade to ``Unknown`` (which passes everything) while the
    schema class can never actually be built. Returns a human-readable detail
    string when the annotation is provably broken, ``None`` otherwise.

    Two crash classes:

    - wrong metadata arity for a known ``pl.<Dtype>`` head — pandera raises
      TypeError at first use (see ``_ANNOTATED_PARAM_NAMES``); simple scalar
      dtypes (``pl.Int64`` …) take zero parameters, so ANY metadata crashes;
    - zero metadata (``Annotated[X]``) — Python's typing module itself
      rejects single-argument ``Annotated`` at import time.

    Conservative by design: unknown heads, non-``pl`` heads (pandera ignores
    pure-metadata Annotated on python types — probed) and statically
    unknowable arities (starred metadata) stay silent.
    """
    node = annotation
    # Unwrap Optional[...] / Series[...] the same way ``_parse_dtype_expr``
    # does — the crash survives both wrappers (probed).
    while True:
        if _is_optional(node):
            inner = _optional_inner(node)
            if inner is None:
                return None
            node = inner
        elif _is_series(node):
            assert isinstance(node, ast.Subscript)
            node = node.slice
        else:
            break
    if not _is_annotated(node):
        return None
    assert isinstance(node, ast.Subscript)
    slice_ = node.slice
    elts = slice_.elts if isinstance(slice_, ast.Tuple) else [slice_]
    if not elts or any(isinstance(e, ast.Starred) for e in elts):
        return None
    head = elts[0]
    meta = elts[1:]
    if not meta:
        return (
            "single-argument `Annotated[...]` is rejected by Python's typing "
            "module — TypeError when the module is imported"
        )
    if not _is_pl_head(head):
        return None
    assert isinstance(head, ast.Attribute)
    name = head.attr
    params = _ANNOTATED_PARAM_NAMES.get(name)
    if params is None:
        if name not in _PL_DTYPE_MAP:
            return None
        params = ()
    if len(meta) == len(params):
        return None
    if not params:
        return (
            f"`Annotated[pl.{name}, ...]` has {len(meta)} metadata argument(s) "
            f"but `pl.{name}` takes none — pandera raises TypeError the first "
            f"time the schema is used; use bare `pl.{name}`"
        )
    return (
        f"`Annotated[pl.{name}, ...]` has {len(meta)} metadata argument(s) but "
        f"pandera requires exactly {len(params)} ({', '.join(params)}) — "
        f"TypeError the first time the schema is used; pass every argument "
        f"(a `None` literal keeps the polars default) or use the call form "
        f"`pl.{name}(...)`"
    )


def _is_pl_head(node: ast.expr) -> bool:
    """``pl.<Attr>`` attribute access (any attr name)."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pl"
    )


def _decimal_annotated_arg(node: ast.expr, *, default: int) -> int | None:
    """One ``Annotated[pl.Decimal, ...]`` metadata element: an int literal,
    or a ``None`` literal standing for the polars default. Non-literals are
    statically unreadable -> ``None``."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return default
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
    return None


def _is_pl_attr(node: ast.expr, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pl"
        and node.attr == attr
    )


def _is_bare_decimal_class(node: ast.expr) -> bool:
    """A bare Decimal CLASS reference: ``pl.Decimal``, ``decimal.Decimal``
    (any non-``pl`` module prefix, mirroring the stdlib temporal forms), or
    the bare ``Decimal`` name (``from decimal import Decimal`` — polars'
    Decimal is always ``pl.``-qualified in annotations, so the bare name is
    the stdlib class; both resolve to pandera's (28, 0) anyway)."""
    if isinstance(node, ast.Name):
        return node.id == "Decimal"
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.attr == "Decimal"
    )


def unrecognized_field_spec(annotation: ast.expr) -> ColumnSpec:
    """Degraded spec for an annotation ``parse_field_annotation`` rejects.

    Issue #77: an unrecognized dtype annotation used to silently DROP the
    field from the schema, producing phantom "extra column" false positives
    on strict schemas and vanished-column false negatives on open ones.
    Instead the column registers with ``Unknown`` dtype — it EXISTS, just
    with an unknowable dtype. Optional/Series wrappers still contribute
    their requiredness. The caller (``pandera_schema._parse_schema``)
    records a definition warning so the degrade stays loud (pplw-unrecognized-annotation).
    """
    node = annotation
    required = True
    while True:
        if _is_optional(node):
            inner = _optional_inner(node)
            if inner is None:
                break
            required = False
            node = inner
        elif _is_series(node):
            assert isinstance(node, ast.Subscript)
            node = node.slice
        else:
            break
    return ColumnSpec(dtype=Unknown(), required=required)


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
        if func.id != FIELD_CALLABLE_NAME:
            return False
    elif isinstance(func, ast.Attribute):
        if func.attr != FIELD_CALLABLE_NAME:
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
