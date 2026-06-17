"""Detect Pandera ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations.

Resolves the schema name through a ``SchemaRegistry`` and returns the
underlying ``FrameType``. Used by the analyzer to attach declared types
to function parameters, return values, and annotated assignments.
"""

from __future__ import annotations

import ast

from polypolarism.compat.pandera_api import FRAME_ANNOTATION_HEADS as _HEAD_NAMES
from polypolarism.pandera_schema import SchemaRegistry
from polypolarism.types import FrameType


def _unwrap_string_annotation(annotation: ast.expr) -> ast.expr:
    """Parse a string (forward-ref) annotation into the expression it spells.

    ``def f(df: "DataFrame[S]")`` carries the annotation as an
    ``ast.Constant(str)`` rather than the ``ast.Subscript`` the detectors
    expect (C-13 gap 2). This is forced whenever ``from __future__ import
    annotations`` or a ``TYPE_CHECKING`` import is in play on Python < 3.12.
    Parse the string in ``eval`` mode and return the resulting expression
    node so the existing annotation detection applies unchanged.

    Soundness: if the string does not ``ast.parse`` as a single expression
    (genuinely malformed, or not an annotation at all) the original node is
    returned untouched — the caller then sees a non-frame annotation and
    skips it, exactly as today. ``ast.parse`` only builds a tree; nothing is
    evaluated (the no-dynamic-execution invariant). Non-string nodes pass
    through unchanged, so already-working annotations are not affected.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value, mode="eval").body
        except (SyntaxError, ValueError):
            return annotation
    return annotation


def frame_annotation_schema_name(
    annotation: ast.expr,
    frame_aliases: dict[str, str] | None = None,
) -> str | None:
    """Return the schema name from a ``DataFrame[X]`` / ``LazyFrame[X]`` shape.

    Returns the name ``X`` — bare (``Schema``) or dotted
    (``mod.Schema``) — regardless of whether the registry knows it;
    purely syntactic. Use this to detect annotations the user *intended*
    as Pandera-backed even when import resolution failed (so we can warn
    instead of silently treating the file as empty).
    Returns ``None`` for annotations not in this shape.

    ``frame_aliases`` maps alias names (e.g. ``DF``) to canonical wrapper
    names (``"DataFrame"``/``"LazyFrame"``) so ``DF[Schema]`` is recognised
    the same as ``DataFrame[Schema]`` (issue #96).
    """
    annotation = _unwrap_string_annotation(annotation)
    if not isinstance(annotation, ast.Subscript):
        return None
    if _dataframe_head_name(annotation.value, frame_aliases) is None:
        return None
    return _extract_schema_name(annotation.slice)


def extract_dataframe_annotation(
    annotation: ast.expr,
    registry: SchemaRegistry,
    with_field_spans: bool = False,
) -> FrameType | None:
    """Return the FrameType from a ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotation.

    Recognised forms:
    - ``DataFrame[Schema]`` / ``LazyFrame[Schema]``
    - Qualified attribute paths whose tail is ``DataFrame``/``LazyFrame``
      (e.g. ``pa.DataFrame[Schema]``, ``pandera.typing.polars.DataFrame[Schema]``)
    - Module-qualified schema references: ``DataFrame[mod.Schema]`` —
      looked up under the flat dotted key the registry records for
      ``import mod`` (see ``pandera_schema._merge_module_imports``)
    - Forward refs as string constants: ``DataFrame["Schema"]``
    - Alias forms: ``DF[Schema]`` when ``from pandera.typing.polars import
      DataFrame as DF`` is in scope (via ``registry.frame_aliases``,
      issue #96).

    The returned FrameType has ``is_lazy=True`` for ``LazyFrame[...]``
    annotations and ``is_lazy=False`` for ``DataFrame[...]``.

    Returns ``None`` if the annotation is not in a recognised form or the
    schema name is unknown to the registry.

    String forward-ref annotations (``"DataFrame[Schema]"``) are unwrapped
    first (C-13 gap 2) so quoted annotations resolve like the bare form.
    """
    annotation = _unwrap_string_annotation(annotation)
    if not isinstance(annotation, ast.Subscript):
        return None

    head_name = _dataframe_head_name(annotation.value, registry.frame_aliases)
    if head_name is None:
        return None

    schema_name = _extract_schema_name(annotation.slice)
    if schema_name is None:
        return None

    # ``with_field_spans`` carries the declared-field ``column_spans`` for the
    # SECONDARY ("declared here") mismatch location — used ONLY for the
    # function declared-RETURN binding (issue #110). Parameter / local /
    # validate bindings stay span-free so an input schema's field spans never
    # masquerade as a body-producing PRIMARY span on a pass-through column.
    base = (
        registry.declared_return_frame(schema_name)
        if with_field_spans
        else registry.to_frame_type(schema_name)
    )
    if base is None:
        return None
    # Stamp the laziness onto a copy so the registry's cached value stays neutral.
    frame = FrameType(
        columns=base.columns,
        strict=base.strict,
        rest=base.rest,
        is_lazy=(head_name == "LazyFrame"),
        coerce=base.coerce,
        nonstrict_schema=base.nonstrict_schema,
        schema_name=base.schema_name,
        column_spans=base.column_spans,
    )
    # ``column_annotation_spans`` is diagnostic-only and not a constructor
    # parameter (issue #113) — carry it across so the declared-RETURN frame
    # keeps the annotation-only spans for the retype quick fix's range.
    frame.column_annotation_spans = dict(base.column_annotation_spans)
    return frame


def bare_frame_annotation(annotation: ast.expr) -> str | None:
    """Return ``"DataFrame"`` / ``"LazyFrame"`` for a bare polars frame
    annotation: ``pl.DataFrame`` / ``polars.LazyFrame`` with no subscript
    (ADR-0006).

    Only the ``pl`` / ``polars`` prefixes are recognized — a bare
    ``DataFrame`` name or any other prefix (``pd.DataFrame``) may be
    pandas, and claiming a polars frame there would be wrong. Such a
    parameter binds an empty OPEN frame: nothing is known about its
    columns, but everything the function body itself determines is
    checked.

    String forward-ref annotations (``"pl.DataFrame"``) are unwrapped first
    (C-13 gap 2) so quoted bare-frame annotations resolve like the bare form.
    """
    annotation = _unwrap_string_annotation(annotation)
    if (
        isinstance(annotation, ast.Attribute)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id in ("pl", "polars")
        and annotation.attr in _HEAD_NAMES
    ):
        return annotation.attr
    return None


def _dataframe_head_name(
    node: ast.expr,
    frame_aliases: dict[str, str] | None = None,
) -> str | None:
    """Return ``"DataFrame"`` / ``"LazyFrame"`` when ``node`` is one of those
    (bare or qualified, e.g. ``pa.DataFrame``); ``None`` otherwise.

    ``frame_aliases`` maps alias names (e.g. ``DF``) to canonical wrapper
    names so ``DF[Schema]`` is recognised the same as ``DataFrame[Schema]``
    (issue #96). Returns the CANONICAL name, not the alias.
    """
    if isinstance(node, ast.Name):
        if node.id in _HEAD_NAMES:
            return node.id
        if frame_aliases:
            canonical = frame_aliases.get(node.id)
            if canonical is not None:
                return canonical
    if isinstance(node, ast.Attribute) and node.attr in _HEAD_NAMES:
        return node.attr
    return None


def _is_dataframe_head(node: ast.expr) -> bool:
    """Check whether ``node`` is ``DataFrame``/``LazyFrame`` (bare or qualified)."""
    return _dataframe_head_name(node) is not None


def _extract_schema_name(slice_: ast.expr) -> str | None:
    """Pull the schema class name out of a subscript slice.

    Handles bare names (``Schema``), dotted attribute chains
    (``mod.Schema``, ``pkg.schemas.Out`` — returned as the dotted path
    exactly as written), and string forward refs (``"Schema"``).
    """
    if isinstance(slice_, ast.Name):
        return slice_.id
    if isinstance(slice_, ast.Attribute):
        return _dotted_name(slice_)
    if isinstance(slice_, ast.Constant) and isinstance(slice_.value, str):
        return slice_.value
    return None


def _dotted_name(node: ast.expr) -> str | None:
    """Render a ``Name``/``Attribute`` chain as ``a.b.c``; ``None`` if the
    chain contains anything else (calls, subscripts, ...)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))
