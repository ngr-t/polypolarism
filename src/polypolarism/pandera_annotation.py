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


def frame_annotation_schema_name(annotation: ast.expr) -> str | None:
    """Return the schema name from a ``DataFrame[X]`` / ``LazyFrame[X]`` shape.

    Returns the bare name ``X`` regardless of whether the registry knows
    it — purely syntactic. Use this to detect annotations the user
    *intended* as Pandera-backed even when import resolution failed
    (so we can warn instead of silently treating the file as empty).
    Returns ``None`` for annotations not in this shape.
    """
    if not isinstance(annotation, ast.Subscript):
        return None
    if _dataframe_head_name(annotation.value) is None:
        return None
    return _extract_schema_name(annotation.slice)


def extract_dataframe_annotation(
    annotation: ast.expr,
    registry: SchemaRegistry,
) -> FrameType | None:
    """Return the FrameType from a ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotation.

    Recognised forms:
    - ``DataFrame[Schema]`` / ``LazyFrame[Schema]``
    - Qualified attribute paths whose tail is ``DataFrame``/``LazyFrame``
      (e.g. ``pa.DataFrame[Schema]``, ``pandera.typing.polars.DataFrame[Schema]``)
    - Forward refs as string constants: ``DataFrame["Schema"]``

    The returned FrameType has ``is_lazy=True`` for ``LazyFrame[...]``
    annotations and ``is_lazy=False`` for ``DataFrame[...]``.

    Returns ``None`` if the annotation is not in a recognised form or the
    schema name is unknown to the registry.
    """
    if not isinstance(annotation, ast.Subscript):
        return None

    head_name = _dataframe_head_name(annotation.value)
    if head_name is None:
        return None

    schema_name = _extract_schema_name(annotation.slice)
    if schema_name is None:
        return None

    base = registry.to_frame_type(schema_name)
    if base is None:
        return None
    # Stamp the laziness onto a copy so the registry's cached value stays neutral.
    return FrameType(
        columns=base.columns,
        strict=base.strict,
        rest=base.rest,
        is_lazy=(head_name == "LazyFrame"),
        coerce=base.coerce,
    )


def _dataframe_head_name(node: ast.expr) -> str | None:
    """Return ``"DataFrame"`` / ``"LazyFrame"`` when ``node`` is one of those
    (bare or qualified, e.g. ``pa.DataFrame``); ``None`` otherwise."""
    if isinstance(node, ast.Name) and node.id in _HEAD_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _HEAD_NAMES:
        return node.attr
    return None


def _is_dataframe_head(node: ast.expr) -> bool:
    """Check whether ``node`` is ``DataFrame``/``LazyFrame`` (bare or qualified)."""
    return _dataframe_head_name(node) is not None


def _extract_schema_name(slice_: ast.expr) -> str | None:
    """Pull the schema class name out of a subscript slice."""
    if isinstance(slice_, ast.Name):
        return slice_.id
    if isinstance(slice_, ast.Constant) and isinstance(slice_.value, str):
        return slice_.value
    return None
