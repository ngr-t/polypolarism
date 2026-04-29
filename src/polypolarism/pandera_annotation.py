"""Detect Pandera ``DataFrame[Schema]`` / ``LazyFrame[Schema]`` annotations.

Resolves the schema name through a ``SchemaRegistry`` and returns the
underlying ``FrameType``. Used by the analyzer to attach declared types
to function parameters, return values, and annotated assignments.
"""

from __future__ import annotations

import ast

from polypolarism.pandera_schema import SchemaRegistry
from polypolarism.types import FrameType

_HEAD_NAMES = frozenset({"DataFrame", "LazyFrame"})


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

    Returns ``None`` if the annotation is not in a recognised form or the
    schema name is unknown to the registry.
    """
    if not isinstance(annotation, ast.Subscript):
        return None

    if not _is_dataframe_head(annotation.value):
        return None

    schema_name = _extract_schema_name(annotation.slice)
    if schema_name is None:
        return None

    return registry.to_frame_type(schema_name)


def _is_dataframe_head(node: ast.expr) -> bool:
    """Check whether ``node`` is ``DataFrame``/``LazyFrame`` (bare or qualified)."""
    if isinstance(node, ast.Name):
        return node.id in _HEAD_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _HEAD_NAMES
    return False


def _extract_schema_name(slice_: ast.expr) -> str | None:
    """Pull the schema class name out of a subscript slice."""
    if isinstance(slice_, ast.Name):
        return slice_.id
    if isinstance(slice_, ast.Constant) and isinstance(slice_.value, str):
        return slice_.value
    return None
