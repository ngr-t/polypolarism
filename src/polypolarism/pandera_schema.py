"""Pandera DataFrameModel class registry.

Walks a module AST, finds classes that inherit from ``pa.DataFrameModel``
(or transitively from another schema), parses each class's field
annotations + ``class Config:`` block, resolves inheritance via
topological sort, and exposes the result as a ``SchemaRegistry``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from polypolarism.pandera_dtype import parse_field_annotation
from polypolarism.types import ColumnSpec, FrameType


# Class names treated as the Pandera schema base. Qualified forms (pa.DataFrameModel,
# pandera.polars.DataFrameModel) are matched on the trailing attribute name.
_BASE_NAMES = frozenset({"DataFrameModel", "SchemaModel"})


@dataclass
class Schema:
    """Parsed Pandera schema class."""

    name: str
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    strict: bool = False
    bases: list[str] = field(default_factory=list)

    def to_frame_type(self) -> FrameType:
        return FrameType(columns=dict(self.columns), strict=self.strict)


@dataclass
class SchemaRegistry:
    """Registry of parsed Pandera schemas keyed by class name."""

    schemas: dict[str, Schema] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Schema]:
        return self.schemas.get(name)

    def to_frame_type(self, name: str) -> Optional[FrameType]:
        schema = self.schemas.get(name)
        if schema is None:
            return None
        return schema.to_frame_type()

    def __contains__(self, name: str) -> bool:
        return name in self.schemas


def collect_schemas(tree: ast.Module) -> SchemaRegistry:
    """Walk a module AST and return the set of Pandera schemas defined at top level."""
    registry = SchemaRegistry()

    candidate_names: set[str] = set()
    candidates: dict[str, ast.ClassDef] = {}

    # Iteratively expand the candidate set so transitive subclasses are recognised
    # even if their parent is defined later in the source.
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name not in candidate_names:
                if _looks_like_schema(node, candidate_names):
                    candidates[node.name] = node
                    candidate_names.add(node.name)
                    changed = True

    sorted_names = _topo_sort(candidates)
    for name in sorted_names:
        registry.schemas[name] = _parse_schema(candidates[name], registry)

    return registry


def _looks_like_schema(node: ast.ClassDef, known_schemas: set[str]) -> bool:
    """Return True if any base of ``node`` is DataFrameModel or a known schema."""
    for base in node.bases:
        if isinstance(base, ast.Name):
            if base.id in _BASE_NAMES or base.id in known_schemas:
                return True
        elif isinstance(base, ast.Attribute):
            if base.attr in _BASE_NAMES:
                return True
    return False


def _parse_schema(node: ast.ClassDef, registry: SchemaRegistry) -> Schema:
    """Parse a single ClassDef into a Schema, merging parent fields."""
    schema = Schema(name=node.name)

    parent_names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in registry.schemas:
            parent_names.append(base.id)
    schema.bases = parent_names

    # Merge parent columns first; leftmost base wins on conflicts (Python MRO-ish).
    for base_name in parent_names:
        parent = registry.schemas[base_name]
        for col_name, col_spec in parent.columns.items():
            schema.columns.setdefault(col_name, col_spec)
        if parent.strict:
            schema.strict = True

    # Parse this class's body.
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            spec = parse_field_annotation(stmt.annotation, stmt.value)
            if spec is not None:
                schema.columns[stmt.target.id] = spec
        elif isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
            _apply_config(schema, stmt)

    return schema


def _apply_config(schema: Schema, config_node: ast.ClassDef) -> None:
    """Read ``class Config:`` settings and apply to the schema."""
    for stmt in config_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "strict":
                    if isinstance(stmt.value, ast.Constant) and isinstance(
                        stmt.value.value, bool
                    ):
                        schema.strict = stmt.value.value


def _topo_sort(candidates: dict[str, ast.ClassDef]) -> list[str]:
    """Topological sort by base class dependencies (parents first)."""
    deps: dict[str, list[str]] = {}
    for name, node in candidates.items():
        parents: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in candidates:
                parents.append(base.id)
        deps[name] = parents

    visited: set[str] = set()
    result: list[str] = []

    def visit(n: str, stack: set[str]) -> None:
        if n in visited or n in stack:
            return
        stack.add(n)
        for parent in deps.get(n, []):
            visit(parent, stack)
        stack.discard(n)
        visited.add(n)
        result.append(n)

    for name in candidates:
        visit(name, set())

    return result
