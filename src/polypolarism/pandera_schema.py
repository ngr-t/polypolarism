"""Pandera DataFrameModel class registry.

Walks a module AST, finds classes that inherit from ``pa.DataFrameModel``
(or transitively from another schema), parses each class's field
annotations + ``class Config:`` block, resolves inheritance via
topological sort, and exposes the result as a ``SchemaRegistry``.
"""

from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

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

    def get(self, name: str) -> Schema | None:
        return self.schemas.get(name)

    def to_frame_type(self, name: str) -> FrameType | None:
        schema = self.schemas.get(name)
        if schema is None:
            return None
        return schema.to_frame_type()

    def __contains__(self, name: str) -> bool:
        return name in self.schemas


def collect_schemas_with_imports(tree: ast.Module, file_path: Path) -> SchemaRegistry:
    """Like ``collect_schemas`` but also resolves project-local imports.

    For each ``from X import Y`` / ``from .X import Y`` in ``tree``,
    locates ``X`` on disk relative to ``file_path`` (or its ancestors up
    to a project root) and merges schemas from that file into the
    registry. Recurses transitively so a schema chain spanning multiple
    files resolves end-to-end. Imports that don't resolve to a project
    file (stdlib, third-party) are skipped silently — only files we can
    actually parse contribute.

    Schemas defined in ``tree`` itself take precedence over imports;
    imported schemas only fill names not already present.
    """
    registry = collect_schemas(tree)
    visited: set[Path] = set()
    with contextlib.suppress(OSError):
        visited.add(file_path.resolve())
    _merge_imports(tree, file_path, registry, visited)
    return registry


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
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, bool):
                        schema.strict = stmt.value.value


_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")


def _project_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing a project marker file.

    Returns ``None`` if no marker is found on the way up to the
    filesystem root. The marker bounds how far the import resolver may
    walk upward — without one we conservatively only consider the file's
    own directory so we don't reach into unrelated trees.
    """
    for ancestor in [start, *start.parents]:
        for marker in _PROJECT_MARKERS:
            if (ancestor / marker).is_file():
                return ancestor
    return None


def _try_module_at(base: Path, parts: list[str]) -> Path | None:
    """Look for ``base/parts.py`` or ``base/parts/__init__.py``."""
    if not parts:
        return None
    module_file = base.joinpath(*parts).with_suffix(".py")
    if module_file.is_file():
        return module_file
    pkg_init = base.joinpath(*parts) / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _resolve_module_path(module: str | None, current_file: Path, level: int = 0) -> Path | None:
    """Best-effort resolution of an import target to a file on disk.

    Restricted to the project tree (the dir containing
    ``pyproject.toml`` / ``setup.py`` and below) so we don't pick up
    stdlib or third-party packages. Both absolute and relative imports
    are supported. Returns ``None`` when the target can't be found.
    """
    parts = module.split(".") if module else []

    if level > 0:
        base = current_file.parent
        for _ in range(level - 1):
            base = base.parent
        return _try_module_at(base, parts)

    if not parts:
        return None

    root = _project_root(current_file.parent)
    bases: list[Path] = []
    seen: set[Path] = set()
    for ancestor in [current_file.parent, *current_file.parents]:
        if ancestor in seen:
            continue
        seen.add(ancestor)
        bases.append(ancestor)
        if root is not None and ancestor == root:
            break
        if root is None:
            # No project marker found — only consider the file's own dir
            # so we don't accidentally import from unrelated trees.
            break

    for base in bases:
        resolved = _try_module_at(base, parts)
        if resolved is not None:
            return resolved
    return None


def _merge_imports(
    tree: ast.Module,
    current_file: Path,
    registry: SchemaRegistry,
    visited: set[Path],
) -> None:
    """Recursively merge schemas from project-local imports into ``registry``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = _resolve_module_path(node.module, current_file, node.level)
        if resolved is None:
            continue
        try:
            real = resolved.resolve()
        except OSError:
            continue
        if real in visited:
            continue
        visited.add(real)
        try:
            sub_source = resolved.read_text()
            sub_tree = ast.parse(sub_source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        sub_registry = collect_schemas(sub_tree)
        for name, schema in sub_registry.schemas.items():
            registry.schemas.setdefault(name, schema)

        # Recurse so chains like app -> schemas -> base resolve fully.
        _merge_imports(sub_tree, resolved, registry, visited)


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
