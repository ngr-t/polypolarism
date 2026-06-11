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

from polypolarism.compat.pandera_api import SCHEMA_BASE_NAMES as _BASE_NAMES
from polypolarism.pandera_dtype import (
    annotated_arity_error,
    parse_field_annotation,
    unrecognized_field_spec,
)
from polypolarism.types import ColumnSpec, FrameType


@dataclass
class Schema:
    """Parsed Pandera schema class."""

    name: str
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    strict: bool = False
    coerce: bool = False
    bases: list[str] = field(default_factory=list)
    # Field annotations that provably crash pandera at runtime (issue #69):
    # field name -> detail from ``annotated_arity_error``. Inherited from
    # parents; a child re-declaring the field with a healthy annotation
    # clears the entry (probed: the override repairs the schema). The
    # analyzer surfaces these as PLY041 on every function that references
    # the schema.
    definition_errors: dict[str, str] = field(default_factory=dict)
    # Field annotations polypolarism cannot translate to a dtype (issue
    # #77): field name -> detail. The column still registers (Unknown
    # dtype, via ``unrecognized_field_spec``) so it neither produces
    # phantom "extra column" FPs on strict schemas nor vanished-column
    # FNs on open ones. NOT an error: a bare name may be a runtime alias
    # of a real dtype (``MyAlias = pl.Int64`` resolves fine in pandera),
    # so unresolvability is not provable statically. The analyzer
    # surfaces these as PLW011 on every function that references the
    # schema. Inheritance/repair semantics mirror ``definition_errors``.
    definition_warnings: dict[str, str] = field(default_factory=dict)

    def to_frame_type(self) -> FrameType:
        return FrameType(columns=dict(self.columns), strict=self.strict, coerce=self.coerce)


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

    Plain ``import X`` / ``import X.Y as z`` statements register the
    imported module's schemas under their *qualified* spelling
    (``X.Schema`` / ``X.Y.Schema`` / ``z.Schema``) so module-qualified
    annotations like ``DataFrame[X.Schema]`` resolve (issue #68).

    Schemas defined in ``tree`` itself take precedence over imports;
    imported schemas only fill names not already present.

    Cross-file INHERITANCE (issue #76): a class whose base is an imported
    schema (``from base import WithId`` + ``class Users(WithId)``) is not
    recognizable by the per-file pass — the base name is unknown until
    the imports are merged. A second fixpoint pass
    (:func:`_collect_inherited_subclasses`) re-scans every parsed tree
    against the merged registry, so such subclasses (and in-file chains
    rooted at them, aliased bases, and dotted ``mod.Schema`` bases) parse
    with the parent's fields merged exactly like the same-file path.
    """
    registry = collect_schemas(tree)
    own_names = set(registry.schemas)
    visited: set[Path] = set()
    with contextlib.suppress(OSError):
        visited.add(file_path.resolve())
    imported_trees: list[ast.Module] = []
    _merge_imports(tree, file_path, registry, visited, imported_trees)
    _merge_module_imports(tree, file_path, registry)
    _collect_inherited_subclasses(tree, imported_trees, registry, own_names)
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
        elif isinstance(base, ast.Attribute):
            # Module-qualified base ``class Users(mod.WithId)`` — resolved
            # through the dotted keys ``import mod`` registers (issues
            # #68/#76).
            dotted = _dotted_base_name(base)
            if dotted is not None and dotted in registry.schemas:
                parent_names.append(dotted)
    schema.bases = parent_names

    # Merge parent columns first; leftmost base wins on conflicts (Python MRO-ish).
    for base_name in parent_names:
        parent = registry.schemas[base_name]
        for col_name, col_spec in parent.columns.items():
            schema.columns.setdefault(col_name, col_spec)
        for col_name, detail in parent.definition_errors.items():
            schema.definition_errors.setdefault(col_name, detail)
        for col_name, detail in parent.definition_warnings.items():
            schema.definition_warnings.setdefault(col_name, detail)
        if parent.strict:
            schema.strict = True
        if parent.coerce:
            schema.coerce = True

    # Parse this class's body.
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_name = stmt.target.id
            # Issue #69: a wrong-arity ``Annotated`` form deterministically
            # crashes pandera at runtime. Record it; a healthy re-declaration
            # shadows (and thereby repairs) an inherited broken one.
            arity_error = annotated_arity_error(stmt.annotation)
            if arity_error is None:
                schema.definition_errors.pop(field_name, None)
            else:
                schema.definition_errors[field_name] = arity_error
            spec = parse_field_annotation(stmt.annotation, stmt.value)
            if spec is not None:
                schema.columns[field_name] = spec
                schema.definition_warnings.pop(field_name, None)
            elif arity_error is None:
                # Issue #77: an unrecognized annotation must not silently
                # drop the field. Register the column with Unknown dtype
                # and record a definition warning (surfaced as PLW011).
                # Arity-broken fields are excluded — PLY041 already
                # carries their verdict.
                schema.columns[field_name] = unrecognized_field_spec(stmt.annotation)
                schema.definition_warnings[field_name] = (
                    f"annotation `{ast.unparse(stmt.annotation)}` is not a recognized dtype form"
                )
            else:
                # Re-declared (broken) field: an inherited unrecognized-
                # annotation warning no longer describes this class.
                schema.definition_warnings.pop(field_name, None)
        elif isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
            _apply_config(schema, stmt)

    return schema


def _apply_config(schema: Schema, config_node: ast.ClassDef) -> None:
    """Read ``class Config:`` settings and apply to the schema."""
    for stmt in config_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in ("strict", "coerce"):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, bool):
                        setattr(schema, target.id, stmt.value.value)


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
    imported_trees: list[ast.Module] | None = None,
) -> None:
    """Recursively merge schemas from project-local imports into ``registry``.

    Each successfully parsed imported tree is appended to
    ``imported_trees`` (when given) so the cross-file inheritance pass
    (issue #76) can re-scan it against the fully merged registry.
    """
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
        if imported_trees is not None:
            imported_trees.append(sub_tree)

        sub_registry = collect_schemas(sub_tree)
        for name, schema in sub_registry.schemas.items():
            registry.schemas.setdefault(name, schema)
        # ``from X import Y as Z`` binds Z — register the alias so both
        # ``DataFrame[Z]`` annotations and ``class C(Z)`` bases resolve
        # (issue #76).
        for alias in node.names:
            if alias.asname is not None:
                target = sub_registry.schemas.get(alias.name)
                if target is not None:
                    registry.schemas.setdefault(alias.asname, target)

        # Recurse so chains like app -> schemas -> base resolve fully.
        _merge_imports(sub_tree, resolved, registry, visited, imported_trees)


def _merge_module_imports(
    tree: ast.Module,
    current_file: Path,
    registry: SchemaRegistry,
) -> None:
    """Register schemas reachable via plain ``import X`` under dotted keys.

    Design note (issue #68): the registry stays *flat* — a
    module-qualified annotation ``DataFrame[mod.Schema]`` is resolved by
    keying the imported module's schemas under the exact dotted path the
    annotation site spells (``mod.Schema``; for ``import pkg.mod as m``,
    ``m.Schema``). This mirrors ``_extract_schema_name`` returning the
    dotted chain as written, and avoids introducing a per-module
    registry for what is a pure name-resolution concern.

    Only top-level ``import`` statements of the file under analysis are
    honoured: a plain import inside an *imported* module does not bind a
    name in the analyzed module at runtime either. Each imported
    module's own ``from``-imports are still followed so its schemas and
    re-exports resolve (``mod.ReExported`` works when ``mod`` does
    ``from base import ReExported`` — faithful to Python attribute
    access on modules). ``import pkg`` followed by ``pkg.sub.Schema``
    (attribute access through an un-imported submodule) is *not*
    resolved and falls through to the PLW006 unresolved-schema warning.
    """
    try:
        current_real = current_file.resolve()
    except OSError:
        current_real = current_file
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            resolved = _resolve_module_path(alias.name, current_file)
            if resolved is None:
                continue
            try:
                real = resolved.resolve()
            except OSError:
                continue
            if real == current_real:
                continue
            try:
                sub_source = resolved.read_text()
                sub_tree = ast.parse(sub_source)
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue

            # Build the imported module's full registry (its own schemas
            # plus from-import chains and cross-file inheritance — issue
            # #76), then mount it under the binding name: the alias if
            # given, else the full dotted module path (``import pkg.mod``
            # binds ``pkg`` but use sites spell ``pkg.mod.Schema``, which
            # is exactly this key).
            sub_registry = collect_schemas(sub_tree)
            sub_own = set(sub_registry.schemas)
            sub_visited = {real}
            sub_trees: list[ast.Module] = []
            _merge_imports(sub_tree, resolved, sub_registry, sub_visited, sub_trees)
            _collect_inherited_subclasses(sub_tree, sub_trees, sub_registry, sub_own)
            prefix = alias.asname or alias.name
            for name, schema in sub_registry.schemas.items():
                registry.schemas.setdefault(f"{prefix}.{name}", schema)


def _dotted_base_name(node: ast.expr) -> str | None:
    """Render a ``Name``/``Attribute`` base as ``a.b.c``; ``None`` for
    anything else. Mirrors ``pandera_annotation._dotted_name`` (not
    imported — that module imports this one)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _bases_resolve(node: ast.ClassDef, registry: SchemaRegistry) -> bool:
    """True when any base marks ``node`` as a schema against the MERGED
    registry: ``DataFrameModel``/``SchemaModel`` (bare or attribute tail),
    a registered schema name, or a dotted module-qualified schema key."""
    for base in node.bases:
        if isinstance(base, ast.Name):
            if base.id in _BASE_NAMES or base.id in registry.schemas:
                return True
        elif isinstance(base, ast.Attribute):
            if base.attr in _BASE_NAMES:
                return True
            dotted = _dotted_base_name(base)
            if dotted is not None and dotted in registry.schemas:
                return True
    return False


def _collect_inherited_subclasses(
    main_tree: ast.Module,
    imported_trees: list[ast.Module],
    registry: SchemaRegistry,
    own_names: set[str],
) -> None:
    """Second pass for cross-file inheritance (issue #76).

    The per-file pass cannot recognize ``class Users(WithId)`` when
    ``WithId`` is imported — the base name is unknown until the imports
    are merged. Re-scan every parsed tree against the merged registry and
    parse the newly recognizable classes (parents merged exactly like the
    same-file path), repeating to fixpoint so in-file chains rooted at an
    imported base (``class A(Imported); class B(A)``) resolve too.

    Precedence mirrors runtime name binding: a class defined in the
    ANALYZED module shadows a same-named schema merged from an import
    (``own_names`` tracks what the analyzed module itself has defined);
    classes from imported trees never overwrite existing entries.
    """
    changed = True
    while changed:
        changed = False
        for node in main_tree.body:
            if not isinstance(node, ast.ClassDef) or node.name in own_names:
                continue
            if not _bases_resolve(node, registry):
                continue
            registry.schemas[node.name] = _parse_schema(node, registry)
            own_names.add(node.name)
            changed = True
        for sub_tree in imported_trees:
            for node in sub_tree.body:
                if not isinstance(node, ast.ClassDef) or node.name in registry.schemas:
                    continue
                if not _bases_resolve(node, registry):
                    continue
                registry.schemas[node.name] = _parse_schema(node, registry)
                changed = True


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
