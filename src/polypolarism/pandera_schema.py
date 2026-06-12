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

from polypolarism.compat.pandera_api import (
    OBJECT_COLUMN_CALLABLE_NAME,
    OBJECT_SCHEMA_CALLABLE_NAME,
)
from polypolarism.compat.pandera_api import SCHEMA_BASE_NAMES as _BASE_NAMES
from polypolarism.pandera_dtype import (
    annotated_arity_error,
    parse_field_annotation,
    unrecognized_field_spec,
)
from polypolarism.types import ColumnSpec, FrameType, Nullable, RowVar, Unknown


@dataclass
class Schema:
    """Parsed Pandera schema class."""

    name: str
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    strict: bool = False
    # ``strict="filter"`` (issue #88): validate REMOVES extra input
    # columns. Input-acceptance-wise it behaves like strict=False (wider
    # frames pass), output-shape-wise like strict=True (the result has
    # exactly the declared columns) — so ``strict`` stays False and the
    # output binds closed without island provenance (removed-column
    # lookups are PLY001 proofs).
    filters_extras: bool = False
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
    # Issue #90: the schema's columns could not be derived statically at
    # all (non-literal remove_columns, update_columns/rename_columns,
    # unreadable DataFrameSchema arguments). The schema still EXISTS and
    # validate runs at runtime, so it binds as a fully OPEN assumption
    # frame (no island lint — nothing is declared to lint against) and
    # PLW011 surfaces the degrade via ``definition_warnings``.
    unresolved: bool = False

    def to_frame_type(self) -> FrameType:
        # Checked-island semantics (issue #83, user-approved): a
        # strict=False schema binds CLOSED with ``nonstrict_schema``
        # provenance — the declaration is the contract; undeclared
        # lookups flag PLY042 with honest wording. strict=True and
        # strict="filter" (issue #88) bind closed and island-free: their
        # outputs hold exactly the declared columns, so missing-column
        # lookups are PLY001 proofs. VALIDATE-RESULT bindings additionally
        # open the non-strict frame (see ``validate_result_frame``) —
        # runtime extras provably flow through a non-strict validate, so
        # frame-level subtyping there is value-dependent, not provable.
        if self.unresolved:
            # Issue #90: columns unknowable — a fully open assumption
            # frame (validate still narrows; PLW011 carries the degrade).
            return FrameType({}, rest=RowVar(self.name), coerce=self.coerce, schema_name=self.name)
        if self.strict or self.filters_extras:
            return FrameType(
                columns=dict(self.columns),
                strict=self.strict,
                coerce=self.coerce,
                schema_name=self.name,
            )
        return FrameType(
            columns=dict(self.columns),
            strict=self.strict,
            coerce=self.coerce,
            nonstrict_schema=self.name,
            schema_name=self.name,
        )

    def validate_result_frame(self) -> FrameType:
        """The FrameType a ``schema.validate(df)`` RESULT binds (issue #88).

        A non-strict validate passes the input's extra columns through —
        the result provably carries them — so the binding is an OPEN
        ISLAND: ``rest`` keeps frame-level subtyping lenient
        (missing-column claims against the result are value-dependent),
        while the ``nonstrict_schema`` provenance keeps undeclared
        lookups flagged as the PLY042 interface lint. strict=True and
        strict="filter" results are exactly the declared columns —
        closed, island-free (filter's removed-column lookups are PLY001
        proofs).
        """
        ft = self.to_frame_type()
        if self.unresolved or self.strict or self.filters_extras:
            return ft
        ft.rest = RowVar(self.name)
        return ft


@dataclass
class SchemaRegistry:
    """Registry of parsed Pandera schemas keyed by class name."""

    schemas: dict[str, Schema] = field(default_factory=dict)
    # Names bound by a from-import whose module did NOT resolve to a
    # project-local file, mapped to the import spelling ("pkg.mod",
    # ".sibling"). Lets PLW006 say the import was seen but unresolved
    # instead of suggesting an import the user already wrote.
    failed_imports: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Schema | None:
        return self.schemas.get(name)

    def to_frame_type(self, name: str) -> FrameType | None:
        schema = self.schemas.get(name)
        if schema is None:
            return None
        return schema.to_frame_type()

    def validate_result_frame(self, name: str) -> FrameType | None:
        """Binding for a ``schema.validate(df)`` RESULT (issue #88) —
        non-strict schemas open (extras provably flow through)."""
        schema = self.schemas.get(name)
        if schema is None:
            return None
        return schema.validate_result_frame()

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

    _collect_object_schemas(tree, registry)

    return registry


def _name_tail_matches(node: ast.expr, name: str) -> bool:
    """``Name(name)`` or any qualified ``X.name`` (``pa.DataFrameSchema``)."""
    if isinstance(node, ast.Name) and node.id == name:
        return True
    return isinstance(node, ast.Attribute) and node.attr == name


def _parse_object_column(node: ast.expr) -> tuple[ColumnSpec, str | None]:
    """Parse one ``pa.Column(dtype, nullable=..., required=...)`` entry.

    Returns ``(spec, warning_detail)``. The dtype expression reuses the
    class-annotation parser (the AST shapes are identical — probed:
    parametrized ``pl.Datetime(...)`` / ``pl.Enum([...])`` and the bare
    ``pl.Decimal`` -> (28, 0) engine default all resolve like class
    fields). Anything statically unreadable — a non-``Column`` value, a
    string dtype alias (``pa.Column("int64")``, accepted by pandera but
    not modeled), a non-literal ``nullable``/``required`` — degrades to
    an ``Unknown`` column with a warning detail (the PLW011 channel,
    mirroring issue #77's loud-degrade rule).
    """
    if not (
        isinstance(node, ast.Call) and _name_tail_matches(node.func, OBJECT_COLUMN_CALLABLE_NAME)
    ):
        return ColumnSpec(dtype=Unknown()), "value is not a pa.Column(...) call"
    dtype_node: ast.expr | None = node.args[0] if node.args else None
    for kw in node.keywords:
        if kw.arg == "dtype":
            dtype_node = kw.value
    if dtype_node is None:
        return ColumnSpec(dtype=Unknown()), "pa.Column(...) has no dtype argument"

    nullable = False
    required = True
    for kw in node.keywords:
        if kw.arg in ("nullable", "required"):
            if not (isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool)):
                return ColumnSpec(dtype=Unknown()), (
                    f"pa.Column(..., {kw.arg}=...) is not a literal bool — statically unreadable"
                )
            if kw.arg == "nullable":
                nullable = kw.value.value
            else:
                required = kw.value.value

    spec = parse_field_annotation(dtype_node)
    if spec is None:
        return ColumnSpec(dtype=Unknown(), required=required), (
            "dtype expression is not statically readable (string aliases "
            'like pa.Column("int64") are not modeled)'
        )
    dtype = spec.dtype
    if nullable and not isinstance(dtype, Nullable):
        dtype = Nullable(dtype)
    return ColumnSpec(dtype=dtype, required=required), None


def _parse_object_columns(
    node: ast.expr,
    str_list_consts: dict[str, list[str]],
    column_dicts: dict[str, tuple[dict[str, ColumnSpec], dict[str, str]]],
) -> tuple[dict[str, ColumnSpec], dict[str, str]] | None:
    """Parse a columns expression of the object API (backlog C-11 tier 2).

    Accepted shapes:
    - a dict literal ``{"a": pa.Column(...), ...}`` — ``**`` spreads of a
      nested dict literal or of a module-level column-dict ``Name`` recurse;
    - a dict comprehension ``{c: pa.Column(...) for c in <list/tuple of
      str literals | module-level str-list const>}`` (single generator, no
      conditions, key is the bare loop variable; a value referencing the
      loop variable is not constant-foldable — the keys still register,
      as Unknown columns with warnings);
    - a bare ``Name`` bound to a previously recorded column dict.

    Returns ``(columns, per-field warning details)`` or ``None`` when the
    shape is not statically readable at all.
    """
    if isinstance(node, ast.Name):
        recorded = column_dicts.get(node.id)
        if recorded is None:
            return None
        columns, warnings = recorded
        return dict(columns), dict(warnings)

    if isinstance(node, ast.Dict):
        columns: dict[str, ColumnSpec] = {}
        warnings: dict[str, str] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                # ``**spread`` — a nested literal or a recorded Name.
                spread = _parse_object_columns(value_node, str_list_consts, column_dicts)
                if spread is None:
                    return None
                columns.update(spread[0])
                warnings.update(spread[1])
                continue
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                return None
            spec, detail = _parse_object_column(value_node)
            columns[key_node.value] = spec
            if detail is not None:
                warnings[key_node.value] = detail
            else:
                warnings.pop(key_node.value, None)
        return columns, warnings

    if isinstance(node, ast.DictComp):
        if len(node.generators) != 1:
            return None
        gen = node.generators[0]
        if gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name):
            return None
        loop_var = gen.target.id
        if not (isinstance(node.key, ast.Name) and node.key.id == loop_var):
            return None
        keys: list[str] | None = None
        if isinstance(gen.iter, (ast.List, ast.Tuple)):
            keys = []
            for elt in gen.iter.elts:
                if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                    return None
                keys.append(elt.value)
        elif isinstance(gen.iter, ast.Name):
            keys = str_list_consts.get(gen.iter.id)
        if keys is None:
            return None
        # A value referencing the loop variable is not constant per key —
        # register the (known) keys as Unknown columns, loudly.
        refs_loop_var = any(
            isinstance(sub, ast.Name) and sub.id == loop_var for sub in ast.walk(node.value)
        )
        if refs_loop_var:
            detail = (
                "dict-comprehension value references the loop variable — "
                "not constant-foldable; the column registers as Unknown"
            )
            return (
                {k: ColumnSpec(dtype=Unknown()) for k in keys},
                {k: detail for k in keys},
            )
        spec, detail = _parse_object_column(node.value)
        columns = {k: spec for k in keys}
        warnings = {k: detail for k in keys} if detail is not None else {}
        return columns, warnings

    return None


_OBJECT_SCHEMA_DERIVATIONS: frozenset[str] = frozenset(
    {
        "add_columns",
        "remove_columns",
        "update_columns",
        "rename_columns",
        "set_index",
        "reset_index",
    }
)


def _unresolved_object_schema(
    name: str, value: ast.expr, registry: SchemaRegistry
) -> Schema | None:
    """Recognize a schema-SHAPED RHS that could not be folded (issue #90).

    Two shapes register as ``unresolved`` (everything else returns
    ``None`` — not schema-related at all):

    - ``pa.DataFrameSchema(<unreadable columns>)``;
    - ``<registered schema>.<derivation>(...)`` for any pandera object-API
      derivation method, with arguments (or a method like
      ``update_columns`` / ``rename_columns``) we do not model.
    """
    if not isinstance(value, ast.Call):
        return None
    detail: str | None = None
    if _name_tail_matches(value.func, OBJECT_SCHEMA_CALLABLE_NAME):
        detail = "DataFrameSchema columns argument is not statically readable"
    elif (
        isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.attr in _OBJECT_SCHEMA_DERIVATIONS
        and registry.get(value.func.value.id) is not None
    ):
        detail = (
            f"derivation '.{value.func.attr}(...)' is not statically foldable "
            f"(only literal add_columns/remove_columns are modeled)"
        )
    if detail is None:
        return None
    return Schema(
        name=name,
        unresolved=True,
        definition_warnings={"<derivation>": detail},
    )


def _collect_object_schemas(tree: ast.Module, registry: SchemaRegistry) -> None:
    """Register module-level ``pa.DataFrameSchema`` object-API schemas
    (backlog C-11, tiers 1–2).

    Processes top-level assignments in source order:

    - ``NAME = pa.DataFrameSchema(<columns>, strict=..., coerce=...)``
      registers like a class schema, keyed by the variable name — the
      whole downstream machinery (``validate`` narrowing, checked-island
      ``nonstrict_schema`` provenance, cross-file import merging, PLW011
      definition warnings) applies uniformly.
    - ``NAME = other.add_columns({...})`` / ``other.remove_columns([...])``
      derive a NEW schema from a registered one (probed: pandera's object
      API is immutable). ``update_columns`` / ``rename_columns`` are not
      modeled.
    - ``NAME = ["a", "b"]`` string-list constants and
      ``NAME = {"a": pa.Column(...)}`` column dicts are tracked so tier-2
      construction (dict comprehensions, ``**`` spreads, direct ``Name``
      arguments) folds against them.

    Function-local schemas are out of scope (the registry is
    module-level). Rebinding a schema name to something unreadable drops
    the registration — last binding wins, mirroring runtime name binding.
    """
    str_list_consts: dict[str, list[str]] = {}
    column_dicts: dict[str, tuple[dict[str, ColumnSpec], dict[str, str]]] = {}

    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        name = target.id

        # Track tier-2 building blocks.
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
        ):
            str_list_consts[name] = [e.value for e in value.elts]  # type: ignore[union-attr]
            continue
        if isinstance(value, (ast.Dict, ast.DictComp)):
            cols = _parse_object_columns(value, str_list_consts, column_dicts)
            if cols is not None:
                column_dicts[name] = cols
            else:
                column_dicts.pop(name, None)
            continue

        schema = _parse_object_schema_value(name, value, registry, str_list_consts, column_dicts)
        if schema is None:
            # Issue #90: a schema-SHAPED value we cannot fold must not
            # vanish silently — register it as unresolved (open
            # assumption frame + PLW011) so validate sites stay loud.
            schema = _unresolved_object_schema(name, value, registry)
        if schema is not None:
            registry.schemas[name] = schema
        elif name in registry.schemas and registry.get(name) is not None:
            # The name was a schema and is rebound to something unreadable
            # — last binding wins (runtime name binding).
            existing = registry.schemas[name]
            if not existing.bases and existing.name == name:
                registry.schemas.pop(name, None)


def _parse_object_schema_value(
    name: str,
    value: ast.expr,
    registry: SchemaRegistry,
    str_list_consts: dict[str, list[str]],
    column_dicts: dict[str, tuple[dict[str, ColumnSpec], dict[str, str]]],
) -> Schema | None:
    """Parse one RHS as an object-API schema; ``None`` if it isn't one."""
    if not isinstance(value, ast.Call):
        return None

    # ``pa.DataFrameSchema(<columns>, strict=..., coerce=...)``
    if _name_tail_matches(value.func, OBJECT_SCHEMA_CALLABLE_NAME):
        if not value.args:
            return None
        parsed = _parse_object_columns(value.args[0], str_list_consts, column_dicts)
        if parsed is None:
            return None
        columns, warnings = parsed
        schema = Schema(name=name, columns=columns, definition_warnings=warnings)
        for kw in value.keywords:
            if kw.arg in ("strict", "coerce"):
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                    setattr(schema, kw.arg, kw.value.value)
                elif (
                    kw.arg == "strict"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "filter"
                ):
                    schema.filters_extras = True
        return schema

    # ``base.add_columns({...})`` / ``base.remove_columns([...])``
    if (
        isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.attr in ("add_columns", "remove_columns")
    ):
        base = registry.get(value.func.value.id)
        if base is None or not value.args:
            return None
        if value.func.attr == "add_columns":
            parsed = _parse_object_columns(value.args[0], str_list_consts, column_dicts)
            if parsed is None:
                return None
            added, added_warnings = parsed
            columns = {**base.columns, **added}
            warnings = {**base.definition_warnings, **added_warnings}
            for col in added:
                if col not in added_warnings:
                    warnings.pop(col, None)
            return Schema(
                name=name,
                columns=columns,
                strict=base.strict,
                coerce=base.coerce,
                definition_errors=dict(base.definition_errors),
                definition_warnings=warnings,
            )
        removed_node = value.args[0]
        if not isinstance(removed_node, (ast.List, ast.Tuple)):
            return None
        removed: set[str] = set()
        for elt in removed_node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return None
            removed.add(elt.value)
        return Schema(
            name=name,
            columns={k: v for k, v in base.columns.items() if k not in removed},
            strict=base.strict,
            coerce=base.coerce,
            definition_errors={k: v for k, v in base.definition_errors.items() if k not in removed},
            definition_warnings={
                k: v for k, v in base.definition_warnings.items() if k not in removed
            },
        )

    return None


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
                    elif (
                        target.id == "strict"
                        and isinstance(stmt.value, ast.Constant)
                        and stmt.value.value == "filter"
                    ):
                        schema.strict = False
                        schema.filters_extras = True


_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")
_PROJECT_MARKER_DIRS = (".git",)


def _project_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing a project marker.

    Markers are the packaging files (``pyproject.toml`` & co.) or a
    ``.git`` directory — most real repos have only the latter, and
    without any marker dotted imports like ``from pkg.mod import X``
    could never resolve from inside ``pkg`` (user report 2026-06-12).

    Returns ``None`` if no marker is found on the way up to the
    filesystem root. The marker bounds how far the import resolver may
    walk upward — without one we conservatively only consider the file's
    own directory (plus the cwd fallback) so we don't reach into
    unrelated trees.
    """
    for ancestor in [start, *start.parents]:
        for marker in _PROJECT_MARKERS:
            if (ancestor / marker).is_file():
                return ancestor
        for marker_dir in _PROJECT_MARKER_DIRS:
            if (ancestor / marker_dir).is_dir():
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

    # The src layout keeps importable packages one level below the
    # project root; try each base's `src/` after the base itself.
    for base in bases:
        for candidate in (base, base / "src"):
            resolved = _try_module_at(candidate, parts)
            if resolved is not None:
                return resolved

    if root is None:
        # Marker-less tree: fall back to the invocation directory when
        # the analyzed file lives under it (the common "run from the
        # project root" case). Files outside the cwd stay bounded to
        # their own directory.
        cwd: Path | None = None
        try:
            cwd = Path.cwd()
            file_in_cwd = current_file.resolve().is_relative_to(cwd)
        except OSError:
            file_in_cwd = False
        if cwd is not None and file_in_cwd:
            for candidate in (cwd, cwd / "src"):
                resolved = _try_module_at(candidate, parts)
                if resolved is not None:
                    return resolved
    return None


def _merge_imports(
    tree: ast.Module,
    current_file: Path,
    registry: SchemaRegistry,
    visited: set[Path],
    imported_trees: list[ast.Module] | None = None,
    sub_registries: dict[Path, SchemaRegistry | None] | None = None,
) -> None:
    """Recursively merge schemas from project-local imports into ``registry``.

    Each successfully parsed imported tree is appended to
    ``imported_trees`` (when given) so the cross-file inheritance pass
    (issue #76) can re-scan it against the fully merged registry.
    ``sub_registries`` caches each file's pass-1 registry per invocation
    so REPEAT import statements of an already-visited module can still
    register their alias bindings (issue #80); ``None`` marks a file
    that failed to parse.
    """
    if sub_registries is None:
        sub_registries = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = _resolve_module_path(node.module, current_file, node.level)
        if resolved is None:
            module_spelled = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    continue
                stmt = f"from {module_spelled} import {alias.name}"
                if alias.asname is not None:
                    stmt += f" as {alias.asname}"
                registry.failed_imports.setdefault(alias.asname or alias.name, stmt)
            continue
        try:
            real = resolved.resolve()
        except OSError:
            continue
        first_visit = real not in visited
        if first_visit:
            visited.add(real)
            try:
                sub_source = resolved.read_text()
                sub_tree = ast.parse(sub_source)
            except (OSError, UnicodeDecodeError, SyntaxError):
                sub_registries[real] = None
                continue
            if imported_trees is not None:
                imported_trees.append(sub_tree)
            sub_registry = collect_schemas(sub_tree)
            sub_registries[real] = sub_registry
            for name, schema in sub_registry.schemas.items():
                registry.schemas.setdefault(name, schema)
        else:
            # A module already merged by an EARLIER import statement: the
            # broad merge and the recursion are done, but THIS statement's
            # alias bindings still need registering (issue #80 — the skip
            # used to drop them, breaking ``from m import Base as B0`` +
            # ``class C(B0)`` whenever m had been imported before).
            cached = sub_registries.get(real)
            if cached is None:
                continue
            sub_registry = cached

        # ``from X import Y as Z`` binds Z — register the alias so both
        # ``DataFrame[Z]`` annotations and ``class C(Z)`` bases resolve
        # (issues #76/#80).
        for alias in node.names:
            if alias.asname is not None:
                target = sub_registry.schemas.get(alias.name)
                if target is not None:
                    registry.schemas.setdefault(alias.asname, target)

        if first_visit:
            # Recurse so chains like app -> schemas -> base resolve fully.
            _merge_imports(sub_tree, resolved, registry, visited, imported_trees, sub_registries)


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
