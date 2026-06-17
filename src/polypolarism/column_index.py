"""Column-reference index for the ``--rename-targets`` capability (Batch C).

This module records every place a column NAME appears in a way the analyzer
can resolve, each as ``(file, Span, column_name, origin)``. ``origin`` is the
column's PROVABLE source — a schema field ``(schema_name, field_name)`` — or
``None`` when the reference cannot be proven to resolve to a known field.

CRITICAL — SOUNDNESS. The extension uses the returned targets to RENAME
(rewrite) source text, so a wrong target corrupts the user's code. The whole
module errs toward FEWER targets:

* A reference gets a provable ``(schema, field)`` origin ONLY when the
  analyzer can follow the dataflow from a schema-typed frame to the
  reference with the column's identity intact.
* Two references are "the same column" ONLY when they share a provable
  ``(schema, field)`` origin. A bare name match is NEVER sufficient.
* A column renamed / aliased mid-pipeline starts a NEW identity from the
  rename point — old-name and new-name refs are DIFFERENT (never merged).
* A reference off an open / unknown frame, or one whose lineage cannot be
  followed, gets ``origin=None`` and only ever matches by exact position
  (single-occurrence fallback), never by name-broadcast.

The dataflow tracked here is intentionally a SOUND SUBSET — it does not
re-implement the full analyzer. It follows, per function:

* a frame-typed parameter ``df: DataFrame[S]`` binds ``df`` to schema ``S``
  with all of ``S``'s declared fields "live";
* pass-through assignment ``y = x`` and identity-preserving method chains
  (``select`` / ``filter`` / ``with_columns`` / ``sort`` / ``head`` / ...)
  keep the binding;
* ``rename({"old": "new"})`` ENDS ``old``'s identity (drops it from live)
  and does NOT make ``new`` provable;
* a column ADDED / ALIASED by ``with_columns(new=...)`` / ``agg(new=...)``
  is a NEW identity (its alias key is recorded ``origin=None``);
* any reassignment the follower does not understand DROPS the binding, so
  later refs off that variable are ``origin=None``.

Indexed reference kinds (the SOUND SUBSET shipped here):

* schema field DECLARATIONS — origin ``(schema, field)`` (the rename anchor);
* ``pl.col("X")`` / ``pl.col("X", "Y")`` string-literal references — origin
  ``(schema, field)`` when the receiver chain provably resolves to that
  schema's frame with ``X`` still live, else ``origin=None``.

The regex form ``pl.col("^...$")`` names no single column and is skipped.
Bare-string column refs (``select("X")`` / ``df["X"]``), ``rename`` keys /
values, and ``with_columns`` / ``agg`` alias OUTPUT names are deliberately
NOT origin-indexed in this subset: a bare ``select("X")`` is followable but
adds no rename safety beyond ``pl.col``, and a rename/alias output name is a
NEW identity (origin restarts) — so to stay sound they are simply left out
(querying one falls to the single-occurrence position fallback). See
``docs/backlog.md`` for the deferred extension.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from polypolarism.pandera_annotation import (
    bare_frame_annotation,
    frame_annotation_schema_name,
)
from polypolarism.pandera_schema import (
    SchemaRegistry,
    _project_root,
    _resolve_module_path,
    collect_schemas_with_imports,
)
from polypolarism.types import Span

# Method chains that preserve a frame's per-column identity (the columns that
# survive keep their original ``(schema, field)`` origin). Conservative: any
# method NOT in this set drops the binding so later refs become origin-less.
# ``rename`` / ``with_columns`` / ``agg`` / ``select`` are handled explicitly.
_IDENTITY_PRESERVING_METHODS: frozenset[str] = frozenset(
    {
        "filter",
        "sort",
        "head",
        "tail",
        "limit",
        "slice",
        "unique",
        "drop_nulls",
        "fill_null",
        "fill_nan",
        "shift",
        "reverse",
        "sample",
        "with_row_index",
        "with_row_count",
        "lazy",
        "collect",
        "clone",
        "cache",
        "set_sorted",
        "shrink_to_fit",
        "rechunk",
        "interpolate",
    }
)


@dataclass(frozen=True)
class ColumnRef:
    """One indexed occurrence of a column name.

    ``origin`` is ``(schema_name, field_name)`` when the reference provably
    resolves to a known schema field along the dataflow, else ``None`` (the
    reference then matches only by exact position — never by name).

    ``schema_source`` is the absolute path of the file that DEFINES the
    origin schema's class (``Schema.source_file``), or ``None`` for an
    origin-less ref. It disambiguates two different schemas that happen to
    share a name across files: the "same column" key is the full triple
    ``(schema_source, schema_name, field_name)``, NOT the name pair — so a
    ``Foo.id`` defined in one module never merges with an unrelated ``Foo.id``
    defined in another.
    """

    file: str  # absolute path
    span: Span  # 1-indexed line / 0-indexed column, like ``ast`` nodes
    column_name: str
    origin: tuple[str, str] | None
    schema_source: str | None = None

    @property
    def identity(self) -> tuple[str | None, str, str] | None:
        """The globally-unique field identity used to merge "same column"
        references, or ``None`` when the ref has no provable origin."""
        if self.origin is None:
            return None
        return (self.schema_source, self.origin[0], self.origin[1])


def _abs(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _str_constant(node: ast.expr | None) -> str | None:
    """The value of a string-constant node, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_regex_col_pattern(value: str) -> bool:
    """``pl.col("^...$")`` — a regex selector, NOT a literal column name.

    polars treats a SINGLE ``pl.col`` string argument that BOTH starts with
    ``^`` AND ends with ``$`` as a regex over the column names. Such a form
    names no single column, so it is never indexed (skip).
    """
    return len(value) >= 2 and value.startswith("^") and value.endswith("$")


def _name_span(node: ast.AST) -> Span:
    return Span.from_node(node)


def _string_literal_span(node: ast.Constant) -> Span:
    """Span covering the column-NAME inside a string literal node.

    ``ast`` reports the literal's span including the surrounding quotes; the
    editor wants to rewrite only the name, so trim one column on each side.
    Falls back to the raw node span when end positions are unavailable.
    """
    base = Span.from_node(node)
    if base.end_line is None or base.end_column is None or base.end_line != base.line:
        return base
    # Single-line ``"name"`` -> inner range is +1 / -1 around the quotes.
    return Span(
        line=base.line,
        column=base.column + 1,
        end_line=base.end_line,
        end_column=base.end_column - 1,
    )


def _pl_col_string_args(node: ast.Call) -> list[ast.Constant]:
    """String-literal arguments of a ``pl.col(...)`` call, REGEX forms removed.

    Returns the constant nodes for ``pl.col("X")`` / ``pl.col("X", "Y")`` so
    each name's precise span can be recorded. A ``pl.col("^...$")`` regex
    selector is excluded (it names no single column). Non-``pl.col`` calls
    return ``[]``.
    """
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "col"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
    ):
        return []
    out: list[ast.Constant] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if _is_regex_col_pattern(arg.value):
                continue
            out.append(arg)
    return out


class _FileIndexer:
    """Builds the column index for ONE file (its own AST), resolving origins
    against ``schema_registry`` (which already carries imported schemas).
    """

    def __init__(self, file_path: Path, registry: SchemaRegistry) -> None:
        self.file = _abs(file_path)
        self.registry = registry
        self.refs: list[ColumnRef] = []

    def _schema_source(self, schema_name: str) -> str | None:
        """Absolute path of the file DEFINING ``schema_name``'s class
        (``Schema.source_file``), used to disambiguate same-named schemas
        across files."""
        schema = self.registry.get(schema_name)
        return schema.source_file if schema is not None else None

    # -- schema field declarations -----------------------------------------

    def index_schema_declarations(self, tree: ast.Module) -> None:
        """Record each schema field DECLARATION in this file as an origin ref.

        Only classes the registry recognises as schemas (defined in THIS
        file) contribute — the annotation target ``Name`` token's precise
        span is recorded so the editor rewrites just the field name.
        """
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            schema = self.registry.get(node.name)
            if schema is None:
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    field_name = stmt.target.id
                    # Only fields the schema actually registered as columns.
                    if field_name not in schema.columns:
                        continue
                    self.refs.append(
                        ColumnRef(
                            file=self.file,
                            span=_name_span(stmt.target),
                            column_name=field_name,
                            origin=(node.name, field_name),
                            schema_source=self._schema_source(node.name) or self.file,
                        )
                    )

    # -- function-body references ------------------------------------------

    def index_functions(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_function(node)

    def _param_schema_bindings(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, str]:
        """Map each frame-typed parameter to its schema name (when the schema
        is known to the registry). A bare ``pl.DataFrame`` param binds nothing
        (open frame -> no provable origin)."""
        bindings: dict[str, str] = {}
        args = func.args
        all_args = [*args.posonlyargs, *args.args, *([] if args.vararg is None else [])]
        if args.kwonlyargs:
            all_args += args.kwonlyargs
        for arg in all_args:
            if arg.annotation is None:
                continue
            schema_name = frame_annotation_schema_name(arg.annotation, self.registry.frame_aliases)
            if schema_name is not None and self.registry.get(schema_name) is not None:
                bindings[arg.arg] = schema_name
            elif bare_frame_annotation(arg.annotation) is not None:
                # Open frame: no provable origin for its columns.
                bindings.pop(arg.arg, None)
        return bindings

    def _index_function(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Walk a function body in source order, tracking which variables are
        schema-origin frames, then record each ``pl.col`` reference's origin.
        """
        param_schemas = self._param_schema_bindings(func)
        # var name -> (schema_name, live_fields). ``live_fields`` is the set of
        # field names still provably mapping to ``(schema, field)``.
        var_state: dict[str, tuple[str, frozenset[str]]] = {}
        for name, schema_name in param_schemas.items():
            schema = self.registry.get(schema_name)
            if schema is None:
                continue
            var_state[name] = (schema_name, frozenset(schema.columns))

        # First, record EVERY pl.col reference's span/name regardless of
        # origin (so the position fallback always finds a token), resolving the
        # origin from the receiver chain's current binding state.
        for stmt in func.body:
            self._index_stmt(stmt, var_state)

    def _index_stmt(
        self,
        stmt: ast.stmt,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> None:
        """Index one statement, updating ``var_state`` for assignments.

        Refs are resolved against the binding state AS OF this statement
        (before any reassignment it performs), which is the correct
        receiver-frame identity for refs INSIDE the RHS.
        """
        # Record refs appearing anywhere in this statement, resolving each
        # against the receiver chain it belongs to.
        self._record_refs_in(stmt, var_state)

        # Then update var_state for simple, followable assignments.
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                self._apply_assignment(target.id, stmt.value, var_state)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                self._apply_assignment(stmt.target.id, stmt.value, var_state)

        # Recurse into nested compound statements (if/for/with/try bodies).
        for child in _nested_body_stmts(stmt):
            self._index_stmt(child, var_state)

    def _apply_assignment(
        self,
        name: str,
        value: ast.expr,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> None:
        """Resolve the new binding state for ``name = <value>``.

        Sound and conservative: anything not understood DROPS the binding.
        """
        new_state = self._eval_frame_state(value, var_state)
        if new_state is None:
            var_state.pop(name, None)
        else:
            var_state[name] = new_state

    def _eval_frame_state(
        self,
        expr: ast.expr,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> tuple[str, frozenset[str]] | None:
        """The (schema, live_fields) state an expression evaluates to, or
        ``None`` when it is not a followable schema-origin frame."""
        # Pass-through: ``y = x``.
        if isinstance(expr, ast.Name):
            return var_state.get(expr.id)

        # Method chain: ``<receiver>.<method>(...)``.
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            receiver_state = self._eval_frame_state(expr.func.value, var_state)
            if receiver_state is None:
                return None
            schema_name, live = receiver_state
            method = expr.func.attr
            if method == "rename":
                return schema_name, self._rename_live(live, expr)
            if method in ("with_columns", "agg"):
                # Added/aliased columns are NEW identities (origin=None). The
                # surviving original columns keep their identity, so the live
                # set is unchanged for the purpose of pl.col resolution; the
                # alias output keys are recorded as origin=None elsewhere.
                return schema_name, live
            if method == "select":
                return schema_name, self._select_live(live, expr)
            if method == "drop":
                return schema_name, self._drop_live(live, expr)
            if method in _IDENTITY_PRESERVING_METHODS:
                return schema_name, live
            # Unknown method -> cannot prove identity survives. Drop.
            return None

        return None

    def _rename_live(self, live: frozenset[str], call: ast.Call) -> frozenset[str]:
        """``rename({"old": "new"})`` ends each renamed source's identity.

        The OLD name leaves the live set (its column identity ended); the NEW
        name is a fresh identity and is NOT added (origin restarts).
        """
        if not call.args or not isinstance(call.args[0], ast.Dict):
            # Non-literal rename mapping -> cannot prove what survived. Drop
            # all provable identities to stay sound.
            return frozenset()
        mapping = call.args[0]
        olds: set[str] = set()
        for key in mapping.keys:
            name = _str_constant(key)
            if name is None:
                # An unreadable key could rename ANY live column away -> drop
                # all to stay sound.
                return frozenset()
            olds.add(name)
        return frozenset(live - olds)

    def _select_live(self, live: frozenset[str], call: ast.Call) -> frozenset[str]:
        """``select(...)`` keeps only the named columns (when all selectors are
        literal ``pl.col("X")`` / ``"X"``). Any non-literal selector -> drop to
        stay sound (the surviving set is unknown)."""
        kept: set[str] = set()
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            names = _selected_column_names(arg)
            if names is None:
                return frozenset()
            kept.update(names)
        return frozenset(live & kept)

    def _drop_live(self, live: frozenset[str], call: ast.Call) -> frozenset[str]:
        """``drop("X", ...)`` removes the named columns; non-literal -> drop."""
        dropped: set[str] = set()
        for arg in call.args:
            names = _selected_column_names(arg)
            if names is None:
                return frozenset()
            dropped.update(names)
        return frozenset(live - dropped)

    # -- reference recording -----------------------------------------------

    def _record_refs_in(
        self,
        node: ast.AST,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> None:
        """Record every ``pl.col`` string-literal column reference reachable
        in ``node`` WITHOUT descending into nested function defs (those are
        indexed separately with their own binding state).

        Each ``pl.col("X")`` is resolved against the binding of the variable
        whose method chain it appears inside. A ``pl.col`` is only matched to
        a single unambiguous frame binding when exactly one schema is live in
        scope at that statement; otherwise it is recorded origin=None (sound).
        """
        # Determine the candidate schema-origin binding for refs in this
        # statement. We resolve a pl.col against the receiver variable of the
        # innermost enclosing method chain (handled in the walk below).
        for descendant in _walk_no_nested_defs(node):
            if isinstance(descendant, ast.Call):
                self._record_pl_col(descendant, var_state)

    def _record_pl_col(
        self,
        call: ast.Call,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> None:
        const_args = _pl_col_string_args(call)
        if not const_args:
            return
        origin_schema = self._unambiguous_live_schema(var_state)
        for const in const_args:
            # ``_pl_col_string_args`` already guarantees a ``str`` value; the
            # annotation re-establishes it for the type checker.
            name: str = const.value  # type: ignore[assignment]
            origin: tuple[str, str] | None = None
            schema_source: str | None = None
            if origin_schema is not None:
                schema_name, live = origin_schema
                if name in live:
                    origin = (schema_name, name)
                    schema_source = self._schema_source(schema_name)
            self.refs.append(
                ColumnRef(
                    file=self.file,
                    span=_string_literal_span(const),
                    column_name=name,
                    origin=origin,
                    schema_source=schema_source,
                )
            )

    def _unambiguous_live_schema(
        self,
        var_state: dict[str, tuple[str, frozenset[str]]],
    ) -> tuple[str, frozenset[str]] | None:
        """When exactly one DISTINCT schema is live among the tracked
        variables, return ``(schema, union-of-live-fields)``; else ``None``.

        Soundness: a ``pl.col("X")`` is ambiguous if two distinct schemas are
        in scope (we cannot know which frame the bare ``pl.col`` targets), so
        we record origin=None rather than guess. A single schema (possibly
        bound to several aliased variables of the SAME schema) is safe — every
        such variable carries the same ``(schema, X)`` origin for a shared
        field, and the merged live set is the intersection across them so a
        field renamed away in any one of them is no longer provable.
        """
        schemas = {s for s, _ in var_state.values()}
        if len(schemas) != 1:
            return None
        (schema_name,) = schemas
        lives = [live for s, live in var_state.values() if s == schema_name]
        if not lives:
            return None
        merged = lives[0]
        for live in lives[1:]:
            merged = merged & live
        return schema_name, merged


def _selected_column_names(node: ast.expr) -> set[str] | None:
    """Literal column names a select/drop selector resolves to, or ``None``
    when the selector is not a plain literal (non-regex ``pl.col("X")`` or a
    bare ``"X"`` string)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _is_regex_col_pattern(node.value):
            return None
        return {node.value}
    if isinstance(node, ast.Call):
        consts = _pl_col_string_args(node)
        # Only a pure ``pl.col("X", ...)`` with all-literal args qualifies; a
        # regex or a chained expression makes the surviving set unknowable.
        if consts and len(consts) == len(node.args) and not node.keywords:
            # ``_pl_col_string_args`` guarantees ``str`` values.
            return {str(c.value) for c in consts}
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for elt in node.elts:
            sub = _selected_column_names(elt)
            if sub is None:
                return None
            names.update(sub)
        return names
    return None


def _nested_body_stmts(stmt: ast.stmt) -> list[ast.stmt]:
    """Statements in the nested bodies of a compound statement (if/for/while/
    with/try), in source order. Nested function/class defs are NOT descended
    into here — they are handled by the function walk."""
    out: list[ast.stmt] = []
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        out.extend(stmt.body)
        out.extend(stmt.orelse)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        out.extend(stmt.body)
    elif isinstance(stmt, ast.Try):
        out.extend(stmt.body)
        for handler in stmt.handlers:
            out.extend(handler.body)
        out.extend(stmt.orelse)
        out.extend(stmt.finalbody)
    return out


def _walk_no_nested_defs(node: ast.AST):
    """Yield ``node`` and its descendants, but DO NOT descend into nested
    function / class definitions (those bodies are indexed separately)."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from _walk_no_nested_defs(child)


def _imported_module_paths(tree: ast.Module, file_path: Path) -> list[Path]:
    """Project-local files imported by ``tree`` (one hop), de-duplicated.

    Reuses the schema importer's resolution rules (project-local only —
    stdlib / third-party / unresolvable imports are skipped). Only files we
    can resolve on disk are returned; reading/parsing happens in the caller.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_module_path(node.module, file_path, node.level)
        elif isinstance(node, ast.Import):
            resolved = None
            for alias in node.names:
                cand = _resolve_module_path(alias.name, file_path)
                if cand is not None:
                    try:
                        real = cand.resolve()
                    except OSError:
                        real = cand
                    if real not in seen:
                        seen.add(real)
                        out.append(cand)
            continue
        else:
            continue
        if resolved is None:
            continue
        try:
            real = resolved.resolve()
        except OSError:
            real = resolved
        if real not in seen:
            seen.add(real)
            out.append(resolved)
    return out


def _schema_identities(registry: SchemaRegistry) -> set[tuple[str | None, str]]:
    """The ``(source_file, name)`` identity of every schema in ``registry``.

    Used to decide whether another project file references the SAME schemas
    as the query file (a reverse-direction match) — by definition, not just
    by name."""
    out: set[tuple[str | None, str]] = set()
    for name, schema in registry.schemas.items():
        # Skip the dotted ``mod.Schema`` re-export keys — the bare name +
        # source_file pair is the canonical identity.
        if "." in name:
            continue
        out.add((schema.source_file, schema.name))
    return out


def _project_py_files(file_path: Path) -> list[Path]:
    """All ``.py`` files under the query file's project root (bounded by a
    project marker). Returns just the query file's directory tree when no
    marker is found, so we never scan unrelated trees.

    This is the reverse-direction discovery surface: a schema field
    declaration's references live in OTHER files that IMPORT the schema, not
    in the files the schema's module imports. We scan the project for such
    referencing files and index the ones that resolve the SAME schema.
    """
    try:
        start = file_path.resolve().parent
    except OSError:
        start = file_path.parent
    root = _project_root(start)
    base = root if root is not None else start
    try:
        return sorted(base.glob("**/*.py"))
    except OSError:
        return []


def build_column_index(file_path: Path) -> list[ColumnRef]:
    """Build the column-reference index for ``file_path``, its project-local
    imports, AND project files that reference the same schemas.

    Reads + ``ast.parse`` only (never executes analyzed code). Each file is
    indexed against ITS OWN import-resolved schema registry, so a schema's
    ``source_file`` is consistent across files — a field declaration in
    ``schemas.py`` and a ``pl.col`` reference in ``app.py`` resolve to the
    same ``(schema_source, schema, field)`` identity regardless of which file
    the query started from.

    Three directions are covered:

    * the query file itself (declarations + references);
    * forward — project-local modules the query file imports (so a query on a
      ``pl.col`` ref reaches the declaration it points at);
    * reverse — project files that import a schema DEFINED in the query file
      (so a query on a field declaration reaches references elsewhere). Only
      files whose registry contains a schema with the SAME ``(source_file,
      name)`` identity are indexed — never a same-named but unrelated schema.
    """
    try:
        main_tree = ast.parse(file_path.read_text())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    main_registry = collect_schemas_with_imports(main_tree, file_path)
    main_identities = _schema_identities(main_registry)

    refs: list[ColumnRef] = []
    seen_files: set[str] = set()

    def index_file(path: Path, tree: ast.Module, registry: SchemaRegistry) -> None:
        key = _abs(path)
        if key in seen_files:
            return
        seen_files.add(key)
        indexer = _FileIndexer(path, registry)
        indexer.index_schema_declarations(tree)
        indexer.index_functions(tree)
        refs.extend(indexer.refs)

    index_file(file_path, main_tree, main_registry)

    # Forward: project-local imported modules (the query file's declarations
    # live there when the query started on a reference).
    for imported in _imported_module_paths(main_tree, file_path):
        try:
            sub_tree = ast.parse(imported.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        index_file(imported, sub_tree, collect_schemas_with_imports(sub_tree, imported))

    # Reverse: every other project file that resolves the SAME schemas — its
    # references must join the rename set. Bounded by the project marker; each
    # file is parsed at most once and only contributes when its registry shares
    # a schema identity with the query file's registry.
    for candidate in _project_py_files(file_path):
        if _abs(candidate) in seen_files:
            continue
        try:
            sub_tree = ast.parse(candidate.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        sub_registry = collect_schemas_with_imports(sub_tree, candidate)
        if _schema_identities(sub_registry) & main_identities:
            index_file(candidate, sub_tree, sub_registry)

    return refs


def _span_covers(span: Span, line: int, col: int) -> bool:
    """True when the 1-indexed ``line`` / 0-indexed ``col`` position falls
    within ``span`` (inclusive of start, exclusive of end column on the same
    line, matching ``ast`` half-open column ranges)."""
    if span.end_line is None or span.end_column is None:
        # Point span with no end: match only the exact start line and a column
        # at-or-after the start (best effort for a single token).
        return line == span.line and col >= span.column
    if line < span.line or line > span.end_line:
        return False
    if span.line == span.end_line:
        return line == span.line and span.column <= col < span.end_column
    if line == span.line:
        return col >= span.column
    if line == span.end_line:
        return col < span.end_column
    return True


def _entry_to_target(ref: ColumnRef) -> dict:
    return {
        "file": ref.file,
        "line": ref.span.line,
        "column": ref.span.column,
        "end_line": ref.span.end_line,
        "end_column": ref.span.end_column,
    }


def rename_targets(file_path: Path, line: int, col: int) -> dict:
    """Resolve the set of source occurrences that PROVABLY refer to the SAME
    column as the token at ``file_path:line:col``.

    Returns a JSON-ready dict::

        {"column": str | None, "schema": str | None, "targets": [<range>, ...]}

    Resolution:

    * Find the index entry whose span COVERS the query position.
    * No entry there -> ``{"targets": []}`` (no column token at the position).
    * Entry has a provable ``(schema, field)`` origin -> return ALL index
      entries (across files) with the SAME origin (declaration + every ref
      resolved to it). This is the safe, useful rename set.
    * Entry has ``origin=None`` -> return ONLY that single occurrence (safe
      fallback) — NEVER name-match other same-named tokens.

    READ-ONLY: returns ranges; the extension performs the edits.
    """
    index = build_column_index(Path(file_path))
    query_file = _abs(Path(file_path))

    hit: ColumnRef | None = None
    for ref in index:
        if ref.file == query_file and _span_covers(ref.span, line, col):
            hit = ref
            break

    if hit is None:
        return {"column": None, "schema": None, "targets": []}

    if hit.origin is None:
        return {
            "column": hit.column_name,
            "schema": None,
            "targets": [_entry_to_target(hit)],
        }

    schema_name, field_name = hit.origin
    # Merge by the full field IDENTITY ``(schema_source, schema, field)`` so a
    # same-named-but-unrelated schema in another file never joins the set.
    identity = hit.identity
    seen: set[tuple[str, int, int]] = set()
    targets: list[dict] = []
    for r in index:
        if r.identity != identity:
            continue
        target = _entry_to_target(r)
        dedup_key = (target["file"], target["line"], target["column"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        targets.append(target)
    return {
        "column": field_name,
        "schema": schema_name,
        "targets": targets,
    }
