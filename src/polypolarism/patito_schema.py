"""Patito ``Model`` class registry (ADR-0010).

Collects Patito model classes into the shared :class:`SchemaRegistry` so the
rest of the analyzer — annotation detection (``pt.DataFrame[Model]``),
``Model.validate(df)`` narrowing, and the declared-vs-inferred checker —
treats them exactly like Pandera schemas. The shared ``Schema`` /
``SchemaRegistry`` / ``FrameType`` IR is dialect-neutral; only detection and
field translation are Patito-specific.

Detection is **import-anchored**: ``Model`` is a collision-prone base-class
name, so a ``class X(Model)`` / ``class X(pt.Model)`` is only treated as a
Patito schema when ``Model`` provably resolves to the ``patito`` package
(``import patito as pt`` + ``pt.Model``, or ``from patito import Model``).
This is the ADR-0009 no-false-positive safeguard.

Single-dialect assumption (ADR-0010): a file mixing Patito and Pandera is not
supported. Patito models register with ``setdefault`` so a Pandera schema of
the same name (should one somehow coexist) wins.
"""

from __future__ import annotations

import ast

from polypolarism.compat.patito_api import PATITO_MODEL_BASE, PATITO_PACKAGE
from polypolarism.pandera_schema import (
    Schema,
    SchemaRegistry,
    _annotation_span,
    _import_statements,
    _topo_sort,
)
from polypolarism.patito_dtype import parse_patito_field
from polypolarism.types import ColumnSpec, Nullable, Span, Struct


def scan_patito_imports(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(module_aliases, model_names)`` anchoring Patito spellings.

    - ``module_aliases``: names that refer to the ``patito`` module so
      ``<alias>.Model`` resolves — ``import patito`` -> ``"patito"``,
      ``import patito as pt`` -> ``"pt"``.
    - ``model_names``: names bound directly to ``patito.Model`` —
      ``from patito import Model`` -> ``"Model"``, ``... as M`` -> ``"M"``.

    Only top-level imports (plus those under ``if TYPE_CHECKING:``) are
    scanned, mirroring the Pandera importer.
    """
    module_aliases: set[str] = set()
    model_names: set[str] = set()
    for node in _import_statements(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PATITO_PACKAGE:
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root == PATITO_PACKAGE:
                for alias in node.names:
                    if alias.name == PATITO_MODEL_BASE:
                        model_names.add(alias.asname or alias.name)
    return frozenset(module_aliases), frozenset(model_names)


def collect_patito_schemas(
    tree: ast.Module,
    registry: SchemaRegistry,
    source_file: str | None = None,
) -> None:
    """Find Patito models in ``tree`` and merge them into ``registry``.

    No-op when ``patito`` is not imported (the import-anchor returns empty),
    so Pandera-only and library-agnostic files are untouched.
    """
    module_aliases, model_names = scan_patito_imports(tree)
    if not module_aliases and not model_names:
        return

    # Iteratively expand the candidate set so a subclass of an already-found
    # Patito model is recognised even when its parent appears later.
    candidates: dict[str, ast.ClassDef] = {}
    candidate_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name in candidate_names:
                continue
            if _looks_like_patito(node, module_aliases, model_names, candidate_names):
                candidates[node.name] = node
                candidate_names.add(node.name)
                changed = True

    frozen_names = frozenset(candidate_names)
    nested_refs: dict[str, dict[str, str]] = {}
    for name in _topo_sort(candidates):
        schema, refs = _parse_patito_schema(candidates[name], registry, frozen_names, source_file)
        registry.schemas.setdefault(name, schema)
        if refs:
            nested_refs[name] = refs

    _resolve_nested_structs(registry, nested_refs)


def _is_patito_base(
    base: ast.expr,
    module_aliases: frozenset[str],
    model_names: frozenset[str],
    known: set[str],
) -> bool:
    """True when ``base`` marks a class as a Patito model.

    Recognised: ``<patito-module>.Model``, a bare name bound to
    ``patito.Model``, or a bare name of an already-found Patito candidate
    (transitive inheritance within the file).
    """
    if (
        isinstance(base, ast.Attribute)
        and base.attr == PATITO_MODEL_BASE
        and isinstance(base.value, ast.Name)
        and base.value.id in module_aliases
    ):
        return True
    if isinstance(base, ast.Name):
        return base.id in model_names or base.id in known
    return False


def _looks_like_patito(
    node: ast.ClassDef,
    module_aliases: frozenset[str],
    model_names: frozenset[str],
    known: set[str],
) -> bool:
    return any(_is_patito_base(b, module_aliases, model_names, known) for b in node.bases)


def _parse_patito_schema(
    node: ast.ClassDef,
    registry: SchemaRegistry,
    candidate_names: frozenset[str],
    source_file: str | None,
) -> tuple[Schema, dict[str, str]]:
    """Parse one Patito model into a ``(Schema, nested_model_refs)`` pair.

    ``nested_model_refs`` maps a field name to the model it references, for the
    second-pass struct resolution. Patito models bind ``strict=True`` (extra
    columns are rejected at validate time — probed).
    """
    schema = Schema(
        name=node.name,
        strict=True,
        source_file=source_file,
        header_line=node.lineno,
    )

    # Merge already-parsed parent models (leftmost base wins on conflict).
    parent_names = [
        b.id for b in node.bases if isinstance(b, ast.Name) and b.id in registry.schemas
    ]
    schema.bases = parent_names
    for base_name in parent_names:
        parent = registry.schemas[base_name]
        for col_name, col_spec in parent.columns.items():
            schema.columns.setdefault(col_name, col_spec)
        for col_name, span in parent.field_spans.items():
            schema.field_spans.setdefault(col_name, span)
        for col_name, span in parent.field_annotation_spans.items():
            schema.field_annotation_spans.setdefault(col_name, span)

    nested_refs: dict[str, str] = {}
    for stmt in node.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        field_name = stmt.target.id
        schema.field_spans[field_name] = Span.from_node(stmt)
        ann_span = _annotation_span(stmt.annotation)
        if ann_span is not None:
            schema.field_annotation_spans[field_name] = ann_span
        else:
            schema.field_annotation_spans.pop(field_name, None)
        spec, nested = parse_patito_field(stmt.annotation, stmt.value, candidate_names)
        schema.columns[field_name] = spec
        if nested is not None:
            nested_refs[field_name] = nested

    return schema, nested_refs


def _resolve_nested_structs(
    registry: SchemaRegistry,
    nested_refs: dict[str, dict[str, str]],
) -> None:
    """Second pass: replace nested-model placeholders with ``Struct`` of the
    referenced model's columns.

    The struct is CLOSED (#118): a nested Patito model has an exact field set,
    so accessing a field it does not declare (``.struct.field("nope")``) is a
    provable miss. Group acceptance inside the struct still works because the
    checker compares closed-struct fields field-by-field via the subtype
    verdict (which honours ``DataTypeGroup`` members), not by exact dict
    equality. A reference whose model is unknown keeps its open-``Struct``
    placeholder (fields genuinely unknown).
    """
    for schema_name, refs in nested_refs.items():
        schema = registry.schemas.get(schema_name)
        if schema is None:
            continue
        for field_name, model_name in refs.items():
            target = registry.schemas.get(model_name)
            if target is None:
                continue
            struct = Struct(
                fields={cn: cs.dtype for cn, cs in target.columns.items()},
                open=False,
            )
            current = schema.columns.get(field_name)
            nullable = current is not None and isinstance(current.dtype, Nullable)
            dtype = Nullable(struct) if nullable else struct
            schema.columns[field_name] = ColumnSpec(dtype=dtype, required=True)
