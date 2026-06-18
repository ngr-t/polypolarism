"""Stable diagnostic codes for analyzer errors.

The codes are intended for IDE / CI consumption — never repurpose a code,
only add new ones. Format: ``[<code>] message``, where ``<code>`` is a
semantic slug: ``pple-<slug>`` for errors, ``pplw-<slug>`` for warnings
(kebab-case). The ``ppl`` stem plus the ``e`` / ``w`` namespace keeps the
codes from colliding with mypy / ruff inside ``# type: ignore[...]``.
"""

from __future__ import annotations

import re

# Expression / column lookup. One shared code for the whole "a referenced
# column does not exist on the frame" family: pl.col / cs.* lookups, drop,
# rename source, cast, drop_nulls subset, sort, and unique subset all describe
# the same provable runtime miss — the message text distinguishes the kind.
COLUMN_NOT_FOUND = "pple-column-not-found"

# Frame reshape — other column-reference errors
COLUMN_NAME_COLLISION = (
    "pple-column-name-collision"  # with_row_index: name collides with existing column
)
NON_BOOLEAN_PREDICATE = (
    "pple-non-boolean-predicate"  # filter predicate / when condition dtype is not Boolean
)
INCOMPATIBLE_OPERANDS = "pple-incompatible-operands"  # binary operation between incompatible dtypes (arithmetic, comparison, is_in)
INVALID_CAST = "pple-invalid-cast"  # cast between structurally incompatible dtypes
DUPLICATE_COLUMN = (
    "pple-duplicate-column"  # duplicate output column name in select/with_columns/rename
)
NON_NUMERIC_OPERAND = (
    "pple-non-numeric-operand"  # numeric-only operation applied to a non-numeric column
)
LIST_LITERAL_MISUSE = (
    "pple-list-literal-misuse"  # list literal mixed with other positional expression arguments
)

# Join
JOIN_KEY = "pple-join-key"  # join key error (missing or dtype mismatch)

# GroupBy / aggregation
GROUPBY = "pple-groupby"  # group_by key missing or aggregation type error

# Expression namespaces
WRONG_NAMESPACE_DTYPE = "pple-wrong-namespace-dtype"  # namespace accessor (.str/.dt/.list/.arr/.struct/.bin/.cat) on a wrong dtype

# Concat / explode / unpivot
CONCAT_MISMATCH = "pple-concat-mismatch"  # concat schema mismatch / horizontal overlap
EXPLODE = "pple-explode"  # explode: column missing or not List/Array
UNPIVOT = "pple-unpivot"  # unpivot: column missing / value dtype unification failure

# Eager / lazy distinction
EAGER_ONLY_METHOD = (
    "pple-eager-only-method"  # eager-only method called on a LazyFrame (suggest .collect())
)
LAZY_ONLY_METHOD = (
    "pple-lazy-only-method"  # lazy-only method called on a DataFrame (suggest .lazy())
)
EAGER_LAZY_MISMATCH = "pple-eager-lazy-mismatch"  # function expected DataFrame[S] but got LazyFrame[S] (or vice versa)
ANNOTATION_CONFLICT = "pple-annotation-conflict"  # variable annotation re-interprets the inferred frame as an unrelated type (ADR-0005)

# Declared vs inferred return type comparison (checker.py). One shared code
# for the whole family (issue #70): missing column, extra column, dtype
# difference, and could-not-infer all describe the same declared-return-type
# check; the message distinguishes the kind.
RETURN_TYPE = "pple-return-type"  # declared return type does not match the inferred return type

# Schema definition
BROKEN_SCHEMA_ANNOTATION = "pple-broken-schema-annotation"  # schema field annotation provably crashes pandera at runtime (Annotated arity, issue #69)

# Declared-schema interface ("checked island", issue #83)
UNDECLARED_COLUMN = "pple-undeclared-column"  # column not declared in the function's (non-strict) schema — an undeclared dependency, not a provable runtime failure

# Row polymorphism (C-14)
ROWPOLY_NOT_PRESERVED = "pple-rowpoly-not-preserved"  # @rowpoly helper body provably drops the row variable — caller's extra columns are not preserved


# Warnings (pplw-*): inference is imprecise here, but the user can usually
# fix it by adding a type annotation or an explicit dtype argument.
MISSING_RETURN_DTYPE = (
    "pplw-missing-return-dtype"  # map_elements / map_batches without ``return_dtype=`` keyword
)
UNRESOLVED_PIPE = (
    "pplw-unresolved-pipe"  # ``df.pipe(callable)`` where the callable can't be resolved
)
UNKNOWN_FUNCTION = (
    "pplw-unknown-function"  # function call to a name that isn't in the analysed module
)
UNTYPED_CALLABLE = (
    "pplw-untyped-callable"  # lambda / inline callable used where its return dtype is unknowable
)
DATA_DEPENDENT_SCHEMA = "pplw-data-dependent-schema"  # pivot / to_dummies result schema is data-dependent; user should annotate
UNKNOWN_SCHEMA = (
    "pplw-unknown-schema"  # DataFrame[X] / LazyFrame[X] annotation references an unknown schema
)
UNMODELED_METHOD = "pplw-unmodeled-method"  # method not modeled (or experimental polars API); result degrades to Unknown
UNBACKED_NARROWING = "pplw-unbacked-narrowing"  # variable annotation narrows the inferred RHS without runtime backing (ADR-0005)

# Environment / version
UNSUPPORTED_VERSION = "pplw-unsupported-version"  # detected polars or pandera version below polypolarism's supported floor

UNRECOGNIZED_ANNOTATION = "pplw-unrecognized-annotation"  # schema field annotation unrecognized; column degrades to Unknown dtype (#77)

ALL_NULL_AGGREGATION = "pplw-all-null-aggregation"  # grouped aggregation provably yields an all-null column (probed; probably a bug) (#91)

IGNORED_CAST = "pplw-ignored-cast"  # typing.cast(DataFrame[Schema], ...) is not honored as a schema assertion (#102)

ROWPOLY_NOT_THREADED = "pplw-rowpoly-not-threaded"  # imported @rowpoly helper does not provably preserve its row variable; its extras are not threaded (#112)


# Registry of every defined slug — used by the fixture coverage gate so a new
# code can't silently lose its end-to-end test.
ALL_CODES: frozenset[str] = frozenset(
    {
        COLUMN_NOT_FOUND,
        COLUMN_NAME_COLLISION,
        NON_BOOLEAN_PREDICATE,
        INCOMPATIBLE_OPERANDS,
        INVALID_CAST,
        DUPLICATE_COLUMN,
        NON_NUMERIC_OPERAND,
        LIST_LITERAL_MISUSE,
        JOIN_KEY,
        GROUPBY,
        WRONG_NAMESPACE_DTYPE,
        CONCAT_MISMATCH,
        EXPLODE,
        UNPIVOT,
        EAGER_ONLY_METHOD,
        LAZY_ONLY_METHOD,
        EAGER_LAZY_MISMATCH,
        ANNOTATION_CONFLICT,
        RETURN_TYPE,
        BROKEN_SCHEMA_ANNOTATION,
        UNDECLARED_COLUMN,
        ROWPOLY_NOT_PRESERVED,
        MISSING_RETURN_DTYPE,
        UNRESOLVED_PIPE,
        UNKNOWN_FUNCTION,
        UNTYPED_CALLABLE,
        DATA_DEPENDENT_SCHEMA,
        UNKNOWN_SCHEMA,
        UNMODELED_METHOD,
        UNBACKED_NARROWING,
        UNSUPPORTED_VERSION,
        UNRECOGNIZED_ANNOTATION,
        ALL_NULL_AGGREGATION,
        IGNORED_CAST,
        ROWPOLY_NOT_THREADED,
    }
)


_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?:\[([^\]]*)\])?")


def parse_type_ignore(line: str) -> frozenset[str] | None:
    """Parse a ``# type: ignore`` or ``# type: ignore[CODE1, CODE2]`` comment.

    Returns:
        ``None``           — blanket ignore (suppress all diagnostics)
        ``frozenset(...)`` — suppress only the listed diagnostic codes
        ``frozenset()``    — no ``type: ignore`` present; nothing suppressed
    """
    m = _TYPE_IGNORE.search(line)
    if m is None:
        return frozenset()
    raw = m.group(1)
    if raw is None:
        return None  # bare # type: ignore
    codes = frozenset(c.strip() for c in raw.split(",") if c.strip())
    return codes or None  # empty brackets treated as blanket


def tag(code: str, message: str) -> str:
    """Return ``"[CODE] message"``; idempotent if message is already tagged."""
    if message.startswith(f"[{code}]"):
        return message
    return f"[{code}] {message}"


class TaggedError(str):
    """A tagged analyzer error string that also carries structured fields.

    The analyzer collects errors as plain ``str`` (``"[<code>] message"``);
    this subclass keeps that exact text — every ``str`` operation (equality,
    ``str(...)``, regex, ``f"at line N: {e}"`` wrapping, suppression code
    extraction) behaves byte-for-byte the same — while attaching the
    structured info the raising site already held. JSON output reads these
    so consumers don't have to regex the message (issue #70 follow-up).

    ``column`` / ``schema`` are ``None`` when not applicable; the JSON layer
    omits absent fields and never invents one a code doesn't have.

    ``fix`` is an optional structured quick-fix payload (Batch B, Request 2):
    the ``pple-undeclared-column`` "declare the column on the schema" object
    (``{schema, column, schema_file, schema_insert_line, suggested_dtype?}``).
    ``None`` when no sound fix could be built; the JSON layer omits it.

    ``param_name`` / ``param_annotation_range`` are the "relax the param"
    helper fields (Batch B, Request 4): the parameter whose annotation the
    diagnostic suggests loosening (e.g. ``pple-undeclared-column``'s "take a
    bare pl.DataFrame parameter") and its annotation's
    ``{line, column, end_line, end_column}`` range. Both ``None`` when not
    cleanly determinable; the JSON layer omits absent ones.
    """

    column: str | None
    schema: str | None
    fix: dict | None
    param_name: str | None
    param_annotation_range: dict | None


def tagged_error(
    code: str,
    message: str,
    *,
    column: str | None = None,
    schema: str | None = None,
    fix: dict | None = None,
    param_name: str | None = None,
    param_annotation_range: dict | None = None,
) -> TaggedError:
    """Build a :class:`TaggedError`: ``tag(code, message)`` text plus fields.

    The text is identical to ``tag(code, message)``; ``column`` / ``schema`` /
    ``fix`` / ``param_name`` / ``param_annotation_range`` are attached for
    structured JSON output and left unset (``None``) when not supplied.
    """
    err = TaggedError(tag(code, message))
    err.column = column
    err.schema = schema
    err.fix = fix
    err.param_name = param_name
    err.param_annotation_range = param_annotation_range
    return err


_TAGGED_MESSAGE = re.compile(r"^\[(ppl[ew]-[a-z-]+)\]")


def extract_code(message: str) -> str | None:
    """Return the leading ``pple-*`` / ``pplw-*`` slug of a tagged message.

    ``None`` for untagged diagnostics (e.g. parse / read failures). Used by
    JSON output to expose the code structurally (issue #70) — consumers
    should not have to regex the message themselves.
    """
    match = _TAGGED_MESSAGE.match(message)
    return match.group(1) if match else None
