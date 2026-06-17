"""Property-based tests for the type-system algebra (hypothesis).

Compiler test suites pair example-based tests with checks of the algebraic
laws the type system is designed around. Hand-written examples can't cover
the dtype lattice exhaustively; these tests generate random (nested) dtypes
and assert the laws that every call site implicitly relies on:

* ``_is_subtype`` is a partial order (reflexive, transitive, antisymmetric)
  on the Unknown-free fragment, plus the documented ``T <: Nullable[T]``
  one-way widening. ``Unknown`` is deliberately excluded from the order
  laws: gradual typing's consistency relation is not transitive by design.
* ``promote_types`` / ``unify_types`` / ``supertype`` are commutative —
  callers pass operands in source order, so any asymmetry in the probed
  tables would surface as an order-dependent diagnostic.
* ``promote_types`` / ``unify_types`` are idempotent (except that
  ``Nullable[Unknown]`` collapses to ``Unknown``, which absorbs wrappers).
* ``infer_cast`` keeps the target base type and never loses nullability.
* ``_is_frame_subtype`` implements row-polymorphic width subtyping:
  extra actual columns are fine unless the expected frame is strict.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polypolarism.analyzer import (
    FunctionSignature,
    _is_frame_subtype,
    _thread_row_poly_extras,
)
from polypolarism.checker import _is_subtype
from polypolarism.expr_infer import (
    TypePromotionError,
    TypeUnificationError,
    infer_cast,
    promote_types,
    supertype,
    unify_types,
)
from polypolarism.types import (
    Array,
    Binary,
    Boolean,
    Categorical,
    ColumnSpec,
    DataType,
    Date,
    Datetime,
    Duration,
    Enum,
    Float32,
    Float64,
    FrameType,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Null,
    Nullable,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Unknown,
    Utf8,
)

_SIMPLE: tuple[DataType, ...] = (
    Int8(),
    Int16(),
    Int32(),
    Int64(),
    Int128(),
    UInt8(),
    UInt16(),
    UInt32(),
    UInt64(),
    Float32(),
    Float64(),
    Utf8(),
    Boolean(),
    Binary(),
    Date(),
    Time(),
    Datetime(),
    Datetime(tz="UTC"),
    Datetime(unit="ns"),
    Datetime(tz="UTC", unit="ms"),
    Duration(),
    Duration(unit="ms"),
    Categorical(),
    # Concrete category tuples only (issue #67): the categories=None
    # wildcard relates to EVERY concrete Enum in both directions (a
    # gradual-consistency rule like Unknown / the unknown Array width),
    # which is deliberately not antisymmetric — keep it out of the order
    # laws, mirroring how Array widths never mix here.
    Enum(categories=("a", "b")),
    Enum(categories=("b", "a")),
)


def _containers(children: st.SearchStrategy[DataType]) -> st.SearchStrategy[DataType]:
    """Container layer for the recursive dtype strategies: nested ``List``
    and ``Array`` (issue #53 — both containers participate in the order
    laws; ``Array`` recurses in ``_is_subtype`` like ``List`` and is never
    cross-compatible with it)."""
    return st.one_of(children.map(List), children.map(Array))


def _dtypes(*, with_unknown: bool, with_null: bool = True) -> st.SearchStrategy[DataType]:
    """Random dtypes: simple leaves, nested ``List``/``Array``, optional top ``Nullable``."""
    leaves = _SIMPLE + ((Unknown(),) if with_unknown else ())
    inner = st.recursive(st.sampled_from(leaves), _containers, max_leaves=3)
    top = st.one_of(inner, inner.map(Nullable))
    if with_null:
        top = st.one_of(top, st.just(Null()))
    return top


dtypes = _dtypes(with_unknown=True)
unknown_free_dtypes = _dtypes(with_unknown=False)
# No Unknown / Null / top-level Nullable — for tests that wrap in Nullable
# themselves (double-Nullable is not a constructible polars dtype).
unknown_free_plain = st.recursive(st.sampled_from(_SIMPLE), _containers, max_leaves=3)


def _outcome(fn, left: DataType, right: DataType):
    """Result-or-exception-class, so commutativity covers the error cases too."""
    try:
        return fn(left, right)
    except (TypePromotionError, TypeUnificationError) as err:
        return type(err)


def _widen(dtype: DataType) -> DataType:
    """One step up the subtype order: ``T -> Nullable[T]`` where legal."""
    if isinstance(dtype, (Nullable, Null)):
        return dtype
    return Nullable(dtype)


# ---------------------------------------------------------------------------
# _is_subtype: partial order + one-way Nullable widening
# ---------------------------------------------------------------------------


class TestSubtypeOrder:
    @given(dtypes)
    def test_reflexive(self, t: DataType):
        assert _is_subtype(t, t)

    @given(unknown_free_plain)
    def test_nullable_widening(self, t: DataType):
        assert _is_subtype(t, Nullable(t))

    @given(unknown_free_plain)
    def test_nullable_narrowing_rejected(self, t: DataType):
        assert not _is_subtype(Nullable(t), t)

    @given(unknown_free_plain, st.lists(st.booleans(), min_size=2, max_size=2))
    def test_transitive_along_widening_chains(self, t: DataType, steps: list[bool]):
        # Build a <: b <: c by construction (identity or Nullable-widening
        # steps), then check a <: c. Unknown is excluded: consistency with
        # Unknown is intentionally not transitive.
        a = t
        b = _widen(a) if steps[0] else a
        c = _widen(b) if steps[1] else b
        assert _is_subtype(a, b) and _is_subtype(b, c)
        assert _is_subtype(a, c)

    @given(unknown_free_plain, st.lists(st.booleans(), min_size=2, max_size=2))
    def test_transitive_inside_list(self, t: DataType, steps: list[bool]):
        a = t
        b = _widen(a) if steps[0] else a
        c = _widen(b) if steps[1] else b
        assert _is_subtype(List(a), List(b)) and _is_subtype(List(b), List(c))
        assert _is_subtype(List(a), List(c))

    @given(unknown_free_dtypes, unknown_free_dtypes)
    def test_antisymmetric(self, a: DataType, b: DataType):
        if _is_subtype(a, b) and _is_subtype(b, a):
            assert a == b


# ---------------------------------------------------------------------------
# promote_types / unify_types / supertype: commutativity + idempotence
# ---------------------------------------------------------------------------


class TestBinaryTypeOperators:
    @pytest.mark.parametrize("op", [promote_types, unify_types, supertype])
    def test_commutative(self, op):
        @given(dtypes, dtypes)
        def check(a: DataType, b: DataType):
            assert _outcome(op, a, b) == _outcome(op, b, a)

        check()

    @pytest.mark.parametrize("op", [promote_types, unify_types])
    def test_idempotent(self, op):
        @given(unknown_free_dtypes)
        def check(t: DataType):
            assert _outcome(op, t, t) in (t, TypePromotionError, TypeUnificationError)

        check()

    @given(dtypes.map(_widen), dtypes)
    def test_supertype_preserves_nullability(self, a: DataType, b: DataType):
        result = supertype(a, b)
        if isinstance(result, DataType) and not isinstance(result, (Unknown, Null)):
            assert isinstance(result, Nullable)

    @given(dtypes, dtypes)
    def test_infer_cast_targets_base_and_keeps_nullability(
        self, source: DataType, target: DataType
    ):
        result = infer_cast(source, target)
        source_nullable = isinstance(source, Nullable)
        target_base = target.inner if isinstance(target, Nullable) else target
        result_base = result.inner if isinstance(result, Nullable) else result
        assert result_base == target_base
        if source_nullable or isinstance(target, Nullable):
            assert isinstance(result, Nullable)


# ---------------------------------------------------------------------------
# _is_frame_subtype: row-polymorphic width/depth subtyping
# ---------------------------------------------------------------------------

_COLUMN_NAMES = st.sampled_from(("id", "name", "amount", "ts", "flag", "score"))

column_specs = st.builds(
    ColumnSpec,
    dtype=unknown_free_dtypes,
    required=st.booleans(),
)

frame_columns = st.dictionaries(_COLUMN_NAMES, column_specs, min_size=1, max_size=4)


class TestFrameSubtyping:
    @given(frame_columns, st.booleans())
    def test_reflexive(self, columns, strict: bool):
        frame = FrameType(columns, strict=strict)
        assert _is_frame_subtype(frame, frame)

    @given(frame_columns, unknown_free_dtypes)
    def test_width_extra_column_ok_when_not_strict(self, columns, extra_dtype: DataType):
        expected = FrameType(dict(columns), strict=False)
        widened = dict(columns)
        widened["__extra__"] = ColumnSpec(dtype=extra_dtype)
        assert _is_frame_subtype(FrameType(widened), expected)

    @given(frame_columns, unknown_free_dtypes)
    def test_width_extra_column_rejected_when_strict(self, columns, extra_dtype: DataType):
        expected = FrameType(dict(columns), strict=True)
        widened = dict(columns)
        widened["__extra__"] = ColumnSpec(dtype=extra_dtype)
        assert not _is_frame_subtype(FrameType(widened), expected)

    @given(frame_columns)
    def test_missing_required_column_rejected(self, columns):
        expected_cols = dict(columns)
        expected_cols["__required__"] = ColumnSpec(dtype=Int64(), required=True)
        expected = FrameType(expected_cols)
        assert not _is_frame_subtype(FrameType(dict(columns)), expected)

    @given(frame_columns)
    def test_depth_widening_expected_column_to_nullable(self, columns):
        # Wrapping every expected dtype in Nullable can only make the
        # expectation easier to satisfy.
        actual = FrameType(dict(columns))
        widened = {
            name: ColumnSpec(dtype=_widen(spec.dtype), required=spec.required)
            for name, spec in columns.items()
        }
        assert _is_frame_subtype(actual, FrameType(widened))


# ---------------------------------------------------------------------------
# _thread_row_poly_extras: row-variable threading laws (C-14 Tier 3/5)
# ---------------------------------------------------------------------------
#
# Threading adds a @rowpoly helper's caller extras (argument columns beyond the
# declared parameter schema) onto the declared-return frame. The laws the
# inference relies on: the declared return is never weakened (its columns and
# their dtypes survive unchanged), and threading only ever ADDS columns.

_EXTRA_NAMES = st.sampled_from(("e1", "e2", "e3", "e4"))


def _single_param_sig(param_columns) -> FunctionSignature:
    """A one-frame-parameter @rowpoly signature for threading tests."""
    return FunctionSignature(
        name="helper",
        parameters={"a": (0, FrameType(dict(param_columns)))},
        return_type=None,
        lineno=1,
        row_var="R",
    )


class TestRowThreading:
    @given(frame_columns, frame_columns, st.dictionaries(_EXTRA_NAMES, column_specs, max_size=3))
    def test_declared_return_columns_are_never_weakened(
        self, param_columns, return_columns, extra_columns
    ):
        # Every declared-return column keeps its exact spec after threading —
        # the helper's output contract wins over any same-named caller extra.
        sig = _single_param_sig(param_columns)
        arg = FrameType({**param_columns, **extra_columns})
        base = FrameType(dict(return_columns))
        result = _thread_row_poly_extras(sig, [arg], base)
        for name, spec in base.columns.items():
            assert result.columns[name] == spec

    @given(frame_columns, frame_columns, st.dictionaries(_EXTRA_NAMES, column_specs, max_size=3))
    def test_threading_only_adds_columns(self, param_columns, return_columns, extra_columns):
        # The result is a superset of the declared-return columns (threading
        # never drops a declared column).
        sig = _single_param_sig(param_columns)
        arg = FrameType({**param_columns, **extra_columns})
        base = FrameType(dict(return_columns))
        result = _thread_row_poly_extras(sig, [arg], base)
        assert set(result.columns) >= set(base.columns)

    @given(frame_columns, frame_columns)
    def test_no_extras_is_identity(self, param_columns, return_columns):
        # When the argument carries nothing beyond the declared parameter
        # schema, threading returns the base unchanged.
        sig = _single_param_sig(param_columns)
        arg = FrameType(dict(param_columns))
        base = FrameType(dict(return_columns))
        result = _thread_row_poly_extras(sig, [arg], base)
        assert result is base

    @given(
        frame_columns,
        frame_columns,
        st.dictionaries(_EXTRA_NAMES, column_specs, min_size=1, max_size=3),
    )
    def test_extras_disjoint_from_return_are_all_present(
        self, param_columns, return_columns, extra_columns
    ):
        # Extras whose names collide with neither the parameter nor the
        # declared return appear verbatim in the result.
        sig = _single_param_sig(param_columns)
        arg = FrameType({**param_columns, **extra_columns})
        base = FrameType(dict(return_columns))
        result = _thread_row_poly_extras(sig, [arg], base)
        for name, spec in extra_columns.items():
            if name not in param_columns and name not in return_columns:
                assert result.columns[name] == spec
