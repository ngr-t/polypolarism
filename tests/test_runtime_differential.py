"""Runtime differential harness: execute the fixture corpus against real polars/pandera.

polypolarism's value proposition is "static verdict == polars/pandera runtime
verdict".  The golden-file harness (``test_fixtures.py``) pins the *static*
side; nothing executed the fixtures, so two bug classes could slip through:

* a ``valid/`` fixture whose code would actually CRASH at runtime
  (static false negative hidden by checker leniency), and
* an ``invalid/`` fixture whose flagged operation would actually be FINE at
  runtime (static false positive).

This module closes the loop.  For every top-level fixture function with
``DataFrame[Model]`` / ``LazyFrame[Model]`` parameters it

1. synthesizes a small input frame per parameter from the Pandera model,
2. calls the function (awaiting awaitables, in a tmp cwd), and
3. compares the runtime outcome against the per-function *static* verdict
   parsed from the fixture's ``.expected`` golden file:

   * static ``OK`` / ``WARN``  -> the call must not raise, and the declared
     return model (if any) must validate against the result;
   * static ``FAIL``           -> the call or the return validation must raise.

Disagreements are the whole point: do NOT silently add a ``SKIP`` entry when a
case fails.  Triage first — a synthesis bug means fixing the generator (or
adding a justified ``VALUE_OVERRIDES`` entry), a mis-specified fixture means
fixing the fixture, and a genuine checker bug must be reported/filed.  Every
``SKIP`` entry needs a one-line justification, and the coverage floor test at
the bottom keeps the skip-list from silently rotting.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import re
import sys
import typing
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("pandera")

import polars as pl  # noqa: E402
from pandera.api.polars.container import DataFrameSchema  # noqa: E402
from pandera.polars import DataFrameModel  # noqa: E402
from pandera.typing.polars import DataFrame as PanderaDataFrame  # noqa: E402
from pandera.typing.polars import LazyFrame as PanderaLazyFrame  # noqa: E402
from polars.datatypes import DataTypeClass  # noqa: E402
from polars.exceptions import PanicException  # noqa: E402

pytestmark = pytest.mark.runtime

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATEGORIES = ("valid", "invalid")
N_ROWS = 3

# pandera warns "'X' support is not guaranteed" for landmark dtypes
# (Float16/Int128/UInt128); the corpus uses them deliberately.
_PANDERA_DTYPE_WARNING = r".*support is not guaranteed.*"

# Mirrors polars' own PolarsDataType: an instance or the (metaclass-typed)
# bare dtype class.
PolarsDType = pl.DataType | DataTypeClass


# ---------------------------------------------------------------------------
# Skip-list: cases the harness cannot meaningfully execute.
#
# Keys are "category/fixture.py" (whole fixture) or
# "category/fixture.py::function" (single function).  Every entry must carry a
# one-line justification; stale keys fail test_skip_list_is_not_stale.
# ---------------------------------------------------------------------------
SKIP: dict[str, str] = {
    # -- annotation contradiction (PLY033) is static-only by design
    #    (ADR-0005): pandera annotations are inert at runtime (validation
    #    happens only via validate/check_types), so the function executes
    #    cleanly while the static verdict is FAIL — same family as the
    #    PLY032 annotation-contract skip below.
    "invalid/variable_annotation_contradiction.py": (
        "ADR-0005: annotations are inert at runtime; the contradiction is static-only"
    ),
    # -- validate-narrowing fixtures: the annotated *input* schema is a lower
    #    bound; the body's Schema.validate() call demands columns beyond it.
    #    Inputs synthesized from the annotation alone can never satisfy the
    #    runtime narrowing check — that check failing IS the feature.
    "valid/pandera_validate_bare.py": (
        "validate-narrowing: body requires columns beyond the annotated input schema"
    ),
    "valid/pandera_validate_assign.py": (
        "validate-narrowing: body requires columns beyond the annotated input schema"
    ),
    "valid/pandera_pipe_validate.py": (
        "validate-narrowing: body requires columns beyond the annotated input schema"
    ),
    "valid/pandera_lazyframe_validate.py": (
        "validate-narrowing: body requires columns beyond the annotated input schema"
    ),
    # -- known modeled divergence: sink_csv(lazy=True) terminates the plan at
    #    runtime (collect() writes the file and yields a 0-column frame);
    #    polypolarism deliberately models sink_* as identity (see fixture).
    "valid/m13_lazy_pipeline.py::streaming_sink": (
        "sink_csv terminates the plan at runtime; statically modeled as identity by design"
    ),
    # -- declaration-level static error with no runtime manifestation:
    #    PLY032 flags passing LazyFrame[S] where DataFrame[S] is declared, but
    #    the helper body is an identity, so nothing raises at runtime.
    "invalid/m15_eager_lazy_arg.py::caller": (
        "PLY032 annotation-contract violation; identity body never crashes at runtime"
    ),
    # -- gradual-typing notion with no runtime counterpart: OrderIn declares
    #    ``items: pl.List(pl.Struct)``, which at runtime is List(Struct([]));
    #    @pa.check_types + coerce casts to it, stripping every struct field
    #    before the body runs.  Statically, bare pl.Struct means "unknown
    #    fields" (the open-frame behavior this fixture exists to test).
    "valid/container_dtypes.py::explode_lines": (
        "bare pl.Struct = unknown fields statically but Struct([]) at runtime; "
        "check_types+coerce strips all fields"
    ),
    # -- documented int-literal leniency IS this function's subject:
    #    then(1).otherwise(0) is Int32 at runtime but polypolarism models int
    #    literals as Int64 (expr_infer.infer_lit), so the declared ``a: int``
    #    only holds statically.
    "valid/unknown_dtype_tracking.py::when_then_otherwise_column": (
        "int literals modeled as Int64 by design; runtime Int32 fails pandera validation"
    ),
    # -- issue #69 (PLY041): the module crashes at IMPORT time — Python's
    #    typing rejects single-argument Annotated — so the harness cannot
    #    load it at all; the import-time crash IS the diagnostic's subject.
    "invalid/dtype_annotated_no_metadata.py": (
        "module crashes at import: single-argument Annotated is a typing-level TypeError"
    ),
    # -- issue #69 (PLY041): the input schema is the broken one, so the
    #    harness cannot synthesize an input frame (to_schema raises the very
    #    TypeError the fixture exists to flag). The return-side siblings in
    #    the same fixture self-verify via the return validation.
    "invalid/dtype_annotated_arity.py::broken_schema_as_input": (
        "input schema is the PLY041 subject: to_schema raises TypeError during input synthesis"
    ),
    # -- issues #94 / #95: the harness cannot synthesize a non-frame parameter
    #    (``flag: bool``); the error is branch-dependent and statically flagged.
    "invalid/early_return_wrong_schema.py::wrong_early_return": (
        "non-frame bool parameter 'flag' has no default; branch-dependent error is static-only"
    ),
    "invalid/ternary_diverging_arms.py::ternary_diverge": (
        "non-frame bool parameter 'flag' has no default; branch-dependent error is static-only"
    ),
    # -- issue #95: branch divergence fixtures have a non-frame bool parameter.
    "invalid/if_branch_diverge.py::if_branch_wrong": (
        "non-frame bool parameter 'flag' has no default; branch-dependent error is static-only"
    ),
    "invalid/if_branch_diverge.py::else_branch_wrong": (
        "non-frame bool parameter 'flag' has no default; branch-dependent error is static-only"
    ),
    "valid/if_branch_converge.py::both_branches_ok": (
        "non-frame bool parameter 'flag' has no default; branch-dependent behavior is static-only"
    ),
    # -- type: ignore suppression: the suppressed functions have real runtime bugs;
    #    the static verdict is OK (suppressed) but runtime would FAIL under check_types.
    "valid/type_ignore_suppress.py::blanket": (
        "type: ignore blanket suppress: runtime would FAIL but static is suppressed by design"
    ),
    "valid/type_ignore_suppress.py::specific": (
        "type: ignore[PLY040] suppress: runtime would FAIL but static is suppressed by design"
    ),
    "valid/type_ignore_suppress.py::multi_code": (
        "type: ignore[PLY040, PLY032] suppress: runtime would FAIL but static is suppressed by design"
    ),
}


# ---------------------------------------------------------------------------
# Value overrides: per-fixture column series for functions whose runtime
# behavior depends on column *values* the schema cannot express.  Each entry
# must say why generic samples are not enough.
# ---------------------------------------------------------------------------
def _value_overrides() -> dict[str, dict[str, pl.Series]]:
    return {
        # bin.decode("hex") needs even-length hex strings.
        "valid/bin_namespace.py": {"hex_repr": pl.Series(["0a", "1b", "2c"])},
        # str.to_datetime parses these with explicit formats; generic "s0"
        # strings raise. The offset column feeds the %::z parse.
        "valid/str_to_datetime_time_unit.py": {
            "s": pl.Series(["2020-01-01 00:00:00", "2021-06-15 12:30:00", "2022-12-31 23:59:59"]),
            "s_off": pl.Series(
                [
                    "2020-01-01 00:00:00 +09:00",
                    "2021-06-15 12:30:00 +00:00",
                    "2022-12-31 23:59:59 -05:00",
                ]
            ),
        },
        # the fixture exercises the value-dependent Utf8 -> Int64 cast (its own
        # docstring calls it out); generic "s0" strings are not numeric.
        "valid/compare_cast_ok.py": {"s": pl.Series(["1", "2", "3"])},
        # pivot output columns come from the values of 'metric'; the declared
        # WideOut(revenue, cost) is only reachable when every (region, metric)
        # combination is present (missing combos pivot to nulls).
        "valid/m12_pivot_annotated.py": {
            "region": pl.Series(["r1", "r1", "r2", "r2"]),
            "metric": pl.Series(["revenue", "cost", "revenue", "cost"]),
        },
        # same pivot as the valid twin: generic metric values would make the
        # return validation fail on *missing* revenue/cost columns instead of
        # the intended Float64-vs-declared-Int64 dtype mismatch.
        "invalid/m12_pivot_wrong_declared.py": {
            "region": pl.Series(["r1", "r1", "r2", "r2"]),
            "metric": pl.Series(["revenue", "cost", "revenue", "cost"]),
        },
        # hstack requires equal heights: unpivot of a 3-row frame over two
        # `on` columns yields 6 rows, so the side frame must have 6 rows.
        "valid/m4_unpivot_and_hstack.py": {
            "label": pl.Series([f"l{i}" for i in range(6)]),
        },
        # same height constraint as the valid twin: without it hstack raises a
        # ShapeError instead of the intended label dtype mismatch.
        "invalid/m4_hstack_wrong_dtype.py": {
            "label": pl.Series([f"l{i}" for i in range(6)]),
        },
        # str.to_integer/to_time/to_decimal parse column *contents*; generic
        # "s0" strings are not parseable (and price is parsed at scale=0 AND
        # scale=2, so integer-valued strings keep both casts lossless).
        "valid/m3_str_namespace.py": {
            "qty": pl.Series(["1", "2", "3"]),
            "price": pl.Series(["10", "25", "37"]),
            "opens_at": pl.Series(["01:00:00", "02:30:00", "03:45:00"]),
        },
        # to_decimal parses column *contents* too; parseable strings keep the
        # runtime failure on the intended dtype mismatch (Decimal(38, 2) vs
        # the declared Decimal(38, 0)), not on an unparseable "s0".
        "invalid/str_to_decimal_wrong_scale.py": {
            "price": pl.Series(["10.50", "20.25", "30.75"]),
        },
        # str.to_datetime("...%:z") parses column *contents*; generic "s0"
        # strings are not offset-stamped datetimes.
        "valid/tz_same_ops.py": {
            "stamp": pl.Series(["2024-01-02T03:04:05+09:00", "2024-01-02T04:04:05+09:00"]),
        },
        # same parse in the false-negative twin: parseable strings keep the
        # runtime failure on the intended dtype mismatch (Datetime[UTC] vs
        # the declared naive Datetime), not on an unparseable "s0".
        "invalid/tz_mixing.py": {
            "stamp": pl.Series(["2024-01-02T03:04:05+09:00", "2024-01-02T04:04:05+09:00"]),
        },
        # `items: pl.List(pl.Struct)` leaves the struct fields unknown, but the
        # body unnests and the declared output requires a `qty` field.
        "valid/container_dtypes.py": {
            "items": pl.Series([[{"qty": 1}], [{"qty": 2}], [{"qty": 3}]]),
        },
    }


# Fixtures whose static FAIL quantifies over *all* legal inputs: the witness
# input must omit required=False columns (absence is legal per the input
# schema), otherwise the runtime sample masks the failure.
OMIT_OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "invalid/pandera_optional_required_mismatch.py",
        # Issue #84: the static OK is value-dependent — the optional sku
        # MAY be absent, and the sku-less input (the issue's runtime
        # counterexample) is the execution the static verdict describes.
        # With sku synthesized present, the strict argument validation
        # correctly raises — that value-dependence is exactly why the
        # static side stays lenient instead of claiming a proof.
        "valid/optional_extra_strict_param.py",
    }
)


class SynthesisError(Exception):
    """The generator cannot build a sample for this schema."""


# ---------------------------------------------------------------------------
# Input synthesis
# ---------------------------------------------------------------------------
def _base_type(dtype: PolarsDType) -> DataTypeClass:
    if isinstance(dtype, pl.DataType):
        return dtype.base_type()
    return dtype


def _sample_values(dtype: PolarsDType, n: int, offset: int = 0) -> list[Any]:
    """Generate n sample values for a polars dtype (ascending where ordered).

    Ordered dtypes get ascending values so sortedness-sensitive operations
    (join_asof, group_by_dynamic) work on synthesized frames.

    ``offset`` shifts the value sequence; the harness gives each frame
    PARAMETER a different offset so multi-frame operations see partially
    overlapping data. Without it, every parameter gets identical values,
    every join key matches, and join-introduced nulls never flow — letting
    nullability-invalid fixtures pass at runtime for the wrong reason.
    """
    base = _base_type(dtype)
    if base in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.UInt128,
    ):
        return [i + 1 + offset for i in range(n)]
    if base in (pl.Float16, pl.Float32, pl.Float64):
        return [i + 0.5 + offset for i in range(n)]
    if base is pl.Boolean:
        return [(i + offset) % 2 == 0 for i in range(n)]
    if base is pl.String:
        return [f"s{i + offset}" for i in range(n)]
    if base is pl.Date:
        return [date(2024, 1, 1) + timedelta(days=i + offset) for i in range(n)]
    if base is pl.Datetime:
        return [datetime(2024, 1, 1, 6, 0) + timedelta(hours=i + offset) for i in range(n)]
    if base is pl.Duration:
        return [timedelta(seconds=i + 1 + offset) for i in range(n)]
    if base is pl.Time:
        return [time(hour=(i + 1 + offset) % 24) for i in range(n)]
    if base is pl.Decimal:
        return [Decimal(i + 1 + offset) for i in range(n)]
    if base is pl.Binary:
        return [bytes([i + 1 + offset]) for i in range(n)]
    if base is pl.Categorical:
        return [f"c{i + offset}" for i in range(n)]
    if base is pl.Enum:
        if isinstance(dtype, pl.Enum):
            categories = dtype.categories.to_list()
            return [categories[(i + offset) % len(categories)] for i in range(n)]
        # Bare `pl.Enum` carries no categories; the caller substitutes a
        # concrete Enum dtype (see _sample_series).
        raise SynthesisError("bare pl.Enum has no categories to sample from")
    if base is pl.List:
        if not isinstance(dtype, pl.List):
            raise SynthesisError("bare pl.List has no inner dtype")
        inner = _sample_values(dtype.inner, n + 1, offset)
        return [[inner[i], inner[i + 1]] for i in range(n)]
    if base is pl.Array:
        if not isinstance(dtype, pl.Array):
            raise SynthesisError("bare pl.Array has no inner dtype/width")
        width = dtype.size
        inner = _sample_values(dtype.inner, max(n, width), offset)
        return [[inner[(i + j) % len(inner)] for j in range(width)] for i in range(n)]
    if base is pl.Struct:
        if not isinstance(dtype, pl.Struct):
            raise SynthesisError("bare pl.Struct has no fields")
        field_values = {
            field.name: _sample_values(field.dtype, n, offset) for field in dtype.fields
        }
        return [{name: values[i] for name, values in field_values.items()} for i in range(n)]
    raise SynthesisError(f"no sampler for dtype {dtype!r}")


# Invented categories for columns declared as bare `pl.Enum`: the input frame
# needs *some* concrete Enum dtype, and no input-side validation compares it
# against the (un-instantiable) declared class.
_BARE_ENUM_FALLBACK = ("alpha", "beta", "gamma")


def _sample_series(
    name: str, dtype: PolarsDType, nullable: bool, n: int, offset: int = 0
) -> pl.Series:
    if _base_type(dtype) is pl.Enum and not isinstance(dtype, pl.Enum):
        dtype = pl.Enum(list(_BARE_ENUM_FALLBACK))
    values = _sample_values(dtype, n, offset)
    if nullable and n > 1:
        # A null must actually flow for nullability invariants to be testable
        # at runtime; keep row 0 non-null so joins/sorts stay well-behaved.
        values[1] = None
    return pl.Series(name, values, dtype=dtype)


def _polars_dtype(column: Any) -> PolarsDType:
    """Extract the polars dtype from a pandera column's engine DataType."""
    dtype = getattr(column.dtype, "type", None)
    if dtype is None:
        raise SynthesisError(f"pandera dtype {column.dtype!r} has no polars type")
    return dtype


def _synthesize_frame(
    model: type[DataFrameModel],
    overrides: dict[str, pl.Series],
    omit_optional: bool,
    offset: int = 0,
) -> pl.DataFrame:
    schema = _to_schema(model)
    columns = {
        name: column
        for name, column in schema.columns.items()
        if not (omit_optional and not column.required)
    }
    n = max(
        [N_ROWS] + [len(series) for name, series in overrides.items() if name in columns],
    )
    series: list[pl.Series] = []
    for name, column in columns.items():
        if name in overrides:
            series.append(overrides[name].rename(name))
        else:
            series.append(
                _sample_series(name, _polars_dtype(column), bool(column.nullable), n, offset)
            )
    return pl.DataFrame(series)


# ---------------------------------------------------------------------------
# Annotation introspection
# ---------------------------------------------------------------------------
def _frame_annotation(annotation: Any) -> tuple[type[DataFrameModel], bool] | None:
    """Return (model, is_lazy) for DataFrame[Model] / LazyFrame[Model], else None."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if not args or not inspect.isclass(origin):
        return None
    if issubclass(origin, PanderaDataFrame):
        return args[0], False
    if issubclass(origin, PanderaLazyFrame):
        return args[0], True
    return None


def _to_schema(model: type[DataFrameModel]) -> DataFrameSchema:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PANDERA_DTYPE_WARNING)
        return model.to_schema()


# ---------------------------------------------------------------------------
# Execution + verdicts
# ---------------------------------------------------------------------------
def _invoke(fn: Any, kwargs: dict[str, Any]) -> Any:
    """Call fn inside an event loop, resolving awaitables (collect_async)."""

    async def run() -> Any:
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    return asyncio.run(run())


def _pandera_can_validate(schema: DataFrameSchema) -> bool:
    """pandera's polars engine mis-handles some landmark dtypes.

    * Float16/Int128/UInt128 raise a false "expected X, got X" SchemaError.
    * bare ``pl.Enum`` (no categories) raises AttributeError during checks.

    For models using these (top-level — the corpus nests none of them), fall
    back to a structural check instead of Model.validate.
    """
    for column in schema.columns.values():
        dtype = _polars_dtype(column)
        base = _base_type(dtype)
        if base in (pl.Float16, pl.Int128, pl.UInt128):
            return False
        if base is pl.Enum and not isinstance(dtype, pl.Enum):
            return False
    return True


def _structural_validate(schema: DataFrameSchema, result: pl.DataFrame) -> None:
    """Minimal mirror of pandera semantics: presence, dtype, null check."""
    for name, column in schema.columns.items():
        if name not in result.columns:
            if column.required:
                raise AssertionError(f"required column {name!r} missing from result")
            continue
        actual = result.schema[name]
        expected = _polars_dtype(column)
        # polars dtype equality is loose for class-vs-instance (pl.Enum == Enum([...])).
        if actual != expected:
            raise AssertionError(f"column {name!r}: expected dtype {expected!r}, got {actual!r}")
        if not column.nullable and result[name].null_count() > 0:
            raise AssertionError(f"non-nullable column {name!r} contains nulls")


def _validate_return(model: type[DataFrameModel], result: Any) -> None:
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    if not isinstance(result, pl.DataFrame):
        raise AssertionError(f"expected a frame result, got {type(result)!r}")
    schema = _to_schema(model)
    if _pandera_can_validate(schema):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_PANDERA_DTYPE_WARNING)
            model.validate(result)
    else:
        _structural_validate(schema, result)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------
VERDICT_RE = re.compile(r"^(?P<name>\w+): (?P<status>OK|FAIL|WARN)$", re.MULTILINE)


def _parse_golden(path: Path) -> dict[str, str]:
    text = path.with_suffix(".expected").read_text()
    return {m.group("name"): m.group("status") for m in VERDICT_RE.finditer(text)}


def _load_fixture_module(path: Path, category: str) -> ModuleType:
    name = f"_runtime_diff_{category}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PANDERA_DTYPE_WARNING)
        spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Case:
    fixture_key: str  # "valid/foo.py"
    func_name: str
    static_passed: bool  # OK/WARN vs FAIL

    @property
    def case_key(self) -> str:
        return f"{self.fixture_key}::{self.func_name}"


_MODULES: dict[str, ModuleType] = {}


def _collect_cases() -> tuple[list[Case], dict[str, str]]:
    """Return (runnable cases, {case_key: skip reason})."""
    cases: list[Case] = []
    skipped: dict[str, str] = {}
    for category in CATEGORIES:
        for path in sorted((FIXTURES_DIR / category).glob("*.py")):
            fixture_key = f"{category}/{path.name}"
            # Whole-fixture skips must be honored BEFORE import: some
            # fixtures (e.g. invalid/dtype_annotated_no_metadata.py) cannot
            # be imported at all — the import-time crash is the static
            # diagnostic's subject. One skipped param per fixture.
            fixture_reason = SKIP.get(fixture_key)
            if fixture_reason is not None:
                skipped[fixture_key] = fixture_reason
                continue
            module = _load_fixture_module(path, category)
            _MODULES[fixture_key] = module
            verdicts = _parse_golden(path)
            for func_name, fn in inspect.getmembers(module, inspect.isfunction):
                if fn.__module__ != module.__name__:
                    continue
                hints = typing.get_type_hints(fn)
                params = inspect.signature(fn).parameters
                frame_params = {
                    name: _frame_annotation(hints[name])
                    for name in params
                    if name in hints and _frame_annotation(hints[name]) is not None
                }
                if not frame_params:
                    continue  # not callable from schemas alone; out of scope
                case = Case(
                    fixture_key=fixture_key,
                    func_name=func_name,
                    static_passed=verdicts.get(func_name) != "FAIL",
                )
                reason = SKIP.get(case.case_key)
                if reason is not None:
                    skipped[case.case_key] = reason
                else:
                    cases.append(case)
    return cases, skipped


CASES, SKIPPED = _collect_cases()


def _params() -> list[Any]:
    params = [pytest.param(case, id=case.case_key) for case in CASES]
    params.extend(
        pytest.param(None, id=case_key, marks=pytest.mark.skip(reason=reason))
        for case_key, reason in SKIPPED.items()
    )
    return params


@pytest.mark.parametrize("case", _params())
def test_runtime_agrees_with_static_verdict(
    case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # any file side effects (sink_*, write_*) land in tmp
    module = _MODULES[case.fixture_key]
    fn = getattr(module, case.func_name)
    hints = typing.get_type_hints(fn)
    overrides = _value_overrides().get(case.fixture_key, {})
    omit_optional = case.fixture_key in OMIT_OPTIONAL_COLUMNS

    kwargs: dict[str, Any] = {}
    frame_index = 0
    for name, param in inspect.signature(fn).parameters.items():
        frame = _frame_annotation(hints.get(name))
        if frame is None:
            if param.default is inspect.Parameter.empty:
                raise SynthesisError(
                    f"{case.case_key}: parameter {name!r} is not a frame and has no "
                    "default; add a SKIP entry with justification"
                )
            continue
        model, is_lazy = frame
        # Each frame parameter gets a shifted value sequence so joins see
        # partially overlapping keys (unmatched rows make join-introduced
        # nulls actually flow at runtime).
        synthesized = _synthesize_frame(model, overrides, omit_optional, offset=frame_index)
        frame_index += 1
        kwargs[name] = synthesized.lazy() if is_lazy else synthesized

    return_frame = _frame_annotation(hints.get("return"))

    if case.static_passed:
        # Static OK/WARN: the call must succeed and the declared return model
        # (if any) must validate.  Exceptions propagate as plain failures.
        result = _invoke(fn, kwargs)
        if return_frame is not None:
            _validate_return(return_frame[0], result)
    else:
        # Static FAIL: the call or the return validation must raise.
        # ``PanicException`` (a rust panic surfaced by pyo3) derives from
        # BaseException, not Exception; a static FAIL predicting a
        # guaranteed crash (e.g. grouped Float16 mean, backlog N-5) is
        # confirmed by it exactly like by a regular polars error. Probed
        # (polars 1.41.2): these panics are catchable and leave the
        # process healthy.
        try:
            result = _invoke(fn, kwargs)
            if return_frame is not None:
                _validate_return(return_frame[0], result)
        except (Exception, PanicException):
            return
        pytest.fail(
            f"{case.case_key}: static verdict is FAIL but the function ran and "
            "validated cleanly at runtime — possible static false positive "
            "(or the synthesized input is not adversarial enough)"
        )


# ---------------------------------------------------------------------------
# Meta-tests: skip-list hygiene and coverage floor
# ---------------------------------------------------------------------------
# Honest floor based on observed numbers (294 covered / 11 skipped = 96.4% as
# of 2026-06; whole-fixture skips now count once per fixture, not per
# function, since they are honored before import); raise it if coverage
# improves, never lower it to silence a failure without triage.
COVERAGE_FLOOR = 0.95


def test_coverage_floor() -> None:
    covered = len(CASES)
    skipped = len(SKIPPED)
    fraction = covered / (covered + skipped)
    assert fraction >= COVERAGE_FLOOR, (
        f"runtime differential coverage dropped to {fraction:.1%} "
        f"({covered} covered / {skipped} skipped); triage before skipping more"
    )


def test_skip_list_is_not_stale() -> None:
    """Every SKIP / override key must still point at an existing fixture."""
    all_case_keys = {case.case_key for case in CASES} | set(SKIPPED)
    fixture_keys = {key.split("::")[0] for key in all_case_keys}
    for key in SKIP:
        target = key.split("::")[0]
        assert target in fixture_keys, f"stale SKIP entry: {key}"
        if "::" in key:
            assert key in SKIPPED or key in all_case_keys, f"stale SKIP entry: {key}"
    for key in _value_overrides():
        assert key in fixture_keys, f"stale VALUE_OVERRIDES entry: {key}"
    for key in OMIT_OPTIONAL_COLUMNS:
        assert key in fixture_keys, f"stale OMIT_OPTIONAL_COLUMNS entry: {key}"
