"""Type definitions for DataType and FrameType."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field


class DataType(ABC):
    """Base class for all data types."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class Int64(DataType):
    """64-bit signed integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Int64)

    def __hash__(self) -> int:
        return hash("Int64")

    def __str__(self) -> str:
        return "Int64"


@dataclass(frozen=True)
class Int32(DataType):
    """32-bit signed integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Int32)

    def __hash__(self) -> int:
        return hash("Int32")

    def __str__(self) -> str:
        return "Int32"


@dataclass(frozen=True)
class Int16(DataType):
    """16-bit signed integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Int16)

    def __hash__(self) -> int:
        return hash("Int16")

    def __str__(self) -> str:
        return "Int16"


@dataclass(frozen=True)
class Int8(DataType):
    """8-bit signed integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Int8)

    def __hash__(self) -> int:
        return hash("Int8")

    def __str__(self) -> str:
        return "Int8"


@dataclass(frozen=True)
class Int128(DataType):
    """128-bit signed integer (polars 1.18+)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Int128)

    def __hash__(self) -> int:
        return hash("Int128")

    def __str__(self) -> str:
        return "Int128"


@dataclass(frozen=True)
class UInt32(DataType):
    """32-bit unsigned integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt32)

    def __hash__(self) -> int:
        return hash("UInt32")

    def __str__(self) -> str:
        return "UInt32"


@dataclass(frozen=True)
class UInt16(DataType):
    """16-bit unsigned integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt16)

    def __hash__(self) -> int:
        return hash("UInt16")

    def __str__(self) -> str:
        return "UInt16"


@dataclass(frozen=True)
class UInt8(DataType):
    """8-bit unsigned integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt8)

    def __hash__(self) -> int:
        return hash("UInt8")

    def __str__(self) -> str:
        return "UInt8"


@dataclass(frozen=True)
class UInt64(DataType):
    """64-bit unsigned integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt64)

    def __hash__(self) -> int:
        return hash("UInt64")

    def __str__(self) -> str:
        return "UInt64"


@dataclass(frozen=True)
class UInt128(DataType):
    """128-bit unsigned integer (polars 1.34+)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt128)

    def __hash__(self) -> int:
        return hash("UInt128")

    def __str__(self) -> str:
        return "UInt128"


@dataclass(frozen=True)
class Float16(DataType):
    """16-bit floating point (polars 1.36+)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Float16)

    def __hash__(self) -> int:
        return hash("Float16")

    def __str__(self) -> str:
        return "Float16"


@dataclass(frozen=True)
class Float32(DataType):
    """32-bit floating point."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Float32)

    def __hash__(self) -> int:
        return hash("Float32")

    def __str__(self) -> str:
        return "Float32"


@dataclass(frozen=True)
class Float64(DataType):
    """64-bit floating point."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Float64)

    def __hash__(self) -> int:
        return hash("Float64")

    def __str__(self) -> str:
        return "Float64"


# Complete set of numeric dtype classes: every signed/unsigned integer
# width plus the float widths. Pandera ``coerce=True`` can cast any of
# these into any other at validation time, so the coercion-compatibility
# check (``checker._is_coercible_difference``) treats differences within
# this set as non-errors. Membership is checked via ``type(dtype)`` —
# these classes are leaf dataclasses with no subclasses.
NUMERIC_DTYPES: frozenset[type[DataType]] = frozenset(
    {
        Int8,
        Int16,
        Int32,
        Int64,
        Int128,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        UInt128,
        Float16,
        Float32,
        Float64,
    }
)


@dataclass(frozen=True)
class Utf8(DataType):
    """UTF-8 string."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Utf8)

    def __hash__(self) -> int:
        return hash("Utf8")

    def __str__(self) -> str:
        return "Utf8"


@dataclass(frozen=True)
class Boolean(DataType):
    """Boolean type."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Boolean)

    def __hash__(self) -> int:
        return hash("Boolean")

    def __str__(self) -> str:
        return "Boolean"


@dataclass(frozen=True)
class Binary(DataType):
    """Binary (bytes) type."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Binary)

    def __hash__(self) -> int:
        return hash("Binary")

    def __str__(self) -> str:
        return "Binary"


@dataclass(frozen=True)
class Date(DataType):
    """Date type (no time component)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Date)

    def __hash__(self) -> int:
        return hash("Date")

    def __str__(self) -> str:
        return "Date"


@dataclass(frozen=True)
class Time(DataType):
    """Time of day (no date component)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Time)

    def __hash__(self) -> int:
        return hash("Time")

    def __str__(self) -> str:
        return "Time"


@dataclass(frozen=True)
class Datetime(DataType):
    """Datetime type with optional timezone and a time unit (issue #66).

    ``unit`` is one of ``"ms"`` / ``"us"`` / ``"ns"``; polars' own default
    is ``"us"`` (bare ``pl.Datetime``, casts, ``str.to_datetime``, literals
    — all probed). The unit participates in equality exactly like ``tz``:
    pandera validation rejects a unit mismatch at runtime. There is no
    "unknown unit" wildcard — statically unreadable units degrade to
    ``Unknown`` at the parse sites instead.
    """

    tz: str | None = None
    unit: str = "us"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Datetime) and self.tz == other.tz and self.unit == other.unit

    def __hash__(self) -> int:
        return hash(("Datetime", self.tz, self.unit))

    def __str__(self) -> str:
        if self.tz:
            return f"Datetime[{self.unit}, {self.tz}]"
        return f"Datetime[{self.unit}]"


@dataclass(frozen=True)
class Duration(DataType):
    """Duration type with a time unit (issue #66; same model as Datetime)."""

    unit: str = "us"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Duration) and self.unit == other.unit

    def __hash__(self) -> int:
        return hash(("Duration", self.unit))

    def __str__(self) -> str:
        return f"Duration[{self.unit}]"


@dataclass(frozen=True)
class Decimal(DataType):
    """Decimal type with precision and scale."""

    precision: int
    scale: int

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Decimal)
            and self.precision == other.precision
            and self.scale == other.scale
        )

    def __hash__(self) -> int:
        return hash(("Decimal", self.precision, self.scale))

    def __str__(self) -> str:
        return f"Decimal({self.precision}, {self.scale})"


@dataclass(frozen=True)
class Categorical(DataType):
    """Categorical type."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Categorical)

    def __hash__(self) -> int:
        return hash("Categorical")

    def __str__(self) -> str:
        return "Categorical"


@dataclass(frozen=True)
class Enum(DataType):
    """Enum type (polars 1.25+ stabilized).

    Distinct from ``Categorical`` because the value space is fixed at
    schema-declaration time. ``categories`` is the ordered variant tuple
    (issue #67) — polars treats Enums with different category *sequences*
    as distinct dtypes (probed: ``pl.Enum(["a","b"]) != pl.Enum(["b","a"])``
    and pandera validation rejects the order swap), so equality is exact
    and order-sensitive. ``categories=None`` means "statically unknown"
    (a bare ``pl.Enum`` reference or a non-literal category list);
    structural equality is exact (``None != ("a",)``) and the wildcard
    treatment of unknown categories lives in the checker's subtype
    verdict, mirroring ``Array.width``.
    """

    categories: tuple[str, ...] | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Enum) and self.categories == other.categories

    def __hash__(self) -> int:
        return hash(("Enum", self.categories))

    def __str__(self) -> str:
        if self.categories is None:
            return "Enum"
        return f"Enum[{', '.join(repr(c) for c in self.categories)}]"


@dataclass(frozen=True)
class Null(DataType):
    """Null type (for null literals)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Null)

    def __hash__(self) -> int:
        return hash("Null")

    def __str__(self) -> str:
        return "Null"


@dataclass(frozen=True)
class Unknown(DataType):
    """Gradual-typing escape hatch for dtypes polypolarism cannot infer.

    Compatible with every dtype in both directions so un-inferable code
    never produces false positives.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unknown)

    def __hash__(self) -> int:
        return hash("Unknown")

    def __str__(self) -> str:
        return "Unknown"


@dataclass(frozen=True)
class Nullable(DataType):
    """Nullable wrapper for any type."""

    inner: DataType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Nullable) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("Nullable", self.inner))

    def __str__(self) -> str:
        return f"{self.inner}?"


def unwrap_nullable(dtype: DataType) -> tuple[DataType, bool]:
    """Unwrap one Nullable layer and return ``(inner_type, was_nullable)``."""
    if isinstance(dtype, Nullable):
        return dtype.inner, True
    return dtype, False


def wrap_nullable(dtype: DataType, is_nullable: bool) -> DataType:
    """Wrap ``dtype`` in Nullable when ``is_nullable`` is true."""
    if is_nullable:
        return Nullable(dtype)
    return dtype


@dataclass(frozen=True)
class List(DataType):
    """List type containing elements of a single type."""

    inner: DataType

    def __eq__(self, other: object) -> bool:
        return isinstance(other, List) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("List", self.inner))

    def __str__(self) -> str:
        return f"List[{self.inner}]"


@dataclass(frozen=True)
class Array(DataType):
    """Fixed-size array type (polars ``pl.Array``) with elements of a single type.

    Distinct from :class:`List` (issue #53): polars is strict about the
    container kind — ``.arr.*`` requires an Array receiver and ``.list.*``
    requires a List receiver, casts behave differently, and pandera
    validation rejects one where the other is declared.

    The fixed size is tracked in ``width`` (backlog C-7). Probed (polars
    1.41.2): widths are part of dtype identity — pandera validation
    rejects a width mismatch (coerce cannot repair it: the underlying cast
    raises "cannot cast Array to a different width"), and ``concat``
    raises on mixed widths. ``width=None`` means "statically unknown";
    structural equality is exact (``None != 3``) and the wildcard
    treatment of unknown widths lives in the checker's subtype verdict,
    mirroring how ``Unknown`` is handled.
    """

    inner: DataType
    width: int | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Array) and self.inner == other.inner and self.width == other.width

    def __hash__(self) -> int:
        return hash(("Array", self.inner, self.width))

    def __str__(self) -> str:
        if self.width is None:
            return f"Array[{self.inner}]"
        return f"Array[{self.inner}, {self.width}]"


@dataclass(frozen=True, eq=False)
class Struct(DataType):
    """Struct type with named fields.

    ``open=True`` is the row-polymorphic "some struct, fields beyond the
    pinned ones unknown" (backlog C-9) — produced by a bare ``pl.Struct``
    annotation or an unreadable ``pl.Struct(...)`` call. Probed (pandera
    0.31.1): a bare ``pl.Struct`` declaration validates ANY struct and
    rejects non-structs, so the struct-ness itself is provable (receiver
    checks like ``.str``-on-struct fire) while field lookups get
    ADR-0006 assumption semantics. Structural equality is exact
    (``open`` participates); the wildcard treatment lives in the
    checker's subtype verdict, mirroring ``Array.width`` /
    ``Enum.categories``. NOT to be confused with the cast target
    ``cast(pl.Struct)``, which polars materializes as the CLOSED empty
    ``struct[0]`` (probed).
    """

    fields: dict[str, DataType] = field(default_factory=dict)
    open: bool = False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Struct) and self.fields == other.fields and self.open == other.open

    def __hash__(self) -> int:
        return hash(("Struct", tuple(sorted(self.fields.items(), key=lambda x: x[0])), self.open))

    def __str__(self) -> str:
        fields_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.fields.items()))
        if self.open:
            return f"Struct{{{fields_str}, ...}}" if fields_str else "Struct{...}"
        return f"Struct{{{fields_str}}}"


@dataclass(frozen=True)
class ColumnSpec:
    """Per-column metadata: dtype + presence semantics.

    `required=True` means the column must exist in the frame.
    `required=False` means the column may be absent (Pandera `Optional[T]`).
    Value-level nullability is encoded by wrapping `dtype` in `Nullable(...)`.
    """

    dtype: DataType
    required: bool = True

    def __str__(self) -> str:
        marker = "" if self.required else "?"
        return f"{self.dtype}{marker}"


@dataclass(init=False)
class FrameType:
    """Type representation for a DataFrame with known columns.

    The constructor accepts ``dict[str, ColumnSpec | DataType]`` for ergonomics
    — bare DataType values are wrapped in ``ColumnSpec(dtype=val)`` — but the
    stored ``columns`` field is always ``dict[str, ColumnSpec]``.

    ``is_lazy`` distinguishes ``LazyFrame[S]`` (``True``) from
    ``DataFrame[S]`` (``False``). It does *not* participate in ``__eq__`` so
    existing schema-shape assertions (``assert ft == FrameType({"x": Int64()})``)
    keep working — laziness is enforced explicitly in the function-call
    argument check, the declared-vs-inferred return-type check, and the
    eager-only / lazy-only method dispatch.

    ``coerce`` mirrors Pandera's ``class Config: coerce = True``: at
    validation time pandera casts coercible dtypes instead of rejecting
    them. Like ``is_lazy`` it is excluded from ``__eq__`` for the same
    reason — coercion is consulted explicitly in the declared-vs-inferred
    dtype check (``checker.py``) and the function-argument check
    (``analyzer._is_frame_subtype``).

    ``absent`` is the open frame's NEGATIVE knowledge (issue #78,
    ADR-0006): column names provably not present — removed by ``drop`` /
    renamed away by ``rename`` — even though ``rest`` says unknown extras
    may exist. A later reference to an absent name is a guaranteed
    runtime ColumnNotFoundError (conditional on reaching the line, the
    ADR's standard), so it is a provable error. Reintroducing the name
    (``with_columns``, a rename target) clears the mark. Meaningless on
    closed frames (``rest is None`` — the pinned set is already exact)
    and always empty there. Excluded from ``__eq__`` like ``is_lazy`` —
    it is consulted explicitly at the column-lookup sites.

    ``nonstrict_schema`` is diagnostic provenance (issue #83): the name
    of the ``strict=False`` pandera schema this CLOSED frame was bound
    from (parameter binding, ``Schema.validate`` narrowing). Such a
    schema admits extra columns at runtime, so a missing-column lookup
    is an interface violation against the declaration ("checked island"
    — flagged PLY042 with honest wording) rather than a provable runtime
    failure (PLY001). Cleared by shape-determining calls (``select``,
    aggregations — their outputs are exact). Excluded from ``__eq__``.
    """

    columns: dict[str, ColumnSpec]
    strict: bool
    rest: RowVar | None  # For future row polymorphism extension
    is_lazy: bool
    coerce: bool
    absent: frozenset[str]
    nonstrict_schema: str | None
    # Diagnostic provenance only (no semantic effect, excluded from
    # ``__eq__``): the pandera schema this frame was bound from, kept
    # through the same column-ops that keep ``nonstrict_schema`` so
    # column-not-found messages can name the violated contract
    # ("... in frame from schema 'Sales'"). Unlike ``nonstrict_schema``
    # it is set for strict bindings too.
    schema_name: str | None

    def __init__(
        self,
        columns: Mapping[str, ColumnSpec | DataType] | None = None,
        strict: bool = False,
        rest: RowVar | None = None,
        is_lazy: bool = False,
        coerce: bool = False,
        absent: frozenset[str] | set[str] | None = None,
        nonstrict_schema: str | None = None,
        schema_name: str | None = None,
    ) -> None:
        normalized: dict[str, ColumnSpec] = {}
        if columns:
            for name, val in columns.items():
                if isinstance(val, ColumnSpec):
                    normalized[name] = val
                elif isinstance(val, DataType):
                    normalized[name] = ColumnSpec(dtype=val)
                else:
                    raise TypeError(
                        f"FrameType column {name!r} must be ColumnSpec or DataType, "
                        f"got {type(val).__name__}"
                    )
        self.columns = normalized
        self.strict = strict
        self.rest = rest
        self.is_lazy = is_lazy
        self.coerce = coerce
        self.nonstrict_schema = nonstrict_schema
        self.schema_name = schema_name
        # A pinned column trumps a stale absence mark; absence is only
        # meaningful while the frame is open.
        if absent and rest is not None:
            self.absent = frozenset(absent) - normalized.keys()
        else:
            self.absent = frozenset()

    def origin_note(self) -> str:
        """Diagnostic suffix naming this frame's schema provenance —
        ``" in frame from schema 'Sales'"`` — or ``""`` when unknown."""
        if self.schema_name is None:
            return ""
        return f" in frame from schema '{self.schema_name}'"

    def lacks(self, name: str) -> bool:
        """True when ``name`` is PROVABLY not a column of this frame:
        unpinned on a closed frame, or marked absent on an open one."""
        if name in self.columns:
            return False
        return self.rest is None or name in self.absent

    def has_column(self, name: str) -> bool:
        """Check if a column exists."""
        return name in self.columns

    def get_column_type(self, name: str) -> DataType | None:
        """Get the dtype of a column, or None if not found."""
        spec = self.columns.get(name)
        return spec.dtype if spec is not None else None

    def get_column_spec(self, name: str) -> ColumnSpec | None:
        """Get the full ColumnSpec for a column, or None if not found."""
        return self.columns.get(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameType):
            return False
        return (
            self.columns == other.columns
            and self.strict == other.strict
            and self.rest == other.rest
        )


@dataclass
class RowVar:
    """Row variable for row polymorphism (future extension)."""

    name: str


@dataclass
class FrameList:
    """A list of frames sharing the same ``element`` FrameType.

    Used to model ``df.partition_by("k")`` and any future operation that
    yields multiple frames at once. Subscript indexing (``parts[0]``) and
    for-loop iteration (``for p in parts:``) bind the element type to the
    target name.
    """

    element: FrameType

    def __str__(self) -> str:
        return f"list[FrameType({len(self.element.columns)} cols)]"
