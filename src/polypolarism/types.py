"""Type definitions for DataType and FrameType."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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
class UInt32(DataType):
    """32-bit unsigned integer."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UInt32)

    def __hash__(self) -> int:
        return hash("UInt32")

    def __str__(self) -> str:
        return "UInt32"


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
class Date(DataType):
    """Date type (no time component)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Date)

    def __hash__(self) -> int:
        return hash("Date")

    def __str__(self) -> str:
        return "Date"


@dataclass(frozen=True)
class Datetime(DataType):
    """Datetime type with optional timezone."""

    tz: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Datetime) and self.tz == other.tz

    def __hash__(self) -> int:
        return hash(("Datetime", self.tz))

    def __str__(self) -> str:
        if self.tz:
            return f"Datetime[{self.tz}]"
        return "Datetime"


@dataclass(frozen=True)
class Duration(DataType):
    """Duration type."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Duration)

    def __hash__(self) -> int:
        return hash("Duration")

    def __str__(self) -> str:
        return "Duration"


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
class Null(DataType):
    """Null type (for null literals)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Null)

    def __hash__(self) -> int:
        return hash("Null")

    def __str__(self) -> str:
        return "Null"


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


@dataclass(frozen=True, eq=False)
class Struct(DataType):
    """Struct type with named fields."""

    fields: dict[str, DataType] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Struct) and self.fields == other.fields

    def __hash__(self) -> int:
        return hash(("Struct", tuple(sorted(self.fields.items(), key=lambda x: x[0]))))

    def __str__(self) -> str:
        fields_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.fields.items()))
        return f"Struct{{{fields_str}}}"


@dataclass
class FrameType:
    """Type representation for a DataFrame with known columns."""

    columns: dict[str, DataType] = field(default_factory=dict)
    rest: Optional["RowVar"] = None  # For future row polymorphism extension

    def has_column(self, name: str) -> bool:
        """Check if a column exists."""
        return name in self.columns

    def get_column_type(self, name: str) -> Optional[DataType]:
        """Get the type of a column, or None if not found."""
        return self.columns.get(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameType):
            return False
        return self.columns == other.columns and self.rest == other.rest


@dataclass
class RowVar:
    """Row variable for row polymorphism (future extension)."""

    name: str
