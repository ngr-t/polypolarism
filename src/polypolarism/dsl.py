"""Schema DSL parser for DF["{col:Type, ...}"] annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from polypolarism.types import (
    DataType,
    Int64,
    Int32,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Utf8,
    Boolean,
    Date,
    Datetime,
    Duration,
    Decimal,
    Categorical,
    List,
    Struct,
    Nullable,
    FrameType,
)


class ParseError(Exception):
    """Error raised when parsing fails."""

    pass


# Mapping of type names to their constructors
PRIMITIVE_TYPES: dict[str, type[DataType]] = {
    "Int64": Int64,
    "Int32": Int32,
    "UInt32": UInt32,
    "UInt64": UInt64,
    "Float32": Float32,
    "Float64": Float64,
    "Utf8": Utf8,
    "String": Utf8,  # Alias for Utf8 (Polars 1.x uses String)
    "Boolean": Boolean,
    "Date": Date,
    "Datetime": Datetime,
    "Duration": Duration,
    "Categorical": Categorical,
}


@dataclass
class Parser:
    """Recursive descent parser for type expressions."""

    text: str
    pos: int = 0

    def peek(self) -> Optional[str]:
        """Look at the current character without consuming it."""
        self.skip_whitespace()
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def consume(self, expected: Optional[str] = None) -> str:
        """Consume and return the current character."""
        self.skip_whitespace()
        if self.pos >= len(self.text):
            raise ParseError(f"Unexpected end of input, expected {expected}")
        char = self.text[self.pos]
        if expected and char != expected:
            raise ParseError(f"Expected '{expected}', got '{char}' at position {self.pos}")
        self.pos += 1
        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def parse_identifier(self) -> str:
        """Parse an identifier (type name or field name)."""
        self.skip_whitespace()
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        if start == self.pos:
            raise ParseError(f"Expected identifier at position {self.pos}")
        return self.text[start : self.pos]

    def parse_type(self) -> DataType:
        """Parse a type expression."""
        self.skip_whitespace()

        # Check for List[T]
        if self.text[self.pos :].startswith("List["):
            self.pos += 5  # Skip "List["
            inner = self.parse_type()
            self.consume("]")
            result: DataType = List(inner)
        # Check for Struct{...}
        elif self.text[self.pos :].startswith("Struct{"):
            self.pos += 7  # Skip "Struct{"
            fields = self.parse_struct_fields()
            self.consume("}")
            result = Struct(fields)
        else:
            # Parse primitive type
            type_name = self.parse_identifier()
            if type_name not in PRIMITIVE_TYPES:
                raise ParseError(f"Unknown type: {type_name}")
            result = PRIMITIVE_TYPES[type_name]()

        # Check for nullable suffix
        self.skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == "?":
            self.pos += 1
            result = Nullable(result)

        return result

    def parse_struct_fields(self) -> dict[str, DataType]:
        """Parse struct fields: name: Type, name: Type, ..."""
        fields: dict[str, DataType] = {}

        self.skip_whitespace()
        if self.peek() == "}":
            return fields

        while True:
            # Parse field name
            field_name = self.parse_identifier()
            self.consume(":")
            # Parse field type
            field_type = self.parse_type()
            fields[field_name] = field_type

            self.skip_whitespace()
            if self.peek() == "}":
                break
            if self.peek() == ",":
                self.consume(",")
            else:
                break

        return fields

    def parse_schema(self) -> FrameType:
        """Parse a schema: {col: Type, col: Type, ...}"""
        self.consume("{")

        self.skip_whitespace()
        if self.peek() == "}":
            self.consume("}")
            return FrameType({})

        columns: dict[str, DataType] = {}

        while True:
            # Parse column name
            col_name = self.parse_identifier()
            self.consume(":")
            # Parse column type
            col_type = self.parse_type()
            columns[col_name] = col_type

            self.skip_whitespace()
            if self.peek() == "}":
                break
            if self.peek() == ",":
                self.consume(",")
            else:
                break

        self.consume("}")
        return FrameType(columns)


def parse_type(type_str: str) -> DataType:
    """Parse a type expression string into a DataType."""
    parser = Parser(type_str.strip())
    result = parser.parse_type()
    parser.skip_whitespace()
    if parser.pos < len(parser.text):
        raise ParseError(f"Unexpected characters after type: {parser.text[parser.pos:]}")
    return result


def parse_schema(schema_str: str) -> FrameType:
    """Parse a schema string into a FrameType."""
    parser = Parser(schema_str.strip())
    try:
        result = parser.parse_schema()
        parser.skip_whitespace()
        if parser.pos < len(parser.text):
            raise ParseError(f"Unexpected characters after schema: {parser.text[parser.pos:]}")
        return result
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Failed to parse schema: {e}") from e
