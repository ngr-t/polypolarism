"""Tests for join operation type inference."""

import pytest

from polypolarism.types import (
    FrameType,
    Int64,
    Utf8,
    Float64,
    Nullable,
)
from polypolarism.ops.join import infer_join, JoinError


class TestInferJoinBasic:
    """Test basic join type inference."""

    def test_inner_join_with_on_key(self):
        """Inner join preserves all columns from both sides."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        assert result.columns["id"] == Int64()
        assert result.columns["name"] == Utf8()
        assert result.columns["value"] == Float64()

    def test_inner_join_columns_not_nullable(self):
        """Inner join columns remain non-nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        # All columns should be non-nullable
        assert result.columns["name"] == Utf8()
        assert result.columns["score"] == Float64()


class TestLeftJoin:
    """Test left join type inference."""

    def test_left_join_right_columns_become_nullable(self):
        """Left join makes right-side columns nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="left")

        # Left columns remain as-is
        assert result.columns["id"] == Int64()
        assert result.columns["name"] == Utf8()
        # Right columns become nullable
        assert result.columns["value"] == Nullable(Float64())

    def test_left_join_preserves_left_nullability(self):
        """Left join preserves left-side column nullability."""
        left = FrameType({"id": Int64(), "name": Nullable(Utf8())})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="left")

        assert result.columns["name"] == Nullable(Utf8())

    def test_left_join_already_nullable_right(self):
        """Left join keeps right columns nullable (no double wrapping)."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Nullable(Float64())})

        result = infer_join(left, right, on="id", how="left")

        # Should remain Nullable(Float64), not Nullable(Nullable(Float64))
        assert result.columns["value"] == Nullable(Float64())


class TestRightJoin:
    """Test right join type inference."""

    def test_right_join_left_columns_become_nullable(self):
        """Right join makes left-side columns nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="right")

        # Left columns become nullable
        assert result.columns["name"] == Nullable(Utf8())
        # Right columns remain as-is
        assert result.columns["id"] == Int64()
        assert result.columns["value"] == Float64()


class TestFullJoin:
    """Test full/outer join type inference."""

    def test_full_join_both_sides_nullable(self):
        """Full join makes both sides nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="full")

        # Key column becomes nullable in full join
        assert result.columns["id"] == Nullable(Int64())
        # Both sides become nullable
        assert result.columns["name"] == Nullable(Utf8())
        assert result.columns["value"] == Nullable(Float64())


class TestJoinColumnConflict:
    """Test column name conflict resolution."""

    def test_conflicting_column_gets_suffix(self):
        """Non-key columns with same name get _right suffix."""
        left = FrameType({"id": Int64(), "value": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        assert result.columns["value"] == Utf8()
        assert result.columns["value_right"] == Float64()

    def test_key_column_not_duplicated(self):
        """Join key column appears only once (from left)."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        # id should appear only once
        assert "id" in result.columns
        assert "id_right" not in result.columns


class TestJoinWithLeftOnRightOn:
    """Test join with separate left_on/right_on keys."""

    def test_different_key_names(self):
        """Join with different key column names."""
        left = FrameType({"user_id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, left_on="user_id", right_on="id", how="inner")

        # Both key columns preserved
        assert result.columns["user_id"] == Int64()
        assert result.columns["name"] == Utf8()
        assert result.columns["id"] == Int64()
        assert result.columns["value"] == Float64()


class TestJoinErrors:
    """Test join error conditions."""

    def test_key_column_missing_from_left(self):
        """Error when join key not found in left frame."""
        left = FrameType({"name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how="inner")

        assert "id" in str(exc_info.value)
        assert "left" in str(exc_info.value).lower()

    def test_key_column_missing_from_right(self):
        """Error when join key not found in right frame."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how="inner")

        assert "id" in str(exc_info.value)
        assert "right" in str(exc_info.value).lower()

    def test_key_dtype_mismatch(self):
        """Error when join key types don't match."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Utf8(), "value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how="inner")

        assert "dtype" in str(exc_info.value).lower() or "type" in str(exc_info.value).lower()

    def test_left_on_column_missing(self):
        """Error when left_on key not found."""
        left = FrameType({"name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, left_on="user_id", right_on="id", how="inner")

        assert "user_id" in str(exc_info.value)

    def test_right_on_column_missing(self):
        """Error when right_on key not found."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, left_on="id", right_on="other_id", how="inner")

        assert "other_id" in str(exc_info.value)

    def test_left_on_right_on_dtype_mismatch(self):
        """Error when left_on/right_on key types don't match."""
        left = FrameType({"user_id": Int64(), "name": Utf8()})
        right = FrameType({"id": Utf8(), "value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, left_on="user_id", right_on="id", how="inner")

        assert "type" in str(exc_info.value).lower() or "dtype" in str(exc_info.value).lower()


class TestJoinNullableKeyComparison:
    """Test join key dtype comparison with nullable types."""

    def test_nullable_and_non_nullable_key_match(self):
        """Nullable and non-nullable keys of same base type should match."""
        left = FrameType({"id": Nullable(Int64()), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        # Should not raise - nullable Int64 can join with Int64
        result = infer_join(left, right, on="id", how="inner")
        assert "id" in result.columns
