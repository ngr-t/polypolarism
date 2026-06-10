"""Tests for join operation type inference."""

import pytest

from polypolarism.ops.join import JoinError, infer_join
from polypolarism.types import (
    Float64,
    FrameType,
    Int64,
    Nullable,
    RowVar,
    Unknown,
    Utf8,
)


class TestInferJoinBasic:
    """Test basic join type inference."""

    def test_inner_join_with_on_key(self):
        """Inner join preserves all columns from both sides."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        assert result.columns["id"].dtype == Int64()
        assert result.columns["name"].dtype == Utf8()
        assert result.columns["value"].dtype == Float64()

    def test_inner_join_columns_not_nullable(self):
        """Inner join columns remain non-nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        # All columns should be non-nullable
        assert result.columns["name"].dtype == Utf8()
        assert result.columns["score"].dtype == Float64()


class TestLeftJoin:
    """Test left join type inference."""

    def test_left_join_right_columns_become_nullable(self):
        """Left join makes right-side columns nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="left")

        # Left columns remain as-is
        assert result.columns["id"].dtype == Int64()
        assert result.columns["name"].dtype == Utf8()
        # Right columns become nullable
        assert result.columns["value"].dtype == Nullable(Float64())

    def test_left_join_preserves_left_nullability(self):
        """Left join preserves left-side column nullability."""
        left = FrameType({"id": Int64(), "name": Nullable(Utf8())})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="left")

        assert result.columns["name"].dtype == Nullable(Utf8())

    def test_left_join_already_nullable_right(self):
        """Left join keeps right columns nullable (no double wrapping)."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Nullable(Float64())})

        result = infer_join(left, right, on="id", how="left")

        # Should remain Nullable(Float64), not Nullable(Nullable(Float64))
        assert result.columns["value"].dtype == Nullable(Float64())


class TestRightJoin:
    """Test right join type inference."""

    def test_right_join_left_columns_become_nullable(self):
        """Right join makes left-side columns nullable."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="right")

        # Left columns become nullable
        assert result.columns["name"].dtype == Nullable(Utf8())
        # Right columns remain as-is
        assert result.columns["id"].dtype == Int64()
        assert result.columns["value"].dtype == Float64()


class TestFullJoin:
    """Test full/outer join type inference.

    polars does NOT coalesce full-join keys by default: both key columns
    are kept (the right one suffixed) and both are nullable.
    """

    def test_full_join_both_sides_nullable(self):
        """Full join makes both sides nullable and keeps both key columns."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="full")

        # Both key columns are kept and become nullable in a full join
        assert result.columns["id"].dtype == Nullable(Int64())
        assert result.columns["id_right"].dtype == Nullable(Int64())
        # Both sides become nullable
        assert result.columns["name"].dtype == Nullable(Utf8())
        assert result.columns["value"].dtype == Nullable(Float64())


class TestJoinColumnConflict:
    """Test column name conflict resolution."""

    def test_conflicting_column_gets_suffix(self):
        """Non-key columns with same name get _right suffix."""
        left = FrameType({"id": Int64(), "value": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        assert result.columns["value"].dtype == Utf8()
        assert result.columns["value_right"].dtype == Float64()

    def test_key_column_not_duplicated(self):
        """Join key column appears only once (from left)."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "score": Float64()})

        result = infer_join(left, right, on="id", how="inner")

        # id should appear only once
        assert "id" in result.columns
        assert "id_right" not in result.columns


class TestJoinSuffix:
    """Test custom suffix= for conflicting right-side columns."""

    def test_custom_suffix_renames_conflict(self):
        """suffix='_new' renames the overlapping right column to value_new."""
        left = FrameType({"id": Int64(), "value": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="inner", suffix="_new")

        assert result.columns["value"].dtype == Utf8()
        assert result.columns["value_new"].dtype == Float64()
        assert "value_right" not in result.columns

    def test_custom_suffix_applies_nullability(self):
        """Left join still makes the suffixed right column nullable."""
        left = FrameType({"id": Int64(), "value": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how="left", suffix="_b")

        assert result.columns["value_b"].dtype == Nullable(Float64())


class TestJoinMultiKey:
    """Test join with a list of key columns."""

    def test_on_list_of_keys(self):
        """on=['x', 'y'] skips both key columns from the right side."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="inner")

        assert result.columns["x"].dtype == Int64()
        assert result.columns["y"].dtype == Utf8()
        assert result.columns["a"].dtype == Float64()
        assert result.columns["b"].dtype == Float64()
        assert "x_right" not in result.columns
        assert "y_right" not in result.columns

    def test_on_list_left_join_keys_stay_non_nullable(self):
        """Left join with multiple keys keeps key columns at left types."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="left")

        assert result.columns["x"].dtype == Int64()
        assert result.columns["y"].dtype == Utf8()
        assert result.columns["b"].dtype == Nullable(Float64())

    def test_on_list_full_join_keys_nullable(self):
        """Full join (uncoalesced default) keeps and null-wraps every key pair."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="full")

        assert result.columns["x"].dtype == Nullable(Int64())
        assert result.columns["y"].dtype == Nullable(Utf8())
        assert result.columns["x_right"].dtype == Nullable(Int64())
        assert result.columns["y_right"].dtype == Nullable(Utf8())

    def test_on_list_missing_key_raises(self):
        """Error when one of the multi-key columns is missing."""
        left = FrameType({"x": Int64(), "z": Utf8()})
        right = FrameType({"x": Int64(), "b": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on=["x", "z"], how="inner")

        assert "z" in str(exc_info.value)

    def test_left_on_right_on_lists(self):
        """left_on/right_on lists drop the right key columns by default
        (polars coalesces inner-join keys unless coalesce=False)."""
        left = FrameType({"x1": Int64(), "y1": Utf8(), "a": Float64()})
        right = FrameType({"x2": Int64(), "y2": Utf8(), "b": Float64()})

        result = infer_join(left, right, left_on=["x1", "y1"], right_on=["x2", "y2"], how="inner")

        assert set(result.columns) == {"x1", "y1", "a", "b"}

    def test_left_on_right_on_length_mismatch_raises(self):
        """Error when left_on and right_on have different lengths."""
        left = FrameType({"x1": Int64(), "y1": Utf8()})
        right = FrameType({"x2": Int64()})

        with pytest.raises(JoinError):
            infer_join(left, right, left_on=["x1", "y1"], right_on=["x2"], how="inner")


class TestJoinWithLeftOnRightOn:
    """Test join with separate left_on/right_on keys."""

    def test_different_key_names(self):
        """Inner join coalesces by default: the right key column is dropped."""
        left = FrameType({"user_id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, left_on="user_id", right_on="id", how="inner")

        assert result.columns["user_id"].dtype == Int64()
        assert result.columns["name"].dtype == Utf8()
        assert result.columns["value"].dtype == Float64()
        assert "id" not in result.columns

    def test_different_key_names_coalesce_false_keeps_both(self):
        """coalesce=False preserves both key columns under their own names."""
        left = FrameType({"user_id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(
            left, right, left_on="user_id", right_on="id", how="inner", coalesce=False
        )

        assert result.columns["user_id"].dtype == Int64()
        assert result.columns["id"].dtype == Int64()
        assert result.columns["value"].dtype == Float64()


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


class TestSemiAntiJoin:
    """Semi/anti joins return the left frame's schema unchanged (#15)."""

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_returns_left_schema_unchanged(self, how):
        """Semi/anti joins keep exactly the left columns, no nullability changes."""
        left = FrameType({"id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how=how)

        assert set(result.columns) == {"id", "name"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["name"].dtype == Utf8()

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_right_columns_never_appear(self, how):
        """Right-side columns (even conflicting names) are not added; suffix is irrelevant."""
        left = FrameType({"id": Int64(), "value": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, on="id", how=how, suffix="_r")

        assert set(result.columns) == {"id", "value"}
        assert result.columns["value"].dtype == Utf8()
        assert "value_r" not in result.columns
        assert "value_right" not in result.columns

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_preserves_strict_and_rest(self, how):
        """Strictness and the row variable of the left frame survive."""
        rest = RowVar("r")
        left = FrameType({"id": Int64()}, strict=True, rest=rest)
        right = FrameType({"id": Int64()})

        result = infer_join(left, right, on="id", how=how)

        assert result.strict is True
        assert result.rest == rest

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_preserves_left_nullability(self, how):
        """Existing nullability on left columns is kept as-is."""
        left = FrameType({"id": Int64(), "name": Nullable(Utf8())})
        right = FrameType({"id": Int64()})

        result = infer_join(left, right, on="id", how=how)

        assert result.columns["name"].dtype == Nullable(Utf8())

    def test_multi_key_semi(self):
        """Semi join with a list of keys validates all pairs and keeps left schema."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="semi")

        assert set(result.columns) == {"x", "y", "a"}

    def test_left_on_right_on_semi(self):
        """Semi join with left_on/right_on keeps only left columns."""
        left = FrameType({"user_id": Int64(), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        result = infer_join(left, right, left_on="user_id", right_on="id", how="semi")

        assert set(result.columns) == {"user_id", "name"}

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_key_missing_from_left_raises(self, how):
        """Key validation still applies: missing left key errors."""
        left = FrameType({"name": Utf8()})
        right = FrameType({"id": Int64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how=how)

        assert "id" in str(exc_info.value)
        assert "left" in str(exc_info.value).lower()

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_key_missing_from_right_raises(self, how):
        """Key validation still applies: missing right key errors."""
        left = FrameType({"id": Int64()})
        right = FrameType({"value": Float64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how=how)

        assert "id" in str(exc_info.value)
        assert "right" in str(exc_info.value).lower()

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_key_dtype_mismatch_raises(self, how):
        """Key validation still applies: dtype mismatch errors."""
        left = FrameType({"id": Int64()})
        right = FrameType({"id": Utf8()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how=how)

        assert "dtype" in str(exc_info.value).lower()

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_missing_keys_spec_raises(self, how):
        """on / left_on+right_on rules still enforced."""
        left = FrameType({"id": Int64()})
        right = FrameType({"id": Int64()})

        with pytest.raises(JoinError):
            infer_join(left, right, how=how)


class TestJoinNullableKeyComparison:
    """Test join key dtype comparison with nullable types."""

    def test_nullable_and_non_nullable_key_match(self):
        """Nullable and non-nullable keys of same base type should match."""
        left = FrameType({"id": Nullable(Int64()), "name": Utf8()})
        right = FrameType({"id": Int64(), "value": Float64()})

        # Should not raise - nullable Int64 can join with Int64
        result = infer_join(left, right, on="id", how="inner")
        assert "id" in result.columns


class TestJoinCoalesce:
    """#24: key coalescing semantics per how x coalesce combination.

    polars coalesces equi-join keys by default for inner/left/right joins
    but NOT for full joins; an explicit ``coalesce=`` overrides the default.
    """

    def test_full_coalesce_true_key_non_nullable(self):
        """Full join with coalesce=True: the coalesced key is never null (#24)."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="full", coalesce=True)

        assert set(result.columns) == {"id", "x", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["x"].dtype == Nullable(Int64())
        assert result.columns["y"].dtype == Nullable(Int64())

    def test_full_coalesce_true_nullable_left_key_stays_nullable(self):
        """A nullable input key stays nullable after full-join coalescing."""
        left = FrameType({"id": Nullable(Int64()), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="full", coalesce=True)

        assert result.columns["id"].dtype == Nullable(Int64())

    def test_full_coalesce_true_nullable_right_key_stays_nullable(self):
        """Nullability from the right key also survives full-join coalescing."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Nullable(Int64()), "y": Int64()})

        result = infer_join(left, right, on="id", how="full", coalesce=True)

        assert result.columns["id"].dtype == Nullable(Int64())

    def test_full_coalesce_false_same_as_default(self):
        """Explicit coalesce=False matches the full-join default shape."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="full", coalesce=False)

        assert set(result.columns) == {"id", "x", "id_right", "y"}
        assert result.columns["id"].dtype == Nullable(Int64())
        assert result.columns["id_right"].dtype == Nullable(Int64())

    def test_left_coalesce_false_keeps_right_key_nullable(self):
        """Left join with coalesce=False keeps id_right (nullable)."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="left", coalesce=False)

        assert set(result.columns) == {"id", "x", "id_right", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["id_right"].dtype == Nullable(Int64())
        assert result.columns["y"].dtype == Nullable(Int64())

    def test_right_coalesce_false_keeps_left_key_nullable(self):
        """Right join with coalesce=False keeps the left key (nullable)."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="right", coalesce=False)

        assert set(result.columns) == {"id", "x", "id_right", "y"}
        assert result.columns["id"].dtype == Nullable(Int64())
        assert result.columns["x"].dtype == Nullable(Int64())
        assert result.columns["id_right"].dtype == Int64()
        assert result.columns["y"].dtype == Int64()

    def test_inner_coalesce_false_keeps_both_keys_non_nullable(self):
        """Inner join with coalesce=False keeps both keys, nothing nullable."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="inner", coalesce=False)

        assert set(result.columns) == {"id", "x", "id_right", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["id_right"].dtype == Int64()

    def test_left_coalesce_true_matches_default(self):
        """Explicit coalesce=True on a left join matches the default shape."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="left", coalesce=True)

        assert set(result.columns) == {"id", "x", "y"}
        assert result.columns["id"].dtype == Int64()

    def test_right_coalesce_true_key_uses_right_type(self):
        """Right join coalescing keeps the right side's key dtype (non-null)."""
        left = FrameType({"id": Nullable(Int64()), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how="right", coalesce=True)

        assert set(result.columns) == {"id", "x", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["x"].dtype == Nullable(Int64())

    def test_multi_key_full_coalesce_true(self):
        """Multi-key full join with coalesce=True: every key is non-null."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="full", coalesce=True)

        assert set(result.columns) == {"x", "y", "a", "b"}
        assert result.columns["x"].dtype == Int64()
        assert result.columns["y"].dtype == Utf8()
        assert result.columns["a"].dtype == Nullable(Float64())
        assert result.columns["b"].dtype == Nullable(Float64())

    def test_multi_key_left_coalesce_false(self):
        """Multi-key left join with coalesce=False keeps every right key."""
        left = FrameType({"x": Int64(), "y": Utf8(), "a": Float64()})
        right = FrameType({"x": Int64(), "y": Utf8(), "b": Float64()})

        result = infer_join(left, right, on=["x", "y"], how="left", coalesce=False)

        assert set(result.columns) == {"x", "y", "a", "x_right", "y_right", "b"}
        assert result.columns["x_right"].dtype == Nullable(Int64())
        assert result.columns["y_right"].dtype == Nullable(Utf8())

    def test_left_on_right_on_full_default_keeps_both_keys(self):
        """Full join with left_on/right_on keeps both keys (nullable) by default."""
        left = FrameType({"uid": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, left_on="uid", right_on="id", how="full")

        assert set(result.columns) == {"uid", "x", "id", "y"}
        assert result.columns["uid"].dtype == Nullable(Int64())
        assert result.columns["id"].dtype == Nullable(Int64())

    def test_left_on_right_on_full_coalesce_true_keeps_left_name(self):
        """Full join coalescing with left_on/right_on keeps the left key name."""
        left = FrameType({"uid": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, left_on="uid", right_on="id", how="full", coalesce=True)

        assert set(result.columns) == {"uid", "x", "y"}
        assert result.columns["uid"].dtype == Int64()

    def test_left_on_right_on_right_coalesce_drops_left_key(self):
        """Right join coalescing with left_on/right_on keeps the RIGHT key name."""
        left = FrameType({"uid": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, left_on="uid", right_on="id", how="right", coalesce=True)

        assert set(result.columns) == {"x", "id", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["x"].dtype == Nullable(Int64())

    def test_left_on_right_on_left_default_drops_right_key(self):
        """Left join with left_on/right_on coalesces (drops right key) by default."""
        left = FrameType({"uid": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, left_on="uid", right_on="id", how="left")

        assert set(result.columns) == {"uid", "x", "y"}
        assert result.columns["uid"].dtype == Int64()
        assert result.columns["y"].dtype == Nullable(Int64())

    @pytest.mark.parametrize("how", ["semi", "anti"])
    @pytest.mark.parametrize("coalesce", [True, False, None])
    def test_semi_anti_ignore_coalesce(self, how, coalesce):
        """Semi/anti joins return the left schema regardless of coalesce."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"id": Int64(), "y": Int64()})

        result = infer_join(left, right, on="id", how=how, coalesce=coalesce)

        assert set(result.columns) == {"id", "x"}


class TestCrossJoin:
    """#26: cross join is left columns + right columns, no nullability."""

    def test_cross_join_disjoint_columns(self):
        """Cross join concatenates both schemas with dtypes unchanged."""
        left = FrameType({"id": Int64(), "x": Int64()})
        right = FrameType({"rid": Int64(), "y": Utf8()})

        result = infer_join(left, right, how="cross")

        assert set(result.columns) == {"id", "x", "rid", "y"}
        assert result.columns["id"].dtype == Int64()
        assert result.columns["x"].dtype == Int64()
        assert result.columns["rid"].dtype == Int64()
        assert result.columns["y"].dtype == Utf8()

    def test_cross_join_collision_gets_suffix(self):
        """Colliding right column names get the suffix, like other joins."""
        left = FrameType({"id": Int64(), "v": Utf8()})
        right = FrameType({"id": Int64(), "v": Float64()})

        result = infer_join(left, right, how="cross")

        assert set(result.columns) == {"id", "v", "id_right", "v_right"}
        assert result.columns["v"].dtype == Utf8()
        assert result.columns["v_right"].dtype == Float64()

    def test_cross_join_custom_suffix(self):
        """suffix= is honored for cross-join collisions."""
        left = FrameType({"v": Utf8()})
        right = FrameType({"v": Float64()})

        result = infer_join(left, right, how="cross", suffix="_b")

        assert set(result.columns) == {"v", "v_b"}
        assert result.columns["v_b"].dtype == Float64()

    def test_cross_join_preserves_nullability(self):
        """Existing nullability passes through unchanged; none is added."""
        left = FrameType({"a": Nullable(Int64())})
        right = FrameType({"b": Float64()})

        result = infer_join(left, right, how="cross")

        assert result.columns["a"].dtype == Nullable(Int64())
        assert result.columns["b"].dtype == Float64()

    def test_cross_join_with_on_raises(self):
        """Providing 'on' with a cross join is an error (polars raises too)."""
        left = FrameType({"id": Int64()})
        right = FrameType({"id": Int64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, on="id", how="cross")

        assert "cross join takes no join keys" in str(exc_info.value)

    def test_cross_join_with_left_on_right_on_raises(self):
        """Providing left_on/right_on with a cross join is an error."""
        left = FrameType({"id": Int64()})
        right = FrameType({"rid": Int64()})

        with pytest.raises(JoinError) as exc_info:
            infer_join(left, right, left_on="id", right_on="rid", how="cross")

        assert "cross join takes no join keys" in str(exc_info.value)


class TestUnknownKeyCompatibility:
    """Issue #39b: an Unknown join key never reports a dtype mismatch
    (gradual typing — uncertainty must not error)."""

    def test_left_unknown_key_joins_against_utf8(self):
        left = FrameType({"name": Unknown(), "step": Int64()})
        right = FrameType({"name": Utf8(), "v": Int64()})

        result = infer_join(left, right, on="name", how="left")

        assert result.columns["name"].dtype == Unknown()
        assert result.columns["v"].dtype == Nullable(Int64())

    def test_right_unknown_key_joins_against_utf8(self):
        left = FrameType({"name": Utf8(), "step": Int64()})
        right = FrameType({"name": Unknown(), "v": Int64()})

        result = infer_join(left, right, on="name", how="inner")

        assert result.columns["name"].dtype == Utf8()

    def test_both_unknown_keys_join(self):
        left = FrameType({"name": Unknown()})
        right = FrameType({"name": Unknown(), "v": Int64()})

        result = infer_join(left, right, on="name", how="inner")

        assert result.columns["name"].dtype == Unknown()

    def test_nullable_unknown_key_joins_against_utf8(self):
        left = FrameType({"name": Nullable(Unknown())})
        right = FrameType({"name": Utf8(), "v": Int64()})

        result = infer_join(left, right, on="name", how="inner")

        assert result.columns["v"].dtype == Int64()

    def test_genuine_dtype_mismatch_still_raises(self):
        left = FrameType({"name": Int64()})
        right = FrameType({"name": Utf8()})

        with pytest.raises(JoinError, match="dtype mismatch"):
            infer_join(left, right, on="name", how="inner")
