# Fixture corpus conventions

Fixtures under `valid/`, `invalid/`, and `warning/` are auto-discovered by
the golden harness (`tests/test_fixtures.py`, see ADR-0002): adding a `.py`
file IS adding a test, and its full diagnostic output is pinned in the
`<name>.expected` golden next to it.

## The pairing convention (ADR-0003)

**Every inference rule ships a PAIR of fixtures:**

1. a **valid** fixture proving the correct declaration passes
   (guards against *false positives* — the checker must not reject
   correct code), and
2. an **invalid** fixture proving a *wrong declaration of the SAME
   operation* fails
   (guards against *false negatives* — the checker must actually check).

### Why the invalid twin is not optional

A valid fixture alone proves nothing about the rule it appears to cover,
because polypolarism is deliberately lenient under uncertainty: an `Unknown`
dtype satisfies any declaration, and an open frame excuses missing columns.
A bug that degrades inference to `Unknown` therefore makes the valid fixture
**keep passing — via leniency instead of via the rule**. That is exactly how
regression #47 stayed hidden: `pl.struct(x=...)` silently produced
`Struct{}` for months, but the surrounding when/then degraded to Unknown and
the open-frame rule let the valid fixture through. The same mechanism caused
#55 (`list.sum()` on `List(String)` regressed to accept-anything when a
probed table left invalid cells silently Unknown).

Only the invalid twin exposes this failure mode: if inference degrades, the
wrong declaration suddenly *passes*, and the `invalid/` category invariant
("at least one function fails") turns that into a test failure immediately.

Two complementary mechanisms back the convention up:

- **Leniency notes in goldens**: a pass that relied on a leniency rule
  renders as an indented `via:` line in the `.expected` file
  (e.g. `via: column 'x': passed via Unknown`). A valid fixture that starts
  passing via leniency instead of precise inference shows up as a golden
  diff in review. If a `via:` line appears in a golden where you did not
  expect leniency, investigate before regenerating.
- **`@pytest.mark.imprecision`**: unit tests that deliberately pin leniency
  (Unknown fallback, open-frame skips, silent matrix cells) carry this
  marker plus an upgrade-trigger comment. When the construct gains precise
  inference, upgrade the assertion — never weaken the feature to keep the
  old test green.

### Writing the invalid twin

- Use the **same operation** as the valid fixture; change only the declared
  schema (wrong dtype, wrong nullability, missing/extra column).
- Mark the wrong line with a `# WRONG: ...` comment naming the rule.
- Reference the valid twin in the module docstring
  (`False-negative twin of ``valid/...``).
- Generate the golden with
  `POLYPOLARISM_UPDATE_EXPECTED=1 uv run pytest tests/test_fixtures.py`
  and check it fails for the *intended* reason (the specific column/dtype),
  not via an unrelated error.

## Pair audit (2026-06)

Rules with both sides present (valid twin -> invalid twin):

| Rule | Valid | Invalid |
| --- | --- | --- |
| basic schema check | `pandera_basic` | `pandera_type_mismatch` |
| strict schemas | `pandera_strict_config` | `pandera_strict_extra` |
| optional columns | (in `pandera_basic` et al.) | `pandera_optional_required_mismatch` |
| arithmetic dtypes | `arithmetic_dtypes`, `decimal_arith` | `arith_incompatible`, `truediv_declared_int`, `decimal_arith_stale_precision` |
| comparisons | `compare_cast_ok` | `compare_incompatible` |
| when/then supertype | `when_supertype_shift` | `when_mixed_branches_declared_int`, `when_nonbool_condition` |
| joins (keys, dtypes) | `basic_join` | `join_missing_column`, `join_type_mismatch` |
| left-join nullability | `left_join_nullable` | `left_join_nonnullable_declared` |
| join suffix | `constants_and_join_suffix` | `join_suffix_wrong_dtype` |
| join coalesce / cross | `join_coalesce_cross` | `join_coalesce_cross_wrong` |
| semi/anti joins | `semi_anti_gather` | `semi_anti_schema_change` |
| group_by/agg | `groupby_agg_basic`, `m2_new_aggs` | `agg_type_error`, `groupby_nonexistent_col` |
| str/dt/list namespaces | `m3_*_namespace` | `namespace_wrong_dtype`, `m3_str_method_missing_column`, `str_to_decimal_wrong_scale` |
| bin / cat namespaces | `bin_namespace`, `cat_namespace` | `bin_on_int`, `cat_on_int` |
| Array vs List | `array_dtype` | `arr_list_mismatch` |
| Array width (C-7) | `array_dtype` (widths flow through) | `array_width_mismatch` |
| arr.eval as_list container | `array_dtype` (`eval_array`) | `arr_eval_as_list_declared_array` |
| explode / concat / unpivot | `m4_*` | `m4_explode_non_list`, `m4_concat_mismatch`, `unpivot_incompatible_values` |
| struct field building | `struct_kwarg_fields`, `m9_struct_and_unnest` | `struct_field_wrong_dtype`, `m9_unnest_non_struct` |
| selectors | `m6_selectors`, `m10_selector_algebra` | `selector_wrong_dtype` |
| map_elements return_dtype | `m7_map_elements_dtype` | `map_elements_dtype_mismatch` |
| validate-narrowing | `pandera_validate_*` | `validate_narrow_wrong_downstream` |
| lazy/eager discipline | `m13_lazy_pipeline` | `m15_lazy_misuse`, `m15_eager_lazy_arg`, `lazy_method_on_dataframe` |
| filter predicates | `m1_filter_chain`, `m2_filter_predicates` | `filter_nonbool_predicate`, `m2_filter_unknown_column` |
| drop/rename/cast | `m1_drop_rename_cast` | `m1_drop_missing`, `m1_rename_unknown`, `cast_impossible`, `cast_column_not_found` |
| nullability propagation | `m2_fill_null_narrowing` | `null_propagation_nonnull` |
| coerce limits | `coerce_len_agg`, `coerce_to_string` | `coerce_limits` |
| multi-expr list args | `expr_list_args` | `expr_list_args_mixed` |
| tz handling | `tz_same_ops` | `tz_mixing` |
| function-call checking | `function_call_*` | `function_call_{missing_column,type_mismatch,nullable_mismatch,untyped_inference_fail}` |
| sort/unique/over/drop_nulls keys | `m5_window_and_rolling`, `m1_drop_nulls_and_row_index` | `sort_missing_column`, `unique_missing_subset`, `over_missing_column`, `drop_nulls_subset_not_found`, `with_row_index_collision` |
| window/rolling/over dtypes | `m5_window_and_rolling` | `window_rolling_wrong_dtype` |
| rolling nullability | `m5_window_and_rolling` | `rolling_nullability_nonnull` |
| numeric-only elementwise (cum/rolling/round families) | `numeric_elementwise` | `cum_sum_on_string`, `round_on_string` |
| group_by_dynamic / join_asof | `m5_time_groupby_and_asof` | `time_groupby_asof_wrong` |
| pivot (annotation flows downstream) | `m12_pivot_annotated` (+ `warning/m12_pivot_unannotated`) | `m12_pivot_wrong_declared` |
| partition_by elements | `m14_partition_by` | `m14_partition_by_wrong_element` |
| landmark dtypes | `dtype_enum`, `dtype_float16`, `dtype_int128`, `dtype_uint128` | `dtype_landmarks_wrong_declared` |
| frame literals | `frame_literal` | `frame_literal_wrong_declared` |
| pl expression constructors | `m6_pl_constructors` | `m6_pl_constructors_wrong_declared` |
| variable annotations | `variable_annotation_basic`, `variable_annotation_chain` | `variable_annotation_wrong_downstream` |
| annotation vs inferred RHS (ADR-0005, PLY033/PLW008) | `warning/annotation_narrowing` (narrowing assertion warns) | `variable_annotation_contradiction` |
| plural col | `m9_plural_col`, `plural_col_exprs` | `plural_col_wrong_dtype` |
| struct rename_fields | `struct_rename_fields` | `struct_rename_wrong_dtype` |
| hstack | `m4_unpivot_and_hstack` | `m4_hstack_wrong_dtype` |
| unmodeled-method degradation (PLW007) | `unmodeled_method_pinned` (cast retracts the warning) | `warning/unmodeled_method` (a warning, not an error — Unknown passes by design) |

Intentionally unpaired:

- `valid/unknown_dtype_tracking` — pins the leniency design itself (Unknown
  columns stay registered). Its golden carries the `via:` notes that make
  the leniency visible; an invalid twin is impossible by construction.

### Known gaps (backlog — add the invalid twin when touching the rule)

(none at the moment — the variable-annotation contradiction gap was
closed by ADR-0005: `invalid/variable_annotation_contradiction` pins the
PLY033 error, `warning/annotation_narrowing` pins the PLW008 narrowing
assertion.)

## Runtime differential harness (ADR-0003, separate module)

A `runtime`-marked test module executes fixtures against real
polars + pandera with synthesized inputs: valid fixtures must succeed at
runtime, invalid ones must fail. It lives behind a dedicated dependency
group and an explicit skip-list for fixtures whose inputs cannot be
synthesized. See `docs/adr/0003-dual-direction-fixture-testing.md`.
