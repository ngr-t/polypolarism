# polypolarism documentation

Reference material for the polypolarism type checker. Start with the
[project README](../README.md) for an overview, installation, and a
quick-start example.

## Reference

- [Schema declaration](schemas.md) — every Pandera field-annotation form,
  `Optional[T]` vs `pa.Field(nullable=True)`, and validation-as-narrowing.
- [Supported operations](operations.md) — join, group-by/agg, select,
  reshape, sub-namespaces (`.str`/`.dt`/`.list`/…), restructuring,
  window/time-series, selectors, and `pl.*` constructors.
- [Diagnostics](diagnostics.md) — the `PLY###` / `PLW###` code tables,
  apply-style helper warnings, JSON output, and the schema diff block.
- [Leniency](leniency.md) — when a clean run is not a proof: the gradual
  rules that let a column pass without a precise check.
- [Supported versions](versions.md) — the Polars / Pandera support window
  and `[tool.polypolarism]` project configuration.

## Design records

- [Architecture Decision Records](adr/) — non-trivial design decisions.
- [Backlog](backlog.md) — planned and in-progress work.
