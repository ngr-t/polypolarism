# Supported versions

`polypolarism` reasons about your code through AST analysis only — it
does not import `polars` or `pandera` at runtime. Even so, the dispatch
tables encode assumptions about the libraries' surface, so the supported
window is narrow and explicit:

- **Polars**: `1.37+` (a fixed floor, set empirically — see ADR-0004: a
  35-example corpus runs unchanged on `1.37`–`1.41`, while `1.36` and
  below hit behavior changes such as `upsample` row counts, `over`
  inside `agg`, and Bool/String schema differences that no analyzer
  guard can absorb). Older minors are best-effort. Pre-1.0 surface is
  **out of scope** — the analyzer doesn't recognize the legacy method
  spellings (`groupby`, `cumsum`, `apply`, `Utf8`, `outer`-join, …) and
  will silently misanalyse code written against those.
- **Pandera**: `0.19+`. Both `DataFrameModel` (post-0.20) and the
  legacy `SchemaModel` are accepted indefinitely; the difference is
  one entry in a name set and costs nothing.

When the version detected from your project's lockfile (`uv.lock` /
`poetry.lock`) or the installed environment falls below the supported
floor, polypolarism emits a `[pplw-unsupported-version]` warning to stderr — it doesn't
fail the run, just tells you that type-check accuracy is best-effort
below the window. Use `--polars-version <ver>` (or the
`[tool.polypolarism]` config below) to opt back in if you've audited
that your code stays within the analyzer's known surface.

Only exact sources can trigger the warning. When the version is merely
inferred from a `>=` floor in `[project.dependencies]`, no pplw-unsupported-version is
emitted — `polars>=1.0` says what the project tolerates, not what it
runs.

## Project-level configuration

Persist the assumed versions in your project's `pyproject.toml`:

```toml
[tool.polypolarism]
polars_version = "1.40"
pandera_version = "0.20"
```

Detection priority (first match wins per package):

1. `--polars-version` / `--pandera-version` CLI flag
2. `[tool.polypolarism]` section in the project's `pyproject.toml`
3. The project's `uv.lock` (exact pinned version)
4. The project's `poetry.lock` (exact pinned version)
5. The version installed in the running environment
   (`importlib.metadata`)
6. `[project.dependencies]` / `[dependency-groups.*]` floor
   (never triggers pplw-unsupported-version — a `>=` floor is not the version in use)

For more on the policy, see
[`adr/0001-polars-pandera-version-support.md`](adr/0001-polars-pandera-version-support.md).
