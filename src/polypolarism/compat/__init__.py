"""Centralized polars / pandera surface knowledge.

The analyzer reasons about polars / pandera codebases by AST text alone —
this package holds the string tables and dispatch knowledge so future
churn lands in one place rather than in scattered dispatch sites across
``analyzer.py``, ``ops/*.py`` and ``pandera_*.py``.

See ADR-0001 (``docs/adr/0001-polars-pandera-version-support.md``) for
the migration plan.
"""
