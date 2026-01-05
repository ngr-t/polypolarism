Run polypolarism type checker on test fixtures.

First, check valid fixtures (should all pass):
```bash
uv run polypolarism tests/fixtures/valid/
```

Then, check invalid fixtures (should all fail with appropriate errors):
```bash
uv run polypolarism tests/fixtures/invalid/
```

Report the results and verify that:
1. All valid fixtures pass
2. All invalid fixtures fail with meaningful error messages
