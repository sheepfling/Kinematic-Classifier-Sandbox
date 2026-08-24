# SEA-SUB tranche validation

Run the repository fixture checks from the repository root:

```bash
python -m pytest -q tests/corpus/real_world/test_sea_subsurface_research_fixtures.py
```

The tests verify:

- YAML and JSON parsing for the committed tranche;
- source-card lifecycle and scorecard arithmetic;
- SHA-256 and byte-size identity for both retained source artifacts and the OOI license;
- exact anchor profile query retention;
- IOOS time monotonicity, measured-pressure versus calculated-depth semantics, and conservative position role;
- OOI DBD_ASC header/row parsing, DDMM conversion, GPS/dead-reckoning separation, and null validity;
- identity-only grouping, classifier-view blocking, registry lifecycle state, and the open G2 gate;
- `SCR-SEA-SUB-001` as a clarification that requires no competing root schema.

These checks validate the restricted research tranche. They do not validate the selected IOOS/UAF
anchor or the Sentry independent source because those numeric artifacts are not committed here.
