# SEA-SUB tranche validation

Run the focused repository checks from the repository root:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/corpus/real_world/test_sea_subsurface_research_fixtures.py \
  tests/corpus/real_world/test_sea_subsurface_selected_anchor.py
python -m ruff check \
  tests/corpus/real_world/test_sea_subsurface_research_fixtures.py \
  tests/corpus/real_world/test_sea_subsurface_selected_anchor.py
python -m ruff format --check \
  tests/corpus/real_world/test_sea_subsurface_research_fixtures.py \
  tests/corpus/real_world/test_sea_subsurface_selected_anchor.py
pyright \
  tests/corpus/real_world/test_sea_subsurface_research_fixtures.py \
  tests/corpus/real_world/test_sea_subsurface_selected_anchor.py
```

The checks verify:

- YAML and JSON parsing for the committed tranche;
- source-card lifecycle and scorecard arithmetic;
- SHA-256 and byte-size identity for retained source artifacts and the OOI license;
- exact anchor profile query retention;
- selected-anchor identity, 99-row shape, mixed channel presence, and artifact inspection;
- same-time rows as distinct asynchronous channel events rather than removable duplicates;
- the selected artifact's GPS units/value mismatch and prohibition on unsafe DDMM conversion;
- measured-pressure versus calculated-depth semantics and conservative position roles;
- OOI DBD_ASC parsing, source-specific DDMM conversion, GPS/dead-reckoning separation, and null validity;
- identity-only grouping, classifier-view blocking, registry lifecycle state, and the open G2 gate;
- `SCR-SEA-SUB-001` as a clarification that requires no competing root schema.

These checks validate the research tranche and selected-anchor artifact mapping. They do not validate
a canonical COMMON-FRONT fixture, a production adapter, or the Sentry independent source.
