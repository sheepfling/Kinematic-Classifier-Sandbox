# Research Evidence

This directory contains source-research and evidence-intake artifacts that support the
repository's product and implementation work without becoming runtime package code.

Research tranches may include source cards, scorecards, acquisition inventories, adapter
mappings, claim boundaries, leakage rules, validation summaries, blockers, and proposed
registry patches. They must not contain large raw datasets, restricted source rows, generated
binary dumps, credentials, or claims that exceed the recorded evidence state.

Product 4 lane returns live under:

```text
research/product_4/<corpus_sublane>/<wave>/
```

The shared Product 4 contract and runtime interfaces remain owned by the common-front and
implementation workstreams. Lane research must reference those contracts rather than fork
them.
