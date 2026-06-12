# Package Utility

The package utility lane keeps the three epics reproducible and auditable. It is not a separate research story; it supplies the shared objects, commands, manifests, validators, and packet exporters.

Shared surfaces:

- `StudySpec`
- `RunManifest`
- `DecisionCard`
- `EvidenceContract`
- `PosteriorHistory`
- `HeroChartManifest`
- `LaneProofMatrix`
- packet validators

The package should stay simple: no path sniffing, no circular imports, no package-root compatibility wrappers, no dynamic `__all__`, and no broad `__init__` reexports. Internal code imports owning modules directly.

