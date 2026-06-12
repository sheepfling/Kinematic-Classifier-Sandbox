# Archive Feature Headroom Witness

The repo now has a second named archive-family witness on a timing-order task:

- study id: `archive_feature_headroom_witness_v1`
- artifacts: `artifacts/archive_feature_headroom_witness_v1/`

## What It Proves

This packet compares:

- `minirocket_family`
- `drcif_interval_forests`
- `dictionary_tde_family`
- `hive_cote`

against:

- `windowed_feature_summary`
- `gradient_boosted_features`

on the feature-headroom dataset where a simple global window misses the timing
order and the engineered boosted baseline recovers it.

## Claim Boundary

This is a generic-TSC timing-order witness, not an archive-family promotion
packet.

The important value is no longer only negative evidence:

- the real external archive backends execute on this task too
- execution alone still is not enough
- the engineered timing-order baseline remains a serious bar
- but the current bounded external archive rows now match that bar instead of
  failing far below it

The current witness therefore upgrades the generic-TSC read from:

`archive methods only execute externally`

to:

`archive methods can now meet a bounded timing-order witness, but the family
still needs a broader closure decision`
