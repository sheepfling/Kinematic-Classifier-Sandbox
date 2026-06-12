# Archive Backend Diagnosis

The repo now has a dedicated diagnosis packet for the generic archive lane:

- study id: `archive_backend_diagnosis_v1`
- artifacts: `artifacts/archive_backend_diagnosis_v1/`

## What It Proves

This packet does not try to promote MiniRocket, DrCIF, WEASEL/TDE, or
HIVE-COTE. It exists to answer a narrower question:

Was the archive lane underperforming because the wrapper path was broken, or
because bounded panel/resampling choices and the current witness regime still do
not give those methods enough signal?

The packet sweeps:

- panel variants
- added kinematic channels
- compact resample lengths
- warning load during fit/predict

across the current shared-corpus and timing-order archive witnesses.

## Current Read

The current read is still conservative and negative:

- the archive lane now has real external execution paths
- bounded panel and resampling variants do not recover the lane enough to beat
  the current baselines
- some rows remain warning-dominated, which means external execution alone is
  not enough to justify promotion

## Claim Boundary

This is a diagnosis packet, not a promotion packet.

It is enough to support the current gate decision:

- keep `minirocket_family`, `drcif_interval_forests`,
  `dictionary_tde_family`, and `hive_cote` at `implemented`
- keep the generic time-series benchmark family gate closed

What remains open:

- broader archive-family robustness beyond the bounded diagnosis sweep
- stronger evidence about whether the failure is witness mismatch, model-form
  mismatch, or still-limited wrapper fidelity
- a witness where one of the real archive families clearly earns promotion over
  the current interpretable or physics-aware baselines
