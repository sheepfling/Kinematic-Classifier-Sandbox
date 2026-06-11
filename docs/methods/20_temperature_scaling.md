# Temperature / Isotonic Calibration

The repo now has a dedicated calibration-shift witness for the uncertainty
wrapper lane:

- study id: `confidence_calibration_shift_v1`
- artifacts: `artifacts/confidence_calibration_shift_v1/`

## What It Proves

This witness uses a sharpened raw posterior from the existing proxy classifier
surface, fits a scalar temperature on a calibration split, and evaluates the
result on a shifted slice.

The current witness is enough to justify:

- `temperature_scaling` moving from `researched` to `witness_supported`

## Claim Boundary

This is not yet a full calibration workbench for all evidence providers or all
shift families.

What remains open:

- broader wrapper coverage beyond the current proxy posterior source
- isotonic, Dirichlet, and conformal comparisons
- stronger temporal-shift and multi-regime calibration studies
