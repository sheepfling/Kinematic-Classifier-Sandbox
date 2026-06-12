# HIVE-COTE 2.0

The repo now tracks the HIVE-COTE lane through the shared modern-TSC execution
packet:

- study id: `tsc_archive_baseline_frontier_v1`
- artifacts: `artifacts/tsc_archive_baseline_frontier_v1/`
- family method id in that packet: `hive_cote`

## What It Proves

This lane now has:

- real bounded external `HIVECOTEV2` execution
- bounded seed-stability and calibration evidence in the shared archive
  frontier
- a positive result on the shared archive-versus-physics witness
- parity with the engineered timing-order champion on the bounded
  feature-headroom witness

The current packet set is enough to justify:

- `hive_cote` moving to `witness_supported`

## Claim Boundary

This is not a claim of full HIVE-COTE benchmark parity or unconstrained
ensemble-budget fidelity.

What remains open:

- broader archive-family benchmark breadth
- larger-budget HIVE-COTE fidelity beyond the compact bounded config used here
- stronger witness coverage beyond the current shared-corpus and timing-order
  packets

The registry therefore now keeps `hive_cote` at `witness_supported` on the
current bounded archive packets.
