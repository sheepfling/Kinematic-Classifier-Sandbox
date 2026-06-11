# TS2Vec

The repo now has a first TS2Vec-style embedding benchmark witness, but it is
still a proxy frontier rather than a claim that the external TS2Vec library
has been installed and benchmarked.

The current packet is enough to justify `ts2vec` as `witness_supported` in the
method-validation registry.

## What It Proves

The current atlas boundary is intentionally conservative:

- `ts2vec` now has a first embedding witness
- `ts_tcc_softclt` remains a follow-on research lane
- `masked_timeseries_autoencoder` remains a follow-on research lane

The coverage matrix already tracks `ts2vec_family`, and the new witness makes
the lane executable instead of only being a registry note.

## Claim Boundary

What remains open before this lane can be promoted:

- matched-budget comparisons against broader external TSC baselines
- robustness sweeps showing that the representation is stable across seeds and
  trajectory families
- a clearer downstream task where the learned embedding buys something the
  simpler 1D classifier ladder does not
- faithful external-library TS2Vec parity rather than a local proxy encoder
