# TS2Vec

The repo now has two bounded TS2Vec surfaces:

- `embedding_baseline_frontier_v1`, which keeps the learned-embedding lane
  executable with an online prefix route and reports the backend actually used
- `ts2vec_backend_parity_v1`, which compares the local proxy route against the
  optional external `ts2vec` package on the same shared 1D witness

The current packet set is enough to justify `ts2vec` as `witness_supported` in
the method-validation registry, but not enough to claim broad library parity or
full Epic 2 promotion.

## What It Proves

The current atlas boundary is intentionally conservative:

- `ts2vec` now has a first embedding witness
- the witness now includes an online prefix route that compares TS2Vec-style
  embeddings against the current online-capable baselines
- the witness now records which backend was used
- the lane now has a bounded proxy-versus-external parity witness rather than
  only an import-time optional-backend note
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
- broader external-library TS2Vec parity beyond the current shared-corpus
  witness
- a faithful streaming encoder rather than only a prefix-route proof on the
  current compact frontier
