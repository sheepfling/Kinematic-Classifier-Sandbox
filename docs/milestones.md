# Milestone Rerun Guide

This guide is the junior-friendly entrypoint for rerunning milestone artifact bundles without reading the source tree first.

## Quick Use

List milestone surfaces:

```bash
python3 scripts/run_milestone.py list
```

Run one milestone:

```bash
python3 scripts/run_milestone.py m6 --output-dir artifacts
```

Run the whole graduated stack:

```bash
python3 scripts/run_milestone.py m1-m9 --output-dir artifacts
```

## Milestones

| Milestone | Status | What it reruns | Main artifact dir |
| --- | --- | --- | --- |
| `m0` | done | contract demo and sample artifact validation surface | `milestone0_contract_demo/` |
| `m1` | done | pointwise baseline benchmark | `pointwise_baseline/` |
| `m2` | done | windowed raw and robust feature benchmark | `windowed_baseline/` |
| `m3` | done | sequential Bayesian accumulator benchmark | `bayes_accumulator/` |
| `m4` | done | Monte Carlo accumulator pack | `monte_carlo_accumulator/` |
| `m5` | done | trajectory generator foundation | `trajectory_generator_v1/` |
| `m6` | done | feature excitation and identifiability analysis | `feature_analysis_v1/` |
| `m7` | done | Kalman filter bank benchmark | `kalman_filter_bank/` |
| `m8` | done | PCA and principal-feature analysis | `pca_analysis_v1/` |
| `m9` | done | current generator-stack graduation surface plus supplemental scenario libraries | `trajectory_generator_v1/` |

## Suggested Commands

Generate one milestone for documentation refresh:

```bash
python3 scripts/run_milestone.py m1 --output-dir artifacts
python3 scripts/run_milestone.py m6 --output-dir artifacts
python3 scripts/run_milestone.py m8 --output-dir artifacts
```

Generate everything that a reviewer usually needs:

```bash
python3 scripts/run_milestone.py m1-m9 --output-dir artifacts
python3 scripts/coverage_report.py --output-dir artifacts
```

## Notes

- `m9` now includes explicit short-horizon, perturbation-sweep, and switching scenario libraries under `trajectory_generator_v1/`.
- For the full repo-wide artifact refresh, `python3 scripts/export_artifacts.py` still exists, but `run_milestone.py` is the simpler entrypoint when you only want milestone-oriented outputs.
