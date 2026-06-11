# Control Surface Backends

The control-surface layer separates how a trajectory is generated from what the classifier is allowed to observe.

Every backend emits a canonical trajectory candidate with:

- truth state: time, position, velocity, acceleration
- observation: measured position for the current 1D workbench
- generation metadata: backend id, parameters, constraints
- optional hidden control trace: acceleration, jerk, mode schedule, or stochastic process trace

The classifier-facing contract is narrower:

- allowed fields: time, observed position, derived velocity, derived acceleration
- hidden fields: backend id, generator parameters, control trace, objective score

This prevents the workbench from proving only that a classifier can exploit one generator signature.

## Implemented 1D Surfaces

| Backend | Surface Type | Main Use |
| --- | --- | --- |
| `direct_kinematic_params` | static parameters | readable witnesses and class-validity probes |
| `acceleration_sequence` | sequential control | PPO/CEM control optimization and posterior-boundary witnesses |
| `jerk_sequence` | sequential control | smooth onset and weak-acceleration ambiguity witnesses |
| `spline_knots` | compact smooth parameters | endpoint ambiguity and smooth feature targeting |
| `hybrid_mode_schedule` | latent mode schedule | switching, transition-matrix, IMM, and RBPF witnesses |
| `stochastic_process` | stochastic dynamics | anti-template stress, calibration, and generalization |

## First Backend-Agnostic Proof

The first proof is `posterior_target__cv_ca_50_50` across all implemented surfaces.

The experiment asks whether different generation mechanisms can produce trajectories whose achieved posterior is close to:

```text
P(constant_velocity | trajectory) = 0.5
P(constant_acceleration | trajectory) = 0.5
```

This is a stronger check than a single `boundary_closeness` score because the target is expressed in posterior space rather than one backend's control variables.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/kcs_pycache PYTHONPATH=src \
python3 -m kinematic_classifier_sandbox trajectory-control-surface-sweep \
  --output-dir artifacts
```

Key outputs:

- `artifacts/control_surfaces/control_surface_manifest.csv`
- `artifacts/control_surfaces/observation_surface_manifest.csv`
- `artifacts/control_surfaces/backend_capability_matrix.csv`
- `artifacts/control_surfaces/backend_objective_achievability.csv`
- `artifacts/control_surfaces/target_vs_achieved_posterior_by_backend.csv`
- `artifacts/control_surfaces/generator_identification_probe.csv`
- `artifacts/control_surfaces/backend_identification_probe.csv`
- `artifacts/control_surfaces/backend_identification_confusion.csv`
- `artifacts/control_surfaces/backend_objective_achievability.png`
- `artifacts/control_surfaces/target_vs_achieved_posterior_by_backend.png`
- `artifacts/control_surfaces/backend_identification_probe.png`
- `artifacts/control_surfaces/control_surface_report.md`

## Remaining Work

Observation surfaces are now cataloged but not fully pluggable. The next layer should split truth generation from sensor generation in the runner:

- clean observation
- Gaussian position noise
- dropout sampler
- quantization
- outlier injection
- low-rate sampler

The current generator identification probe is a deterministic nearest-centroid classifier over visible trajectory summaries. The stronger audit is to train a backend-ID classifier using the same feature matrix available to the real classifier and then run backend holdout studies.
