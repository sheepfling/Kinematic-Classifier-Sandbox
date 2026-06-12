# Sequential Control Generator Frontier

The repo now has a dedicated proxy frontier packet for the sequential-control
lane:

- study id: `sequential_control_generator_frontier_v1`
- artifacts: `artifacts/sequential_control_generator_frontier_v1/`

## What It Proves

This packet sits on top of the existing PPO boundary-control surface and
compares it against the current sequential-control baselines:

- `ppo_policy`
- `random_control`
- `scripted_profiles`
- `doe_schedule_bank`
- `guided_schedule_mutation`

The current packet keeps the sequential-control lane explicit and is now
paired with the off-policy smoke frontier, so SAC and TD3 are represented by
real runs rather than roadmap notes alone.

The current packet is enough to keep the lane tracked and to justify:

- `sac_td3` as a first-class implemented comparison surface, while the claim
  remains limited to witness-specific smoke frontier evidence

## Claim Boundary

This is not yet a full off-policy sequential-control workbench.

What remains open:

- the companion off-policy smoke packet now exists, but it still needs broader
  seed and budget sweeps
- broader objective-family sweeps beyond the current proxy frontier
- stronger sample-efficiency comparisons against the PPO proxy on matched
  sequential-control budgets
