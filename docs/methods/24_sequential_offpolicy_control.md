# SAC / TD3 Sequential Frontier

The repo now has a dedicated smoke frontier packet for the off-policy
sequential-control lane:

- study id: `sequential_offpolicy_control_frontier_v1`
- artifacts: `artifacts/sequential_offpolicy_control_frontier_v1/`

The packet writes `frontier_summary.csv`, `budget_sweep_summary.csv`,
`seed_sweep_summary.csv`, and the usual report/decision-card artifacts.

## What It Proves

This packet trains SAC and TD3 on the same sequential-control witness surface
used by the PPO boundary-control work and compares them against:

- `ppo_policy`
- `random_control`
- `scripted_profiles`
- `doe_schedule_bank`
- `guided_schedule_mutation`

The key addition is not a grand claim of superiority. It is a concrete
sample-efficiency comparison for the off-policy lane on the same generated
objectives.

The current packet also includes a small budget sweep so SAC and TD3 can be
compared at more than one evaluation budget rather than only at a single run
length, plus a narrow seed sweep so the sample-efficiency comparison is not
just a one-off artifact of a single random seed.

The current packet is enough to keep the lane explicit and to justify:

- `sac_td3` moving from roadmap-only status to an implemented sequential-control
  comparison surface

## Claim Boundary

This is not yet a full off-policy sequential-control workbench.

What remains open:

- broader objective-family sweeps
- longer budgets and stability checks
- a stronger decision on whether SAC or TD3 should be promoted over PPO for a
  given witness family
