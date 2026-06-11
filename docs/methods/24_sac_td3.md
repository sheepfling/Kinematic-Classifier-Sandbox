# SAC / TD3

The repo tracks SAC and TD3 as off-policy sequential-control candidates, and it
now has a dedicated smoke frontier packet for them:

- study id: `sequential_offpolicy_control_frontier_v1`
- artifacts: `artifacts/sequential_offpolicy_control_frontier_v1/`

The packet writes `frontier_summary.csv`, `budget_sweep_summary.csv`,
`seed_sweep_summary.csv`, and the report/decision-card artifacts.

## What It Proves

The earlier sequential-control packet is still the PPO proxy:

- study id: `sequential_control_generator_frontier_v1`
- artifacts: `artifacts/sequential_control_generator_frontier_v1/`

That packet defines the baseline contract SAC/TD3 would need to beat. The
off-policy smoke frontier then trains SAC and TD3 on the same sequential
control surface and compares them against PPO and the baseline control
families.

The current registry status is therefore:

- `sac_td3` is `implemented`

## Claim Boundary

This is not yet a full off-policy sequential-control workbench.

What remains open:

- broader objective-family sweeps
- longer budgets and stability checks
- a stronger decision on whether SAC or TD3 should be promoted over PPO for a
  given witness family
