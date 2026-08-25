# Labels and Claim Boundary

TGSIM and pNEUMA are the only selected sources with a current path to road-user class
assertions. Every class value must become a `LabelAssertion` with source evidence, strength,
proxy status, vocabulary version, and dependency channels. The label and raw identity must not
enter ordinary kinematic features.

ReV-StED supports only:

```text
land.collection_role = instrumented_test_vehicle
```

That assertion documents collection role. It is not a road-user class target. Amazon Precision
GNSS likewise provides no vehicle-class assertion; its role is paired estimate/reference quality
validation.

This tranche supports source-portfolio and contract-semantics claims. It does not support:

- car-versus-truck classifier performance;
- fleet or population diversity conclusions;
- cross-source generalization;
- tracked-vehicle coverage;
- armored-vehicle semantics;
- tank identification;
- a prepared or study-ready pilot.
