# Product Test Matrix

The repository has four connected analysis products plus shared infrastructure. Their tests are
separated so a deep classifier-ladder investigation does not become an accidental prerequisite
for a corpus-adapter change.

The machine-readable source is [`test_matrix.yaml`](test_matrix.yaml). Pytest marker assignment is
implemented in [`tests/conftest.py`](../../tests/conftest.py), and marker declarations live in
[`pyproject.toml`](../../pyproject.toml).

| Surface | Product | Validation tier | Parallel-safe | Cross-product gate | Current state |
| --- | --- | --- | --- | --- | --- |
| Static admissibility | Product 1 | product | yes | no | active |
| Classifier evidence ladder | Product 2 | product | no | no | active |
| RL corpus exploration | Product 3 | product | no | no | active |
| Product 4 common contract | Product 4 | product | yes | no | active |
| Product 4 cross-domain portfolio | Product 4 | product | yes | no | active |
| Product 4 analysis-product boundaries | Product 4 | product | yes | no | active |
| LAND / SEA-SURF / SEA-SUB / AIR / SPACE-NEAR / SPACE-ORB | Product 4 | product | yes, per lane | no | active |
| Shared analysis and repository contracts | Shared | contract | no | yes | active |
| Real-world corpus → classifier ladder | Products 2 + 4 | release | no | yes | blocked until prepared snapshot |
| Full repository regression | All | release | no | yes | active |

Useful targeted commands:

```bash
PYTHONPATH=src python3 -m pytest -q -m product1
PYTHONPATH=src python3 -m pytest -q -m product2
PYTHONPATH=src python3 -m pytest -q -m product3
PYTHONPATH=src python3 scripts/run/run_product4_tests.py --workers 4
PYTHONPATH=src python3 -m pytest -q -m cross_product
```

The Product 4 worker runs each common, cross-domain, analysis-product, and domain-lane marker in a
separate pytest process. Product 2, Product 3, shared analysis, and the aggregate/full gates
remain sequential until their artifact namespaces are proven isolated.

The Product 4 cross-domain gate also exercises `Product4GateReport`: provenance, rights, immutable
snapshot, lane coverage, quality, grouped leakage, and classifier-view readiness remain separate
signals. A coherent registry is therefore allowed to pass its own product gate while the
real-world bridge stays blocked.

The real-world bridge is intentionally not runnable yet. It becomes runnable only when Product 4
has a prepared immutable snapshot with leakage-safe splits and Product 2 can evaluate that snapshot
as a held-out corpus without source identity or audit-only fields entering the classifier view.
