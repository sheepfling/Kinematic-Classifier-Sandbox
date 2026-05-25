# Formal Math Visual Registry

This registry makes the math/evaluation equations visually auditable.

It answers three questions for each equation:

1. What is the representative visual?
2. What exact source artifact and source data support it?
3. How do I rerun the bundle?

## Rerun

```bash
python3 scripts/render/render_formal_math_visual_registry.py --output-dir artifacts
```

The generated bundle includes:

- `formal_math_visual_registry_report.md`
- `formal_math_visual_registry_summary.json`
- `formal_math_visual_registry.csv`
- `formal_math_visual_registry_provenance.csv`
- `formal_math_visual_registry_runbook.md`
- `formal_math_visual_registry_coverage.png`

## Provenance Rules

- Implemented equations are counted only when the visual comes from a real artifact or a real-data generator.
- Conceptual equations remain explicitly labeled `illustrative`.
- Missing source artifacts are regenerated from the underlying analysis code rather than replaced with a generic placeholder chart.
