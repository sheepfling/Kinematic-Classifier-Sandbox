from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "surveys" / "methodology_doc_coverage.yaml"
ARTIFACT_DIR = ROOT / "artifacts" / "latex"
JSON_OUT = ARTIFACT_DIR / "methodology_doc_coverage.json"
MD_OUT = ARTIFACT_DIR / "methodology_doc_coverage.md"


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def render_markdown(payload: dict[str, object]) -> str:
    modules = payload["modules"]
    lines = [
        "# Methodology Documentation Coverage",
        "",
        "This artifact is generated from `docs/surveys/methodology_doc_coverage.yaml`.",
        "",
        f"- Declared docs: {', '.join(f'`{name}`' for name in payload['documents'])}",
        f"- Covered modules: {len(modules)}",
        "",
        "| Module | Coverage Kind | Primary Doc | Primary Section | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in modules:
        lines.append(
            f"| `{row['module_path']}` | `{row['coverage_kind']}` | "
            f"`{row['primary_doc']}` | `{row['primary_section']}` | `{row['status']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = load_manifest()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(render_markdown(payload), encoding="utf-8")
    print(JSON_OUT)
    print(MD_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
