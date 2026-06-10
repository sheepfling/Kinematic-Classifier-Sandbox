from __future__ import annotations

import json
from pathlib import Path

import yaml

from _bootstrap import bootstrap_repo

ROOT = bootstrap_repo(configure_runtime=True)
DOCS_MATH_DIR = ROOT / "docs" / "math"
SYMBOL_GLOSSARY_PATH = DOCS_MATH_DIR / "symbol_glossary.tex"
EQUATION_REGISTRY_PATH = DOCS_MATH_DIR / "equation_registry.yaml"
CROSSWALK_PATH = DOCS_MATH_DIR / "code_equation_crosswalk.md"
ARTIFACT_DIR = ROOT / "artifacts" / "latex"
SYMBOL_JSON_OUT = ARTIFACT_DIR / "symbol_glossary.json"
SYMBOL_MD_OUT = ARTIFACT_DIR / "symbol_glossary.md"
REGISTRY_JSON_OUT = ARTIFACT_DIR / "equation_registry.json"
REGISTRY_MD_OUT = ARTIFACT_DIR / "equation_registry.md"
CROSSWALK_MD_OUT = ARTIFACT_DIR / "code_equation_crosswalk.md"


def load_registry() -> list[dict[str, object]]:
    payload = yaml.safe_load(EQUATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def parse_symbol_glossary() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in SYMBOL_GLOSSARY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "&" not in line or line.startswith("\\") and "toprule" in line:
            continue
        if line.startswith("Symbol &") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if not line.endswith(r"\\"):
            continue
        parts = [part.strip() for part in line[:-2].split("&")]
        if len(parts) != 5:
            continue
        symbol, meaning, shape, example, source = parts
        rows.append(
            {
                "symbol": symbol,
                "meaning": meaning,
                "shape": shape,
                "example": example,
                "source": source,
            }
        )
    return rows


def render_symbol_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Symbol Glossary",
        "",
        "Generated from `docs/math/symbol_glossary.tex`.",
        "",
        f"- Declared symbols: {len(rows)}",
        "",
        "| Symbol | Meaning | Shape / type | Example | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['symbol']}` | {row['meaning']} | {row['shape']} | {row['example']} | {row['source']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_registry_markdown(rows: list[dict[str, object]]) -> str:
    implemented = sum(1 for row in rows if row["status"] == "implemented")
    conceptual = sum(1 for row in rows if row["status"] == "conceptual")
    lines = [
        "# Equation Registry",
        "",
        "Generated from `docs/math/equation_registry.yaml`.",
        "",
        f"- Registered equations: {len(rows)}",
        f"- Implemented equations: {implemented}",
        f"- Conceptual equations: {conceptual}",
        "",
        "| Equation ID | Status | Implementation | Artifacts | Tests |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        implementation = row["implementation"]
        artifacts = "<br>".join(f"`{item}`" for item in row.get("artifacts", []))
        tests = "<br>".join(f"`{item}`" for item in row.get("tests", [])) or "none"
        lines.append(
            f"| `{row['id']}` | `{row['status']}` | "
            f"`{implementation['module']}::{implementation['function']}` | {artifacts} | {tests} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_registry(rows: list[dict[str, object]]) -> None:
    seen: set[str] = set()
    for row in rows:
        equation_id = row["id"]
        if equation_id in seen:
            raise ValueError(f"Duplicate equation id: {equation_id}")
        seen.add(equation_id)
        if row["status"] not in {"implemented", "conceptual"}:
            raise ValueError(f"Unexpected equation status for {equation_id}: {row['status']}")
        implementation = row["implementation"]
        if "module" not in implementation or "function" not in implementation:
            raise ValueError(f"Missing implementation mapping for {equation_id}")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = parse_symbol_glossary()
    registry = load_registry()
    validate_registry(registry)

    SYMBOL_JSON_OUT.write_text(json.dumps(symbols, indent=2) + "\n", encoding="utf-8")
    SYMBOL_MD_OUT.write_text(render_symbol_markdown(symbols), encoding="utf-8")
    REGISTRY_JSON_OUT.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    REGISTRY_MD_OUT.write_text(render_registry_markdown(registry), encoding="utf-8")
    CROSSWALK_MD_OUT.write_text(CROSSWALK_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    print(SYMBOL_JSON_OUT)
    print(SYMBOL_MD_OUT)
    print(REGISTRY_JSON_OUT)
    print(REGISTRY_MD_OUT)
    print(CROSSWALK_MD_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
