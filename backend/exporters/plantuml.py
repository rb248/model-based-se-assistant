"""PlantUML export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

RELATION_ARROWS = {
    "inheritance": "--|>",
    "extends": "--|>",
    "implements": "..|>",
    "composition": "*--",
    "aggregation": "o--",
    "dependency": "..>",
    "association": "--",
}


def ir_to_plantuml(model_ir: Dict[str, Any]) -> str:
    """Convert a normalized IR dictionary back into PlantUML text."""

    lines: List[str] = ["@startuml"]

    for cls in model_ir.get("classes", []):
        name = cls.get("name", "Unnamed")
        stereotype = cls.get("stereotype")
        description = cls.get("description", "")

        header = f"class {name}"
        if stereotype:
            header += f" <<{stereotype}>>"
        lines.append(header + " {")

        for attr in cls.get("attributes", []):
            attr_name = attr.get("name")
            attr_type = attr.get("type", "any")
            visibility = attr.get("visibility", "+")
            lines.append(f"  {visibility}{attr_name}: {attr_type}")

        for method in cls.get("methods", []):
            method_name = method.get("name")
            params = method.get("params", [])
            returns = method.get("returns", "void")
            visibility = method.get("visibility", "+")
            param_str = ", ".join(params)
            lines.append(f"  {visibility}{method_name}({param_str}): {returns}")

        if description:
            lines.append(f"  ' {description}")

        lines.append("}")

    for note in model_ir.get("notes", []):
        lines.append(f"note \"{note}\"")

    for rel in model_ir.get("relationships", []):
        from_cls = rel.get("from")
        to_cls = rel.get("to")
        rel_type = rel.get("type", "association").lower()
        label = rel.get("label") or rel.get("description")
        multiplicity = rel.get("multiplicity")

        arrow = RELATION_ARROWS.get(rel_type, RELATION_ARROWS["association"])
        relation = f"{from_cls} {arrow} {to_cls}"
        if multiplicity:
            relation += f" : {multiplicity}"
        if label:
            relation += f" \"{label}\""
        lines.append(relation)

    lines.append("@enduml")
    return "\n".join(lines)


def write_plantuml(model_ir: Dict[str, Any], path: Path) -> Path:
    """Write PlantUML text to the provided path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plantuml_text = ir_to_plantuml(model_ir)
    path.write_text(plantuml_text)
    return path
