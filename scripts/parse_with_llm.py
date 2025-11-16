"""Small helper to run ParserAgent against a local PlantUML model using the configured LLM.

Usage (example):
RUN_LLM_INTEGRATION=true OPENAI_API_KEY=sk-... python scripts/parse_with_llm.py sample_models/simple_class.puml

If no file is passed, this script uses a small builtin PlantUML example.
"""
import json
import sys
from pathlib import Path

from backend.llms import get_llm
from backend.agents import ParserAgent


def run(path: Path | None = None):
    if path and path.exists():
        text = path.read_text()
    else:
        text = """
        @startuml
        class User {
           - name: str
           + greet(name): str
        }

        class Order {
           - order_id: int
           + total(): float
        }

        User "1" --> "*" Order
        @enduml
        """

    try:
        llm = get_llm()
    except Exception as e:
        print(f"Could not initialize LLM: {e}")
        raise

    agent = ParserAgent(llm=llm)
    parsed = agent.parse_model(text, model_format="plantuml")

    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(p)
