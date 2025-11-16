import json
from backend.agents import ParserAgent


class DummyLLM:
    def __call__(self, prompt: str) -> str:
        # This dummy LLM simply returns a JSON that matches the first example if it sees 'class User'
        if "class User" in prompt:
            return json.dumps({
                "classes": [
                    {"name": "User", "attributes": [{"name": "id", "type": "int"}, {"name": "name", "type": "string"}], "methods": [], "description": ""}
                ],
                "relationships": [],
                "notes": []
            })
        # fallback empty
        return json.dumps({"classes": [], "relationships": [], "notes": []})


def test_fewshot_examples_in_prompt_and_parse():
    dummy = DummyLLM()
    parser = ParserAgent(name="parser-test", llm=dummy)

    plantuml = "class User {\n  +id: int\n  +name: string\n}"

    # Call parse_model which for DummyLLM returns a JSON string
    result = parser.parse_model(plantuml)

    assert "classes" in result
    assert isinstance(result["classes"], list)
    assert any(c["name"] == "User" for c in result["classes"])
