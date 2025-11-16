import json
from backend.agents import ParserAgent


class DummyLLM:
    """A configurable Dummy LLM for deterministic testing.

    Initialize with a response string to return (JSON or non-JSON). The callable
    ignores the incoming prompt and returns the configured response.
    """

    def __init__(self, response: str):
        self.response = response

    def __call__(self, prompt: str) -> str:
        return self.response


def make_plantuml_inheritance():
    return """
    @startuml
    class Animal {
      +eat(): void
    }

    class Dog {
      +bark(): void
    }

    Dog --|> Animal
    @enduml
    """


def make_expected_inheritance_json():
    return {
        "classes": [
            {"name": "Animal", "attributes": [], "methods": [{"name": "eat", "params": [], "returns": "void"}], "description": ""},
            {"name": "Dog", "attributes": [], "methods": [{"name": "bark", "params": [], "returns": "void"}], "description": ""},
        ],
        "relationships": [{"from": "Dog", "to": "Animal", "type": "inheritance", "multiplicity": "1"}],
        "notes": []
    }


def make_plantuml_composition():
    return """
    @startuml
    class Engine {
      - horsepower: int
    }

    class Car {
      - make: string
    }

    Car *-- Engine
    @enduml
    """


def make_expected_composition_json():
    return {
        "classes": [
            {"name": "Engine", "attributes": [{"name": "horsepower", "type": "int"}], "methods": [], "description": ""},
            {"name": "Car", "attributes": [{"name": "make", "type": "string"}], "methods": [], "description": ""},
        ],
        "relationships": [{"from": "Car", "to": "Engine", "type": "composition", "multiplicity": "1"}],
        "notes": []
    }


def make_plantuml_methods_with_params():
    return """
    @startuml
    class Service {
      +process(data): bool
    }
    @enduml
    """


def make_expected_methods_json():
    return {
        "classes": [
            {"name": "Service", "attributes": [], "methods": [{"name": "process", "params": ["data"], "returns": "bool"}], "description": ""}
        ],
        "relationships": [],
        "notes": []
    }


def test_inheritance_parsing():
    plantuml = make_plantuml_inheritance()
    expected = make_expected_inheritance_json()
    agent = ParserAgent(llm=DummyLLM(json.dumps(expected)))

    parsed = agent.parse_model(plantuml)
    assert isinstance(parsed, dict)
    assert len(parsed.get("classes", [])) == 2
    names = {c["name"] for c in parsed.get("classes", [])}
    assert "Dog" in names and "Animal" in names
    # relationship type check
    rels = parsed.get("relationships", [])
    assert any(r["type"] == "inheritance" for r in rels)


def test_composition_parsing():
    plantuml = make_plantuml_composition()
    expected = make_expected_composition_json()
    agent = ParserAgent(llm=DummyLLM(json.dumps(expected)))

    parsed = agent.parse_model(plantuml)
    assert isinstance(parsed, dict)
    assert any(c["name"] == "Car" for c in parsed.get("classes", []))
    assert any(any(attr["name"] == "horsepower" for attr in c.get("attributes", [])) for c in parsed.get("classes", []))
    rels = parsed.get("relationships", [])
    assert any(r["type"] == "composition" for r in rels)


def test_methods_with_params_parsing():
    plantuml = make_plantuml_methods_with_params()
    expected = make_expected_methods_json()
    agent = ParserAgent(llm=DummyLLM(json.dumps(expected)))

    parsed = agent.parse_model(plantuml)
    assert isinstance(parsed, dict)
    methods = parsed.get("classes", [])[0].get("methods", [])
    assert methods and methods[0]["name"] == "process"
    assert methods[0]["params"] == ["data"]


def test_malformed_llm_response_results_in_error():
    plantuml = "@startuml\nthis is not plantuml\n@enduml"
    # Dummy LLM returns non-JSON
    agent = ParserAgent(llm=DummyLLM("I couldn't parse that"))

    parsed = agent.parse_model(plantuml)
    assert parsed.get("classes") == []
    assert "error" in parsed


def test_validation_failure_on_missing_class_name():
    plantuml = "@startuml\nclass X { }\n@enduml"
    # Return JSON where a class is missing the 'name' field -> should fail validation
    bad_json = {"classes": [{"attributes": [], "methods": [], "description": ""}], "relationships": [], "notes": []}
    agent = ParserAgent(llm=DummyLLM(json.dumps(bad_json)))

    parsed = agent.parse_model(plantuml)
    assert parsed.get("classes") == []
    assert "error" in parsed


def test_empty_input_returns_empty_model():
    plantuml = ""
    empty_json = {"classes": [], "relationships": [], "notes": []}
    agent = ParserAgent(llm=DummyLLM(json.dumps(empty_json)))

    parsed = agent.parse_model(plantuml)
    assert isinstance(parsed, dict)
    assert parsed.get("classes") == []
    assert parsed.get("relationships") == []
