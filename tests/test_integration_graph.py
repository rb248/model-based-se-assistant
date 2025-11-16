import json
import sys
import pytest
import types
import pydantic

# If the langchain_core.pydantic_v1 module is missing, add a shim for it
if "langchain_core.pydantic_v1" not in sys.modules:
    mod = types.ModuleType("langchain_core.pydantic_v1")
    mod.BaseModel = pydantic.BaseModel
    mod.Field = pydantic.Field
    sys.modules["langchain_core.pydantic_v1"] = mod

from backend.graph import get_compiled_graph, WorkflowState


class DummyLLM:
    def __init__(self, response: str):
        self.response = response

    def __call__(self, prompt: str) -> str:
        return self.response


@pytest.mark.integration
def test_langgraph_end_to_end_with_real_llm():
    """Run the LangGraph workflow end-to-end with real LLM."""
    compiled = get_compiled_graph()

    initial_state = {
        "project_id": "graph-test-real-llm",
        "model_text": """@startuml
class OrderManager {
  +createOrder(userId, items): Order
  +sendEmail(email, message): void
  +saveToDatabase(order): bool
  +logActivity(message): void
  +generateInvoice(orderId): Invoice
}
@enduml""",
        "model_format": "plantuml",
        "description": "E-commerce order management system with god class that needs refactoring"
    }

    result = compiled.invoke(initial_state)

    # Assertions
    assert result.get("errors") == [] or result.get("errors") == None, f"Workflow had errors: {result.get('errors')}"
    final_report = result.get("final_report")
    assert final_report is not None, "Final report should exist"
    
    # Check that parsing succeeded
    assert final_report.get("model_ir_classes", 0) >= 1, "Should have parsed at least one class"
    
    # Check that analysis ran
    assert final_report.get("analysis_findings", 0) >= 1, "Should have found at least one design issue"
    
    # Check that code generation succeeded
    assert final_report.get("generated_files", 0) >= 1, "Should have generated at least one file"
    
    print("\n" + "="*80)
    print("LANGGRAPH END-TO-END TEST RESULTS (Real LLM)")
    print("="*80)
    print(json.dumps(result.get("final_report"), indent=2))
    print("="*80 + "\n")


def test_langgraph_end_to_end_with_tests_dummy(monkeypatch):
    """End-to-end workflow with dummy agents including test generation and execution."""
    # Deterministic parser response
    parser_response = json.dumps({
        "classes": [
            {
                "name": "OrderService",
                "methods": [
                    {"name": "create_order", "params": ["user_id", "items"], "returns": "Order"}
                ],
                "attributes": []
            },
            {
                "name": "OrderRepository",
                "methods": [
                    {"name": "save", "params": ["order"], "returns": "bool"}
                ],
                "attributes": []
            }
        ],
        "relationships": []
    })

    analysis_response = json.dumps({
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderService should not handle persistence",
                "affected_entities": ["OrderService", "OrderRepository"],
                "violated_principle": "SRP",
                "category": "solid"
            }
        ],
        "recommendations": [
            {
                "title": "Verify DI",
                "description": "Ensure repo is injected",
                "affected_entities": ["OrderService"]
            }
        ]
    })

    codegen_response = json.dumps({
        "files": [
            {"path": "src/order_service.py", "content": "class OrderService:\n    def __init__(self, repository):\n        self.repository = repository\n    def create_order(self, user_id, items):\n        order = {'order_id':'1'}\n        self.repository.save(order)\n        return order"},
            {"path": "src/order_repository.py", "content": "class OrderRepository:\n    def save(self, order):\n        return True"}
        ]
    })

    testgen_response = json.dumps({
        "test_files": [
            {
                "path": "tests/test_order_service.py",
                "content": "import pytest\nfrom src.order_service import OrderService\nfrom src.order_repository import OrderRepository\n\nclass DummyRepo:\n    def save(self, order):\n        return True\n\ndef test_create_order_integration():\n    repo = DummyRepo()\n    svc = OrderService(repository=repo)\n    order = svc.create_order('user', [])\n    assert order is not None\ndef test_repository_save():\n    repo = OrderRepository()\n    assert repo.save({'order_id':'1'}) is True"
            }
        ],
        "total_tests": 2
    })

    from backend.agents import ParserAgent, AnalysisAgent, CodeGenerationAgent, TestGenerationAgent

    monkeypatch.setattr("backend.graph.ParserAgent", lambda: ParserAgent(llm=DummyLLM(parser_response)))
    monkeypatch.setattr("backend.graph.AnalysisAgent", lambda: AnalysisAgent(llm=DummyLLM(analysis_response)))
    monkeypatch.setattr("backend.graph.CodeGenerationAgent", lambda: CodeGenerationAgent(llm=DummyLLM(codegen_response)))
    monkeypatch.setattr("backend.graph.TestGenerationAgent", lambda: TestGenerationAgent(llm=DummyLLM(testgen_response)))

    compiled = get_compiled_graph()

    initial_state = {
        "project_id": "graph-test-dummy-tests",
        "model_text": "@startuml\n class OrderService {\n +create_order(user_id, items): Order\n}\n@enduml",
        "model_format": "plantuml",
        "description": "Test model for LangGraph end-to-end with test generation"
    }

    result = compiled.invoke(initial_state)

    # Assertions
    assert result.get("errors") == [] or result.get("errors") == None
    final_report = result.get("final_report")
    assert final_report is not None
    assert final_report.get("model_ir_classes", 0) >= 1
    assert final_report.get("analysis_findings", 0) >= 1
    assert final_report.get("generated_files", 0) >= 1
    assert final_report.get("test_cases", 0) >= 1
    # Test execution results should be present
    assert result.get("test_results") is not None
    assert result.get("test_results").get("status") in ("completed", "skipped", "timeout", "error")
    
    print(json.dumps(final_report, indent=2))
