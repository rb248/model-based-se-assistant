"""
Tests for TestGenerationAgent.

This module tests the test generation agent's ability to:
1. Generate unit tests for simple code
2. Generate analysis-aware tests (cohesion, coupling, interface tests)
3. Extract test strategies from analysis reports
4. Generate integration tests
5. Handle real LLM integration
"""

import json
import pytest
from unittest.mock import Mock
from backend.agents import TestGenerationAgent


class DummyLLM:
    """Mock LLM for deterministic testing that's compatible with LangChain."""
    
    def __init__(self, response_dict: dict):
        self.response_dict = response_dict
    
    def invoke(self, prompt):
        """Return fixed response dict (not Runnable, used for testing)."""
        return self.response_dict


# ============================================================================
# Unit Tests with DummyLLM
# ============================================================================

def test_simple_test_generation():
    """Test basic test generation with minimal code."""
    
    mock_response = {
        "test_files": [
            {
                "path": "tests/test_calculator.py",
                "content": """import pytest
from src.calculator import Calculator

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
"""
            }
        ]
    }
    
    agent = TestGenerationAgent(llm=DummyLLM(mock_response))
    
    model_ir = {
        "classes": [
            {
                "name": "Calculator",
                "methods": [
                    {"name": "add", "params": ["a", "b"], "returns": "int"},
                    {"name": "subtract", "params": ["a", "b"], "returns": "int"}
                ],
                "attributes": []
            }
        ]
    }
    
    generated_code = {
        "files": [
            {
                "path": "src/calculator.py",
                "content": "class Calculator:\n    def add(self, a, b):\n        return a + b"
            }
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        framework="pytest"
    )
    
    assert result["test_files"] is not None
    assert len(result["test_files"]) == 1
    assert result["framework"] == "pytest"
    assert result["total_tests"] == 2  # 2 'def test_' occurrences
    assert "test_calculator.py" in result["test_files"][0]["path"]


def test_extract_test_strategy_with_god_class():
    """Test strategy extraction identifies god classes for cohesion tests."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderManager is a god class with multiple responsibilities",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP",
                "category": "solid"
            }
        ],
        "recommendations": [
            {
                "title": "Split OrderManager",
                "description": "Decompose into OrderService, OrderRepository",
                "affected_entities": ["OrderManager"]
            }
        ]
    }
    
    strategy = agent._extract_test_strategy(analysis_report)
    
    assert "cohesion" in strategy["categories"]
    assert "cohesion" in strategy["focus_areas"]
    assert "OrderManager" in strategy["god_classes"]


def test_extract_test_strategy_with_missing_abstraction():
    """Test strategy extraction identifies missing abstractions for interface tests."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    analysis_report = {
        "findings": [
            {
                "severity": "warning",
                "issue": "Missing abstraction for payment processors",
                "affected_entities": ["PayPalProcessor", "StripeProcessor"],
                "violated_principle": "",
                "category": "pattern"
            }
        ],
        "recommendations": []
    }
    
    strategy = agent._extract_test_strategy(analysis_report)
    
    assert "interface" in strategy["categories"]
    assert "abstraction" in strategy["focus_areas"]
    assert "PayPalProcessor" in strategy["missing_abstractions"]
    assert "StripeProcessor" in strategy["missing_abstractions"]


def test_extract_test_strategy_with_dip_violation():
    """Test strategy extraction identifies DIP violations for dependency injection tests."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    analysis_report = {
        "findings": [
            {
                "severity": "warning",
                "issue": "OrderService depends directly on MySQLRepository",
                "affected_entities": ["OrderService"],
                "violated_principle": "DIP",
                "category": "solid"
            }
        ],
        "recommendations": []
    }
    
    strategy = agent._extract_test_strategy(analysis_report)
    
    assert "dependency_injection" in strategy["categories"]
    assert "coupling" in strategy["focus_areas"]
    assert "OrderService" in strategy["dip_violations"]


def test_generate_tests_with_analysis_awareness():
    """Test that analysis-aware test generation includes cohesion tests."""
    
    mock_response = {
        "test_files": [
            {
                "path": "tests/test_order_service.py",
                "content": """import pytest
from src.order_service import OrderService

def test_order_service_handles_only_order_logic():
    '''Cohesion test: verify OrderService focuses on orders only.'''
    service = OrderService()
    assert hasattr(service, 'create_order')
    assert hasattr(service, 'update_order')
    assert not hasattr(service, 'send_email')  # Should not have email logic

def test_order_service_accepts_repository_injection():
    '''DI test: verify repository is injected, not hardcoded.'''
    mock_repo = Mock()
    service = OrderService(repository=mock_repo)
    assert service.repository is mock_repo
"""
            }
        ]
    }
    
    agent = TestGenerationAgent(llm=DummyLLM(mock_response))
    
    model_ir = {
        "classes": [
            {"name": "OrderService", "methods": [], "attributes": []}
        ]
    }
    
    generated_code = {
        "files": [
            {"path": "src/order_service.py", "content": "class OrderService: pass"}
        ]
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderManager was a god class",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP",
                "category": "solid"
            }
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        analysis_report=analysis_report
    )
    
    assert result["analysis_aware"] is True
    assert result["total_tests"] == 2
    assert "cohesion" in result["test_categories"]


def test_build_system_prompt_includes_cohesion_tests():
    """Test that system prompt includes cohesion test instructions."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    test_strategy = {
        "categories": ["unit", "cohesion"],
        "focus_areas": ["cohesion"]
    }
    
    prompt = agent._build_system_prompt("pytest", test_strategy)
    
    assert "COHESION TESTS" in prompt
    assert "focused, single responsibilities" in prompt.lower()
    assert "separation of concerns" in prompt.lower()


def test_build_system_prompt_includes_dependency_injection_tests():
    """Test that system prompt includes DI test instructions."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    test_strategy = {
        "categories": ["dependency_injection"],
        "focus_areas": ["coupling"]
    }
    
    prompt = agent._build_system_prompt("pytest", test_strategy)
    
    assert "DEPENDENCY INJECTION TESTS" in prompt
    assert "accept dependencies" in prompt.lower()
    assert "mock injected dependencies" in prompt.lower()


def test_build_system_prompt_includes_interface_tests():
    """Test that system prompt includes interface test instructions."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    test_strategy = {
        "categories": ["interface"],
        "focus_areas": ["abstraction"]
    }
    
    prompt = agent._build_system_prompt("pytest", test_strategy)
    
    assert "INTERFACE TESTS" in prompt
    assert "interface implementations" in prompt.lower()
    assert "polymorphic behavior" in prompt.lower()


def test_build_user_message_includes_analysis_context():
    """Test that user message includes analysis findings for context."""
    
    agent = TestGenerationAgent(llm=DummyLLM({}))
    
    model_ir = {"classes": [{"name": "OrderService"}]}
    generated_code = {"files": [{"path": "src/order_service.py", "content": "pass"}]}
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "God class",
                "affected_entities": ["OrderManager"]
            }
        ],
        "recommendations": [{"title": "Split class"}]
    }
    
    test_strategy = {
        "categories": ["cohesion"],
        "god_classes": ["OrderManager"],
        "missing_abstractions": [],
        "dip_violations": []
    }
    
    message = agent._build_user_message(
        model_ir,
        generated_code,
        analysis_report,
        test_strategy,
        include_integration_tests=True
    )
    
    assert "ANALYSIS REPORT" in message
    assert "1 design issues detected" in message
    assert "GOD CLASSES THAT WERE SPLIT" in message
    assert "OrderManager" in message


def test_error_handling_on_llm_failure():
    """Test that agent handles LLM failures gracefully."""
    
    class FailingLLM:
        def invoke(self, prompt):
            raise Exception("LLM API error")
    
    agent = TestGenerationAgent(llm=FailingLLM())
    
    result = agent.generate_tests(
        model_ir={"classes": []},
        generated_code={"files": []},
        framework="pytest"
    )
    
    assert "error" in result
    assert result["total_tests"] == 0
    assert result["test_files"] == []


# ============================================================================
# Integration Tests with Real LLM
# ============================================================================

@pytest.mark.integration
def test_real_llm_simple_test_generation():
    """Integration test: Generate tests for simple Calculator class."""
    
    agent = TestGenerationAgent()  # Uses real LLM
    
    model_ir = {
        "classes": [
            {
                "name": "Calculator",
                "methods": [
                    {"name": "add", "params": ["a", "b"], "returns": "int"},
                    {"name": "subtract", "params": ["a", "b"], "returns": "int"},
                    {"name": "multiply", "params": ["a", "b"], "returns": "int"}
                ],
                "attributes": []
            }
        ]
    }
    
    generated_code = {
        "files": [
            {
                "path": "src/calculator.py",
                "content": """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
"""
            }
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code
    )
    
    # Assertions
    assert "test_files" in result
    assert len(result["test_files"]) >= 1, "Should generate at least one test file"
    assert result["framework"] == "pytest"
    assert result["total_tests"] >= 3, "Should generate at least 3 tests (one per method)"
    
    # Check test file structure
    test_file = result["test_files"][0]
    assert "path" in test_file
    assert "content" in test_file
    assert "test_" in test_file["path"].lower()
    
    # Check test content
    content = test_file["content"]
    assert "def test_" in content, "Should contain test functions"
    assert "import pytest" in content or "import unittest" in content
    assert "Calculator" in content


@pytest.mark.integration
def test_real_llm_analysis_aware_test_generation():
    """Integration test: Generate analysis-aware tests for refactored OrderManager."""
    
    agent = TestGenerationAgent()
    
    model_ir = {
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
        ]
    }
    
    generated_code = {
        "files": [
            {
                "path": "src/order_service.py",
                "content": """class OrderService:
    def __init__(self, repository):
        self.repository = repository
    
    def create_order(self, user_id, items):
        order = Order(user_id, items)
        self.repository.save(order)
        return order
"""
            },
            {
                "path": "src/order_repository.py",
                "content": """class OrderRepository:
    def save(self, order):
        # Save to database
        return True
"""
            }
        ]
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderManager was a god class combining order logic, email, and persistence",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP",
                "category": "solid"
            }
        ],
        "recommendations": [
            {
                "title": "Split OrderManager",
                "description": "Separated into OrderService and OrderRepository",
                "affected_entities": ["OrderManager"]
            }
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        analysis_report=analysis_report
    )
    
    # Assertions
    assert result["analysis_aware"] is True
    assert len(result["test_files"]) >= 1
    assert result["total_tests"] >= 2, "Should generate multiple tests"
    assert "cohesion" in result["test_categories"]
    
    # Check that tests mention cohesion/separation
    content = result["test_files"][0]["content"]
    assert "def test_" in content


@pytest.mark.integration
def test_real_llm_interface_test_generation():
    """Integration test: Generate interface implementation tests."""
    
    agent = TestGenerationAgent()
    
    model_ir = {
        "classes": [
            {
                "name": "IPaymentProcessor",
                "methods": [
                    {"name": "process_payment", "params": ["amount"], "returns": "bool"}
                ],
                "attributes": []
            },
            {
                "name": "PayPalProcessor",
                "methods": [
                    {"name": "process_payment", "params": ["amount"], "returns": "bool"}
                ],
                "attributes": []
            },
            {
                "name": "StripeProcessor",
                "methods": [
                    {"name": "process_payment", "params": ["amount"], "returns": "bool"}
                ],
                "attributes": []
            }
        ]
    }
    
    generated_code = {
        "files": [
            {
                "path": "src/payment.py",
                "content": """from abc import ABC, abstractmethod

class IPaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass

class PayPalProcessor(IPaymentProcessor):
    def process_payment(self, amount):
        return True

class StripeProcessor(IPaymentProcessor):
    def process_payment(self, amount):
        return True
"""
            }
        ]
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "warning",
                "issue": "Missing abstraction for payment processors",
                "affected_entities": ["PayPalProcessor", "StripeProcessor"],
                "violated_principle": "",
                "category": "pattern"
            }
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        analysis_report=analysis_report
    )
    
    # Assertions
    assert result["analysis_aware"] is True
    assert len(result["test_files"]) >= 1
    assert "interface" in result["test_categories"]
    
    # Check content mentions interface or polymorphism
    content = result["test_files"][0]["content"]
    assert "IPaymentProcessor" in content or "PayPal" in content or "Stripe" in content


@pytest.mark.integration
def test_real_llm_integration_test_generation():
    """Integration test: Generate integration tests for multi-component workflow."""
    
    agent = TestGenerationAgent()
    
    model_ir = {
        "classes": [
            {"name": "OrderService", "methods": [], "attributes": []},
            {"name": "OrderRepository", "methods": [], "attributes": []},
            {"name": "EmailService", "methods": [], "attributes": []}
        ]
    }
    
    generated_code = {
        "files": [
            {"path": "src/order_service.py", "content": "class OrderService: pass"},
            {"path": "src/order_repository.py", "content": "class OrderRepository: pass"},
            {"path": "src/email_service.py", "content": "class EmailService: pass"}
        ]
    }
    
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        include_integration_tests=True
    )
    
    # Assertions
    assert len(result["test_files"]) >= 1
    assert "integration" in result["test_categories"]
    assert result["total_tests"] >= 1
