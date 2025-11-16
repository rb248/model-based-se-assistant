"""
Comprehensive test suite for CodeGenerationAgent.

Tests cover:
- Unit tests with DummyLLM (deterministic, fast)
- Integration tests with real LLM (validates end-to-end)
- Analysis-aware code generation
- Refactoring application (god classes, missing abstractions, DIP violations)
- Code quality validation
"""

import json
import os
import pytest
from typing import Any, Dict

from backend.agents import CodeGenerationAgent
from backend.config import GOOGLE_API_KEY, OPENAI_API_KEY


# ============================================================================
# Test Helpers
# ============================================================================

class DummyLLM:
    """Mock LLM that returns pre-configured responses for testing."""
    
    def __init__(self, response: str):
        self.response = response
    
    def __call__(self, prompt: str) -> str:
        return self.response


def _should_run_integration() -> bool:
    """Check if integration tests should run."""
    if os.getenv("RUN_LLM_INTEGRATION"):
        return True
    return bool(GOOGLE_API_KEY or OPENAI_API_KEY)


# ============================================================================
# Unit Tests with DummyLLM
# ============================================================================

def test_extract_refactoring_opportunities():
    """Test extraction of refactoring opportunities from analysis report."""
    analysis_report = {
        "findings": [
            {
                "issue": "Class 'OrderManager' is a God Class with 12 methods",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP"
            },
            {
                "issue": "Multiple classes implement method 'processPayment'. Consider introducing a common interface",
                "affected_entities": ["PayPalProcessor", "StripeProcessor", "CreditCardProcessor"],
                "violated_principle": None
            },
            {
                "issue": "OrderManager depends on concrete implementation 'MySQLDatabase'",
                "affected_entities": ["OrderManager", "MySQLDatabase"],
                "violated_principle": "DIP"
            }
        ]
    }
    
    agent = CodeGenerationAgent(name="test_agent")
    opportunities = agent._extract_refactoring_opportunities(analysis_report)
    
    assert "OrderManager" in opportunities["god_classes"]
    assert len(opportunities["missing_abstractions"]) > 0
    assert any("PayPalProcessor" in ab["entities"] for ab in opportunities["missing_abstractions"])
    assert len(opportunities["dip_violations"]) > 0


def test_generate_code_simple_model():
    """Test basic code generation for simple model."""
    model_ir = {
        "classes": [
            {
                "name": "User",
                "methods": [{"name": "getName", "params": [], "returns": "str"}],
                "attributes": [{"name": "name", "type": "str"}]
            }
        ],
        "relationships": []
    }
    
    mock_response = json.dumps({
        "files": [
            {
                "path": "user.py",
                "content": "class User:\n    def __init__(self, name: str):\n        self.name = name\n    \n    def getName(self) -> str:\n        return self.name"
            }
        ]
    })
    
    agent = CodeGenerationAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.generate_code(model_ir, language="python")
    
    assert "error" not in result
    assert "files" in result
    assert len(result["files"]) == 1
    assert result["files"][0]["path"] == "user.py"
    assert "class User" in result["files"][0]["content"]


def test_generate_code_with_analysis_no_refactoring():
    """Test code generation with analysis but refactorings disabled."""
    model_ir = {
        "classes": [
            {
                "name": "OrderManager",
                "methods": [{"name": "createOrder", "params": [], "returns": "Order"}],
                "attributes": []
            }
        ],
        "relationships": []
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderManager is a God Class",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP"
            }
        ],
        "recommendations": []
    }
    
    mock_response = json.dumps({
        "files": [
            {
                "path": "order_manager.py",
                "content": "class OrderManager:\n    pass"
            }
        ]
    })
    
    agent = CodeGenerationAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.generate_code(model_ir, analysis_report=analysis_report, apply_refactorings=False)
    
    assert "error" not in result
    assert len(result["files"]) > 0


def test_generate_code_with_refactorings():
    """Test that refactoring instructions are built correctly."""
    model_ir = {
        "classes": [
            {
                "name": "OrderManager",
                "methods": [{"name": "createOrder"}, {"name": "sendEmail"}, {"name": "saveToDatabase"}],
                "attributes": []
            }
        ],
        "relationships": []
    }
    
    analysis_report = {
        "findings": [
            {
                "issue": "OrderManager is a God Class with multiple responsibilities",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP"
            }
        ],
        "recommendations": [
            {
                "title": "Split OrderManager into focused classes",
                "description": "Create OrderService, EmailService, OrderRepository",
                "priority": "high"
            }
        ]
    }
    
    mock_response = json.dumps({
        "files": [
            {
                "path": "order_service.py",
                "content": "class OrderService:\n    pass"
            },
            {
                "path": "email_service.py",
                "content": "class EmailService:\n    pass"
            },
            {
                "path": "order_repository.py",
                "content": "class OrderRepository:\n    pass"
            }
        ]
    })
    
    agent = CodeGenerationAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.generate_code(model_ir, analysis_report=analysis_report, apply_refactorings=True)
    
    assert "error" not in result
    assert len(result["files"]) >= 1


def test_build_refactoring_instructions_god_class():
    """Test that god class refactoring instructions are generated."""
    refactorings = {
        "god_classes": ["OrderManager"],
        "missing_abstractions": [],
        "dip_violations": [],
        "srp_violations": []
    }
    
    agent = CodeGenerationAgent(name="test_agent")
    instructions = agent._build_refactoring_instructions(refactorings, {})
    
    assert "GOD CLASS" in instructions
    assert "OrderManager" in instructions
    assert "Split" in instructions or "split" in instructions


def test_build_refactoring_instructions_missing_abstraction():
    """Test that missing abstraction instructions are generated."""
    refactorings = {
        "god_classes": [],
        "missing_abstractions": [
            {
                "entities": ["PayPalProcessor", "StripeProcessor"],
                "issue": "Missing payment interface"
            }
        ],
        "dip_violations": [],
        "srp_violations": []
    }
    
    agent = CodeGenerationAgent(name="test_agent")
    instructions = agent._build_refactoring_instructions(refactorings, {})
    
    assert "MISSING ABSTRACTION" in instructions
    assert "interface" in instructions.lower() or "abstract" in instructions.lower()


def test_build_refactoring_instructions_dip_violation():
    """Test that DIP violation instructions are generated."""
    refactorings = {
        "god_classes": [],
        "missing_abstractions": [],
        "dip_violations": [
            {
                "entities": ["OrderManager", "MySQLDatabase"],
                "issue": "Direct MySQL database coupling"
            }
        ],
        "srp_violations": []
    }
    
    agent = CodeGenerationAgent(name="test_agent")
    instructions = agent._build_refactoring_instructions(refactorings, {})
    
    assert "DIP VIOLATION" in instructions
    assert "IRepository" in instructions or "interface" in instructions.lower()


def test_generate_code_empty_model():
    """Test graceful handling of empty model."""
    model_ir = {"classes": [], "relationships": []}
    
    mock_response = json.dumps({"files": []})
    
    agent = CodeGenerationAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.generate_code(model_ir)
    
    assert "error" not in result
    assert isinstance(result["files"], list)


def test_generate_code_invalid_llm_response():
    """Test error handling when LLM returns invalid response."""
    model_ir = {
        "classes": [{"name": "Test", "methods": [], "attributes": []}],
        "relationships": []
    }
    
    # Invalid JSON
    mock_response = "This is not valid JSON"
    
    agent = CodeGenerationAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.generate_code(model_ir)
    
    assert "error" in result
    assert isinstance(result["files"], list)


# ============================================================================
# Integration Tests with Real LLM
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_simple_code_generation():
    """Test code generation with real LLM for simple model."""
    model_ir = {
        "classes": [
            {
                "name": "Product",
                "methods": [
                    {"name": "getPrice", "params": [], "returns": "float"},
                    {"name": "setPrice", "params": ["price"], "returns": "void"}
                ],
                "attributes": [
                    {"name": "name", "type": "str"},
                    {"name": "price", "type": "float"}
                ]
            }
        ],
        "relationships": []
    }
    
    agent = CodeGenerationAgent()
    result = agent.generate_code(model_ir, language="python")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_simple_code_generation")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result
    assert "files" in result
    assert len(result["files"]) > 0
    
    # Validate generated code structure
    for file in result["files"]:
        assert "path" in file
        assert "content" in file
        assert len(file["content"]) > 0


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_god_class_refactoring():
    """Test that god class is properly refactored with real LLM."""
    model_ir = {
        "classes": [
            {
                "name": "OrderManager",
                "methods": [
                    {"name": "createOrder", "params": ["userId", "items"], "returns": "Order"},
                    {"name": "sendEmail", "params": ["email", "message"], "returns": "void"},
                    {"name": "saveToDatabase", "params": ["order"], "returns": "bool"},
                    {"name": "logActivity", "params": ["message"], "returns": "void"},
                    {"name": "generateInvoice", "params": ["orderId"], "returns": "Invoice"}
                ],
                "attributes": [
                    {"name": "database", "type": "MySQLConnection"},
                    {"name": "emailService", "type": "SMTPClient"},
                    {"name": "logger", "type": "Logger"}
                ]
            }
        ],
        "relationships": []
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "critical",
                "issue": "OrderManager is a God Class with 5 methods spanning multiple responsibilities: data access, communication, business logic",
                "affected_entities": ["OrderManager"],
                "violated_principle": "SRP",
                "category": "solid"
            }
        ],
        "recommendations": [
            {
                "title": "Split OrderManager into focused classes",
                "description": "Create OrderService (business logic), EmailService (communication), OrderRepository (data access), LogService (logging)",
                "priority": "high",
                "affected_entities": ["OrderManager"],
                "design_pattern": "Repository, Service Layer",
                "rationale": "Improves cohesion and follows SRP"
            }
        ],
        "quality_score": 0.4
    }
    
    agent = CodeGenerationAgent()
    result = agent.generate_code(model_ir, analysis_report=analysis_report, apply_refactorings=True)
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_god_class_refactoring")
    print("="*80)
    print(json.dumps(result, indent=2)[:2000])
    print("="*80 + "\n")
    
    assert "error" not in result
    assert len(result["files"]) >= 2, "Should generate multiple files for split classes"
    
    # Check that files suggest proper separation (relaxed - accept any separation pattern)
    file_paths = [f["path"] for f in result["files"]]
    file_contents = " ".join([f["content"] for f in result["files"]])
    
    # Should have separated concerns - accept service, repository, interface, or models patterns
    has_separation = any(keyword in path.lower() for path in file_paths 
                        for keyword in ["service", "repository", "interface", "model"])
    assert has_separation, f"Should generate separated files, got: {file_paths}"


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_missing_abstraction():
    """Test that missing abstractions are properly created with real LLM."""
    model_ir = {
        "classes": [
            {
                "name": "PayPalProcessor",
                "methods": [{"name": "processPayment", "params": ["amount"], "returns": "bool"}],
                "attributes": []
            },
            {
                "name": "StripeProcessor",
                "methods": [{"name": "processPayment", "params": ["amount"], "returns": "bool"}],
                "attributes": []
            },
            {
                "name": "Order",
                "methods": [{"name": "checkout", "params": [], "returns": "void"}],
                "attributes": []
            }
        ],
        "relationships": [
            {"from": "Order", "to": "PayPalProcessor", "type": "dependency", "multiplicity": "1"},
            {"from": "Order", "to": "StripeProcessor", "type": "dependency", "multiplicity": "1"}
        ]
    }
    
    analysis_report = {
        "findings": [
            {
                "severity": "info",
                "issue": "Multiple classes (PayPalProcessor, StripeProcessor) implement method 'processPayment'. Consider introducing a common interface or abstract class.",
                "affected_entities": ["PayPalProcessor", "StripeProcessor"],
                "violated_principle": None,
                "category": "pattern"
            },
            {
                "severity": "warning",
                "issue": "Order depends on concrete implementations instead of abstraction (DIP violation)",
                "affected_entities": ["Order", "PayPalProcessor", "StripeProcessor"],
                "violated_principle": "DIP",
                "category": "solid"
            }
        ],
        "recommendations": [
            {
                "title": "Introduce PaymentProcessor interface",
                "description": "Create IPaymentProcessor interface. Have PayPal and Stripe implement it. Order should depend on interface.",
                "priority": "high",
                "design_pattern": "Strategy",
                "rationale": "Follows DIP, allows adding new payment methods without modifying Order"
            }
        ]
    }
    
    agent = CodeGenerationAgent()
    result = agent.generate_code(model_ir, analysis_report=analysis_report, apply_refactorings=True)
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_missing_abstraction")
    print("="*80)
    print(json.dumps(result, indent=2)[:2000])
    print("="*80 + "\n")
    
    assert "error" not in result
    assert len(result["files"]) > 0
    
    # Check for interface/protocol creation
    all_content = " ".join([f["content"] for f in result["files"]])
    assert ("interface" in all_content.lower() or 
            "protocol" in all_content.lower() or 
            "ABC" in all_content or
            "abstractmethod" in all_content.lower()), \
        "Should create interface/protocol for payment processors"


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_code_quality_validation():
    """Test that generated code meets quality standards."""
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
        ],
        "relationships": []
    }
    
    agent = CodeGenerationAgent()
    result = agent.generate_code(model_ir, language="python")
    
    assert "error" not in result
    assert len(result["files"]) > 0
    
    # Validate code quality
    for file in result["files"]:
        content = file["content"]
        
        # Should have class definition
        assert "class " in content, "Should contain class definition"
        
        # Should have proper Python structure
        assert "def " in content, "Should contain method definitions"
        
        # Check for quality indicators (at least some should be present)
        quality_indicators = [
            ":" in content,  # Type hints or proper syntax
            '"""' in content or "'''" in content,  # Docstrings
            "return" in content or "pass" in content  # Method bodies
        ]
        assert sum(quality_indicators) >= 2, "Should have quality code indicators"
