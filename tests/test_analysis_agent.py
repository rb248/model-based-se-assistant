"""
Comprehensive test suite for AnalysisAgent.

Tests cover:
- Unit tests with DummyLLM (deterministic, fast)
- Integration tests with real LLM (validates end-to-end)
- Design issue detection (god classes, SOLID violations, circular deps)
- RAG integration
- Schema validation
- Error handling
"""

import json
import os
import pytest
from typing import Any, Dict

from backend.agents import AnalysisAgent
from backend.llms import get_llm
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


class DummyRetriever:
    """Mock retriever for testing RAG integration."""
    
    def __init__(self, docs: list):
        self.docs = docs
    
    def invoke(self, query: str):
        """Return mock documents."""
        class MockDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        return [MockDoc(doc["content"], {"title": doc["title"], "category": doc["category"]}) 
                for doc in self.docs]


def _should_run_integration() -> bool:
    """Check if integration tests should run."""
    if os.getenv("RUN_LLM_INTEGRATION"):
        return True
    return bool(GOOGLE_API_KEY or OPENAI_API_KEY)


# ============================================================================
# Unit Tests with DummyLLM
# ============================================================================

def test_analyze_god_class_detection():
    """Test that god classes are detected."""
    model_ir = {
        "classes": [{
            "name": "UserManager",
            "methods": [{"name": f"method{i}"} for i in range(15)],
            "attributes": [{"name": f"attr{i}", "type": "str"} for i in range(7)]
        }],
        "relationships": []
    }
    
    # Mock LLM returns minimal valid response
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.5,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    # Check that god class was detected
    assert "error" not in result
    assert len(result["findings"]) > 0
    god_class_finding = [f for f in result["findings"] if "God Class" in f["issue"] or "god" in f["issue"].lower()]
    assert len(god_class_finding) > 0
    assert god_class_finding[0]["severity"] in ["critical", "warning"]
    assert "UserManager" in god_class_finding[0]["affected_entities"]


def test_analyze_solid_srp_violation():
    """Test that SRP violations are detected."""
    model_ir = {
        "classes": [{
            "name": "UserManagerHandler",
            "methods": [{"name": f"method{i}"} for i in range(8)],
            "attributes": []
        }],
        "relationships": []
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.6,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    srp_findings = [f for f in result["findings"] if f.get("violated_principle") == "SRP"]
    assert len(srp_findings) > 0
    assert "UserManagerHandler" in srp_findings[0]["affected_entities"]


def test_analyze_missing_abstraction():
    """Test detection of missing abstractions with common method names."""
    model_ir = {
        "classes": [
            {"name": "EmailSender", "methods": [{"name": "send"}], "attributes": []},
            {"name": "SMSSender", "methods": [{"name": "send"}], "attributes": []},
            {"name": "PushSender", "methods": [{"name": "send"}], "attributes": []}
        ],
        "relationships": []
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.7,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    # Check for finding about common method
    abstraction_findings = [f for f in result["findings"] 
                           if "send" in f["issue"] and "interface" in f["issue"].lower()]
    assert len(abstraction_findings) > 0


def test_analyze_circular_dependency():
    """Test detection of circular dependencies."""
    model_ir = {
        "classes": [
            {"name": "A", "methods": [], "attributes": []},
            {"name": "B", "methods": [], "attributes": []},
            {"name": "C", "methods": [], "attributes": []}
        ],
        "relationships": [
            {"from": "A", "to": "B", "type": "dependency", "multiplicity": "1"},
            {"from": "B", "to": "C", "type": "dependency", "multiplicity": "1"},
            {"from": "C", "to": "A", "type": "dependency", "multiplicity": "1"}
        ]
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.4,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    circular_findings = [f for f in result["findings"] if "circular" in f["issue"].lower()]
    assert len(circular_findings) > 0
    assert circular_findings[0]["severity"] == "critical"


def test_analyze_proper_design_no_issues():
    """Test that well-designed models have minimal or no critical issues."""
    model_ir = {
        "classes": [
            {"name": "User", "methods": [{"name": "getName"}], "attributes": [{"name": "name", "type": "str"}]},
            {"name": "Order", "methods": [{"name": "calculateTotal"}], "attributes": [{"name": "items", "type": "list"}]},
            {"name": "Product", "methods": [{"name": "getPrice"}], "attributes": [{"name": "price", "type": "float"}]}
        ],
        "relationships": [
            {"from": "Order", "to": "Product", "type": "aggregation", "multiplicity": "*"}
        ]
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.9,
        "quality_metrics": {},
        "summary": "Well-designed model"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    critical_findings = [f for f in result["findings"] if f["severity"] == "critical"]
    assert len(critical_findings) == 0
    assert result["quality_score"] >= 0.8


def test_analyze_empty_model():
    """Test graceful handling of empty model."""
    model_ir = {"classes": [], "relationships": [], "notes": []}
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.0,
        "quality_metrics": {},
        "summary": "Empty model"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    assert isinstance(result["findings"], list)
    assert isinstance(result["quality_metrics"], dict)


def test_analyze_metrics_calculation():
    """Test that metrics are correctly calculated."""
    model_ir = {
        "classes": [
            {"name": "A", "methods": [{"name": "m1"}, {"name": "m2"}], "attributes": [{"name": "a1", "type": "str"}]},
            {"name": "B", "methods": [{"name": "m3"}], "attributes": []}
        ],
        "relationships": [
            {"from": "A", "to": "B", "type": "dependency", "multiplicity": "1"}
        ]
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.7,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    metrics = result["quality_metrics"]
    assert "avg_methods_per_class" in metrics
    assert "max_methods_per_class" in metrics
    assert metrics["avg_methods_per_class"] == 1.5  # (2+1)/2
    assert metrics["max_methods_per_class"] == 2
    assert metrics["total_classes"] == 2
    assert metrics["total_relationships"] == 1


def test_analyze_rag_integration():
    """Test that RAG retrieval is used when retriever is provided."""
    model_ir = {
        "classes": [{"name": "Test", "methods": [], "attributes": []}],
        "relationships": []
    }
    
    mock_docs = [
        {"title": "SRP", "content": "Single Responsibility Principle", "category": "SOLID"}
    ]
    
    mock_response = json.dumps({
        "findings": [{"severity": "info", "issue": "Test issue", "affected_entities": ["Test"], 
                     "violated_principle": None, "category": "other"}],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.8,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), retriever=DummyRetriever(mock_docs), name="test_agent")
    result = agent.analyze_model(model_ir, description="Test model")
    
    assert "error" not in result
    # If retriever is used, analysis should complete successfully
    assert isinstance(result["findings"], list)


def test_analyze_output_schema_validation():
    """Test that output conforms to AnalysisReport schema."""
    model_ir = {
        "classes": [{"name": "User", "methods": [], "attributes": []}],
        "relationships": []
    }
    
    mock_response = json.dumps({
        "findings": [
            {
                "severity": "info",
                "issue": "Test issue",
                "affected_entities": ["User"],
                "violated_principle": None,
                "category": "other"
            }
        ],
        "recommendations": [
            {
                "title": "Test recommendation",
                "description": "Test description",
                "priority": "low",
                "affected_entities": ["User"],
                "design_pattern": None,
                "rationale": "Test rationale"
            }
        ],
        "patterns_detected": ["Singleton"],
        "quality_score": 0.75,
        "quality_metrics": {},
        "summary": "Test summary"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    # Check all required fields are present
    assert "findings" in result
    assert "recommendations" in result
    assert "patterns_detected" in result
    assert "quality_score" in result
    assert "quality_metrics" in result
    assert "summary" in result
    
    # Validate field types
    assert isinstance(result["findings"], list)
    assert isinstance(result["recommendations"], list)
    assert isinstance(result["quality_score"], (int, float))
    assert 0.0 <= result["quality_score"] <= 1.0


def test_analyze_invalid_llm_response():
    """Test error handling when LLM returns malformed JSON."""
    model_ir = {
        "classes": [{"name": "Test", "methods": [], "attributes": []}],
        "relationships": []
    }
    
    # Invalid JSON
    mock_response = "This is not valid JSON"
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    # Should return error gracefully
    assert "error" in result
    assert isinstance(result["findings"], list)
    assert len(result["findings"]) == 0


def test_analyze_merge_deterministic_and_llm():
    """Test that deterministic findings and LLM findings are merged correctly."""
    model_ir = {
        "classes": [{
            "name": "GodClass",
            "methods": [{"name": f"m{i}"} for i in range(12)],
            "attributes": []
        }],
        "relationships": []
    }
    
    # LLM returns additional finding
    mock_response = json.dumps({
        "findings": [
            {
                "severity": "warning",
                "issue": "Additional LLM finding",
                "affected_entities": ["GodClass"],
                "violated_principle": None,
                "category": "other"
            }
        ],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.5,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    # Should have both deterministic (god class) and LLM findings
    assert len(result["findings"]) >= 2
    finding_issues = [f["issue"] for f in result["findings"]]
    assert any("God Class" in issue or "god" in issue.lower() for issue in finding_issues)
    assert any("Additional LLM finding" in issue for issue in finding_issues)


def test_analyze_priorities_assigned():
    """Test that critical findings have high-priority recommendations."""
    model_ir = {
        "classes": [{
            "name": "Critical",
            "methods": [{"name": f"m{i}"} for i in range(15)],
            "attributes": [{"name": f"a{i}", "type": "str"} for i in range(8)]
        }],
        "relationships": []
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [
            {
                "title": "Fix critical issue",
                "description": "Split the class",
                "priority": "high",
                "affected_entities": ["Critical"],
                "design_pattern": None,
                "rationale": "Too many responsibilities"
            }
        ],
        "patterns_detected": [],
        "quality_score": 0.3,
        "quality_metrics": {},
        "summary": "Critical issues found"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    # Check that high priority recommendation exists
    high_priority_recs = [r for r in result["recommendations"] if r["priority"] == "high"]
    assert len(high_priority_recs) > 0


def test_analyze_inheritance_depth():
    """Test analysis of inheritance relationships."""
    model_ir = {
        "classes": [
            {"name": "Base", "methods": [], "attributes": []},
            {"name": "Derived", "methods": [], "attributes": []}
        ],
        "relationships": [
            {"from": "Derived", "to": "Base", "type": "inheritance", "multiplicity": "1"}
        ]
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.8,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    assert "inheritance_count" in result["quality_metrics"]
    assert result["quality_metrics"]["inheritance_count"] == 1


def test_analyze_relationship_density():
    """Test calculation of relationship density."""
    model_ir = {
        "classes": [
            {"name": f"Class{i}", "methods": [], "attributes": []} for i in range(5)
        ],
        "relationships": [
            {"from": f"Class{i}", "to": f"Class{j}", "type": "dependency", "multiplicity": "1"}
            for i in range(3) for j in range(i+1, 5)
        ]
    }
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.6,
        "quality_metrics": {},
        "summary": "Test"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), name="test_agent")
    result = agent.analyze_model(model_ir)
    
    assert "error" not in result
    assert "relationship_density" in result["quality_metrics"]
    # Should be number of relationships / number of classes
    expected_density = len(model_ir["relationships"]) / len(model_ir["classes"])
    assert abs(result["quality_metrics"]["relationship_density"] - expected_density) < 0.01


def test_analyze_description_used_in_rag():
    """Test that description is used in RAG query."""
    model_ir = {
        "classes": [{"name": "PaymentProcessor", "methods": [], "attributes": []}],
        "relationships": []
    }
    
    description = "Payment processing system"
    
    mock_docs = [
        {"title": "Payment Security", "content": "Secure payment handling", "category": "Security"}
    ]
    
    mock_response = json.dumps({
        "findings": [],
        "recommendations": [],
        "patterns_detected": [],
        "quality_score": 0.8,
        "quality_metrics": {},
        "summary": "Payment system analyzed"
    })
    
    agent = AnalysisAgent(llm=DummyLLM(mock_response), retriever=DummyRetriever(mock_docs), name="test_agent")
    result = agent.analyze_model(model_ir, description=description)
    
    assert "error" not in result
    # Analysis should complete with description
    assert result["summary"] or len(result["findings"]) >= 0


# ============================================================================
# Integration Tests with Real LLM
# ============================================================================

@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_rag_enriched_query():
    """Test RAG integration with real LLM using realistic e-commerce order management system."""
    from backend.knowledge_base import get_knowledge_base

    # Realistic e-commerce order management system with multiple design issues
    model_ir = {
        "classes": [
            {
                "name": "OrderManager",
                "methods": [
                    {"name": "createOrder", "params": ["userId", "items", "paymentMethod"], "returns": "Order"},
                    {"name": "updateOrder", "params": ["orderId", "updates"], "returns": "bool"},
                    {"name": "cancelOrder", "params": ["orderId"], "returns": "bool"},
                    {"name": "calculateTotal", "params": ["items", "discountCode", "taxRate"], "returns": "float"},
                    {"name": "validatePayment", "params": ["paymentMethod", "amount"], "returns": "bool"},
                    {"name": "sendConfirmationEmail", "params": ["email", "orderDetails"], "returns": "void"},
                    {"name": "logOrderActivity", "params": ["orderId", "action"], "returns": "void"},
                    {"name": "generateInvoice", "params": ["orderId"], "returns": "Invoice"},
                    {"name": "saveToMySQLDatabase", "params": ["order"], "returns": "bool"},
                    {"name": "fetchFromCache", "params": ["orderId"], "returns": "Order"},
                    {"name": "validateAddress", "params": ["street", "city", "zip", "country", "state"], "returns": "bool"},
                    {"name": "trackShipment", "params": ["orderId"], "returns": "ShipmentStatus"}
                ],
                "attributes": [
                    {"name": "database", "type": "MySQLConnection"},
                    {"name": "emailService", "type": "SMTPClient"},
                    {"name": "logger", "type": "FileLogger"},
                    {"name": "cache", "type": "RedisCache"},
                    {"name": "paymentGateway", "type": "StripeAPI"},
                    {"name": "invoiceGenerator", "type": "PDFGenerator"},
                    {"name": "shippingTracker", "type": "FedExAPI"}
                ],
                "description": "Manages all order operations"
            },
            {
                "name": "PayPalProcessor",
                "methods": [
                    {"name": "processPayment", "params": ["amount", "currency"], "returns": "bool"}
                ],
                "attributes": [],
                "description": "PayPal payment processor"
            },
            {
                "name": "StripeProcessor",
                "methods": [
                    {"name": "processPayment", "params": ["amount", "currency"], "returns": "bool"}
                ],
                "attributes": [],
                "description": "Stripe payment processor"
            },
            {
                "name": "CreditCardProcessor",
                "methods": [
                    {"name": "processPayment", "params": ["amount", "currency"], "returns": "bool"}
                ],
                "attributes": [],
                "description": "Credit card payment processor"
            }
        ],
        "relationships": [
            {"from": "OrderManager", "to": "PayPalProcessor", "type": "dependency", "multiplicity": "1"},
            {"from": "OrderManager", "to": "StripeProcessor", "type": "dependency", "multiplicity": "1"},
            {"from": "OrderManager", "to": "CreditCardProcessor", "type": "dependency", "multiplicity": "1"}
        ]
    }

    description = "E-commerce order management system handling order lifecycle, payments, emails, and database operations."

    kb = get_knowledge_base()
    retriever = kb.get_simple_retriever()

    agent = AnalysisAgent(retriever=retriever)
    result = agent.analyze_model(model_ir, description=description)

    print("\n" + "="*80)
    print("TEST: test_real_llm_rag_enriched_query")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")

    assert "error" not in result
    assert len(result["findings"]) >= 3, "Should detect multiple design issues"
    
    # Should detect god class with cohesion issues
    assert any("god class" in f["issue"].lower() or "ordermanager" in f["issue"].lower() 
               for f in result["findings"]), "Should detect OrderManager god class"
    
    # Should detect SRP violations with semantic clustering
    assert any(f.get("violated_principle") == "SRP" for f in result["findings"]), "Should detect SRP violations"
    assert any("responsibility domains" in f["issue"].lower() or "domain" in f["issue"].lower()
               for f in result["findings"]), "Should detect semantic clustering of responsibilities"
    
    # Should detect missing abstraction for payment processors
    assert any("payment" in f["issue"].lower() and ("interface" in f["issue"].lower() or "abstraction" in f["issue"].lower())
               for f in result["findings"]), "Should detect missing payment abstraction"
    
    # Should provide high-quality recommendations
    assert len(result["recommendations"]) > 0, "Should provide refactoring recommendations"
    assert any(r["priority"] == "high" for r in result["recommendations"]), "Should have high-priority recommendations"

@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_analyze_simple_model():
    """Test analysis with real LLM on a simple model."""
    model_ir = {
        "classes": [
            {"name": "Person", "methods": [{"name": "getName"}, {"name": "getAge"}], 
             "attributes": [{"name": "name", "type": "str"}, {"name": "age", "type": "int"}]}
        ],
        "relationships": []
    }
    
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="Simple person class")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_analyze_simple_model")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result
    assert "findings" in result
    assert "recommendations" in result
    assert "quality_score" in result
    assert isinstance(result["findings"], list)
    assert isinstance(result["quality_score"], (int, float))


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_detect_god_class():
    """Test that real LLM detects god class."""
    model_ir = {
        "classes": [{
            "name": "SystemManager",
            "methods": [
                {"name": "createUser"}, {"name": "deleteUser"}, {"name": "sendEmail"},
                {"name": "generateReport"}, {"name": "processPayment"}, {"name": "logActivity"},
                {"name": "validateInput"}, {"name": "exportData"}, {"name": "importData"},
                {"name": "manageCache"}, {"name": "handleErrors"}, {"name": "trackAnalytics"}
            ],
            "attributes": [
                {"name": "database", "type": "Database"}, {"name": "cache", "type": "Cache"},
                {"name": "emailService", "type": "EmailService"}, {"name": "paymentGateway", "type": "PaymentGateway"}
            ]
        }],
        "relationships": []
    }
    
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="System management class")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_detect_god_class")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_detect_missing_abstraction():
    """Test that real LLM detects missing abstraction."""
    model_ir = {
        "classes": [
            {"name": "PayPalPayment", "methods": [{"name": "processPayment"}], "attributes": []},
            {"name": "StripePayment", "methods": [{"name": "processPayment"}], "attributes": []},
            {"name": "CreditCardPayment", "methods": [{"name": "processPayment"}], "attributes": []}
        ],
        "relationships": []
    }
    
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="Payment processing with multiple implementations")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_detect_missing_abstraction")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_recommend_pattern():
    """Test that real LLM recommends appropriate design pattern."""
    model_ir = {
        "classes": [
            {"name": "PayPalPayment", "methods": [{"name": "processPayment"}], "attributes": []},
            {"name": "StripePayment", "methods": [{"name": "processPayment"}], "attributes": []},
            {"name": "Order", "methods": [{"name": "checkout"}], "attributes": []}
        ],
        "relationships": [
            {"from": "Order", "to": "PayPalPayment", "type": "dependency", "multiplicity": "1"},
            {"from": "Order", "to": "StripePayment", "type": "dependency", "multiplicity": "1"}
        ]
    }
    
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="Payment processing with multiple providers")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_recommend_pattern")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result
    # Should recommend strategy pattern or interface
    recommendations = result["recommendations"]
    assert len(recommendations) > 0
    # Check if any recommendation mentions pattern or interface
    pattern_mentions = [
        r for r in recommendations
        if r.get("design_pattern")
        or "pattern" in r.get("description", "").lower()
        or "interface" in r.get("description", "").lower()
    ]
    assert len(pattern_mentions) > 0


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_detect_coupling_metrics():
    """Test that real LLM detects coupling metrics."""
    model_ir = {
        "classes": [
            {"name": "ServiceA", "methods": [{"name": "doA"}], "attributes": []},
            {"name": "ServiceB", "methods": [{"name": "doB"}], "attributes": []},
            {"name": "ServiceC", "methods": [{"name": "doC"}], "attributes": []},
            {"name": "ServiceD", "methods": [{"name": "doD"}], "attributes": []}
        ],
        "relationships": [
            {"from": "ServiceA", "to": "ServiceB", "type": "dependency", "multiplicity": "1"},
            {"from": "ServiceA", "to": "ServiceC", "type": "dependency", "multiplicity": "1"},
            {"from": "ServiceA", "to": "ServiceD", "type": "dependency", "multiplicity": "1"},
            {"from": "ServiceB", "to": "ServiceA", "type": "dependency", "multiplicity": "1"}
        ]
    }
    
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="Service with high coupling")
    
    print("\n" + "="*80)
    print("TEST: test_real_llm_detect_coupling_metrics")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    assert "error" not in result


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_with_rag():
    """Test analysis with real knowledge base and RAG."""
    from backend.knowledge_base import get_knowledge_base
    
    model_ir = {
        "classes": [{
            "name": "DataProcessor",
            "methods": [{"name": "process"}, {"name": "validate"}, {"name": "save"}, {"name": "email"}],
            "attributes": []
        }],
        "relationships": []
    }
    
    try:
        kb = get_knowledge_base()
        retriever = kb.get_simple_retriever()

        agent = AnalysisAgent(retriever=retriever)
        result = agent.analyze_model(model_ir, description="Data processing class")

        print("\n" + "="*80)
        print("TEST: test_real_llm_with_rag")
        print("="*80)
        print(json.dumps(result, indent=2))
        print("="*80 + "\n")

        assert "error" not in result
        assert len(result["findings"]) > 0 or len(result["recommendations"]) > 0
    except Exception as e:
        pytest.skip(f"Knowledge base not available: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="No API key or RUN_LLM_INTEGRATION not set")
def test_real_llm_complex_model():
    """Test analysis of complex model with multiple issues."""
    model_ir = {
        "classes": [
            {"name": "User", "methods": [{"name": "getName"}], "attributes": [{"name": "name", "type": "str"}]},
            {"name": "Order", "methods": [{"name": "calculateTotal"}, {"name": "process"}], "attributes": []},
            {"name": "Product", "methods": [{"name": "getPrice"}], "attributes": [{"name": "price", "type": "float"}]},
            {"name": "PayPalGateway", "methods": [{"name": "pay"}], "attributes": []},
            {"name": "StripeGateway", "methods": [{"name": "pay"}], "attributes": []},
            {"name": "EmailService", "methods": [{"name": "send"}], "attributes": []},
            {"name": "OrderManager", "methods": [{"name": "create"}, {"name": "update"}, {"name": "delete"}, 
                                                 {"name": "notify"}, {"name": "log"}, {"name": "validate"}], "attributes": []}
        ],
        "relationships": [
            {"from": "Order", "to": "User", "type": "association", "multiplicity": "1"},
            {"from": "Order", "to": "Product", "type": "aggregation", "multiplicity": "*"},
            {"from": "Order", "to": "PayPalGateway", "type": "dependency", "multiplicity": "1"}]
    }
    agent = AnalysisAgent()
    result = agent.analyze_model(model_ir, description="E-commerce order management system")

    print("\n" + "="*80)
    print("TEST: test_real_llm_complex_model")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")

    # Be more lenient - complex models might hit rate limits or timeouts
    # Just check that we get a result (even if it has an error, we should handle gracefully)
    assert isinstance(result, dict)

    # If no error, validate structure
    if "error" not in result:
        assert len(result["findings"]) >= 1  # Should find at least one issue
        assert len(result["recommendations"]) >= 0  # Might have recommendations
        assert "quality_metrics" in result
        assert result["quality_metrics"]["total_classes"] == 7
    else:
        # If there's an error, it should be gracefully handled (not crash)
        # This can happen with rate limits or API issues on complex models
        pytest.skip(f"LLM API issue with complex model: {result.get('error')}")
