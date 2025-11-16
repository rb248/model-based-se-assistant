"""
Debug script for TestGenerationAgent with complex model (like the full workflow).
"""
import json
import logging
import sys
sys.path.insert(0, '/Users/rishubbhatia/projects/llm_engineering')

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from backend.agents import TestGenerationAgent
from backend.llms import create_base_llm

# Complex model IR (like OrderManager from full workflow)
model_ir = {
    "classes": [
        {
            "name": "OrderManager",
            "methods": [
                {"name": "createOrder", "params": ["userId", "items"]},
                {"name": "calculateTotal", "params": ["order"]},
                {"name": "sendConfirmationEmail", "params": ["order"]},
                {"name": "saveToDatabase", "params": ["order"]},
                {"name": "logActivity", "params": ["action"]},
                {"name": "generateInvoice", "params": ["order"]}
            ],
            "attributes": [
                {"name": "database", "type": "Database"},
                {"name": "emailService", "type": "EmailService"},
                {"name": "logger", "type": "Logger"}
            ]
        },
        {
            "name": "Database",
            "methods": [{"name": "save", "params": ["data"]}],
            "attributes": []
        },
        {
            "name": "EmailService",
            "methods": [{"name": "send", "params": ["to", "subject", "body"]}],
            "attributes": []
        }
    ],
    "relationships": [
        {"from": "OrderManager", "to": "Database", "type": "dependency"},
        {"from": "OrderManager", "to": "EmailService", "type": "dependency"}
    ]
}

# Generated code (simulated refactored code)
generated_code = {
    "files": [
        {
            "path": "models.py",
            "content": """from dataclasses import dataclass
from typing import List

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    price: float

@dataclass
class Order:
    order_id: str
    user_id: str
    items: List[OrderItem]
    total_amount: float
    status: str = "pending"
"""
        },
        {
            "path": "interfaces.py",
            "content": """from abc import ABC, abstractmethod
from typing import Any

class DatabaseInterface(ABC):
    @abstractmethod
    def save(self, data: Any) -> bool:
        pass

class EmailServiceInterface(ABC):
    @abstractmethod
    def send(self, to: str, subject: str, body: str) -> bool:
        pass

class LoggerInterface(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        pass
"""
        }
    ]
}

# Analysis report (simulated)
analysis_report = {
    "findings": [
        {
            "severity": "critical",
            "issue": "Class 'OrderManager' is a God Class with 6 methods",
            "affected_entities": ["OrderManager"],
            "violated_principle": "SRP",
            "category": "solid"
        },
        {
            "severity": "warning",
            "issue": "OrderManager directly depends on concrete implementations",
            "affected_entities": ["OrderManager", "Database", "EmailService"],
            "violated_principle": "DIP",
            "category": "solid"
        }
    ],
    "recommendations": [
        {
            "title": "Split OrderManager into focused classes",
            "description": "Extract email and database logic",
            "priority": "high",
            "affected_entities": ["OrderManager"]
        }
    ],
    "quality_score": 0.4
}

print("\n" + "="*80)
print("DEBUGGING TESTGENERATIONAGENT - COMPLEX CASE")
print("="*80)
print("\nThis simulates the full workflow scenario where Gemini returns empty responses")
print("Model: OrderManager (God Class) with multiple responsibilities")
print("Code: Refactored with interfaces")

print("\n1. Creating LLM with fallback...")
llm = create_base_llm()

print("\n2. Creating TestGenerationAgent...")
agent = TestGenerationAgent(llm=llm)

print("\n3. Generating tests with full context...")
print("   - 3 classes in model")
print("   - 2 generated files")
print("   - 2 findings in analysis")
print("   - This might trigger empty response from Gemini...")

try:
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        analysis_report=analysis_report,
        framework="pytest",
        include_integration_tests=True
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nGenerated {len(result.get('test_files', []))} test files")
    print(f"Total tests: {result.get('total_tests', 0)}")
    print(f"Test categories: {result.get('test_categories', [])}")
    
    for i, test_file in enumerate(result.get('test_files', []), 1):
        print(f"\n--- Test File {i}: {test_file.get('path')} ---")
        content = test_file.get('content', '')
        lines = content.split('\n')
        print(f"Lines: {len(lines)}")
        print('\n'.join(lines[:20]))
        if len(lines) > 20:
            print(f"... ({len(lines) - 20} more lines)")
    
except Exception as e:
    print("\n" + "="*80)
    print("ERROR!")
    print("="*80)
    print(f"\nException: {type(e).__name__}: {e}")
    
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
