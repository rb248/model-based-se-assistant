"""
Debug script for TestGenerationAgent to inspect prompts and responses.
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

# Simple model IR
model_ir = {
    "classes": [
        {
            "name": "Calculator",
            "methods": [
                {"name": "add", "params": ["a", "b"]},
                {"name": "subtract", "params": ["a", "b"]}
            ],
            "attributes": []
        }
    ],
    "relationships": []
}

# Simple generated code
generated_code = {
    "files": [
        {
            "path": "calculator.py",
            "content": """class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        return a - b
"""
        }
    ]
}

# Simple analysis report
analysis_report = {
    "findings": [],
    "recommendations": [],
    "quality_score": 0.8
}

print("\n" + "="*80)
print("DEBUGGING TESTGENERATIONAGENT")
print("="*80)

print("\n1. Creating LLM with fallback...")
llm = create_base_llm()

print("\n2. Creating TestGenerationAgent...")
agent = TestGenerationAgent(llm=llm)

print("\n3. Generating tests...")
print("   Model: Calculator with add/subtract methods")
print("   This should be a simple case that works...")

try:
    result = agent.generate_tests(
        model_ir=model_ir,
        generated_code=generated_code,
        analysis_report=analysis_report,
        framework="pytest",
        include_integration_tests=False
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nGenerated {len(result.get('test_files', []))} test files")
    print(f"Total tests: {result.get('total_tests', 0)}")
    
    for i, test_file in enumerate(result.get('test_files', []), 1):
        print(f"\n--- Test File {i}: {test_file.get('path')} ---")
        content = test_file.get('content', '')
        print(content[:500])
        if len(content) > 500:
            print(f"... ({len(content) - 500} more chars)")
    
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
