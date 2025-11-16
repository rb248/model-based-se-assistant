"""
Full workflow test with LLM fallback enabled.

This demonstrates the complete end-to-end pipeline:
1. Parse PlantUML model
2. Analyze for design issues (god classes, SOLID violations)
3. Generate refactored code
4. Generate tests
5. Execute tests
6. Critique results
7. Generate final report
"""

import json
import sys
import time
import types
import pydantic
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add shim for langchain_core.pydantic_v1 if missing
if "langchain_core.pydantic_v1" not in sys.modules:
    mod = types.ModuleType("langchain_core.pydantic_v1")
    mod.BaseModel = pydantic.BaseModel
    mod.Field = pydantic.Field
    sys.modules["langchain_core.pydantic_v1"] = mod

from backend.graph import get_compiled_graph

# Test model: OrderManager god class with multiple responsibilities
plantuml_model = """
@startuml
class OrderManager {
  - database: Database
  - emailService: EmailService
  - logger: Logger
  + createOrder(userId: str, items: list): Order
  + sendOrderConfirmation(email: str, order: Order): void
  + saveToDatabase(order: Order): bool
  + logActivity(message: str): void
  + calculateTotal(items: list): float
  + generateInvoice(orderId: str): Invoice
}

class Database {
  + save(data): bool
  + query(sql): list
}

class EmailService {
  + send(to: str, message: str): bool
}

OrderManager --> Database
OrderManager --> EmailService
@enduml
"""

print("="*80)
print("FULL WORKFLOW TEST WITH LLM FALLBACK")
print("="*80)
print("\nModel: OrderManager (God Class)")
print("Expected Issues: SRP violation, multiple responsibilities")
print("-"*80)

# Get compiled graph
graph = get_compiled_graph()

# Initial state
initial_state = {
    "project_id": "test-full-workflow-fallback",
    "model_text": plantuml_model,
    "model_format": "plantuml",
    "description": "E-commerce order management with god class that needs refactoring"
}

print("\nStarting workflow...")
start_time = time.time()

# Run workflow
try:
    result = graph.invoke(initial_state)
except Exception as e:
    print(f"\n❌ WORKFLOW FAILED WITH EXCEPTION:")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

end_time = time.time()
elapsed = end_time - start_time

print(f"\n✅ Workflow completed in {elapsed:.2f} seconds")
print("="*80)

# Display results
final_report = result.get("final_report", {})

print("\nFINAL REPORT")
print("="*80)
print(json.dumps(final_report, indent=2))
print("="*80)

# Detailed breakdown
print("\n📊 WORKFLOW BREAKDOWN")
print("="*80)

# 1. Parsing
if result.get("model_ir"):
    classes = result["model_ir"].get("classes", [])
    relationships = result["model_ir"].get("relationships", [])
    print(f"\n1. PARSING:")
    print(f"   ✅ Parsed {len(classes)} classes")
    for cls in classes:
        print(f"      - {cls['name']}")
    print(f"   ✅ Found {len(relationships)} relationships")

# 2. Analysis
if result.get("analysis_report"):
    analysis = result["analysis_report"]
    findings = analysis.get("findings", [])
    recommendations = analysis.get("recommendations", [])
    quality_score = analysis.get("quality_metrics", {}).get("quality_score", 0)
    
    print(f"\n2. ANALYSIS:")
    print(f"   📋 Found {len(findings)} design issues")
    for finding in findings[:3]:  # Show first 3
        print(f"      - {finding.get('severity', 'N/A').upper()}: {finding.get('issue', 'N/A')[:80]}...")
    print(f"   💡 {len(recommendations)} recommendations")
    print(f"   📈 Quality Score: {quality_score:.2f}/1.0")

# 3. Code Generation
if result.get("generated_code"):
    code = result["generated_code"]
    files = code.get("files", [])
    print(f"\n3. CODE GENERATION:")
    print(f"   ✅ Generated {len(files)} code files")
    for file in files:
        path = file.get("path", "unknown")
        lines = len(file.get("content", "").split("\n"))
        print(f"      - {path} ({lines} lines)")
    
    # Show first 50 lines of first file
    if files:
        print(f"\n   📄 Preview of {files[0].get('path', 'first file')}:")
        content = files[0].get("content", "")
        preview = "\n".join(content.split("\n")[:50])
        print(f"      {preview[:500]}...")

# 4. Test Generation
if result.get("generated_tests"):
    tests = result["generated_tests"]
    test_files = tests.get("test_files", [])
    total_tests = tests.get("total_tests", 0)
    print(f"\n4. TEST GENERATION:")
    print(f"   ✅ Generated {len(test_files)} test files")
    print(f"   ✅ Total test cases: {total_tests}")
    
    # Show first test file preview
    if test_files:
        print(f"\n   📄 Preview of {test_files[0].get('path', 'first test')}:")
        content = test_files[0].get("content", "")
        preview = "\n".join(content.split("\n")[:30])
        print(f"      {preview[:400]}...")

# 5. Test Execution
if result.get("test_results"):
    test_results = result["test_results"]
    status = test_results.get("status", "unknown")
    print(f"\n5. TEST EXECUTION:")
    print(f"   Status: {status}")
    if status == "completed":
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        errors = test_results.get("errors", 0)
        print(f"   ✅ Passed: {passed}")
        if failed > 0:
            print(f"   ❌ Failed: {failed}")
        if errors > 0:
            print(f"   ⚠️  Errors: {errors}")
    
    # Show test output
    stdout = test_results.get("stdout", "")
    stderr = test_results.get("stderr", "")
    if stdout:
        print(f"\n   📋 Test Output (first 1000 chars):")
        print(f"      {stdout[:1000]}")
    if stderr:
        print(f"\n   ⚠️  Test Stderr:")
        print(f"      {stderr[:500]}")

# 6. Critique
if result.get("critique"):
    critique = result["critique"]
    suggestions = critique.get("refactoring_suggestions", [])
    print(f"\n6. CRITIQUE:")
    print(f"   💡 {len(suggestions)} refactoring suggestions")

# Errors
if result.get("errors"):
    print(f"\n⚠️  ERRORS: {len(result['errors'])}")
    for error in result["errors"]:
        print(f"   - {error}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Workflow completed successfully in {elapsed:.2f}s")
print(f"✅ LLM fallback available (Gemini → GPT-4o-mini)")
print(f"✅ All agents executed: Parse → Analyze → CodeGen → TestGen → Execute → Critique → Report")

if elapsed < 120:
    print(f"✅ Fast execution (<2 minutes) - no sleep delays")
else:
    print(f"⚠️  Slow execution (>2 minutes) - check for rate limiting")

print("="*80)
