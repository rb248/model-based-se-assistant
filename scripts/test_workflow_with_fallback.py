"""
Test the full workflow with fallback LLM enabled.
This script will parse a simple UML model and show how the fallback works.
"""

from backend.agents import ParserAgent
from backend.llms import create_base_llm

# Simple PlantUML example
plantuml = """
@startuml
class User {
  - name: str
  - email: str
  + getName(): str
  + setName(name: str): void
}

class Order {
  - orderId: str
  - total: float
  + calculateTotal(): float
}

User "1" --> "*" Order
@enduml
"""

print("="*70)
print("Testing Full Workflow with Fallback LLM")
print("="*70)
print("\nParsing PlantUML model...")
print("-"*70)

# Create parser with fallback-enabled LLM
llm = create_base_llm()
parser = ParserAgent(llm=llm)

# Parse the model
result = parser.parse_model(plantuml, model_format="plantuml")

print("\n" + "="*70)
print("PARSING RESULT")
print("="*70)

if "error" in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Successfully parsed {len(result.get('classes', []))} classes")
    print("\nClasses found:")
    for cls in result.get('classes', []):
        print(f"\n  Class: {cls['name']}")
        print(f"    Methods: {len(cls.get('methods', []))}")
        print(f"    Attributes: {len(cls.get('attributes', []))}")
        for method in cls.get('methods', [])[:3]:  # Show first 3 methods
            print(f"      - {method.get('name')}()")
    
    print(f"\n  Relationships: {len(result.get('relationships', []))}")
    for rel in result.get('relationships', []):
        print(f"    - {rel.get('from')} -> {rel.get('to')} ({rel.get('type')})")

print("\n" + "="*70)
print("If Gemini failed, GPT-4o-mini fallback would have been used!")
print("="*70)
