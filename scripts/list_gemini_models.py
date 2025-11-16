"""List available Gemini models and test ParserAgent with a real model.

Usage:
    PYTHONPATH=. python scripts/list_gemini_models.py
"""
import sys
from backend.config import GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY not set. Please configure your .env file.")
    sys.exit(1)

print(f"✓ GOOGLE_API_KEY configured: {GOOGLE_API_KEY[:8]}...")

# Available Gemini models (as of Nov 2025):
print("\n📋 Current Gemini API Models (stable):")
print("  - gemini-2.5-pro       (most advanced, thinking model)")
print("  - gemini-2.5-flash     (best price-performance)")
print("  - gemini-2.5-flash-lite (fastest, cost-efficient)")
print("  - gemini-2.0-flash     (2nd gen workhorse)")
print("  - gemini-2.0-flash-lite (2nd gen small)")

print("\n📝 Your current config:")
from backend.config import LLM_MODEL, LLM_PROVIDER
print(f"  LLM_PROVIDER: {LLM_PROVIDER}")
print(f"  LLM_MODEL: {LLM_MODEL}")

print("\n🧪 Testing ParserAgent with configured model...")

try:
    from backend.llms import get_llm
    from backend.agents import ParserAgent
    
    llm = get_llm()
    print(f"✓ LLM instantiated: {type(llm).__name__} from {type(llm).__module__}")
    
    agent = ParserAgent(llm=llm)
    
    sample_plantuml = """
    @startuml
    class User {
      - name: str
      + get_name(): str
    }
    @enduml
    """
    
    print("\n🚀 Parsing sample PlantUML...")
    parsed = agent.parse_model(sample_plantuml, model_format="plantuml")
    
    if parsed.get("error"):
        print(f"❌ Error: {parsed['error']}")
    elif parsed.get("classes"):
        print(f"✅ Successfully parsed {len(parsed['classes'])} class(es):")
        for cls in parsed["classes"]:
            print(f"   - {cls.get('name', 'Unknown')}")
    else:
        print("⚠️  No classes found in result")
        print(f"Result: {parsed}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
