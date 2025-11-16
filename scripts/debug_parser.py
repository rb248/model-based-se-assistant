"""Debug script to see raw LLM responses for different PlantUML inputs."""
import sys
import os
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents import ParserAgent
from backend.llms import get_llm

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_complex_diagram():
    """Test the complex diagram that's failing."""
    plantuml = """
    @startuml
    abstract class Animal {
      # species: string
      + {abstract} makeSound(): void
      + sleep(): void
    }

    class Dog {
      - breed: string
      + makeSound(): void
      + fetch(): void
    }

    class Cat {
      - color: string
      + makeSound(): void
      + climb(): void
    }

    class Owner {
      - name: string
      + adopt(pet): void
    }

    Dog --|> Animal
    Cat --|> Animal
    Owner "1" --> "*" Animal : owns
    @enduml
    """
    
    print("=" * 80)
    print("Testing complex diagram parsing...")
    print("=" * 80)
    
    llm = get_llm()
    agent = ParserAgent(llm=llm)
    
    # Capture the raw response by temporarily modifying the agent
    parsed = agent.parse_model(plantuml)
    
    print("\n" + "=" * 80)
    print("PARSED RESULT:")
    print("=" * 80)
    print(json.dumps(parsed, indent=2))
    
    if "error" in parsed:
        print(f"\n❌ ERROR: {parsed['error']}")
    else:
        print(f"\n✅ SUCCESS: Parsed {len(parsed.get('classes', []))} classes")
        print(f"   Classes: {[c['name'] for c in parsed.get('classes', [])]}")
        print(f"   Relationships: {len(parsed.get('relationships', []))}")


def test_methods_with_params():
    """Test methods with parameters."""
    plantuml = """
    @startuml
    class MathOperations {
      + add(x, y): int
      + multiply(a, b, c): float
      + divide(numerator, denominator): float
    }
    @enduml
    """
    
    print("\n" + "=" * 80)
    print("Testing methods with parameters...")
    print("=" * 80)
    
    llm = get_llm()
    agent = ParserAgent(llm=llm)
    parsed = agent.parse_model(plantuml)
    
    print("\n" + "=" * 80)
    print("PARSED RESULT:")
    print("=" * 80)
    print(json.dumps(parsed, indent=2))
    
    if "error" in parsed:
        print(f"\n❌ ERROR: {parsed['error']}")
    else:
        print(f"\n✅ SUCCESS")
        for cls in parsed.get('classes', []):
            print(f"\nClass: {cls['name']}")
            for method in cls.get('methods', []):
                params = method.get('params', [])
                print(f"  - {method['name']}({', '.join(params)}): {method.get('returns', 'void')}")
                if len(params) == 0:
                    print(f"    ⚠️  WARNING: No parameters parsed!")


if __name__ == "__main__":
    test_complex_diagram()
    print("\n\n")
    test_methods_with_params()
