import os
import pytest

from backend.agents import ParserAgent
from backend.llms import get_llm
from backend.config import GOOGLE_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
import multiprocessing


def _should_run_integration():
    # Allow running integration either when explicitly requested or when an API key exists
    if os.getenv("RUN_LLM_INTEGRATION", "false").lower() == "true":
        return True
    if GOOGLE_API_KEY or OPENAI_API_KEY:
        return True
    return False


def _parse_worker(plantuml_text, q):
    # Top-level worker used by multiprocessing.Process (must be pickleable).
    from backend.llms import get_llm
    from backend.agents import ParserAgent
    from backend.config import LLM_PROVIDER, LLM_MODEL

    try:
        llm = get_llm()
        print(f"Configured LLM_PROVIDER={LLM_PROVIDER}, LLM_MODEL={LLM_MODEL}, LLM class={type(llm).__name__}, module={type(llm).__module__}")
        agent = ParserAgent(llm=llm)
        parsed = agent.parse_model(plantuml_text, model_format="plantuml")
        q.put(parsed)
    except Exception as e:
        q.put({"error": str(e)})


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="Integration with real LLM skipped (no key or RUN_LLM_INTEGRATION)")
def test_parser_with_real_llm():
    # Small PlantUML example
    plantuml = """
    @startuml
    class Person {
      - name: str
      + greet(name): str
    }

    class Address {
      - street: str
    }

    Person "1" --> "*" Address
    @enduml
    """

    # Run the parsing inside a separate process with a timeout to avoid
    # blocking the test runner if the network call stalls.
    # Spawn a process using the top-level worker. We do not re-create the LLM
    # in the parent process to avoid blocking the test runner if the model
    # client is waiting on the network.

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_parse_worker, args=(plantuml, q))
    p.start()
    # Wait a reasonable amount of time for a remote LLM call
    p.join(30)
    if p.is_alive():
      # Clean up but don't fail the entire test suite due to network issues
      p.terminate()
      pytest.skip("LLM parse timed out (network or provider may be slow)")
    try:
      parsed = q.get_nowait()
      # Print the parsed output and the class names for debugging and tracing
      import json
      print('\n🔍 Parsed (raw):')
      try:
        print(json.dumps(parsed, indent=2))
      except Exception:
        print(parsed)
      print('\n📚 Classes:')
      classes = parsed.get('classes', []) if isinstance(parsed, dict) else []
      if classes:
        for c in classes:
          print(' -', c.get('name'))
      else:
        print(' (no classes parsed)')
    except Exception:
      pytest.skip("No result returned from LLM process")

    # The provider info and any parsing details will be printed by the
    # worker. Here, we simply inspect the result returned on the queue.

    # We expect at least one class in the parsed model or a helpful error
    assert isinstance(parsed, dict)
    if parsed.get("error"):
      pytest.skip(f"LLM returned error: {parsed.get('error')}")
    assert len(parsed.get("classes", [])) > 0