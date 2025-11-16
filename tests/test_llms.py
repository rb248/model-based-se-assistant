import pytest
from backend.llms import LLMFallbackWrapper

class AlwaysFailLLM:
    def invoke(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")
    def __call__(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")

class DummyFallbackLLM:
    def invoke(self, *args, **kwargs):
        return "fallback-success"
    def __call__(self, *args, **kwargs):
        return "fallback-success"


def test_llm_fallback_invoked():
    """
    Test that LLMFallbackWrapper calls fallback LLM when primary fails.
    """
    wrapper = LLMFallbackWrapper(AlwaysFailLLM(), DummyFallbackLLM())
    result = wrapper.invoke("test prompt")
    assert result == "fallback-success"

    # Also test __call__ interface
    result2 = wrapper("test prompt")
    assert result2 == "fallback-success"


def test_llm_fallback_no_fallback():
    """
    Test that LLMFallbackWrapper raises if no fallback is provided.
    """
    wrapper = LLMFallbackWrapper(AlwaysFailLLM(), None)
    with pytest.raises(Exception) as exc:
        wrapper.invoke("test prompt")
    assert "Primary LLM failed intentionally" in str(exc.value)
