from backend.llms import LLMFallbackWrapper

class AlwaysFailLLM:
    def invoke(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")
    def __call__(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")

class DummyFallbackLLM:
    def invoke(self, *args, **kwargs):
        return "This is the fallback LLM output!"
    def __call__(self, *args, **kwargs):
        return "This is the fallback LLM output!"

if __name__ == "__main__":
    wrapper = LLMFallbackWrapper(AlwaysFailLLM(), DummyFallbackLLM())
    result = wrapper.invoke("test prompt")
    print("Result from fallback LLM:", result)
