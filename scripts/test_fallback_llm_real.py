from backend.llms import LLMFallbackWrapper
from backend.config import OPENAI_API_KEY, LLM_FALLBACK_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Import the real OpenAI LLM
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError("langchain_openai is not installed. Please install it with 'pip install langchain-openai'.")

class AlwaysFailLLM:
    def invoke(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")
    def __call__(self, *args, **kwargs):
        raise Exception("Primary LLM failed intentionally")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in your .env file. Please add your OpenAI key.")

fallback_llm = ChatOpenAI(
    model=LLM_FALLBACK_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

if __name__ == "__main__":
    wrapper = LLMFallbackWrapper(AlwaysFailLLM(), fallback_llm)
    prompt = "Say hello from GPT-4o-mini."
    print("Sending prompt to fallback LLM (GPT-4o-mini):", prompt)
    result = wrapper.invoke(prompt)
    
    # Extract content from LangChain message object
    if hasattr(result, 'content'):
        output = result.content
    else:
        output = str(result)
    
    print("\n" + "="*60)
    print("Result from fallback LLM (GPT-4o-mini):")
    print("="*60)
    print(output)
    print("="*60)
