"""
Small script to verify keys are loaded and LLM can be instantiated.

Usage:
    python scripts/check_keys.py

It will print which keys are present (masked) and attempt to instantiate the LLM factory.
"""

import os
from backend.config import GOOGLE_API_KEY, OPENAI_API_KEY

print("Checking API keys loaded from environment or .env...")
print(f"GOOGLE_API_KEY present: {'Yes' if bool(GOOGLE_API_KEY) else 'No'}")
print(f"OPENAI_API_KEY present: {'Yes' if bool(OPENAI_API_KEY) else 'No'}")

# Masked printing
def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "..."
    return key[:4] + "..." + key[-4:]

print('\nMasked values:')
print('GOOGLE_API_KEY:', mask_key(GOOGLE_API_KEY))
print('OPENAI_API_KEY:', mask_key(OPENAI_API_KEY))

# Try to construct LLM instances (no network calls unless explicitly used)
try:
    from backend.llms import get_llm
    print('\nTesting LLM factory...')
    try:
        llm = get_llm(provider='gemini')
        print('Gemini LLM: OK (client created)')
    except Exception as e:
        print('Gemini LLM: Error -', str(e))
    try:
        llm2 = get_llm(provider='openai')
        print('OpenAI LLM: OK (client created)')
    except Exception as e:
        print('OpenAI LLM: Error -', str(e))
except Exception as e:
    print('LLM check skipped: get_llm or dependencies not available', e)