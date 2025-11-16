"""
LLM and Embeddings factory for the Model-Based Software Engineering Assistant.

Supports multiple providers:
- Gemini (free tier, recommended)
- OpenAI (paid, optional)
- Ollama (local, free)
"""

import logging
import time
from typing import Optional, Union, Any, Callable
from functools import wraps

from backend.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    USE_GEMINI,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAYS,
    LLM_FALLBACK_PROVIDER,
    LLM_FALLBACK_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class LLMFallbackWrapper:
    """
    Wrapper that provides retry logic with exponential backoff and automatic
    fallback to GPT-4o-mini when primary LLM fails.
    """
    
    def __init__(self, primary_llm: Any, fallback_llm: Optional[Any] = None):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.primary_failures = 0
        
    def invoke(self, *args, **kwargs):
        """Invoke with retry logic and fallback."""
        last_error = None
        
        # Try primary LLM with retries
        for attempt in range(LLM_MAX_RETRIES):
            try:
                result = self.primary_llm.invoke(*args, **kwargs)
                
                # Log detailed response information
                logger.debug(f"LLM response type: {type(result)}")
                logger.debug(f"LLM response attributes: {dir(result)[:20]}")  # First 20 attributes
                
                # Check for response metadata (status codes, headers, etc.)
                if hasattr(result, 'response_metadata'):
                    metadata = result.response_metadata
                    logger.info(f"Response metadata: {metadata}")
                
                # Check for usage information
                if hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    logger.info(f"Token usage: {usage}")
                
                # Extract content
                if hasattr(result, 'content'):
                    content = result.content.strip()
                elif isinstance(result, str):
                    content = result.strip()
                else:
                    content = str(result).strip()
                
                logger.debug(f"LLM response content length: {len(content)}")
                
                if not content:
                    logger.warning("Empty response detected from primary LLM")
                    # Log full result object for debugging
                    logger.error(f"Empty response details - Full result object: {result}")
                    if hasattr(result, '__dict__'):
                        logger.error(f"Result __dict__: {result.__dict__}")
                    raise ValueError("Empty response from LLM")
                
                # Success - reset failure counter
                if self.primary_failures > 0:
                    logger.info(f"Primary LLM recovered after {self.primary_failures} failures")
                    self.primary_failures = 0
                    
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit or quota error
                is_rate_limit = any(x in error_str for x in [
                    '429', 'rate limit', 'quota', 'resource_exhausted',
                    'too many requests', 'empty response'
                ])
                
                if is_rate_limit and attempt < LLM_MAX_RETRIES - 1:
                    delay = LLM_RETRY_DELAYS[min(attempt, len(LLM_RETRY_DELAYS) - 1)]
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                    
        # Primary LLM failed all retries
        self.primary_failures += 1
        logger.warning(
            f"Primary LLM failed after {LLM_MAX_RETRIES} attempts. "
            f"Total failures: {self.primary_failures}"
        )
        
        # Try fallback if available
        if self.fallback_llm:
            try:
                logger.info("Attempting fallback to GPT-4o-mini...")
                result = self.fallback_llm.invoke(*args, **kwargs)
                logger.info("Fallback LLM succeeded")
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")
                raise Exception(
                    f"Primary LLM failed: {last_error}. "
                    f"Fallback LLM also failed: {fallback_error}"
                )
        
        # No fallback available
        raise last_error
    
    def __call__(self, *args, **kwargs):
        """Allow callable interface for compatibility."""
        result = self.invoke(*args, **kwargs)
        if hasattr(result, 'content'):
            return result.content
        return str(result)
    
    def __or__(self, other):
        """Support pipe operator for LangChain chains."""
        # Delegate to primary LLM's pipe behavior
        return self.primary_llm | other


def get_llm(provider: Optional[str] = None, temperature: Optional[float] = None, 
            enable_fallback: bool = True, **kwargs):
    """
    Get an LLM instance based on configuration with automatic fallback.

    Args:
        provider: "gemini", "openai", or "ollama". If None, uses LLM_PROVIDER.
        temperature: Override default temperature.
        enable_fallback: Whether to enable automatic fallback to GPT-4o-mini (default: True).
        **kwargs: Additional arguments to pass to the LLM.

    Returns:
        LLM instance wrapped with retry logic and fallback support.
    """
    provider = provider or LLM_PROVIDER
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    if provider == "gemini" and USE_GEMINI and GOOGLE_API_KEY:
        logger.info("Initializing Gemini LLM (free tier)")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=temp,
                max_output_tokens=LLM_MAX_TOKENS,
                **kwargs
            )
        except ImportError:
            logger.warning("langchain_google_genai not available, falling back to Ollama")
            return get_llm(provider="ollama", temperature=temp, **kwargs)

    elif provider == "openai" and OPENAI_API_KEY:
        logger.info("Initializing OpenAI LLM")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=temp,
                max_tokens=LLM_MAX_TOKENS,
                **kwargs
            )
        except ImportError:
            logger.warning("langchain_openai not available, falling back to Ollama")
            return get_llm(provider="ollama", temperature=temp, **kwargs)

    elif provider == "ollama":
        logger.info(f"Initializing Ollama LLM ({OLLAMA_MODEL})")
        try:
            from langchain_community.llms import Ollama

            return Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=temp,
                **kwargs
            )
        except ImportError:
            logger.error("langchain_community not available. Please install it.")
            raise

    else:
        raise ValueError(
            f"Invalid LLM provider: {provider}. "
            f"Available: gemini (requires GOOGLE_API_KEY), "
            f"openai (requires OPENAI_API_KEY), ollama (requires local Ollama)"
        )


def create_base_llm(enable_fallback: bool = True, **kwargs):
    """
    Create base LLM with fallback support.
    
    This is a convenience wrapper that creates the primary LLM and optionally
    adds a fallback LLM wrapper for improved reliability.
    
    Args:
        enable_fallback: Whether to enable GPT-4o-mini fallback (default: True)
        **kwargs: Additional arguments passed to get_llm
        
    Returns:
        LLM instance (wrapped with fallback if enabled)
    """
    # Get primary LLM
    primary_llm = get_llm(enable_fallback=False, **kwargs)
    
    # If fallback disabled or no OpenAI key, return primary only
    if not enable_fallback or not OPENAI_API_KEY:
        if enable_fallback and not OPENAI_API_KEY:
            logger.warning(
                "Fallback requested but OPENAI_API_KEY not set. "
                "Primary LLM will be used without fallback."
            )
        return primary_llm
    
    # Create fallback LLM (GPT-4o-mini)
    try:
        from langchain_openai import ChatOpenAI
        
        fallback_llm = ChatOpenAI(
            model=LLM_FALLBACK_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=kwargs.get('temperature', LLM_TEMPERATURE),
            max_tokens=LLM_MAX_TOKENS,
        )
        
        logger.info(
            f"LLM initialized with fallback: {LLM_PROVIDER} -> {LLM_FALLBACK_MODEL}"
        )
        
        return LLMFallbackWrapper(primary_llm, fallback_llm)
        
    except ImportError:
        logger.warning(
            "langchain_openai not available. Fallback disabled."
        )
        return primary_llm
    except Exception as e:
        logger.warning(f"Failed to create fallback LLM: {e}. Using primary only.")
        return primary_llm


def get_embeddings(provider: Optional[str] = None):
    """
    Get embeddings instance based on configuration.

    Args:
        provider: "google", "ollama", or "huggingface". If None, uses EMBEDDING_PROVIDER.

    Returns:
        Embeddings instance.
    """
    provider = provider or EMBEDDING_PROVIDER

    if provider == "google" and GOOGLE_API_KEY:
        logger.info("Initializing Google Embeddings")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )
        except ImportError:
            logger.warning("langchain_google_genai not available, falling back to Ollama")
            return get_embeddings(provider="ollama")

    elif provider == "ollama":
        logger.info(f"Initializing Ollama Embeddings ({OLLAMA_EMBEDDING_MODEL})")
        try:
            from langchain_community.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_EMBEDDING_MODEL,
            )
        except ImportError:
            logger.error("langchain_community not available. Please install it.")
            raise

    elif provider == "huggingface":
        logger.info("Initializing HuggingFace Embeddings")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        except ImportError:
            logger.error("langchain_community not available. Please install it.")
            raise

    else:
        raise ValueError(
            f"Invalid embeddings provider: {provider}. "
            f"Available: google, ollama, huggingface"
        )
