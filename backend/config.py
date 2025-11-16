"""
Configuration and constants for the Model-Based Software Engineering Assistant.

This module handles all environment variables, model parameters, and system settings.
"""

import os
from pathlib import Path
from typing import Optional

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
PROJECTS_DIR = PROJECT_ROOT / "projects"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# Load local .env if present (allows developers to keep API keys out of source control)
try:
	from dotenv import load_dotenv

	_env_path = PROJECT_ROOT / ".env"
	if _env_path.exists():
		load_dotenv(dotenv_path=str(_env_path))
except Exception:
	# dotenv is optional; if not installed environment vars will come from system environment
	pass

# If python-dotenv wasn't available, try a simple .env parser so development is smooth
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
	try:
		with open(_env_path, "r") as f:
			for line in f:
				s = line.strip()
				# skip comments and empty lines
				if not s or s.startswith("#"):
					continue
				if "=" not in s:
					continue
				key, value = s.split("=", 1)
				key = key.strip()
				value = value.strip().strip('"').strip("'")
				# Only set if not already in environment to avoid overriding
				if key and os.getenv(key) is None:
					os.environ[key] = value
	except Exception:
		# If manual parsing fails, ignore — we still fall back to environment
		pass

# LLM Configuration
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-pro")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: Optional[int] = int(os.getenv("LLM_MAX_TOKENS", "8192")) or None  # Increased for test generation

# Gemini API key (free tier)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
USE_GEMINI: bool = os.getenv("USE_GEMINI", "true").lower() == "true"

# OpenAI API key (fallback)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# LLM Retry Configuration
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAYS: list = [5, 10, 20]  # Exponential backoff in seconds
LLM_FALLBACK_PROVIDER: str = os.getenv("LLM_FALLBACK_PROVIDER", "openai")
LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")

# RAG Configuration
RAG_BACKEND: str = os.getenv("RAG_BACKEND", "faiss")  # "faiss" or "chroma"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "google")  # "google" or "ollama"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Ollama Configuration (optional local fallback)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Code execution and testing
SANDBOX_TIMEOUT: int = int(os.getenv("SANDBOX_TIMEOUT", "30"))
TEST_FRAMEWORK: str = os.getenv("TEST_FRAMEWORK", "pytest")

# Memory and project settings
MAX_PROJECT_MEMORY_SIZE: int = int(os.getenv("MAX_PROJECT_MEMORY_SIZE", "1000"))
ENABLE_LANGSMITH: bool = os.getenv("ENABLE_LANGSMITH", "false").lower() == "true"
LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT", None)

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # or "text"

# Default model generation language
DEFAULT_CODE_LANGUAGE: str = os.getenv("DEFAULT_CODE_LANGUAGE", "python")

# Debug mode
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
