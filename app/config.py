"""
Centralized configuration for GraphRAG Engine
"""
import os
from typing import Optional

# Environment variables with defaults
def get_env_str(key: str, default: str) -> str:
    """Get string environment variable with default."""
    return os.environ.get(key, default)

def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def get_env_float(key: str, default: float) -> float:
    """Get float environment variable with default."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

# LLM Configuration
LLM_MODEL = get_env_str("LLM_MODEL", "qwen2.5-coder:3b")
LLM_TEMPERATURE = get_env_float("LLM_TEMPERATURE", 0.1)
LLM_TEMPERATURE_REWRITE = get_env_float("LLM_TEMPERATURE_REWRITE", 0.3)
LLM_BASE_URL = get_env_str("LLM_BASE_URL", "http://localhost:11434")

# Neo4j Configuration
NEO4J_URL = get_env_str("NEO4J_URL", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = get_env_str("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_env_str("NEO4J_PASSWORD", "password")

# FAISS Configuration
FAISS_MODEL = get_env_str("FAISS_MODEL", "all-MiniLM-L6-v2")
FAISS_K = get_env_int("FAISS_K", 5)
FAISS_SIMILARITY_THRESHOLD = get_env_float("FAISS_SIMILARITY_THRESHOLD", 0.3)

# Workflow Configuration
MAX_REWRITE_ITERATIONS = get_env_int("MAX_REWRITE_ITERATIONS", 3)
MIN_DOC_LENGTH = get_env_int("MIN_DOC_LENGTH", 50)
PDF_CHUNK_SIZE = get_env_int("PDF_CHUNK_SIZE", 1000)
PDF_CHUNK_OVERLAP = get_env_int("PDF_CHUNK_OVERLAP", 100)

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Agentic RAG Engine"
STREAMLIT_LAYOUT = "wide"

# Logging Configuration
LOG_LEVEL = get_env_str("LOG_LEVEL", "INFO")
ENABLE_DEBUG_LOGGING = get_env_str("ENABLE_DEBUG_LOGGING", "false").lower() == "true"

def get_llm_config() -> dict:
    """Get LLM configuration as dictionary."""
    return {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "baseUrl": LLM_BASE_URL,
    }

def get_neo4j_config() -> dict:
    """Get Neo4j configuration as dictionary."""
    return {
        "url": NEO4J_URL,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASSWORD,
    }

def get_faiss_config() -> dict:
    """Get FAISS configuration as dictionary."""
    return {
        "model_name": FAISS_MODEL,
        "k": FAISS_K,
        "similarity_threshold": FAISS_SIMILARITY_THRESHOLD,
    }
