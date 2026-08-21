import os
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class GatewaySettings(BaseSettings):
        APP_NAME: str = "High-Throughput Local LLM Inference Gateway"
        VERSION: str = "1.0.0"
        
        # Redis 8 Vector Semantic Cache
        REDIS_URL: str = "redis://localhost:6379/0"
        CACHE_SIMILARITY_THRESHOLD: float = 0.60
        CACHE_TTL_SECONDS: int = 86400  # 24 hours
        VECTOR_DIMENSION: int = 384     # Lightweight 384-D embeddings
        
        # Local SLM Settings (High-Throughput SLM Tier)
        VLLM_API_BASE: str = "http://localhost:8000/v1"
        OLLAMA_API_BASE: str = "http://localhost:11434/v1"
        DEFAULT_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"
        SUPPORTED_MODELS: List[str] = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "gpt-4o-mini",
            "groq/llama-3.2-1b-preview"
        ]
        MAX_TOKENS: int = 1024
        TEMPERATURE: float = 0.7
        ENABLE_SAFETY_GUARDRAILS: bool = True
        MAX_INPUT_CHARS: int = 4000
        
        # Optional External Fallback API Keys (Works 100% offline without them)
        OPENAI_API_KEY: Optional[str] = None
        GROQ_API_KEY: Optional[str] = None
        OPENROUTER_API_KEY: Optional[str] = None
        
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    settings = GatewaySettings()
except ImportError:
    class StandaloneSettings:
        APP_NAME: str = os.getenv("APP_NAME", "High-Throughput Local LLM Inference Gateway")
        VERSION: str = "1.0.0"
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.60"))
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
        VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))
        VLLM_API_BASE: str = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
        DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
        SUPPORTED_MODELS: List[str] = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "gpt-4o-mini",
            "groq/llama-3.2-1b-preview"
        ]
        MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
        TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        ENABLE_SAFETY_GUARDRAILS: bool = True
        MAX_INPUT_CHARS: int = 4000
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
        OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    settings = StandaloneSettings()
