from pydantic_settings import BaseSettings, SettingsConfigDict

class GatewaySettings(BaseSettings):
    APP_NAME: str = "Local LLM Inference Gateway"
    VERSION: str = "1.0.0"
    
    # Redis Semantic Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    
    # vLLM Server Settings
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = GatewaySettings()
