import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config.gateway_config import settings
from src.gateway.router import router as gateway_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="High-Throughput Local LLM Inference Gateway with Redis 8 Vector Semantic Caching, vLLM PagedAttention, and NeMo Guardrails."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gateway_router, prefix="/v1")

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serves the interactive dark-mode developer console."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Local LLM Inference Gateway is running</h1>"

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "default_model": settings.DEFAULT_MODEL,
        "memory_optimized": "High-Throughput SLM Tier"
    }
