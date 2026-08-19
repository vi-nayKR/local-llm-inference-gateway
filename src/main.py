from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.gateway_config import settings
from src.gateway.router import router as gateway_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="High-Throughput Local LLM Inference Gateway with Redis 8 Vector Semantic Caching & vLLM"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gateway_router, prefix="/v1")

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": settings.APP_NAME, "version": settings.VERSION}
