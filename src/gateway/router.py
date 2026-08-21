import time
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from config.gateway_config import settings
from src.gateway.cache import semantic_cache
from src.gateway.vllm_client import inference_client

router = APIRouter()

class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="Explain vLLM PagedAttention")

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = settings.DEFAULT_MODEL
    messages: List[ChatMessage]
    temperature: Optional[float] = settings.TEMPERATURE
    max_tokens: Optional[int] = settings.MAX_TOKENS
    stream: Optional[bool] = False
    bypass_cache: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class UsageMetrics(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageMetrics
    cached: bool
    latency_ms: float
    tokens_per_second: Optional[float] = None


@router.get("/models", tags=["Models"])
async def list_models():
    """Returns list of supported local lightweight SLMs optimized for 16GB RAM."""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "local-gateway",
                "permission": [],
                "root": m,
                "parent": None
            } for m in settings.SUPPORTED_MODELS
        ]
    }


@router.post("/chat/completions", tags=["Inference"])
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    1. Evaluates Redis 8 vector semantic cache for sub-5ms return (unless bypass_cache=True).
    2. Fallbacks to local vLLM / SLM inference client with continuous batching.
    3. Supports both standard JSON and real-time Server-Sent Events (SSE) streaming.
    """
    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found in messages array.")

    # 1. Check Redis Semantic Cache (if not streaming and not bypassing cache)
    if not req.stream and not req.bypass_cache:
        cached = await semantic_cache.get_cached_response(user_prompt)
        if cached:
            return ChatCompletionResponse(
                id=f"chatcmpl-cached-{int(time.time()*1000)}",
                created=int(time.time()),
                model=req.model or settings.DEFAULT_MODEL,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=cached["content"]),
                        finish_reason="stop"
                    )
                ],
                usage=UsageMetrics(prompt_tokens=len(user_prompt.split()), completion_tokens=len(cached["content"].split()), total_tokens=len(user_prompt.split()) + len(cached["content"].split())),
                cached=True,
                latency_ms=cached["latency_ms"],
                tokens_per_second=1000.0  # Instantaneous cache delivery
            )

    # 2. Handle Streaming Completion (SSE)
    if req.stream:
        async def stream_generator():
            full_response = []
            async for chunk in inference_client.stream_completion(
                messages=[m.model_dump() for m in req.messages],
                model=req.model,
                max_tokens=req.max_tokens or settings.MAX_TOKENS,
                temperature=req.temperature or settings.TEMPERATURE
            ):
                yield chunk
                if chunk.startswith("data: {"):
                    try:
                        parsed = json.loads(chunk[6:])
                        delta = parsed["choices"][0]["delta"].get("content", "")
                        full_response.append(delta)
                    except Exception:
                        pass
            
            if full_response and not req.bypass_cache:
                await semantic_cache.set_cached_response(user_prompt, "".join(full_response))

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 3. Handle Non-Streaming Inference
    result = await inference_client.generate_completion(
        messages=[m.model_dump() for m in req.messages],
        model=req.model,
        max_tokens=req.max_tokens or settings.MAX_TOKENS,
        temperature=req.temperature or settings.TEMPERATURE
    )

    if not req.bypass_cache:
        await semantic_cache.set_cached_response(user_prompt, result["content"])

    prompt_toks = len(user_prompt.split())
    comp_toks = result["tokens_generated"]

    return ChatCompletionResponse(
        id=f"chatcmpl-vllm-{int(time.time()*1000)}",
        created=int(time.time()),
        model=result["model"],
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["content"]),
                finish_reason=result["finish_reason"]
            )
        ],
        usage=UsageMetrics(prompt_tokens=prompt_toks, completion_tokens=comp_toks, total_tokens=prompt_toks + comp_toks),
        cached=False,
        latency_ms=result["latency_ms"],
        tokens_per_second=result["tokens_per_second"]
    )


@router.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Returns telemetry statistics from the Redis 8 Vector Semantic Cache."""
    return semantic_cache.get_stats()


@router.post("/cache/clear", tags=["Cache"])
async def cache_clear():
    """Clears all stored entries in the semantic cache."""
    semantic_cache.clear()
    return {"status": "success", "message": "Semantic cache cleared successfully."}
