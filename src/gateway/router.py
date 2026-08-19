from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.gateway.cache import semantic_cache

router = APIRouter()

class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="What is RAG?")

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatChoice]
    cached: bool
    latency_ms: float

@router.post("/chat/completions", response_model=ChatCompletionResponse, tags=["Inference"])
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible inference endpoint with Redis semantic cache acceleration."""
    user_prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user message found in messages array.")

    # 1. Check Redis Semantic Cache
    cached = await semantic_cache.get_cached_response(user_prompt)
    if cached:
        return ChatCompletionResponse(
            id="chatcmpl-cached-001",
            model=req.model or "cached",
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=cached["content"]), finish_reason="stop")],
            cached=True,
            latency_ms=cached["latency_ms"]
        )

    # 2. Mock vLLM inference fallback
    generated_content = f"Synthesized response via local vLLM PagedAttention engine for query: '{user_prompt}'."
    await semantic_cache.set_cached_response(user_prompt, generated_content)

    return ChatCompletionResponse(
        id="chatcmpl-vllm-002",
        model=req.model or "vllm-llama3",
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=generated_content), finish_reason="stop")],
        cached=False,
        latency_ms=48.2
    )
