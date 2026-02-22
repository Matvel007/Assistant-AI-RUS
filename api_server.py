#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import time
import json
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))
from main import Assistant

CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

assistant = Assistant(config, api_mode=True)

# Предварительная загрузка модели эмбеддингов, чтобы первый запрос был быстрым
print("⏳ Предварительная загрузка модели эмбеддингов...")
assistant.preload_embedder()
print("✅ Модель эмбеддингов загружена.")

app = FastAPI(title="Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]

def extract_text_from_message(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        texts = []
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return " ".join(texts)
    else:
        return str(message.content)

async def generate_stream_response(request: ChatCompletionRequest):
    """Генерирует потоковый ответ в формате OpenAI."""
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        yield "data: " + json.dumps({"error": "No user message"}) + "\n\n"
        return
    last_user_message = extract_text_from_message(user_messages[-1])

    loop = asyncio.get_event_loop()
    response_text, _ = await loop.run_in_executor(None, assistant.process_input, last_user_message)

    chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"delta": {"role": "assistant", "content": response_text}, "index": 0, "finish_reason": "stop"}]
    }
    yield "data: " + json.dumps(chunk) + "\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(generate_stream_response(request), media_type="text/event-stream")
    else:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        last_user_message = extract_text_from_message(user_messages[-1])

        loop = asyncio.get_event_loop()
        response_text, elapsed = await loop.run_in_executor(None, assistant.process_input, last_user_message)

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ]
        )
        return response

@app.get("/health")
async def health():
    return {"status": "ok", "assistant_ready": True}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск API-сервера ассистента на http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)