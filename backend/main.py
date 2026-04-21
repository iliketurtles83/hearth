from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import anthropic
import os
import json
from dotenv import load_dotenv
from router import should_use_cloud, LOCAL_MODEL, CLOUD_MODEL

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful personal assistant. Be concise and accurate."

async def stream_local(request: ChatRequest):
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", "http://ollama:11434/api/generate", json={
            "model": LOCAL_MODEL,
            "prompt": request.message,
            "system": request.system,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break

async def stream_cloud(request: ChatRequest):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=CLOUD_MODEL,
        max_tokens=2048,
        system=request.system,
        messages=[{"role": "user", "content": request.message}],
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat")
async def chat(request: ChatRequest):
    use_cloud = should_use_cloud(request.message)
    streamer = stream_cloud(request) if use_cloud else stream_local(request)
    model_used = CLOUD_MODEL if use_cloud else LOCAL_MODEL

    async def generate():
        yield f"data: {json.dumps({'model': model_used})}\n\n"
        async for chunk in streamer:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok"}