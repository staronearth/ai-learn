import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
import json
import time
from functools import wraps

def chat_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper
app = FastAPI(title="大模型对话 API", version="1.0")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-..."),
    base_url="https://api.deepseek.com"
)

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一个乐于助人的助手。"
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    reply: str
    model: str


@app.post("/chat")
@chat_timer
async def chat(request: ChatRequest):
    start = time.time()
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ],
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    elapsed = time.time() - start
    reply = response.choices[0].message.content
    print(f"[{elapsed:.2f}s] Q: {request.message[:30]}... → A: {reply[:30]}...")
    return {"reply": reply, "elapsed": round(elapsed, 2)}

@app.get("/health")
async def health():
    return {"status": "healthy"}

class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "英文"

@app.post("/translate")
async def translate(request: TranslateRequest):
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{
            "role": "user",
            "content": f"将以下文本翻译为{request.target_lang}：{request.text}"
        }],
        temperature=0.3,
        max_tokens=512
    )
    return {"translation": response.choices[0].message.content}

@app.post("/chat/stream")
@chat_timer
async def chat_stream(request: ChatRequest):
    def generate():
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.message}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)