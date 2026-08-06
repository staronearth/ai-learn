from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path
import json
import os
import time
from typing import List, Any, AsyncIterator

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

client = OpenAI(
  api_key=os.getenv("DEEPSEEK_API_KEY", "sk-..."),
  base_url="https://api.deepseek.com"
)

LOG_PATH = Path("chat.log")

class ChatReq(BaseModel):
  message: str
  history: List[Any] = Field(default_factory=list)
  sys_prompt: str = "你是一个乐于助人的助手。"
  temperature: float = 0.7

def buildMessages(req: ChatReq) -> List[dict]:
  messages = [{"role": "system", "content": req.sys_prompt}]

  for msg in req.history:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
      messages.append({
        "role": msg["role"],
        "content": msg["content"]
      })
    elif isinstance(msg, (list, tuple)) and len(msg) == 2:
      messages.append({"role": "user", "content": msg[0]})
      messages.append({"role": "assistant", "content": msg[1]})
    else:
      continue

  messages.append({"role": "user", "content": req.message})
  return messages

def writeLog(question: str, answer: str, elapsedMs: int, status: str, errorMessage: str | None = None):
  entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "question": question,
    "answer": answer,
    "elapsedMs": elapsedMs,
    "status": status
  }

  if errorMessage is not None:
    entry["errorMessage"] = errorMessage

  with LOG_PATH.open("a", encoding="utf-8") as logFile:
    logFile.write(json.dumps(entry, ensure_ascii=False) + "\n")

@app.post("/chat")
async def chat(req: ChatReq):
  startTime = time.perf_counter()
  try:
    messages = buildMessages(req)

    resp = client.chat.completions.create(
      model="deepseek-v4-flash",
      messages=messages,
      temperature=req.temperature
    )
    reply = resp.choices[0].message.content

    elapsedMs = int((time.perf_counter() - startTime) * 1000)
    writeLog(req.message, reply, elapsedMs, "success")
    return {"reply": reply}

  except Exception as e:
    elapsedMs = int((time.perf_counter() - startTime) * 1000)
    writeLog(req.message, "", elapsedMs, "error", str(e))
    raise HTTPException(status_code=500, detail=f"调用模型失败: {str(e)}")

@app.post("/chat/stream")
async def chatStream(req: ChatReq):
  startTime = time.perf_counter()
  fullReply = ""

  try:
    messages = buildMessages(req)

    stream = client.chat.completions.create(
      model="deepseek-v4-flash",
      messages=messages,
      temperature=req.temperature,
      stream=True
    )

    async def eventStream() -> AsyncIterator[str]:
      nonlocal fullReply

      try:
        for chunk in stream:
          delta = chunk.choices[0].delta.content
          if delta is None:
            continue

          fullReply += delta

          payload = json.dumps({"content": delta}, ensure_ascii=False)
          yield f"data: {payload}\n\n"
      finally:
        elapsedMs = int((time.perf_counter() - startTime) * 1000)
        writeLog(req.message, fullReply, elapsedMs, "success")

    return StreamingResponse(
      eventStream(),
      media_type="text/event-stream",
      headers={"Cache-Control": "no-cache"}
    )

  except Exception as e:
    elapsedMs = int((time.perf_counter() - startTime) * 1000)
    writeLog(req.message, "", elapsedMs, "error", str(e))
    raise HTTPException(status_code=500, detail=f"调用模型失败: {str(e)}")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")