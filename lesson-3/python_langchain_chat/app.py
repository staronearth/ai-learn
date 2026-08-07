from fastapi import FastAPI,Body
from fastapi.responses import StreamingResponse
from agent import stream_agent_response

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "欢迎使用工具智能体！访问 /chat 开始对话"}

@app.post("/chat")
async def chat(query: str = Body(...,embed=True)):
    return StreamingResponse(stream_agent_response(query), media_type="text/plain")
