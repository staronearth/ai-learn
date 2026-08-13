### Flask 后端：为 Gradio 前端提供学习助手接口（SSE 流式输出）
import asyncio
import json
import queue
import threading

from flask import Flask, Response, jsonify, request

from agent import STYLE_NAMESPACE, ask_stream, get_store

app = Flask(__name__)

# 专用后台事件循环线程，驱动 agent 的异步流式调用
EVENT_LOOP = asyncio.new_event_loop()
threading.Thread(target=EVENT_LOOP.run_forever, daemon=True).start()


@app.post("/api/chat")
def chat():
  data = request.get_json(silent=True) or {}
  question = data.get("question", "").strip()
  if not question:
    return jsonify({"error": "question 不能为空"}), 400

  event_queue = queue.Queue()

  async def producer():
    try:
      async for event in ask_stream(
        user_id=data.get("user_id", "user_001"),
        subject=data.get("subject", "Python"),
        thread_id=data.get("thread_id", "learn-001"),
        question=question,
      ):
        event_queue.put(event)
    except Exception as e:
      event_queue.put({"type": "error", "error": str(e)})
    finally:
      event_queue.put(None)

  asyncio.run_coroutine_threadsafe(producer(), EVENT_LOOP)

  def generate():
    while True:
      event = event_queue.get()
      if event is None:
        break
      yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

  return Response(
    generate(),
    mimetype="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
  )


@app.get("/api/style/<user_id>")
def get_style(user_id):
  item = get_store().get(STYLE_NAMESPACE, user_id)
  style = item.value.get("style") if item else None
  return jsonify({"user_id": user_id, "style": style})


@app.post("/api/style")
def set_style():
  data = request.get_json(silent=True) or {}
  user_id = data.get("user_id", "user_001")
  style = data.get("style", "").strip()
  if not style:
    return jsonify({"error": "style 不能为空"}), 400
  get_store().put(STYLE_NAMESPACE, user_id, {"style": style})
  return jsonify({"user_id": user_id, "style": style})


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, threaded=True)
