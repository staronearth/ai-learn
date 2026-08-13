### Gradio 前端：学习助手界面，通过 SSE 流式调用 Flask 后端接口
import json

import httpx
import gradio as gr

BASE_URL = "http://127.0.0.1:5000"
USER_ID = "user_001"
THREAD_ID = "learn-001"


async def chat(message, history, subject):
  """异步生成器：逐 token 消费后端 SSE 流，边收边渲染"""
  text = message["content"] if isinstance(message, dict) else message
  content = ""
  question_num = None
  async with httpx.AsyncClient(timeout=120) as client:
    async with client.stream(
      "POST",
      f"{BASE_URL}/api/chat",
      json={
        "user_id": USER_ID,
        "subject": subject,
        "thread_id": THREAD_ID,
        "question": text,
      },
    ) as resp:
      resp.raise_for_status()
      async for line in resp.aiter_lines():
        if not line.startswith("data: "):
          continue
        event = json.loads(line[6:])
        if event["type"] == "token":
          content += event["content"]
          yield content
        elif event["type"] == "done":
          question_num = event.get("question_num")
        elif event["type"] == "error":
          yield content + f"\n\n**出错**：{event['error']}"
          return
  yield f"【第 {question_num} 问 · {subject}】\n\n{content}"


async def save_style(style):
  if not style.strip():
    return "请输入学习风格"
  async with httpx.AsyncClient(timeout=30) as client:
    resp = await client.post(
      f"{BASE_URL}/api/style",
      json={"user_id": USER_ID, "style": style.strip()},
    )
    resp.raise_for_status()
    return f"已保存学习风格：{resp.json()['style']}"


async def show_style():
  async with httpx.AsyncClient(timeout=30) as client:
    resp = await client.get(f"{BASE_URL}/api/style/{USER_ID}")
    resp.raise_for_status()
    style = resp.json().get("style")
    return f"当前学习风格：{style}" if style else "尚未设定学习风格"


with gr.Blocks(title="学习助手") as demo:
  gr.Markdown("# 学习助手\nState 记录提问数 · Store 保存学习风格 · Context 传入学习科目")
  subject = gr.Dropdown(
    ["Python", "LangGraph", "数学", "物理", "英语"],
    value="Python",
    label="当前学习科目（Context）",
  )
  gr.ChatInterface(fn=chat, additional_inputs=[subject])
  gr.Markdown("## 学习风格设置（Store）")
  with gr.Row():
    style_input = gr.Textbox(label="学习风格", placeholder="例如：喜欢举例、学术严谨、通俗易懂")
    save_btn = gr.Button("保存风格")
    show_btn = gr.Button("查看当前风格")
  style_output = gr.Markdown()
  save_btn.click(save_style, inputs=[style_input], outputs=[style_output])
  show_btn.click(show_style, outputs=[style_output])


if __name__ == "__main__":
  demo.launch()
