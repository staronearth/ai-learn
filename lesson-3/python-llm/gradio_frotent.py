import json
import requests
import gradio as gr

def chatWithBackend(message, history, sysPrompt, temperature):
    payload = {
        "message": message,
        "history": history,
        "sys_prompt": sysPrompt,
        "temperature": temperature
    }

    with requests.post(
        "http://localhost:8000/chat/stream",
        json=payload,
        stream=True,
        timeout=None
    ) as response:
        if response.status_code != 200:
            yield f"错误：{response.status_code}"
            return

        full_response = ""   # 累积完整回复
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            dataStr = line[len("data: "):].strip()
            if not dataStr:
                continue
            try:
                data = json.loads(dataStr)
                content = data.get("content", "")
                if content:
                    full_response += content
                    yield full_response   # 每次产出完整累积结果
            except Exception:
                continue

demo = gr.ChatInterface(
  fn=chatWithBackend,
  title="前后端分离演示",
  chatbot=gr.Chatbot(),
  additional_inputs=[
    gr.Textbox(label="系统提示词", value="你是一个乐于助人的助手。"),
    gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
  ]
)

demo.queue()

if __name__ == "__main__":
  demo.launch()