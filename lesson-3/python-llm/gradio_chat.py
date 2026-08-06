import gradio as gr
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-..."), 
    base_url="https://api.deepseek.com"
)

def chat_cloud(message, history, sys_prompt, temperature):
    if not message.strip():
        gr.Warning("请输入有效的问题")
        yield "请输入有效的问题"
    # 构造消息列表
    messages = [{"role": "system", "content": sys_prompt}]
    
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({
                "role": str(msg["role"]),
                "content": str(msg["content"])
            })
            
    messages.append({"role": "user", "content": str(message)})

    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        partial = ""
        for chunk in stream:
            # choices 和 delta 是否存在
            if chunk.choices and chunk.choices[0].delta:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    partial += delta_content
                    yield partial
                
    except Exception as e:
        gr.Error(f"发生错误：{str(e)}")
        yield f"⚠️ API 调用失败: {str(e)}"

# 界面配置
demo = gr.ChatInterface(
    fn=chat_cloud,
    title="🌐 云端大模型对话",
    description="基于 DeepSeek 部署的 deepseek-v4-flash 模型 (Gradio 6.18.0)",
    additional_inputs=[
        gr.Textbox(
            label="系统提示词", 
            value="你是一个乐于助人的助手。", 
            lines=3
        ),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.3, step=0.1, label="Temperature")
    ]
)

if __name__ == "__main__":
    demo.launch()