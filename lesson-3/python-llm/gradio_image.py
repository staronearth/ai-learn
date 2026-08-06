import gradio as gr
import requests
import base64
import json
import os
from io import BytesIO
from PIL import Image

# 请设置环境变量 TE CENT_API_KEY，或直接替换为您的密钥（不推荐硬编码）
API_KEY = os.getenv("TECENT_API_KEY", "sk-...")
API_URL = "https://tokenhub.tencentmaas.com/v1/chat/completions"
MODEL_NAME = "hy-vision-2.0-instruct"

def classify(img: Image.Image):
    # 将 PIL 图像转为 base64（不带头部）
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 设计 prompt：要求返回 top-3 类别及置信度（JSON 格式）
    prompt = """请分析这张图片，识别出最可能的三个物体类别（例如猫、狗、汽车等），并以 JSON 格式输出置信度，格式如下：
{"类别1": 置信度(0-1), "类别2": 置信度, "类别3": 置信度}
只输出 JSON，不要有其他文字。"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
    }
    
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # 提取模型回复内容
        content = result["choices"][0]["message"]["content"]
        # 尝试解析 JSON（模型可能输出带 markdown 的代码块）
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        scores = json.loads(content.strip())
        # 确保返回字典，键为类别字符串，值为浮点数概率
        return scores
    except Exception as e:
        # 出错时返回一个默认分类（模拟）
        print(f"API 调用失败: {e}")
        return {"错误": 1.0}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(label="上传图片", type="pil"),
    outputs=gr.Label(num_top_classes=3)
)

if __name__ == "__main__":
    demo.launch()