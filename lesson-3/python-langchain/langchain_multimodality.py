from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import os
import base64
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

model=init_chat_model(
    model=os.getenv("TECENT_MODEL_MULTI_NAME"),
    temperature=0.3,
    top_p=0.9,
    model_provider="openai",
    base_url=os.getenv("TECENT_API_URL"),
    api_key=os.getenv("TECENT_API_KEY"),
)

agent=create_agent(
    model=model,
)

def cloud_image():
    multimodal_message = HumanMessage(
        content=[
            {"type": "image", "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
            {"type": "text", "text": "这张图片描绘了什么内容？"}
        ]
    )
    for token, metadata in agent.stream(
        {"messages": [multimodal_message]},
        stream_mode="messages"
    ):
        if token.content:
            print(token.content, end="", flush=True)

def local_image():
    with open("cat.jpeg", "rb") as f:
        image_bytes = f.read()
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    multimodal_message = HumanMessage(content=[
        {"type": "image", "base64": img_b64, "mime_type": "image/jpeg"},
        {"type": "text", "text": "描述一下这张图片"}
    ])
    for token, metadata in agent.stream(
        {"messages": [multimodal_message]},
        stream_mode="messages"
    ):
        if token.content:
            print(token.content, end="", flush=True)

def homework1():
    messages=[
        SystemMessage(content="你是一个乐于助人的助手。"),
        HumanMessage(content="你好，我是图片管理员"),
        HumanMessage(content=[
            {"type": "image", "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
            {"type": "text", "text": "请描述这张图片的内容"}
        ]),
        # AIMessage(content=[
        #     {"type": "text", "text": "这张图片描绘了一个小女孩和一只狗。小女孩穿着红色的衣服，坐在草地上，微笑着看着狗。狗是一只棕色的拉布拉多犬，正坐在小女孩的旁边，似乎在享受阳光和温暖的氛围。背景是绿色的草地和蓝天，给人一种宁静和愉快的感觉。"}
        # ]),
        HumanMessage(content="我是谁？")
    ]
    for token, metadata in agent.stream(
        {"messages": messages},
        stream_mode="messages"
    ):
        if token.content:
            print(token.content, end="", flush=True)
if __name__ == "__main__":
    # cloud_image()
    # local_image()
    homework1()
