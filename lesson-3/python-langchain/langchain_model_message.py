from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage,AIMessage
from langchain.tools import tool
from langchain.agents import create_agent
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
import os
@tool
def get_current_weather(location: str) -> str:
    """
    获取当前天气信息
    Args:
        location (str): 地点名称
    """
    # 这里可以调用实际的天气API获取天气信息，这里为了演示，返回一个固定的天气信息
    return f"当前{location}的天气是：晴，温度25摄氏度"
model=init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    top_p=0.9,
)

agent=create_agent(
    model=model,
    tools=[get_current_weather]
)

def model_invoke(messages):
    response=model.invoke(messages)
    print(response.content)
def model_stream(messages):
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

def agent_invoke(question):
    answer=agent.invoke({
        "messages":[{"role":"user","content":question}]
    })
    print(f"问题: {question}\n回答: {answer}")

def agent_stream(question):
    for chunk,metadata in agent.stream({
        "messages":[{"role":"user","content":question}]},
        stream_mode="messages"
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)

def message_many_times():
    response = agent.invoke({
        "messages": [
            HumanMessage(content="你好，我是张三"),
            AIMessage(content="你好，张三！很高兴认识你。"),
            HumanMessage(content="我的名字是什么？")
            ]
        })

    # 优雅地打印整个对话历史
    for message in response['messages']:
        message.pretty_print()
if __name__=="__main__":
    # model invoke调用
    messages=[
        SystemMessage(content="你是一个乐于助人的助手。"),
        HumanMessage(content="月亮的首都是哪里？")
    ]
    # model_invoke(messages)
    # model stream调用
    # model_stream(messages)

    # agent invoke调用
    question="请告诉我北京的天气情况"
    # agent_invoke(question)

    #agent stream调用
    question="请告诉我上海的天气情况"
    # agent_stream(question)

    #多轮对话
    message_many_times()
