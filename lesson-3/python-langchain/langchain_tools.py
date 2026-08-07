from tempfile import tempdir
from venv import create
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel,Field
from typing import Literal
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from pydantic.types import Base64Bytes
load_dotenv()
@tool
def get_current_weather(location: str) -> str:
    """
    获取当前天气信息
    Args:
        location (str): 地点名称
    """
    # 这里可以调用实际的天气API获取天气信息，这里为了演示，返回一个固定的天气信息
    return f"当前{location}的天气是：晴，温度25摄氏度"
@tool("square_number", description="计算一个数字的平方")
def square_number(number: int) -> int:
    return number * number
@tool("square_root", description="计算一个数字的平方根")
def tool1(num: int) -> int:
    return num ** 0.5
class WeatherInput(BaseModel):
    location: str=Field(..., description="地点名称")
    units: Literal["celsius","fahrenheit"]=Field("celsius", description="温度单位")
    include_forecast: bool=Field(False, description="是否包含预报")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: Literal["celsius","fahrenheit"] = "celsius", include_forecast: bool = False) -> str:
    """
    获取当前天气信息
    Args:
        location (str): 地点名称
        units (str): 温度单位，默认为摄氏度
        include_forecast (bool): 是否包含预报，默认为False
    """
    temp=22 if units == "celsius" else 72
    result=f"当前{location}的天气是：晴，温度{temp}摄氏度"
    if include_forecast:
        result += "，预报：Next 5 days is sunny"
    return result

tavily = TavilySearch(max_results=5, topic="general")

@tool
def web_search(query: str):
    """Search the web for information"""
    return tavily.invoke(query)

agent = create_agent(
    "deepseek-v4-flash",
    tools=[get_current_weather, square_number,get_weather,tool1,web_search]
)
def agent_invoke(question):
    answer = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print(f"问题: {question}\n回答: {answer}")

def user_agents():
    # 调用智能体
    for chunk in agent.stream(
        {"messages": [HumanMessage(content="467的平方根是多少?")]},
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
            print()


    for chunk in agent.stream(
        {"messages": [HumanMessage(content="北京和杭州接下来几天天气如何?")]},
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
            print()

    for chunk in agent.stream(
        {"messages": [HumanMessage(content="搜索一下大模型今天的最新消息?")]},
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
            print()



def structed_response_agent():
    class Reference(BaseModel):
        title: str = Field(description="The title of the cited web page")
        url: str = Field(description="The url of the cited web page")

    class AnswerInfo(BaseModel):
        answer: str = Field(description="The final answer for user")
        reference: list[Reference] = Field(description="The web pages cited")
    structed_agent = create_agent(
        "deepseek-chat",
        tools=[web_search],
        system_prompt="你是一个智能助手，使用工具解决问题。回答时请引用信息来源。",
        response_format=AnswerInfo,
    )
    result = structed_agent.invoke({"messages": [HumanMessage(content="蒸蚌是什么梗？")]})
    print(result)

if __name__ == "__main__":
    structed_response_agent()
