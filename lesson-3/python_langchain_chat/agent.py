from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import requests
from datetime import datetime
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。

    Args:
        city (str): 城市名称。

    Returns:
        str: 天气信息。
    """
    url=f"http://wttr.in/{city}?format=j1"
    try:
        response=requests.get(url).json()
        current = response.get("current_condition", [{}])[0]
        if current:
            return f"{city}当前温度：{current['temp_C']}°C，天气：{current['weatherDesc'][0]['value']}"
        return f"未找到{city}的天气信息"
    except Exception as e:
        return f"获取天气信息时出错：{str(e)}"

@tool
def get_current_time() -> str:
    """
    获取当前时间。

    Returns:
        str: 当前时间。
    """

    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

tavily = TavilySearch(max_results=5, topic="general")
@tool
def web_search(query: str) -> str:
    """
    搜索互联网。

    Args:
        query (str): 搜索查询。

    Returns:
        str: 搜索结果。
    """
    return tavily.invoke(query)

@tool
def calculate(query: str) -> str:
    """
    计算。

    Args:
        query (str): 计算表达式。

    Returns:
        str: 计算结果。
    """
    return str(eval(query))

@tool
def load_skill(skill_name: str) -> str:
    """当需要特定专业知识时，使用此工具加载对应的技能。"""
    # 这里实现从文件系统加载 SKILL.md 内容的逻辑
    if skill_name == "translate-expert":
        return """# 翻译专家 ... (这里是 SKILL.md 的全部内容) ..."""
    if skill_name == "news-briefing":
        return """# 新闻简报生成专家 ... (这里是 SKILL.md 的全部内容) ..."""
    return "技能未找到。"

tecent_model = init_chat_model(
    model=os.getenv("TECENT_MODEL_MULTI_NAME"),
    temperature=0.3,
    top_p=0.9,
    model_provider="openai",
    base_url=os.getenv("TECENT_API_URL"),
    api_key=os.getenv("TECENT_API_KEY"),
)
deepseek_model = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.3,
    top_p=0.9,
)

agent = create_agent(
    model=deepseek_model,
    tools=[get_weather, get_current_time, web_search, calculate,load_skill],

    system_prompt=(
        "你是一个智能助手，使用工具解决问题。回答时请引用信息来源。"
         "当你需要进行专业翻译时，可以使用 `load_skill` 工具加载 'translate-expert' 技能。"
         "当你需要进行新闻简报生成时，可以使用 `load_skill` 工具加载 'news-briefing' 技能。"
    )
)

async def stream_agent_response(query: str):
    """
    流式响应代理。

    Args:
        query (str): 用户查询。

    Yields:
        str: 流式响应块。
    """
    async for chunk,metadata in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages"
    ):
        if hasattr(chunk, "content"):
            yield chunk.content
