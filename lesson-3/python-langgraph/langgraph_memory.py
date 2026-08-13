from datetime import datetime
from typing import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph_cli.schemas import IndexConfig
from langgraph.store.memory import InMemoryStore
from langchain_community.embeddings import DashScopeEmbeddings
from dataclasses import dataclass
import os
load_dotenv()

class CustomState(AgentState):
    """agent自定义任务状态"""
    model_call_count:NotRequired[int]
    session_start:NotRequired[str]


def update_state(runtime: ToolRuntime):
    """Update custom state: increment model_call_count and record session_start."""
    count = runtime.state.get("model_call_count", 0)
    command={
        "model_call_count":count+1,
        "messages":[ToolMessage("状态已经更新",tool_call_id=runtime.tool_call_id)]
    }
    if count==0:
        command["session_start"]=datetime.now().isoformat()

    return Command(update=command)

def test_use_custom_state():
    agent=create_agent(
        "deepseek-chat",
        tools=[update_state],
        state_schema=CustomState,
        checkpointer=InMemorySaver(),
        system_prompt="你是一个助手，每次收到用户消息都必须调用 update_state 工具。",
    )
    config={"configurable":{"thread_id":"hualu-001"}}
    result=agent.invoke({"messages":[HumanMessage("您好，我是startonearth")]},config)
    for m in result['messages']:
        m.pretty_print()

    print("\n\n第 2 轮对话")
    result = agent.invoke({"messages": [HumanMessage("讲个小红帽的故事")]}, config)
    for m in result['messages']:
        m.pretty_print()
    print("\n\n第 3 轮对话")
    result = agent.invoke({"messages": [HumanMessage("你知道我是谁吗？")]}, config)
    for m in result['messages']:
        m.pretty_print()


def define_store():
    from langgraph.store.memory import InMemoryStore
    store=InMemoryStore()
    store.put(("preferences",), "user_001", {"style": "business", "language": "zh-CN"})

    result=store.get(("preferences",), "user_001")  # 返回 {"style": "business", "language": "zh-CN"}
    print(result)

    result=store.search(("preferences",), filter={"style": "business"})  # 返回 [("user_001", {"style": "business", "language": "zh-CN"})]
    print(result)

def use_embedding_store():

    store=InMemoryStore(index=IndexConfig(
        embed=DashScopeEmbeddings(model="qwen3.7-text-embedding",dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")),
        dimis=1024,
    ))
    store.put(("documents",), "doc_001", "这是一个关于人工智能的文档。")
    store.put(("documents",), "doc_002", "这是一个关于机器学习的文档。")

    result=store.get(("documents",), "doc_001")  # 返回 "这是一个关于人工智能的文档。"
    print(result)

    result=store.search(("documents",), query="人工智能")  # 返回 [("doc_001", "这是一个关于人工智能的文档。")]
    print(result)

def user_store_example():

    # 1. 创建语义搜索 Store
    store = InMemoryStore(index=IndexConfig(
        embed=DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")),
        dims=1024  # 向量维度
    ))

    # 2. 存储数据（与基础 Store 完全一样）
    store.put(("users",), "user_001", {"name": "白泽", "department": "总裁办"})
    store.put(("users",), "user_002", {"name": "张三", "department": "技术部"})
    store.put(("users",), "user_003", {"name": "李四", "department": "市场部"})

    # 3. 语义搜索（用自然语言查询，而不是死板的字段匹配）
    results = store.search(("users",), query="技术人员", limit=5)
    # 张三会因“技术部”与查询“技术人员”语义相近而排在首位
    print(results)
    return store


@tool
def get_user_info(user_id, runtime: ToolRuntime):
    """获取用户信息"""
    # 从 Store 中获取用户信息
    if runtime.store is None:
        return "Store not available"

    # 通过runtime获取store，读取其中的数据
    user_info = runtime.store.get(("users",), user_id)

    if user_info is None:
        return "没有找到用户"

    return f"用户信息: {user_info.value}"

def test_user_store():
    store=user_store_example()
    agent=create_agent(
        "deepseek-chat",
        tools=[get_user_info],
        store=store,
        state_schema=CustomState,
        checkpointer=InMemorySaver(),
        system_prompt="你是一个助手，用户可以通过用户ID查询用户信息。",
    )
    config={"configurable":{"thread_id":"hualu-001"}}
    result=agent.invoke({"messages":[HumanMessage("请帮我查询用户ID为 user_002 的信息。")]},config)
    for m in result['messages']:
        m.pretty_print()

@dataclass
class UserProfile:
    user_id: str

@tool
def get_my_profile(runtime:ToolRuntime[UserProfile]):
    """获取当前用户的偏好设置（从偏好存储中读取）"""
    if runtime.state is None:
        return "没有用户信息"
    user_id=runtime.context.user_id
    preferences=runtime.store.get(("preferences",), user_id)
    if preferences is None:
        return "没有找到用户偏好信息"
    return f"您的偏好信息: {preferences.value}"

@tool
def set_my_profile(key: str, value: str,runtime:ToolRuntime[UserProfile]):
    """
    保存用户的偏好设置（键值对形式），例如：
    key: "style", value: "简洁易懂"
    key: "language", value: "中文"
    """
    user_id=runtime.context.user_id
    item=runtime.store.get(("preferences",), user_id)
    preferences=item.value if item is not None else {"style": "default", "language": "zh-CN"}
    preferences[key]=value
    runtime.store.put(("preferences",), user_id, preferences)
    return f"已更新您的偏好信息: {preferences}"

def test_user_profile():
    store=InMemoryStore()
    # 初始化用户偏好信息
    store.put(("preferences",), "user_001", {"style": "business", "language": "zh-CN"})

    agent=create_agent(
        "deepseek-chat",
        tools=[get_my_profile, set_my_profile],
        store=store,
        checkpointer=InMemorySaver(),
        context_schema=UserProfile,
        system_prompt="你是一个助手，用户可以查询和设置自己的偏好信息。",
    )
    config={"configurable":{"thread_id":"hualu-001"}}
    context=UserProfile(user_id="user_001")
    result=agent.invoke({"messages":[HumanMessage("请帮我查询我的偏好设置。")]},config,context=context)
    for m in result['messages']:
        m.pretty_print()

    result=agent.invoke({"messages":[HumanMessage("请帮我设置我的偏好：style为简洁易懂，language为中文。")]},config,context=context)
    for m in result['messages']:
        m.pretty_print()

    result=agent.invoke({"messages":[HumanMessage("请再次帮我查询我的偏好设置。")]},config,context=context)
    for m in result['messages']:
        m.pretty_print()
    config={"configurable":{"thread_id":"hualu-002"}}
    result=agent.invoke({"messages":[HumanMessage("请再次帮我查询我的偏好设置。")]},config,context=context)
    for m in result['messages']:
        m.pretty_print()
if __name__=="__main__":
    # test_use_custom_state()
    # define_store()
    # use_embedding_store()
    # store=user_store_example()
    test_user_profile()
    # test_user_store()
