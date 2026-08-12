from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, CachePolicy, interrupt
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.errors import NodeError

from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, ToolMessage
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from IPython.display import Image, display
from operator import add
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from sqlalchemy.sql.type_api import TypeDecorator
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from operator import add
from typing import Annotated, TypedDict
load_dotenv()

def display_with_matplotlib(image_path: str):
    img = PILImage.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def test_langgraph_helloworld():
    #定义state
    class SimpleState(TypedDict):
        name:str
        greeting:str

    def greeting_node(state:SimpleState):
        print(f"接收到name:{state['name']}")
        return {'greeting':f"Hello,{state['name']}!"}

    def uppercase_node(state:SimpleState):
        print(f"接收到问候语气{state['greeting']}")
        return {"greeting":state['greeting'].upper()}

    graph_builder=StateGraph(SimpleState)

    graph_builder.add_node('a',greeting_node)
    graph_builder.add_node('b',uppercase_node)

    graph_builder.add_edge(START,'a')
    graph_builder.add_edge('a','b')
    graph_builder.add_edge('b',END)

    graph=graph_builder.compile()

    result=graph.invoke({"name":"langgraph"})
    print(result['greeting'])

def test_langgraph_overrite_state():
    class DefaultReducerState(TypedDict):
        val:int

    def first_node(state:DefaultReducerState):
        print(f"节点1的接收值:{state['val']}")
        return {"val":"first_node"}

    def second_node(state:DefaultReducerState):
        print(f"节点2的接收值:{state['val']}")
        return {"val":"second_node"}

    graph=(
        StateGraph(DefaultReducerState)
        .add_node("first_node",first_node)
        .add_node("second_node",second_node)
        .add_edge(START,"first_node")
        .add_edge("first_node","second_node")
        .add_edge("second_node",END)
    ).compile()
    result=graph.invoke({"val":"atsrt"})
    print(result)


def define_reduce():
    class DefineReducerState(TypedDict):
        num:int
        nodes:Annotated[list[str],add]

    # 2. Node: 定义处理逻辑
    def node_1(state: DefineReducerState):
        print(f"节点1接收num: {state['num']}")
        return {"num": 1, "nodes": ["node_1"]} # 直接修改nodes值，不用自己拼接

    def node_2(state: DefineReducerState):
        print(f"节点2接收num: {state['num']}")
        return {"num": 2, "nodes": ["node_2"]} # 直接修改nodes值，不用自己拼接

    # 3. Edge: 编排执行顺序
    custom_reducer_graph = (
        StateGraph(DefineReducerState)
        .add_node("node_1", node_1)
        .add_node("node_2", node_2)
        .add_edge(START, "node_1")
        .add_edge("node_1", "node_2")
        .add_edge("node_2", END)
        .compile()
    )

    result = custom_reducer_graph.invoke(DefineReducerState(num=0, nodes=[]))
    print(result)

def test_parallel_edge():
    class ParallelState(TypedDict):
        nodes:Annotated[list[str],add]
    def node_a(state:ParallelState):
       return {"nodes":["node_a"]}

    def node_b(state:ParallelState):
        return {"nodes":["node_b"]}

    def node_c(state:ParallelState):
        return {"nodes":["node_c"]}

    graph=(
        StateGraph(ParallelState)
        .add_node("node_a",node_a)
        .add_node("node_b",node_b)
        .add_node("node_c",node_c)
        .add_edge(START,"node_a")
        .add_edge(START,"node_b")
        .add_edge("node_b","node_c")
        .add_edge("node_a","node_c")
        .add_edge("node_c",END)
    ).compile()
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("图形已保存到 graph.png")
    display_with_matplotlib("graph.png")
    result=graph.invoke(ParallelState(nodes=[]))
    print(result)

def test_condition_edge():
    class CondationState(TypedDict):
        score:int
        grade:Annotated[list[str],add]
    def node_a(state:CondationState):
       return {"grade":["优秀"]}

    def node_b(state:CondationState):
        return {"grade":["普通"]}

    def node_c(state:CondationState):
        return {"grade":["及格"]}

    def node_d(state:CondationState):
        return {"grade":["不及格"]}

    def score(state:CondationState):
        score = int(input("请输入分数:"))
        print(score)
        return {"score": score}
    def judge_grade(state:CondationState)->Literal["node_a","node_b","node_c","node_d"]:
        score=state['score']
        if score>=90:
            return "node_a"
        elif score>=80:
            return "node_b"
        elif score>=60:
            return "node_c"
        else:
            return "node_d"
    graph=(
        StateGraph(CondationState)
        .add_node("score",score)
        .add_node("node_a",node_a)
        .add_node("node_b",node_b)
        .add_node("node_c",node_c)
        .add_node("node_d",node_d)
        .add_edge(START,"score")
        .add_conditional_edges("score",judge_grade)
        .add_edge("node_a",END)
        .add_edge("node_b",END)
        .add_edge("node_c",END)
        .add_edge("node_d",END)
    ).compile()
    png_data = graph.get_graph().draw_mermaid_png()
    with open("condation_graph.png", "wb") as f:
        f.write(png_data)
    print("图形已保存到 condation_graph.png")
    display_with_matplotlib("condation_graph.png")
    result=graph.invoke(CondationState(score=0,grade=[]))
    print(result)

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    weather_data = {"北京": "晴天 25度", "上海": "多云 28度"}
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def square_root(x: float) -> str:
    """计算平方根"""
    return str(x ** 0.5)

def create_agent_graph():
    from langgraph.graph import StateGraph, START, END
    tools=[get_weather,square_root]
    tools_by_name={t.name: t for t in tools}

    llm=init_chat_model(
        "deepseek-v4-pro"
    )
    llm_tool=llm.bind_tools(tools)
    def llm_node(state:MessagesState):
        """LLM 节点：调用模型，可能产生工具调用请求"""
        response=llm_tool.invoke(state['messages'])
        return {"messages":response}

    def tool_node(state:MessagesState):
        """工具节点：执行 LLM 请求的工具，并封装结果"""
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        return {"messages": results}

    def condation(state:MessagesState)->Literal["tools",END]:
        last_messages=state['messages'][-1]
        if hasattr(last_messages,"tool_calls") and last_messages.tool_calls:
            return "tools"
        return END

    agent_graph=(
        StateGraph(MessagesState)
        .add_node("llm_node",llm_node)
        .add_node("tools",tool_node)
        .add_edge(START,"llm_node")
        .add_conditional_edges("llm_node",condation)
        .add_edge("tools","llm_node")
        .compile()
    )
    png_data = agent_graph.get_graph().draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_data)
    print("图形已保存到 agent_graph.png")
    display_with_matplotlib("agent_graph.png")
    print("测试1: 不需要工具调用\n")
    result = agent_graph.invoke({
        "messages": [HumanMessage(content="你好")]
    })
    for m in result['messages']:
        m.pretty_print()

    print("测试2: 需要工具调用\n")
    result = agent_graph.invoke({
        "messages": [HumanMessage(content="北京今天天气怎么样？")]
    })
    for m in result['messages']:
        m.pretty_print()

if __name__=="__main__":
    # test_langgraph_helloworld()
    # test_langgraph_overrite_state()
    # define_reduce()
    # test_parallel_edge()
    # test_condition_edge()
    create_agent_graph()
