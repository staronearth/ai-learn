from operator import ge
from webbrowser import get

from langchain_core.language_models import llms
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from datetime import datetime
import os
from dotenv import load_dotenv
from pandas._libs.index import multiindex_nulls_shift
from streamlit import progress
from urllib3 import response
import requests

load_dotenv()

tavily = TavilySearch(max_results=5, topic="general")

def web_search(query: str) -> str:
    return tavily.invoke(query)

def get_current_time() -> str:
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

def get_weather(city: str) -> str:
    url=f"http://wttr.in/{city}?format=j1"
    try:
        response=requests.get(url).json()
        current = response.get("current_condition", [{}])[0]
        if current:
            return f"{city}当前温度：{current['temp_C']}°C，天气：{current['weatherDesc'][0]['value']}"
        return f"未找到{city}的天气信息"
    except Exception as e:
        return f"获取天气信息时出错：{str(e)}"
def sample_chain( topic: str, question: str) -> str:

    llm = ChatDeepSeek(
        model="deepseek-chat",
    )

    prompt = ChatPromptTemplate.from_template(
        f"回答关于{topic}的问题{question}"
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({ "topic": topic, "question": question})

def get_context(topic: str) -> str:
    """模拟从数据库或知识库或联网搜索上下文"""
    context_map={
        "世界杯": "2026年世界杯的冠军是西班牙",
        "人工智能": "人工智能是一门研究如何让计算机模拟人类智能的学科",
    }
    return context_map.get(topic, f"关于{topic}的通用信息")

def runnable_pallel_chain():
    llm = ChatDeepSeek(
        model="deepseek-chat",
    )

    prompt = ChatPromptTemplate.from_template(
        "你是一个问答助手，请根据以下上下文回答问题：\n\n{context}\n\n关于{topic}问题：{question}"
    )

    multi_input_chain = (
        RunnableParallel({
            "context": lambda x: get_context(x["topic"]),
            "topic": lambda x: x["topic"],
            "question": lambda x: x["question"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    response = multi_input_chain.invoke({ "topic": "世界杯", "question": "2026年世界杯的冠军是谁？" })
    print(response)

def runnable_pass_through_chain():
    llm = ChatDeepSeek(
        model="deepseek-chat",
    )

    prompt = ChatPromptTemplate.from_template(
        "你是一个问答助手，请根据以下上下文回答问题：\n\n{context}\n\n关于{topic}问题：{question}"
    )

    chain=RunnablePassthrough.assign(
        context=lambda x: get_context(x["topic"]),
    )|prompt|llm|StrOutputParser()

    response = chain.stream({ "topic": "人工智能", "question": "详细介绍一下人工智能" })
    for chunk in response:
        print(chunk, end="", flush=True)

def product_design_chain():
    llm = ChatDeepSeek(
        model="deepseek-chat",
    )

    name_prompt = ChatPromptTemplate.from_template(
        "为{product}起一个简洁的产品名称，只需输出名称"
    )
    name_chain = name_prompt | llm | StrOutputParser()
    design_prompt = ChatPromptTemplate.from_template(
        "根据产品类型{product}和产品的{name},生成一段50字以内的产品设计描述"
    )
    design_chain = RunnablePassthrough.assign(
        name=name_chain,
    ) | design_prompt | llm | StrOutputParser()

    full_chain=RunnableParallel(
        product_name=name_chain,
        design=design_chain,
    )

    response = full_chain.invoke({ "product": "人脑智能芯片"})
    # result = full_chain.invoke({"product": "智能手表"})
    print(f"产品名称: {response['product_name']}")
    print(f"产品描述: {response['design']}")

def use_tool_chain():
    llm = ChatDeepSeek(
        model="deepseek-chat",
    )

    prompt = ChatPromptTemplate.from_template(
        "当前时间是：{current_time}\n"
        "网络搜索结果：\n{search_result}\n\n"
        "用户原始问题：{query}\n"
        "请结合当前时间点和最新搜索结果，准确回答用户的问题。"
    )
    chain = RunnablePassthrough.assign(
        current_time=lambda _: datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'),
    )|RunnablePassthrough.assign(
        search_result= lambda x: tavily.invoke(x)
    ) | prompt | llm | StrOutputParser()

    result = chain.stream({"query": "今年世界杯冠军的每场的得分"})
    for chunk in result:
        print(chunk, end="", flush=True)

def runnable_branch_chain():
    llm = ChatDeepSeek(
        model="deepseek-chat",
    )
    python_prompt = ChatPromptTemplate.from_template(
        "你是一个python专家，请用技术术语回答，包含示例：\n问题:{question}"
    )
    python_chain = python_prompt | llm | StrOutputParser()

    city_prompt = ChatPromptTemplate.from_template("获取{question}中的城市只输出城市名称")
    city_chain = city_prompt | llm | StrOutputParser()
    weather_prompt = ChatPromptTemplate.from_template("获取{city}的天气,通过查询的情况{weather}")
    weather_chain =   RunnableParallel(
        {"question": lambda x: x}
    ) | RunnablePassthrough.assign(
        city=city_chain
    ) | RunnablePassthrough.assign(
        weather=lambda x: get_weather(x["city"])
    ) | weather_prompt | llm | StrOutputParser()
    general_prompt = ChatPromptTemplate.from_template("请简洁回答（不超过100字）：\n问题：{question}")
    general_chain = general_prompt | llm | StrOutputParser()

    # branch_chain=RunnableParallel(
    #     {"question": lambda x: x["question"]}
    # )|RunnableBranch(
    #     (lambda x: "python" in x["question"], python_chain),
    #     (lambda x: "天气" in x["question"], weather_chain),
    #     general_chain,
    # )
    def route_by_keyword(question: str):
        if "python" in question.lower():
            return python_chain
        elif "天气" in question:
            return weather_chain
        else:
            return general_chain
    branch_chain= RunnableLambda(route_by_keyword)
    # print(branch_chain.invoke({"question":"Python中如何反转列表？"}))
    # print(branch_chain.invoke({"question":"西安今天天气怎么样？"}))
    # print(branch_chain.invoke({"question":"你好"}))
    print(branch_chain.invoke("Python中如何反转列表？"))
    print(branch_chain.invoke("西安今天天气怎么样？"))
    print(branch_chain.invoke("你好"))

def loop_chain():
    class Idiomchain:
        """成语接龙——每次用上一个成语作为输入"""
        def __init__(self, llm):
            self.prompt = ChatPromptTemplate.from_template("接龙成语'{prev}'的下一个成语是：\n注意：你只需要输出下一个成语，无需解释说明")
            self.chain = self.prompt | llm | StrOutputParser()
        def __call__(self, prev: str):
            return self.chain.invoke({"prev": prev})
    llm=ChatDeepSeek(
        model="deepseek-chat",
    )
    chain = Idiomchain(llm)

    start="一心一意"

    for i in range(10):
        print(chain(start))
        start = chain(start)
if __name__ == "__main__":
    # result = sample_chain("世界杯", "今年世界杯的冠军是谁？")
    # print(result)
    # runnable_pallel_chain()
    # runnable_pass_through_chain()
    # product_design_chain()
    # use_tool_chain()
    # runnable_branch_chain()
    loop_chain()
