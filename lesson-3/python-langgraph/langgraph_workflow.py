from operator import add
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.runtime import Runtime
from langgraph.types import (
    CachePolicy,
    Command,
    RetryPolicy,
    Send,
    TimeoutPolicy,
    interrupt,
)
from pydantic import BaseModel, Field
import os
load_dotenv()


def display_graph(graph, xray=False):
    """显示图结构，xray=True 可展开子图内部结构"""
    # proxies={} 强制直连 mermaid.ink，绕过本地代理（代理下易超时）
    display(
        Image(
            graph.get_graph(xray=xray).draw_mermaid_png(
                proxies={}, max_retries=3, retry_delay=2.0
            )
        )
    )


llm = init_chat_model("deepseek-chat")
deepseek_llm=init_chat_model("deepseek-v4-pro")
qwen_llm=init_chat_model(
    model_provider="openai",
    model="qwen3.8-max",
    base_url=os.environ.get("DASHSCOPE_BASE_URL"),
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)
tecent_llm=init_chat_model(
    model=os.getenv("TECENT_MODEL_MULTI_NAME"),
    temperature=0.3,
    top_p=0.9,
    model_provider="openai",
    base_url=os.getenv("TECENT_API_URL"),
    api_key=os.getenv("TECENT_API_KEY"),
)
def prompt_chain():
    # 全局状态
    class State(TypedDict):
        topic: str
        joke: str
        improved_joke: str
        final_joke: str

    # 节点函数
    def generate_joke(state: State):
        """第一步：生成初始笑话"""
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}

    def check_punchline(state: State):
        """分支判断：校验是否包含笑点符号 ?/!"""
        if "?" in state["joke"] or "!" in state["joke"]:
            return "Pass"
        return "Fail"

    def improve_joke(state: State):
        """第二步：增加文字游戏优化笑话"""
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}

    def polish_joke(state: State):
        """第三步：增加反转收尾"""
        msg = llm.invoke(
            f"Add a surprising twist to this joke: {state['improved_joke']}"
        )
        return {"final_joke": msg.content}

    # 构建图
    workflow = chaining_graph = (
        StateGraph(State)
        .add_node("generate_joke", generate_joke)
        .add_node("improve_joke", improve_joke)
        .add_node("polish_joke", polish_joke)
        .add_edge(START, "generate_joke")
        # 条件边：通过则直接结束，不通过则进入改进流程
        .add_conditional_edges(
            "generate_joke", check_punchline, {"Pass": END, "Fail": "improve_joke"}
        )
        .add_edge("improve_joke", "polish_joke")
        .add_edge("polish_joke", END)
        .compile()
    )

    display_graph(workflow)

    # 执行
    state = workflow.invoke({"topic": "夜晚的星星"})
    print("初始的笑话:")
    print(state["joke"])
    print("\n--- --- ---\n")
    if "improved_joke" in state:
        print("优化的笑话:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")
        print("最终的笑话:")
        print(state["final_joke"])
    else:
        print("最终的笑话:")
        print(state["joke"])


def parallelization():
    class ParallelState(TypedDict):
        topic: str
        joke: str
        story: str
        poem: str
        combined_output: str

    def joke_node(state):
        return {"joke": llm.invoke(f"写一个笑话，主题词为： {state['topic']}").content}

    def story_node(state):
        return {"story": llm.invoke(f"写一个故事，主题词为： {state['topic']}").content}

    def poem_node(state):
        return {"poem": llm.invoke(f"写一首诗歌，主题词为： {state['topic']}").content}

    def aggregator(state):
        combined = f"STORY:\n{state['story']}\n\nJOKE:\n{state['joke']}\n\nPOEM:\n{state['poem']}"
        return {"combined_output": combined}

    parallel_workflow = (
        StateGraph(ParallelState)
        .add_node("joke", joke_node)
        .add_node("story", story_node)
        .add_node("poem", poem_node)
        .add_node("aggregator", aggregator)
        # START 同时连接到三个节点，实现并行
        .add_edge(START, "joke")
        .add_edge(START, "story")
        .add_edge(START, "poem")
        .add_edge("joke", "aggregator")
        .add_edge("story", "aggregator")
        .add_edge("poem", "aggregator")
        .add_edge("aggregator", END)
        .compile()
    )

    display_graph(parallel_workflow)
    state = parallel_workflow.invoke({"topic": "程序员"})
    print(state["combined_output"])


def route_workflow():
    class IntentState(TypedDict):
        query: str
        intent: str
        result: str

    class Route(BaseModel):
        step: Literal["weather", "translate", "chat"]

    router = init_chat_model("deepseek-chat").with_structured_output(Route)

    def classify_intent(state: IntentState):
        """意图识别"""
        intent = router.invoke(
            [
                SystemMessage(
                    content="你是一个意图识别助手，用户可能会问天气、翻译或闲聊。"
                ),
                HumanMessage(content=state["query"]),
            ]
        )
        return {"intent": intent.step}

    def router_fn(state):
        return state["intent"]

    def handle_weather(state):
        return {"result": f"天气查询: {state['query']} -> 晴 25度"}

    def handle_translate(state):
        return {"result": f"翻译: {state['query']} -> Hello World"}

    def handle_chat(state):
        return {"result": f"闲聊: {state['query']} -> 你好呀！"}

    router_workflow = (
        StateGraph(IntentState)
        .add_node("classify_intent", classify_intent)
        .add_node("handle_weather", handle_weather)
        .add_node("handle_translate", handle_translate)
        .add_node("handle_chat", handle_chat)
        .add_edge(START, "classify_intent")
        # 条件路由
        .add_conditional_edges(
            "classify_intent",
            router_fn,
            {
                "weather": "handle_weather",
                "translate": "handle_translate",
                "chat": "handle_chat",
            },
        )
        .add_edge("handle_weather", END)
        .add_edge("handle_translate", END)
        .add_edge("handle_chat", END)
        .compile()
    )
    display_graph(router_workflow)
    for q in ["今天北京天气如何？", "翻译hello", "你好吗？"]:
        r = router_workflow.invoke({"query": q, "intent": "", "result": ""})
        print(f"'{q}' -> intent={r['intent']} -> {r['result']}")


def orchestrator_worker_workflow():
    # 1.设置每章的章节结构体
    class Section(BaseModel):
        name: str = Field(description="报告章节名称")
        description: str = Field(description="章节内容概述")

    # 2.整个报告的章节结构体
    class Sections(BaseModel):
        sections: List[Section] = Field(description="报告全部章节列表")

    planner = llm.with_structured_output(Sections)

    # 3.全局图的状态
    class ReportState(TypedDict):
        topic: str
        sections_num: int
        sections: List[Section]
        completed_sections: Annotated[list, add]
        final_report: str

    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, add]

    # 4.Orchestrator：生成报告大纲（加强提示词）
    def orchestrator(state: ReportState):
        """Orchestrator：生成报告大纲"""
        sections = planner.invoke(
            [
                SystemMessage(
                    content="你是一个资深报告撰写专家，负责根据主题生成报告章节大纲。"
                ),
                HumanMessage(
                    content=f"请根据主题 '{state['topic']}' 生成报告的{state['sections_num']}个章节大纲，返回章节名称和内容概述。"
                ),
            ]
        )
        return {"sections": sections.sections}

    # 5.abstract_worker：根据章节大纲生成章节内容
    def abstract_worker(state: WorkerState):
        """Worker：根据章节大纲生成章节内容，通过 add reducer 汇回主状态"""
        section_content = llm.invoke(
            [
                SystemMessage(
                    content="你是一个资深报告撰写专家，负责根据章节大纲生成章节内容。"
                ),
                HumanMessage(
                    content=f"请根据章节名称 '{state['section'].name}' 和内容概述 '{state['section'].description}' 生成完整的章节内容。"
                ),
            ]
        )
        return {"completed_sections": [section_content.content]}

    # 6.abstract_aggregator：汇总所有章节内容生成最终报告
    def abstract_aggregator(state: ReportState):
        # 检查章节是否完成
        if len(state["completed_sections"]) < len(state["sections"]):
            raise NodeError("章节内容未全部生成，无法汇总最终报告。")
        final_report = "\n\n".join(state["completed_sections"])
        return {"final_report": final_report}

    # 7.动态分发工作流
    def dispatch_work(state: ReportState):
        # 为每个章节创建一个 Send 任务，目标节点为 "worker"，状态为 {"section": s}
        return [Send("worker", {"section": s}) for s in state["sections"]]

    # 8.构建workflow
    orchestrator_worker_graph = (
        StateGraph(ReportState)
        .add_node("orchestrator", orchestrator)
        .add_node("worker", abstract_worker)
        .add_node("aggregator", abstract_aggregator)
        .add_edge(START, "orchestrator")
        .add_conditional_edges("orchestrator", dispatch_work, ["worker"])
        .add_edge("worker", "aggregator")
        .add_edge("aggregator", END)
        .compile()
    )
    # 显示图
    display_graph(orchestrator_worker_graph)
    # 执行
    state = orchestrator_worker_graph.invoke(
        {
            "topic": "创建一份关于 LLM scaling laws 的报告",
            "sections_num": 3,
            "sections": [],
            "completed_sections": [],
            "final_report": "",
        }
    )

    print(state["final_report"])
    from IPython.display import Markdown

    Markdown(state["final_report"])


def evaluator_optimizer():
    class SloganScore(BaseModel):
        score: float = Field(..., description="广告语评分,范围0-10")
        grade: str = Field(..., description="广告语等级")
        feedback: str = Field(..., description="广告语反馈")

    class AdState(TypedDict):
        product: str
        slogan: str
        score: float
        grade: str
        feedback: str
        iteration: int

    qwen_llm_structured = qwen_llm.with_structured_output(SloganScore)
    tecent_llm_structured = tecent_llm.with_structured_output(SloganScore)

    def generator_slogan(state: AdState):
        msg = llm.invoke(f"为产品 {state['product']} 生成一句广告语")
        print(f"生成广告语: {msg.content}")
        return {"slogan": msg.content, "iteration": state["iteration"] + 1}

    def qwen_evaluate(state: AdState) -> SloganScore:
        return qwen_llm_structured.invoke(f"请评价广告语 {state['slogan']} 的优劣，评分的分数在0-10之间，并给出反馈")

    def tecent_evaluate(state: AdState) -> SloganScore:
        return tecent_llm_structured.invoke(f"请评价广告语 {state['slogan']} 的优劣，评分的分数在0-10之间，并给出反馈")

    def evaluator(state: AdState):
        # 多个 LLM 评分后归一化：取平均分
        evaluators = [qwen_evaluate, tecent_evaluate]
        total_score = 0.0
        feedback = ""
        for fn in evaluators:
            try:
                result = fn(state)
                total_score += result.score
                feedback += result.feedback + " "
                print(f"{fn.__name__} 评估结果: 评分={result.score}, 等级={result.grade}, 反馈={result.feedback}")
            except Exception as e:
                print(f"评估器 {fn.__name__} 出现异常: {e}")
        avg_score = total_score / len(evaluators)
        grade = "优秀" if avg_score >= 8 else "良好" if avg_score >= 5 else "一般"
        print(f"综合评估结果: 评分={avg_score}, 等级={grade}, 反馈={feedback.strip()}")
        return {"score": avg_score, "grade": grade, "feedback": feedback.strip()}

    def router_after_evaluator(state: AdState):
        print(f"广告语: {state['slogan']}, 评分: {state['score']}, 等级: {state['grade']}, 反馈: {state['feedback']}")
        if state["score"] >= 7.0:
            return "Pass"
        else:
            return "Fail"

    def human_feedback(state: AdState):
        print(f"广告语: {state['slogan']}, 评分: {state['score']}, 等级: {state['grade']}, 反馈: {state['feedback']}")
        feedback = input("请输入人工反馈(或直接回车跳过):")
        if feedback:
            state["feedback"] = feedback
        print(f"人工反馈: {state['feedback']}")
        return {"feedback": state["feedback"]}

    evaluator_optimizer_graph = (
        StateGraph(AdState)
        .add_node("generator_slogan", generator_slogan)
        .add_node("evaluator", evaluator)
        .add_node("human_feedback", human_feedback)
        .add_edge(START, "generator_slogan")
        .add_edge("generator_slogan", "evaluator")
        .add_conditional_edges(
            "evaluator",
            router_after_evaluator,
            {"Pass": END, "Fail": "human_feedback"}
        )
        .add_edge("human_feedback", "generator_slogan")
        .compile()
    )
    display_graph(evaluator_optimizer_graph)
    state = evaluator_optimizer_graph.invoke({"product": "智能手表", "slogan": "", "score": 0, "grade": "", "feedback": "", "iteration": 0})
    print(f"最终广告语: {state['slogan']}, 迭代次数: {state['iteration']}")

if __name__ == "__main__":

    # prompt_chain()
    # parallelization()
    # route_workflow()
    # orchestrator_worker_workflow()
    evaluator_optimizer()
