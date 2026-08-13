### 综合运用：学习助手 Agent
# ○ State：记录本次学习中问了多少个问题
# ○ Store：保存用户设定的"学习风格"（如"喜欢举例"、"学术严谨"）
# ○ Context：调用时传入"当前学习的科目"（如"Python"）
# ○ 功能：结合"学习风格"和"当前科目"给出个性化回答

from dataclasses import dataclass
from typing import NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, before_agent, dynamic_prompt
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

STYLE_NAMESPACE = ("learn_style",)


class QuestionState(AgentState):
  """State：记录本次学习累计提问数"""
  question_num: NotRequired[int]


@dataclass
class LearnContext:
  """Context：当前用户与正在学习的科目"""
  user_id: str
  subject: str


@before_agent
def count_question(state, runtime):
  """agent 每被调用一次即代表用户提了一个问题，自动累加提问数。
  风格设置走 REST 接口、科目切换走 Context，均不经过 agent，不会被计数"""
  return {"question_num": state.get("question_num", 0) + 1}


def get_learn_style(store: InMemoryStore, user_id: str) -> str:
  item = store.get(STYLE_NAMESPACE, user_id)
  return item.value.get("style", "未设置") if item else "未设置"


@dynamic_prompt
def learn_prompt(request: ModelRequest) -> str:
  """动态系统提示词：注入当前科目（Context）与学习风格（Store）"""
  context = request.runtime.context
  subject = context.subject if context else "未知科目"
  style = "未设置"
  store = request.runtime.store
  if store is not None and context is not None:
    style = get_learn_style(store, context.user_id)
  question_num = request.state.get("question_num", 0)
  return (
    f"你是一个学习助手。当前学习科目: {subject}；用户学习风格偏好: {style}。"
    f"回答时必须紧密结合该科目和学习风格，让讲解更有针对性。"
    f"本次会话已累计提问 {question_num} 次。"
    "学习风格的设置由页面设置区完成，若用户在对话中要求修改风格，请引导其使用页面设置。"
  )


_store = InMemoryStore()
_agent = None


def get_store() -> InMemoryStore:
  return _store


def get_agent():
  global _agent
  if _agent is None:
    _agent = create_agent(
      "deepseek-chat",
      middleware=[count_question, learn_prompt],
      state_schema=QuestionState,
      context_schema=LearnContext,
      store=_store,
      checkpointer=InMemorySaver(),
    )
  return _agent


def ask(user_id: str, subject: str, thread_id: str, question: str) -> dict:
  """调用 agent 回答一个问题（同步，一次性返回）"""
  config = {"configurable": {"thread_id": thread_id}}
  context = LearnContext(user_id=user_id, subject=subject)
  result = get_agent().invoke(
    {"messages": [HumanMessage(question)]}, config, context=context
  )
  return {
    "answer": result["messages"][-1].content,
    "question_num": result.get("question_num", 0),
  }


async def ask_stream(user_id: str, subject: str, thread_id: str, question: str):
  """异步流式调用 agent：逐 token 产出事件，最后产出累计提问数"""
  config = {"configurable": {"thread_id": thread_id}}
  context = LearnContext(user_id=user_id, subject=subject)
  agent = get_agent()
  async for chunk, metadata in agent.astream(
    {"messages": [HumanMessage(question)]},
    config,
    context=context,
    stream_mode="messages",
  ):
    if metadata.get("langgraph_node") == "model" and isinstance(chunk.content, str):
      yield {"type": "token", "content": chunk.content}
  state = await agent.aget_state(config)
  yield {"type": "done", "question_num": state.values.get("question_num", 0)}


if __name__ == "__main__":
  store = get_store()
  store.put(STYLE_NAMESPACE, "user_001", {"style": "喜欢举例"})
  print(ask("user_001", "Python", "learn-001", "什么是装饰器？"))
  print(ask("user_001", "Python", "learn-001", "再讲讲生成器？"))
