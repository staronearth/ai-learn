import json
import os
import sys
import time
import types
from telnetlib import AYT

sys.path.insert(0, os.path.dirname(__file__))

# langchain-community>=0.4 移除了 chat_models.vertexai 子模块，
# 但 ragas 0.4.3 仍从该旧路径导入 ChatVertexAI。
# 注入兼容 shim，重定向到已安装的 langchain-google-vertexai。
_chat_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
from langchain_google_vertexai import ChatVertexAI  # noqa: E402

_chat_vertexai.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _chat_vertexai

import httpx
from agentic_rag import rag_agent, recorder
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import Dataset
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

load_dotenv()


def create_dataset():
    """创建dataset，先尝试本地加载，如果没有则重新生成"""

    # 1. 创建 ragas Dataset
    dataset = Dataset(name="rag_eval", backend="local/csv", root_dir="./experiments")
    if os.path.exists("./experiments/datasets/rag_eval.csv"):
        # 尝试读取本地数据
        dataset.reload()
    # 如果有数据，直接返回，没有则重新生成数据集
    if len(dataset) > 0:
        print(f"加载了 {len(dataset)} 个测试数据集，不再重新生成")
        return dataset

    # 2. 加载评估问题（《白鹿原》相关）
    with open("./docs/ragas_eval_dataset.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"加载了 {len(eval_data)} 个评估问题")

    # 3.调用Agent，生成response和context
    # 每次 Agent 调用后的等待时间（秒），避免触发模型提供商限流
    # DeepSeek 免费版建议 1~2s，付费版可适当降低，根据实际报错调整
    REQUEST_DELAY = 1

    print(f"开始收集回答和上下文（共 {len(eval_data)} 题，间隔 {REQUEST_DELAY}s）...")
    print("-" * 50)

    for i, item in enumerate(eval_data):
        query = item["question"]

        # 调用 Agent —— 内部 search_knowledge_base tool 会将检索结果存入 contexts_store
        response = rag_agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": f"eval-{i}"}},
        )
        answer = response["messages"][-1].content

        # 从 tool 的 store 中取 context（一次检索，不重复）
        contexts = recorder.responses

        # 写入数据集
        dataset.append(
            {
                "user_input": query,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )

        print(
            f"[{i + 1}/{len(eval_data)}] {query[:40]}... ✓ (contexts: {len(contexts)}条)"
        )

        # 等待一会儿再发下一个请求，避免触发限流
        if i < len(eval_data) - 1:  # 最后一条不用等
            time.sleep(REQUEST_DELAY)

    dataset.save()
    print(f"\n评估数据集已保存: {len(eval_data)} 个样本 -> ./experiments/rag_eval/")
    return dataset


def create_judge_mode():
    timeout = httpx.Timeout(120.0, connect=60.0)
    http_client = httpx.AsyncClient(timeout=timeout)

    qwen_client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        http_client=http_client,
        max_retries=3,
    )

    # 推理模型（qwen 支持的模型）
    evaluator_llm = llm_factory(
        model="qwen3.8-max",  # 可更换为 "deepseek-ai/DeepSeek-V3" 等
        client=qwen_client,
        max_tokens=4096,  # 适当减小避免超时
    )

    # Embedding 模型（qwen 也提供）
    evaluator_embeddings = embedding_factory(
        model="Qwen/Qwen3-Embedding-0.6B",  # 或其他 SiliconFlow 上的 embedding 模型
        client=qwen_client,
    )

    print("评估器已配置完成（使用 qwen）")

    return evaluator_llm, evaluator_embeddings


def test_agent_ragas():
    # 调用函数，创建dataset
    pass


def judge_answer(dataset, evaluator_llm, evaluator_embeddings):
    results = []
    for i in range(len(dataset)):
        row = dataset.__getitem__(i)
        user_input = row.get("user_input")
        response = row.get("response")
        retrieved_contexts = row.get("retrieved_contexts")
        reference = row.get("reference")
        # 6个核心指标（从 collections 导入）
        from ragas.metrics.collections import (
            AnswerRelevancy,  # 回答相关性
            ContextEntityRecall,  # 上下文实体召回
            ContextPrecision,  # 上下文精度
            ContextRecall,  # 上下文召回
            Faithfulness,  # 忠实度
            NoiseSensitivity,  # 噪声敏感度
        )

        print(f"========开始评估：input:{user_input}===========")
        # ---- 检索阶段指标 ----
        cp = ContextPrecision(llm=evaluator_llm)

        cp_result = await cp.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        print(f"运行完成context precision评估,score: {cp_result},进度1/6\n")

        cr = ContextRecall(llm=evaluator_llm)
        cr_result = await cr.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        print(f"运行完成context recall评估,score: {cr_result},进度2/6\n")

        cer = ContextEntityRecall(llm=evaluator_llm)
        cer_result = await cer.ascore(
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        print(f"运行完成ContextEntityRecall评估,score: {cer_result},进度3/6\n")
        # ---- 生成阶段指标 ----
        faith = Faithfulness(llm=evaluator_llm)

        faith_result = await faith.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        print(f"运行完成Faithfulness评估,score: {faith_result},进度4/6\n")

        ar = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        ar_result = await ar.ascore(
            user_input=user_input,
            response=response,
        )
        print(f"运行完成AnswerRelevancy评估,score: {ar_result},进度5/6\n")

        ns = NoiseSensitivity(llm=evaluator_llm)
        ns_result = await ns.ascore(
            user_input=user_input,
            response=response,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        print(f"运行完成NoiseSensitivity评估,score: {ns_result},进度6/6\n")
        print(f"========================评估完成==========================")
    return results


if __name__ == "__main__":
    dataset = create_dataset()
    print(dataset.__getitem__(3))
    evaluator_llm, evaluator_embeddings = create_judge_mode()
    results = judge_answer(dataset, evaluator_llm, evaluator_embeddings)
    for result in results:
        print(result)
