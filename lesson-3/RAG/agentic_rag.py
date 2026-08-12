import os
import sys
import threading
import time
from typing import Dict, List, Tuple

import bm25s
import jieba
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from numpy._core.numeric import promote_types

sys.path.insert(0, os.path.dirname(__file__))
from qwen_ranker import QWENRanker

load_dotenv()


# 实现一个计时装饰器
def timmer(func):
    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time} 秒")
        return result

    return wrapper


idx = 1


def handle_doc_header(doc: Document):
    global idx
    m = doc.metadata
    # 安全获取 Header 3
    header = m.get("Header 3", "")
    doc.page_content = f"### {header}\n{doc.page_content}"
    doc.id = f"doc_{idx}"
    idx += 1
    return doc


def recursive_char_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # chunk切分的片段大小
        chunk_overlap=100,  # 重复片段大小
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],  # 优先级从高到低
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"原始文档数: {len(split_docs)}，切分后块数: {len(split_docs)}")
    # for doc in split_docs:
    #     print(f"[{doc.metadata}] {doc.page_content[:500]}")
    return split_docs


def splitter_markdown():
    # 1.loader加载数据
    global idx
    idx = 1
    with open("docs/白鹿原.md", "r", encoding="utf-8") as f:
        data = "".join(line for line in f.readlines())
    # 2.文本分割
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    chunks = splitter.split_text(data)
    # 这里还需要切分处理一下，文本太长了。需要按照段落或者句号切分开来

    # 3.给每个章节添加id
    ds = [handle_doc_header(doc) for doc in recursive_char_split(chunks)]
    return ds


@timmer
def retrieve_dense(ds):
    # 稠密检索

    # 4.构建embeddings
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3.7-text-embedding",
    )

    # 5.向量存储
    vectorstore = Chroma(
        collection_name="bailuyuan",
        embedding_function=embeddings,
        persist_directory="./db/chroma_db",
    )
    # 6.将数据添加到向量存储（DashScope 单次最多 20 条，需分批）
    #   注意：langchain Chroma 不会保留 doc.id，需存入 metadata 以便回读
    BATCH_SIZE = 20
    for i in range(0, len(ds), BATCH_SIZE):
        batch = ds[i : i + BATCH_SIZE]
        for doc in batch:
            doc.metadata["doc_id"] = doc.id
        vectorstore.add_documents(batch)
    # print(f"知识库已就绪，{len(chunks)} 个文档")

    return vectorstore


@timmer
def retrieve_bm25(ds, top_k, index_path="./db/my_index.bm25", k1=1.5, b=0.75):
    metadata_corpus = [{"id": doc.id, "content": doc.page_content} for doc in ds]
    if os.path.exists(index_path):
        print(f"索引文件已存在，正在加载: {index_path}")
        retriever = bm25s.BM25.load(index_path, load_corpus=True)
    else:
        print(f"未找到索引文件，正在创建新索引...")
        # 对语料进行分词（注意：jieba.cut 返回生成器，需用 list() 转为列表）
        corpus_tokens = [list(jieba.cut(doc["content"])) for doc in metadata_corpus]

        # 基于原始文档创建索引库，将来检索出来的也是原始文档
        retriever = bm25s.BM25(k1=k1, b=b, corpus=metadata_corpus)
        # 创建索引
        retriever.index(corpus_tokens)
        # 保存到本地
        retriever.save(index_path)
        print(f"索引已保存至: {index_path}")

    return retriever


@timmer
def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k=60):
    """
    ranked_lists: List[List[Dict]]
                  每个检索器返回的文档dict列表，包含id和content，按排名从高到低。
                  注意：这里不需要score，在 RRF 中只使用排名位置。
    """
    results = {}
    rrf_scores = {}

    for _, ranked_list in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            results[doc_id] = doc

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [results[doc_id] for doc_id, score in sorted_docs]


@timmer
def cross_encoder_rerank(query: str, docs: List[Dict], top_k: int = 3):
    reranker = QWENRanker(top_n=top_k, score_threshold=0.5)
    # 将 docs 转换为 Document 对象列表（保留 id）
    documents = [
        Document(page_content=doc["content"], metadata={"id": doc["id"]})
        for doc in docs
    ]
    # 调用 reranker 的 compress_documents，它会返回排序后的 Document 列表，并附带 relevance_score
    scored_docs = reranker.compress_documents(documents=documents, query=query)
    # 只取 top_k 个（compress_documents 已按 top_n 截断，但为了保险）
    scored_docs = scored_docs[:top_k]
    # 构建返回列表
    result = []
    for doc in scored_docs:
        result.append(
            {
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "score": doc.metadata.get("relevance_score", 0.0),
            }
        )
    return result


class RagRecorder:
    """记录 agent 真实运行时的检索结果与回答，供 ragas 评测使用。"""

    def __init__(self):
        self.calls = []  # 每次工具调用的检索结果
        self.responses = []  # 每次 agent 回答

    def record_call(self, query, contexts, scores):
        self.calls.append({"query": query, "contexts": contexts, "scores": scores})

    def record_response(self, query, response):
        self.responses.append({"query": query, "response": response})


recorder = RagRecorder()


# 全局缓存：文档切分、稠密向量库、稀疏 BM25 索引只需构建一次，之后工具调用直接复用
_docs_cache = None
_vectorstore_cache = None
_bm25_cache = None
_index_lock = threading.Lock()


def ensure_indexes(top_k=5):
    """惰性构建并缓存《白鹿原》文档、稠密向量库与稀疏 BM25 索引。

    首次调用工具时才会真正执行 splitter_markdown / retrieve_dense / retrieve_bm25，
    之后的所有检索都复用首次构建好的对象，避免每次都重建。
    """
    global _docs_cache, _vectorstore_cache, _bm25_cache
    with _index_lock:
        if _docs_cache is None:
            _docs_cache = splitter_markdown()
        if _vectorstore_cache is None:
            _vectorstore_cache = retrieve_dense(_docs_cache)
        if _bm25_cache is None:
            _bm25_cache = retrieve_bm25(_docs_cache, top_k)
    return _docs_cache, _vectorstore_cache, _bm25_cache


def make_search_knowledge_rag(recorder):
    """依赖注入 recorder 的工具工厂，工具签名不变。"""

    @tool
    def search_knowledge_rag(query, top_k=5):
        """搜索《白鹿原》知识库，获取小说情节、人物命运、历史背景等知识。需要查找资料时调用。"""
        # 复用全局缓存，避免每次调用都重新切分文档、重建向量库与 BM25 索引
        _, dense_vector, bm25_retrieve = ensure_indexes(top_k)
        # 1.稠密检索
        dense_results = dense_vector.similarity_search(query, top_k)
        dense_vr = [
            {"id": doc.metadata.get("doc_id", doc.id), "content": doc.page_content}
            for doc in dense_results
        ]

        # 2.稀疏检索
        query_tokens = [list(jieba.cut(query))]
        bm25_retrieve_results, bm25_retrieve_scores = bm25_retrieve.retrieve(
            query_tokens, k=top_k
        )
        bm25_tuple_reslut = [
            (bm25_retrieve_results[0, i], bm25_retrieve_scores[0, i])
            for i in range(bm25_retrieve_results.shape[1])
        ]
        bm25_vr = [doc for doc, _ in bm25_tuple_reslut]
        # 3.混合检索+重排序 RFF
        rrf_results = reciprocal_rank_fusion([dense_vr, bm25_vr])

        # 4.cross-encoder精排序
        cross_encoder_results = cross_encoder_rerank(query, rrf_results, top_k)
        # 5.拼接文档
        docs_content = "\n\n".join(doc["content"] for doc in cross_encoder_results)

        # 6.记录本次真实检索结果（供 ragas 评测）
        recorder.record_call(
            query,
            [doc["content"] for doc in cross_encoder_results],
            [doc["score"] for doc in cross_encoder_results],
        )

        # 7.输出日志
        print(f"\n\n{'=' * 30}Tool Message{'=' * 30}")
        print(f"检索到与问题'{query}'相关文档：")
        print(
            f"\n\n".join(
                [
                    f"=====score: {doc['score']}=======\n\n{doc['content']}"
                    for doc in cross_encoder_results
                ]
            )
        )
        print("=" * 30 + "AI Message" + "=" * 30)

        return docs_content

    return search_knowledge_rag


search_knowledge_rag = make_search_knowledge_rag(recorder)

rag_agent = create_agent(
    model="deepseek-chat",
    tools=[search_knowledge_rag],
    system_prompt="""
        你是一个专业的《白鹿原》文学知识专家。您的职责是帮助用户解决有关小说《白鹿原》的相关问题。
        产品说明:
        1. 如果用户问了一个你不确定的问题，或者涉及小说具体情节、人物关系、历史背景等专业知识，你必须使用`search_knowledge_base`工具来查阅相关文档。
        2. 在引用文档时，要清楚地总结包括内容中的相关上下文。
        3. 如果获取文档失败，请告诉用户，并以您最好的专家理解继续进行。
        在回答用户关于《白鹿原》的问题之前，您必须查阅工具以获取最新信息。你的回答应该清晰、简洁、准确。不要有过多解释除非用户询问。
    """,
)
if __name__ == "__main__":
    from langchain.messages import AIMessage

    # def ask(query):
    #     """向 agent 提问，流式打印回答并记录到 recorder.responses"""
    #     print(f"\n{'-' * 40}User: {query}")
    #     answer = ""
    #     response = rag_agent.stream(
    #         {"messages": [{"role": "user", "content": query}]}, stream_mode="messages"
    #     )
    #     for chunk, metadata in response:
    #         if isinstance(chunk, AIMessage) and chunk.content:
    #             print(chunk.content, end="")
    #             answer += chunk.content
    #     print()
    #     # recorder.record_response(query, answer)

    # # 测试问候（不触发检索）
    # ask("你好")
    # ask("白孝文的结局如何？")

    # # 打印 ragas 评测数据
    # print("\n" + "=" * 40 + "Ragas Eval Data" + "=" * 40)
    # for call in recorder.calls:
    #     print(f"\nquery: {call['query']}")
    #     print(f"scores: {[round(s, 4) for s in call['scores']]}")
    #     for i, ctx in enumerate(call["contexts"]):
    #         print(f"  [{i}] {ctx[:]}...")
    # print("\nresponses:")
    # for r in recorder.responses:
    #     print(f"  query: {r['query']} -> {r['response'][:]}...")
