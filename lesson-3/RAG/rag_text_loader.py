from cmath import e
from this import s
from threading import local
from tomllib import load

from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_mineru import MinerULoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from dotenv import load_dotenv
import numpy as np
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import dashscope
load_dotenv()

def load_text(file_path):
    docs_loader = TextLoader(file_path,encoding="utf-8")
    docs = docs_loader.load()
    print(f"加载 {len(docs)} 个文档，内容: {docs[0].page_content[:100]}")

def load_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"来源: {docs[0].metadata.get('source')}, 长度: {len(docs[0].page_content)} 字符")

def load_csv(file_path):
    docs_loader = CSVLoader(file_path)
    docs = docs_loader.load()
    for doc in docs:
        print(f"[{doc.metadata.get('source')}] {doc.page_content[:100]}")

def batch_load_text(file_paths,glob="**/*.pdf"):
    loader = DirectoryLoader(file_paths, glob=glob,loader_cls=PyPDFLoader,
        show_progress=True,
    use_multithreading=True)
    docs = loader.load()
    for doc in docs:
        print(f"[{doc.metadata.get('source')}] {doc.page_content[:100]}")

def mineru_loader(file_paths):
    loader = MinerULoader(source=file_paths, mode="flash")
    docs = loader.load()
    return docs

def ocr_loader(file_paths):
    loader = MinerULoader(source=file_paths, mode="precision", ocr=True, token=os.getenv("MINERU_TOKEN"))
    docs = loader.load()
    for doc in docs:
        print(f"[{doc.metadata.get('source')}] {doc.page_content[:500]}")

def recursive_char_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, #chunk切分的片段大小
        chunk_overlap=100, #重复片段大小
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 优先级从高到低
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"原始文档数: {len(split_docs)}，切分后块数: {len(split_docs)}")
    for doc in split_docs:
        print(f"[{doc.metadata}] {doc.page_content[:500]}")
    return split_docs

def struct_split():
    docs="""
    # 第一章：概述
    这是第一章的引言部分，介绍整体框架。

    ## 第一节：背景
    本节描述项目的背景和动机。

    ### 1.1 现状分析
    当前存在的一些问题和挑战。

    ### 1.2 研究目标
    明确本项目的核心目标。

    ## 第二节：相关工作
    回顾已有研究和相关技术。

    # 第二章：方法论
    本章详细介绍所采用的方法和实验设计。

    ## 第一节：数据采集
    描述数据来源和预处理步骤。

    ## 第二节：模型架构
    阐述模型的具体结构和训练策略。
    """
    structured_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    )
    split_docs = structured_splitter.split_text(docs)
    print(f"原始文档数: {len(split_docs)}，切分后块数: {len(split_docs)}")
    for doc in split_docs:
        print(f"{doc.metadata} → {doc.page_content[:80]}...")
    return split_docs

def length_splitter():
    from langchain_text_splitters import CharacterTextSplitter

    # ---------- 定义一段长文本 ----------
    long_text = """
    LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的构建块，可以轻松组合成强大的工作流。

    LCEL（LangChain 表达式语言）是 LangChain 中一种声明式编排方式，允许使用管道符（|）将不同的组件串联起来，形成可执行的链。这种语法简洁直观，并且天然支持流式输出。

    Runnable 协议是 LCEL 的基石，所有主要组件（如模型、提示词模板、输出解析器）都实现了这个接口，使得它们可以无缝组合。

    与传统的 Agent 不同，确定性链（Chain）可以预先定义执行步骤，从而避免模型自主决策带来的不稳定性。例如，在需要联网搜索的场景中，可以强制先获取当前时间，再基于时间构造搜索词，最后将结果传递给 LLM 生成回答。

    输出解析器负责将模型的原始输出转换为程序可用的格式，例如字符串、JSON 或 Pydantic 对象。结构化输出（Structured Output）通过 with_structured_output 方法可以更方便地实现。

    Agent 则更加灵活，它允许 LLM 自主决定调用哪些工具以及调用顺序，适用于需要复杂推理和动态规划的任务。然而，这种灵活性也带来了不确定性，需要通过精心设计的系统提示词和工具描述来约束。

    在构建生产级应用时，通常需要结合链和 Agent 的优势：用链处理确定性流程，用 Agent 处理开放式决策。LangGraph 进一步扩展了这种能力，允许构建具有循环、条件分支和多角色协作的复杂状态机。

    RAG（检索增强生成）是另一个重要应用场景，它通过向量数据库检索相关文档，然后将这些文档作为上下文注入到 LLM 的提示词中，从而生成更加准确和事实性的回答。LangChain 提供了完整的 RAG 工具链，包括文档加载器、文本分割器、向量存储和检索器。

    文本分割是 RAG 流程中的关键步骤。LangChain 提供了多种分割器，例如按字符、按 token、按 Markdown 标题等。CharacterTextSplitter 是最基础的分割器，而 RecursiveCharacterTextSplitter 会递归尝试不同分隔符（如换行、句号、逗号）以保持语义完整性。

    对于中文文本，建议使用 RecursiveCharacterTextSplitter 并指定分隔符为 ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]，这样能更好地保留句子和段落的完整性。

    此外，LangChain 还支持异步调用、流式输出、回调处理等高级特性，可以轻松集成到 FastAPI、Streamlit 等 Web 框架中，构建实时响应的 AI 应用。

    总之，LangChain 为构建复杂的 AI 应用提供了强大而灵活的工具集，无论你是初学者还是资深开发者，都能从中受益。
    """

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # 使用 cl100k_base 编码（对应 GPT-4 等模型）
        chunk_size=500,               # 每个块的最大 token 数
        chunk_overlap=50              # 块之间的重叠 token 数
    )
    chunks = splitter.split_text(long_text)

    # ---------- 打印分割结果 ----------
    print(f"总共分割为 {len(chunks)} 个块\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"块 {i} (字符数: {len(chunk)}，预估 token 约 {len(chunk)//4})")
        print(chunk)
        print("-" * 50)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def qwen_embedding():
    from langchain_community.embeddings import DashScopeEmbeddings
    import dashscope

    embeddings = DashScopeEmbeddings(
        model="qwen3.7-text-embedding",
        dashscope_api_key=os.environ.get("QWEN_API_KEY")
    )
    text="我爱上班"
    vector = embeddings.embed_query(text)
    print(f"文本: {text}")
    print(f"向量维度: {len(vector)}")
    print(f"向量前5维: {vector[:5]}")

    texts=["我要躺平", "我爱工作", "拒绝加班"]

    vectors = embeddings.embed_documents(texts)
    print(f"\n批量向量化: {len(vectors)} 条, 维度: {len(vectors[0])}")
    for v in vectors:
        similarity = cosine_similarity(vector, v)
        print("Cosine Similarity:", similarity)

def save_vector(chunks):
    embeddings = DashScopeEmbeddings(
        model="qwen3.7-text-embedding",
        dashscope_api_key=os.environ.get("QWEN_API_KEY")
    )
    vectorstore = Chroma(
        collection_name="my_knowledge_base",
        embedding_function=embeddings,        # 复用之前的向量模型
        persist_directory="./db/chroma_db"    # 持久化目录
    )

    # 添加文档（自动完成向量化+存储）
    vectorstore.add_documents(chunks)

    results = vectorstore.similarity_search(
        "高血压饮食注意事项",  # 查询文本
        k=3  # 返回 Top-3 最相关的片段

    )
    for i, doc in enumerate(results):
        print(f"结果 {i+1}: {doc.page_content[:200]}...")
    results = vectorstore.similarity_search_with_relevance_scores(
        "高血压饮食注意事项",  # 查询文本
        k=5,
        score_threshold=0.4  # 低于此分的结果丢弃

    )
    for doc, score in results:
        print(f"得分 {score:.4f}: {doc.page_content[:150]}...")

    results = vectorstore.search(
        query="高血压饮食注意事项",
        search_type="similarity",
        k=3,
        filter={"filename": "老年健康指南.pdf"}  # 仅搜索特定文件
    )
    for i, doc in enumerate(results):
        print(f"结果 {i+1}: {doc.page_content[:200]}...")

    # 将 VectorStore 转为 Retriever，固化检索参数
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 后续检索一行代码搞定
    docs = retriever.invoke("高血压饮食注意事项")
    print(f"检索到 {len(docs)} 个相关文档，内容: {docs[0].page_content[:100]}...")

def complete_retrieval():
    from langchain_mineru import MinerULoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 1. 加载文档
    loader = MinerULoader(source="./docs/老年健康指南.pdf", mode="flash")
    docs = loader.load()

    # 2. 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 3. 向量化 + 存入向量库
    embeddings = DashScopeEmbeddings(
        model="qwen3.7-text-embedding",
        dashscope_api_key=os.environ.get("QWEN_API_KEY")
    )
    vectorstore = Chroma(
        collection_name="my_rag_db",
        embedding_function=embeddings,
        persist_directory="./db/chroma_db"
    )
    vectorstore.add_documents(chunks)
    print(f"✅ 知识库构建完成！共 {len(chunks)} 个片段")

    # 4. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke("高血压饮食注意事项")
    for doc in docs:
        print(f"检索到 {len(docs)} 个相关文档，内容: {doc.page_content[:]}...")


if __name__ == "__main__":
    # load_text("file.txt")
    # load_web("https://docs.langchain.com/oss/python/langchain/rag")
    # load_csv("docs/cozev2_template_agent.csv")
    # batch_load_text("docs")
    # mineru_loader("./docs/老年健康指南.pdf")
    # ocr_loader("./docs/test.jpg")
    # docs = mineru_loader("./docs/老年健康指南.pdf")
    # chunks = recursive_char_split(docs)
    # struct_split()
    # length_splitter()
    # qwen_embedding()
    # save_vector(chunks)
    # complete_retrieval()

    with open("./docs/白鹿原.md") as f:
        data = "".join(line for line in f.readlines())
    #2.文本分割
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
    ])
    chunks = splitter.split_text(data)
    # print(f"原始文档数: {len(chunks)}，切分后块数: {len(chunks)}")
    # for doc in chunks:
    #     print(f"{doc.metadata} → {doc.page_content[:80]}...")

    text_chunks = recursive_char_split(chunks)
    # print(f"原始文档数: {len(text_chunks)}，切分后块数: {len(text_chunks)}")
    # for doc in text_chunks:
    #     print(f"{doc.metadata} → {doc.page_content[:80]}...")
