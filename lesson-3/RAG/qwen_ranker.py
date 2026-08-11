from csv import QUOTE_ALL
import os
import requests
from typing import Sequence, List
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field
from dotenv import load_dotenv
import pytest
load_dotenv()

class QWENRanker(BaseDocumentCompressor):
    """Compressor that uses QWEN ranker to compress documents."""
    api_key: str = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    base_url: str = Field(default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank", description="QWEN base URL")
    model: str = Field(default="qwen3-rerank", description="QWEN model name")
    top_n: int = Field(..., description="Number of top documents to return")
    score_threshold: float = Field(..., description="Score threshold for filtering documents")


    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        if not self.api_key:
            raise ValueError("QWEN API key is not set")
        dc=[doc.page_content for doc in documents]
        print(f"dc: {dc}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": dc,
            },
            "parameters": {
                "top_n": self.top_n,
                "return_documents": True,
            }
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code != 200:
            raise ValueError(f"Failed to compress documents: {response.text}")

        results = response.json()["output"]["results"]
        # 按分数排序并过滤阈值
        reranked_docs = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            if score >= self.score_threshold:
                doc = documents[idx]
                doc.metadata["relevance_score"] = score
                reranked_docs.append(doc)

        return reranked_docs


def test_qwen_ranker():
    reranker = QWENRanker(
        top_n=3,
        score_threshold=0.1,
    )

    # ========== 准备《白鹿原》数据 ==========
    query = "朱先生是什么人物？"
    passages = [
        "朱先生是白鹿原上的圣人，德高望重，曾编纂县志，劝退清军，名震关中。",
        "白嘉轩是白鹿村的族长，一生娶过七房女人，腰杆挺直，性格刚毅。",
        "鹿子霖是白鹿村的乡约，善于钻营，好色成性，与白嘉轩明争暗斗。",
        "黑娃原名鹿兆谦，曾当土匪，后归顺保安团，最终被镇压。",
    ]
    documents = [Document(page_content=p) for p in passages]

    print(f"documents: {documents}")
    # ========== 执行精排 ==========
    scored_docs = reranker.compress_documents(documents=documents, query=query)

    # ========== 输出结果 ==========
    for doc in scored_docs:
        print(f"[{doc.metadata['relevance_score']:.4f}] {doc.page_content}")


if __name__ == "__main__":
    test_qwen_ranker()
