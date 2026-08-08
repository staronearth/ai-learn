from itertools import product
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage,AIMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models import llms
from langchain_openai.chat_models.base import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
import requests
import os

load_dotenv()
class QwenImageInput(BaseModel):
    """工具输入参数模型"""
    prompt: str = Field(description="用于生成图像的详细文本描述")

class QwenImageTool(BaseTool):
    name: str = "qwen_image_generator"
    description: str = "使用阿里云通义千问图像生成模型（qwen-image-3.0）根据文本生成图片。"
    args_schema: Type[BaseModel] = QwenImageInput

    # API 地址和模型名
    api_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    model: str = "qwen-image-3.0"

    def _run(self, prompt: str) -> str:
        """同步执行图像生成并返回图片 URL"""
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("环境变量 QWEN_API_KEY= 未设置")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            },
            "parameters": {
                "prompt_extend": True   # 启用提示词优化
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # 根据 DashScope 官方文档，返回结构为：
            # {"output": {"choices": [{"message": {"content": [{"image": "https://..."}]}}]}}
            # 也可能直接是 {"data": [{"url": "..."}]}，此处兼容两种常见格式
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                # content 可能是列表，每个元素是 {"image": url} 或 {"text": ...}
                for item in content:
                    if "image" in item:
                        return item["image"]
            elif "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                return result["data"][0].get("url")

            # 若未找到 URL，则返回错误信息或完整响应（便于调试）
            return f"未能从响应中提取图片 URL，原始响应：{result}"

        except requests.exceptions.RequestException as e:
            return f"调用 DashScope API 时发生错误: {e}"



"""
将之前在 COZE 中完成的跨境电商文案工作流（产品名 → 卖点 → 标题 → 完整文案）
转换为 LangChain 的顺序链实现（使用 RunnablePassthrough.assign 串联），
并尝试使用 with_structured_output 确保每一步输出为结构化对象。
【跨境电商文案生成器】
流程：输入产品名 → LLM 生成英文卖点 → 生成英文商品标题 → 生成完整电商文案 → 文生图插件生成配套商品图。
"""
def homework1():
    llm=init_chat_model(
        "deepseek-chat"
    )
    image_tool = QwenImageTool()

    # 2. 构建 LCEL 链
    def generate_image(text: str) -> str:
        return image_tool.run(text)




def user_test():

    # 1. 定义组件
    llm =init_chat_model(
        "deepseek-chat"
    )
    # prompt = ChatPromptTemplate.from_template("详细描述：{idea}")
    image_tool = QwenImageTool()

    # 2. 构建 LCEL 链
    def generate_image(text: str) -> str:
        return image_tool.run(text)

    # chain = (
    #     {"idea": RunnablePassthrough()}           # 接收用户输入
    #     | prompt                                  # 格式化为消息
    #     | llm                                     # 调用 LLM
    #     | (lambda msg: msg.content)               # 提取文本
    #     | RunnableLambda(generate_image)          # 调用图像工具
    # )
    class ProductInfo(BaseModel):
        title: str = Field(description="生成创新的英文商品标题，要求吸引人，不超过10个字")
        tags: list[str] = Field(description="产品的3个核心英文卖点标签")
        content: str = Field(description="生成完整电商文案,不少于200字")

    # 3. 定义 Prompt 模板
    prompt = ChatPromptTemplate.from_template(
        "你是一位资深的英文电商产品经理。面向的客户是英文群体，请为以下概念产品名后名称生成核心卖点标签。\n"
        "产品名称：{name}"
    )
    # 4. ✅ 构建 LCEL 链：绑定结构化输出
    # with_structured_output 会自动将 ProductInfo 的 Field 描述转换为模型的 tool/function 定义
    product_chain = (
        prompt | llm.with_structured_output(ProductInfo)
    )
    picture_prompt = ChatPromptTemplate.from_template(
        "产品的名称:{name}\n"
        "产品核心卖点:{tags}\n"
        "产品标题:{title}\n"
        "产品描述:{content}\n"
        "请为这个产品生成文生图插件生成配套商品图"
    )
    picture_chain = (
        RunnablePassthrough.assign(
            name=lambda x: x["name"],
            product_info=product_chain,
        )|RunnablePassthrough.assign(
            tags=lambda x: x["product_info"].tags,
            title=lambda x: x["product_info"].title,
            content=lambda x: x["product_info"].content,
        )|picture_prompt|llm|(lambda msg: msg.content)|RunnableLambda(generate_image)
    )
    full_chain=RunnableParallel(
        product_chain=product_chain,
        picture_chain=picture_chain,
    )


    # 5. 执行测试
    print("正在调用大模型生成结构化数据...\n")

    # invoke 传入的字典 key 必须与 prompt 模板中的变量名一致
    result = full_chain.invoke({
        "name": "智能眼罩"
    })

    # 6. 直接使用 Pydantic 对象的属性，享受代码提示和类型安全
    print(f"📦 电商商品图: {result['picture_chain']}")
    print(f"💰 商品标题: {result['product_chain'].title}")
    print(f"🏷️ 核心标签: {', '.join(result['product_chain'].tags)}")

    # # 3. 执行
    # url = chain.invoke("月光下的城堡")
    # print(url)

def question_route_chain():
    llm=init_chat_model(
        "deepseek-chat"
    )
    #创建一个“专业问答路由链”，
    # 使用 RunnableBranch 根据问题领域（技术/法律/医学/通用）自动选择不同的 System Prompt 和回答风格。
    # 要求：领域分类必须使用 with_structured_output 实现。
    #定义各个子链
    technology_prompt = ChatPromptTemplate.from_template("你是一个电脑技术专业的问答助手，你的任务是根据问题的领域{input}选择合适的回答风格。")
    law_prompt = ChatPromptTemplate.from_template("你是一个法律专业的问答助手，你的任务是根据问题的领域{input}选择合适的回答风格。")
    medical_prompt = ChatPromptTemplate.from_template("你是一个医学专业的问答助手，你的任务是根据问题的领域{input}选择合适的回答风格。")
    general_prompt = ChatPromptTemplate.from_template("你是一个通用问答助手，你的任务是根据问题的领域{input}选择合适的回答风格。")

    #定义各个子链
    technology_chain = technology_prompt | llm | StrOutputParser()
    law_chain = law_prompt | llm | StrOutputParser()
    medical_chain = medical_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()

    class IntentResult(BaseModel):
        intent: str = Field(description="用户意图，仅限'技术'、'法律'、'医疗'、'闲聊'四者之一")
    #意图识别链
    intent_prompt = ChatPromptTemplate.from_template("识别用户意图：\n用户消息：{input}。")
    intent_chain = intent_prompt | llm.with_structured_output(IntentResult)

    branch_chain = RunnableBranch(
        (lambda x: x["intent_result"].intent == "技术", technology_chain),
        (lambda x: x["intent_result"].intent == "法律", law_chain),
        (lambda x: x["intent_result"].intent == "医疗", medical_chain),
        general_chain,
    )

    custom_service_chain=RunnablePassthrough.assign(
        intent_result=lambda x: intent_chain.invoke({"input": x})
    )| branch_chain

    print(custom_service_chain.invoke({"input":"如何快速配置ubuntu？"}))
    print(custom_service_chain.invoke({"input":"眼睛不舒服怎么办？"}))
    print(custom_service_chain.invoke({"input":"我们目前996,我该怎么办法？"}))
    print(custom_service_chain.invoke({"input":"今天心情真好！"}))
if __name__ == "__main__":
    # homework1()
    # user_test()
    question_route_chain()
