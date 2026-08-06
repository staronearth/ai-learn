from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import (ChatPromptTemplate,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate,
AIMessagePromptTemplate)
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, AIMessage

def basic_prompt_template():
    prompt_template = PromptTemplate(
        input_variables=["name"],
        template="Hello, {name}!"
    )
    print(prompt_template.format(name="Alice"))  # 输出: Hello, Alice!

def multi_input_prompt_template():
    prompt_template = PromptTemplate(
        input_variables=["name", "age"],
        template="Hello, {name}! You are {age} years old."
    )
    print(prompt_template.format(name="Bob", age=30))  # 输出: Hello, Bob! You are 30 years old.

def chat_prompt_template():
    # 定义角色和用户输入
    system_msg = SystemMessagePromptTemplate.from_template(
        "你是一位资深{role}，用{style}风格回答问题。"
    )
    human_msg = HumanMessagePromptTemplate.from_template(
        "我的问题是：{question}"
    )

    # 组合成聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

    # 格式化
    messages = chat_prompt.format_messages(
        role="机器学习工程师",
        style="专业严谨",
        question="如何理解 Transformer 的注意力机制？"
    )

    for msg in messages:
        print(f"[{msg.type}]: {msg.content}")

def ai_history_template():
    history = [
        HumanMessagePromptTemplate.from_template("Python 怎么定义函数？"),
        AIMessagePromptTemplate.from_template("使用 def 关键字，例如：def my_func():")
    ]

    # 新问题
    new_question = HumanMessagePromptTemplate.from_template("那如何定义带参数的函数？")

    # 组合完整对话
    full_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("你是一位专业的 Python 编程助手"),
        *history,
        new_question
    ])

    messages = full_prompt.format_messages()
    for msg in messages:
        print(f"{msg.type}: {msg.content}")

def keyword_prompt_template():
    context = {"date": "2026年7月15日", "event": "AI开发者大会", "speaker": "Yann LeCun"}
    template = "欢迎参加{event}！今天是{date}，主讲嘉宾是{speaker}。"
    prompt = PromptTemplate.from_template(template)
    print(prompt.format(**context))  # ** 解包字典

def template_prompt_template():
    header = PromptTemplate.from_template("主题：{title}\n")
    body = PromptTemplate.from_template("主要内容：{content}\n")
    footer = PromptTemplate.from_template("---\n联系人：{contact}")

    print(header.format(title="AI技术分享"))
    print(body.format(content="本次分享将深入探讨AI技术的最新进展。"))
    print(footer.format(contact="张三"))
def verify_variable_template():
    prompt = PromptTemplate(
        template="欢迎{name}，您的会员等级是{level}",
        input_variables=["name", "level"]  # 显式声明必需变量
    )
    # prompt.format(name="张三")  # 缺少 level 会抛出 KeyError
    print(prompt.format(name="白泽", level=3))

def special_characters_template():
    prompt = PromptTemplate(
        template="请注意，特殊字符如{{}}和[]需要正确处理。",
        input_variables=[]
    )
    print(prompt.format())  # 输出: 请注意，特殊字符如{}和[]需要正确处理。

def agent_prompt_template():
    from langchain.agents import create_agent
    from langchain.messages import HumanMessage

    agent = create_agent(
        model="deepseek-chat",
        system_prompt="像海盗一样说话."
    )

    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="你是谁？")]},
        stream_mode="messages"
    ):
        print(token.content, end="", flush=True)
    
def format_response():
    from pydantic import BaseModel

    class CapitalInfo(BaseModel):
        name: str
        location: str
        vibe: str
        economy: str

    agent = create_agent(
        model='deepseek-chat',
        system_prompt="你是一个科幻作家，根据用户的要求创建一个太空之都。",
        response_format=CapitalInfo
    )

    response = agent.invoke({"messages": [HumanMessage(content="月球的首都是什么?")]})
    city = response['structured_response']
    print(f"{city.name}位于{city.location}，是一座{city.vibe}的城市。")

def homework_prompt_template():
    # 定义角色和用户输入
    system_msg = SystemMessagePromptTemplate.from_template(
        """
        #身份
        你是一位资深{role}，用{style}风格回答问题。
        #指令
        - 输出包含城市名、三道特色菜和一句总结
        #示例
        例如：
        问题:成都
        回答:成都的特色美食有火锅、串串香、兔头等，麻辣鲜香，口味独特。
        """
    )
    human_msg = HumanMessagePromptTemplate.from_template(
        "我的问题是：{question}"
    )


    # 组合成聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

    # 格式化
    messages = chat_prompt.format_messages(
        role="美食评论家",
        style="专业且热情",
        question="西安"
    )

    for msg in messages:
        print(f"[{msg.type}]: {msg.content}")
    
    agent = create_agent(
        model='deepseek-chat',
        # system_prompt=chat_prompt
    )

    for token, metadata in agent.stream(
        {"messages": messages},
        stream_mode="messages"
    ):
        if token.content:
            print(token.content, end="", flush=True)

def book_info_template():
    from pydantic import BaseModel, Field

    class BookInfo(BaseModel):
        title: str = Field(..., description="书名")
        year: int = Field(..., description="出版年份")
        article: str = Field(..., description="文章内容")
        author: str = Field(..., description="作者")
    
    agent=create_agent(
        model='deepseek-chat',
        system_prompt="你是一个图书管理员，根据用户的要求提供图书信息。",
        response_format=BookInfo
    )

    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="请提供《三体》的信息")]},
        stream_mode="messages"
    ):
        if token.content:
            print(token.content, end="", flush=True)


if __name__ == "__main__":
    # basic_prompt_template()
    # multi_input_prompt_template()
    # chat_prompt_template()
    # ai_history_template()
    # keyword_prompt_template()
    # template_prompt_template()
    # verify_variable_template()
    # special_characters_template()
    # agent_prompt_template()
    # format_response()
    # homework_prompt_template()
    book_info_template()