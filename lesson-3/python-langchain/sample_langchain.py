from langchain.agents import create_agent
from langchain.tools import tool
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool as core_tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type
@tool
def get_current_weather(location: str) -> str:
    """
    获取当前天气
    Args:
        location (str): 地点名称
    """
    return f"当前{location}的天气是：晴，温度25摄氏度"



def sum(a: int, b: int) -> int:
    """
    计算两个数字的和
    Args:
        a (int): 第一个数字
        b (int): 第二个数字
    """
    return a + b
calculate_sum=core_tool(sum)

class CalculateSubInput(BaseModel):
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")


def calculate_sub(input: CalculateSubInput) -> int:
    return input.a - input.b

calculate_sub=StructuredTool.from_function(
    func=calculate_sub,
    name="calculate_sub",
    description="计算两个数字的差",
    input_schema=CalculateSubInput)

@tool
def calculate_mul(a: int, b: int) -> int:
    """
    计算两个数字的积
    Args:
        a (int): 第一个数字
        b (int): 第二个数字
    """
    return a * b

# 外部定义输入模型（名称建议带上 Tool 前缀，避免与 builtins.Input 混淆）
class CalculateDivInput(BaseModel):
    a: int = Field(..., description="第一个数字（被除数）")
    b: int = Field(..., description="第二个数字（除数）")

class CalculateDiv(BaseTool):
    name: str = "calculate_div"
    description: str = "计算两个数字的商"
    args_schema: Type[BaseModel] = CalculateDivInput  # 直接引用外部类

    def _run(self, a: int, b: int) -> str:
        if b == 0:
            return "除数不能为零"
        return f"商是 {a / b}"



if __name__ == "__main__":
    agent = create_agent(
        "deepseek-v4-flash",
        tools=[get_current_weather, calculate_sum, calculate_sub, calculate_mul, CalculateDiv()]
    )

    # 测试调用
    # question = "请告诉我北京的天气情况"
    question = "请计算20/5"
    answer = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print(f"问题: {question}\n回答: {answer}")
