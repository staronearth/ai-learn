# Please install OpenAI SDK first: `pip3 install openai`
#python有丰富的第三方包
import os
from openai import OpenAI

#创建一个openai的客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

#发送api请求
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)
#解析api请求
print(response)
print(response.choices[0].message.content)
