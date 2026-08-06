import requests
import os
from openai import OpenAI
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
def use_requests():
    api_key=os.environ.get('DEEPSEEK_API_KEY')
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": "用一句话介绍 Python"}
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    response = requests.post(url, headers=headers, json=data)
    #[{'index': 0, 'message': {...}, 'logprobs': None, 'finish_reason': 'length'}]
    if response.status_code == 200:
        choices_answer = response.json()["choices"]
        answer = choices_answer[0]["message"]
        if answer.get("content", ""):
            content = answer["content"]
        else:
            content = answer.get("reasoning_content", "")
        print(f"回答: {content}")
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
def user_openai_sdk():
    

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"  # 指向第三方平台
    )

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "用一句话介绍 Python"}],
        temperature=0.7
    )

    print(response.choices[0].message.content)

def user_openai_sdk_stream():
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"  # 指向第三方平台
    )
    stream = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "讲个程序员笑话"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
app = FastAPI(title="我的 API", version="1.0")

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/greet/{name}")
async def greet(name: str, age: int = 18):
    return {"message": f"你好 {name}！", "age": age}
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
@app.get("/search")
async def search(keyword: str, page: int = 1, page_size: int = 20):
    return {"keyword": keyword, "page": page, "page_size": page_size}


class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1

@app.post("/items/")
async def create_item(item: Item):
    total = item.price * item.quantity
    return {"item": item, "total_price": total}
if __name__ == "__main__":
    # use_requests()
    user_openai_sdk()
    # user_openai_sdk_stream()
    
    # uvicorn.run(app, host="127.0.0.1", port=8001)


