from openai import OpenAI
import os
#创建一个openai的客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

messages=[{"role": "system", "content": "You are a helpful assistant"},]

#开启多伦对话
while True:
    user_input=input("你:")
    if user_input=="exit":
        break
    if "生成文案" in user_input:
        user_input+="\n要求:简洁有力，适合朋友圈发布，不超过50字"
    messages.append({"role": "user", "content": user_input})
    #发送api请求
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    #解析api请求
    ai_reply=response.choices[0].message.content
    print(ai_reply)
    #将请求再次写入messages
    messages.append({"role": "user", "content": ai_reply})
    with open("chat_history.txt", "a") as f:
        f.write(f"你: {user_input}\nAI: {ai_reply}\n")
