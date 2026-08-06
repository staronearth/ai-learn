import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=greet,           # 要包装的函数
    inputs="text",      # 输入组件类型（简写）
    outputs="text"      # 输出组件类型（简写）
)
def weather(city, humidity, unit):
    greeting = f"{city}的天气："
    temp = 25 if unit == "摄氏度" else 77
    detail = f"温度{temp}{unit}" + (f"，湿度60%" if humidity else "")
    return greeting, detail

weather_demo = gr.Interface(
    fn=weather,
    inputs=[
        gr.Textbox(label="城市", placeholder="输入城市名"),
        gr.Checkbox(label="显示湿度"),
        gr.Radio(["摄氏度", "华氏度"], label="温度单位")
    ],
    outputs=[
        gr.Textbox(label="标题"),
        gr.Textbox(label="详情")
    ]
)
if __name__ == "__main__":
    weather_demo.launch()