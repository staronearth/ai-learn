import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------- 加载本地 T5-small 模型 ----------
model_name = "./t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ---------- 辅助函数：语言映射 ----------
# T5 要求目标语言为英文名，且采用 "translate English to X" 前缀
lang_map = {
    "英语": "English",
    "日语": "Japanese",
    "法语": "French"
}

def translate(text: str, target_lang: str):
    """翻译函数：将输入的英文文本翻译为目标语言"""
    if not text or not text.strip():
        return "请输入需要翻译的文本（英文）。"
    
    # 构建任务前缀
    target = lang_map.get(target_lang, "English")
    prompt = f"translate English to {target}: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=200, num_beams=4, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def summarize(text: str, max_length: int):
    """摘要函数：对长文本进行摘要"""
    if not text or not text.strip():
        return "请输入需要摘要的长文本。"
    
    prompt = f"summarize: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# ---------- Gradio 界面 ----------
with gr.Blocks(title="AI 工具箱") as demo:
    gr.Markdown("# 🤖 AI 文本处理")
    
    with gr.Tab("翻译"):
        text_in = gr.Textbox(label="输入文本（英文）", lines=5, placeholder="输入英文内容...")
        with gr.Row():
            lang = gr.Dropdown(["英语", "日语", "法语"], label="目标语言", value="英语")
            btn = gr.Button("翻译", variant="primary")
        text_out = gr.Textbox(label="翻译结果", lines=5)
        btn.click(fn=translate, inputs=[text_in, lang], outputs=text_out)
    
    with gr.Tab("摘要"):
        long_text = gr.Textbox(label="长文本", lines=8)
        max_len = gr.Slider(50, 500, value=200, label="摘要长度")
        sum_btn = gr.Button("生成摘要", variant="primary")
        sum_out = gr.Textbox(label="摘要结果", lines=5)
        sum_btn.click(fn=summarize, inputs=[long_text, max_len], outputs=sum_out)

demo.launch()