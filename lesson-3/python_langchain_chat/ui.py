import streamlit as st
import requests

st.set_page_config(page_title="三工具智能体", page_icon="🤖")
st.title("三工具智能体")

backend_url = "http://127.0.0.1:6788/chat"

if 'history' not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("来和我聊聊吧~~~"):
    st.chat_message('user').markdown(prompt)
    st.session_state.history.append(('user', prompt))

    with st.chat_message('ai'):
        placeholder = st.empty()
        full_text = ""
        # 使用流式请求
        with requests.post(backend_url, json={"query": prompt}, stream=True) as response:
            if response.status_code == 200:
                # 每次读取 1 字节，确保实时性（或者使用 response.iter_lines() 等）
                for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:  # 避免空块
                        full_text += chunk
                        placeholder.markdown(full_text)
                st.session_state.history.append(('ai', full_text))
            else:
                st.error(f"请求失败，状态码：{response.status_code}")
