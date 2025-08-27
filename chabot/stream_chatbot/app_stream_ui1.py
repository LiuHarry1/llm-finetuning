# app_stream_ui.py
import os
import time
import uuid
import streamlit as st
from backend_stream import ChatBackend

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# ======== CSS =========
st.markdown("""
<style>
.main {padding: 2rem 2rem;}
.chat-card {background: rgba(255,255,255,0.65); backdrop-filter: blur(8px); border-radius: 20px; padding: 1.25rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08);} 
.msg {border-radius: 16px; padding: 0.8rem 1rem; margin: 0.35rem 0; line-height: 1.5;}
.human {background: #eef2ff;}
.ai {background: #ecfeff;}
.small {font-size: 0.86rem; color: #6b7280;}
.footer-note {color:#9ca3af; font-size:.85rem}
.block-container { padding-top: 1.5rem !important; }
html, body, [class*="css"] { font-size: 15px; }
.msg { font-size: 14px; }
.small { font-size: 12.5px; color: #6b7280; }
h1 { font-size: 1.6rem !important; }
textarea { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ======== 初始化 ======
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# ======== 主逻辑 ======
api_key = st.sidebar.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
model_name = st.sidebar.selectbox("选择模型", ["qwen-plus", "qwen-turbo", "qwen-max"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

if not api_key:
    st.warning("请先在左侧输入 DashScope API Key 才能开始对话。")
    st.stop()

backend = ChatBackend(api_key=api_key, model_name=model_name, temperature=temperature)

# ======== 左侧历史会话列表 ======
st.sidebar.markdown("## 🗂 历史会话")

# 获取所有线程 ID
cursor = backend.conn.cursor()
cursor.execute("SELECT thread_id FROM threads ORDER BY rowid DESC")
all_threads = [row[0] for row in cursor.fetchall()]

# 会话选择
selected_thread = st.sidebar.selectbox("选择会话", ["新建会话"] + all_threads)

# 新建会话
if selected_thread == "新建会话":
    new_id = str(uuid.uuid4())
    st.session_state.thread_id = new_id
    st.session_state.chat_display = []
else:
    st.session_state.thread_id = selected_thread
    # 加载对应历史
    state = backend.load_thread(selected_thread)
    st.session_state.chat_display = [(msg.get("role", "assistant"), msg.get("content", "")) for msg in state.get("messages", [])]

# 清空当前会话
if st.sidebar.button("🗑️ 清空当前会话"):
    st.session_state.chat_display = []

# ======== 主区域聊天显示 ======
st.markdown("# 💬 Chatbot")
chat_area = st.container()
history_ph = chat_area.empty()

def render_history(ph):
    html = '<div class="chat-card">'
    for role, content in st.session_state.chat_display:
        css_class = "human" if role == "user" else "ai"
        avatar = "🧑‍💻" if role == "user" else "🤖"
        html += f"<div class='msg {css_class}'><span class='small'>{avatar} {role}</span><br/>{content}</div>"
    html += "</div>"
    ph.markdown(html, unsafe_allow_html=True)

# 初始渲染
render_history(history_ph)

# ====== 用户输入表单 ======
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_area("输入你的问题/指令：", height=100, placeholder="比如：帮我写一个带注释的二分查找函数。")
    submitted = st.form_submit_button("发送 ➤")

if submitted and user_input.strip():
    # 记录用户消息
    st.session_state.chat_display.append(("user", user_input))
    # 插入 assistant 占位
    st.session_state.chat_display.append(("assistant", ""))
    ai_index = len(st.session_state.chat_display) - 1
    render_history(history_ph)

    # 流式生成
    try:
        for token in backend.chat_stream(user_input, st.session_state.thread_id):
            st.session_state.chat_display[ai_index] = ("assistant", st.session_state.chat_display[ai_index][1] + token)
            render_history(history_ph)
            time.sleep(0.01)
    except Exception as e:
        st.session_state.chat_display[ai_index] = ("assistant", f"[生成出错] {e}")
        render_history(history_ph)

# 页脚显示 session_id
st.markdown(f"<div class='footer-note'>Session: <code>{st.session_state.thread_id}</code></div>", unsafe_allow_html=True)
