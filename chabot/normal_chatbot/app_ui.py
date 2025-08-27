import os
import uuid
import streamlit as st
from backend import ChatBackend

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# ======== Streamlit CSS =========
st.markdown(
    """
    <style>
      .main {padding: 2rem 2rem;}
      .chat-card {background: rgba(255,255,255,0.65); backdrop-filter: blur(8px); border-radius: 20px; padding: 1.25rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08);} 
      .msg {border-radius: 16px; padding: 0.8rem 1rem; margin: 0.35rem 0; line-height: 1.5;}
      .human {background: #eef2ff;}
      .ai {background: #ecfeff;}
      .small {font-size: 0.86rem; color: #6b7280;}
      .tag {display:inline-block; padding: .25rem .6rem; border-radius: 9999px; border: 1px solid #e5e7eb; margin-right:.4rem; cursor:pointer}
      .tag:hover {background:#f3f4f6}
      .footer-note {color:#9ca3af; font-size:.85rem}



      [data-testid="stSidebarHeader"] {
        margin-bottom: 0px
      }

      /* 让主标题上移，和侧边栏对齐 */
      .block-container {
          padding-top: 1.5rem !important;
      }
    </style>

    """,
    unsafe_allow_html=True,
)

# ======== 侧边栏配置 =========
with st.sidebar:
    st.markdown("## ⚙️ 配置")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    model_name = st.selectbox("选择模型", ["qwen-plus", "qwen-turbo", "qwen-max"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    if st.button("➕ 新建会话"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_display = []
        st.rerun()

    if st.button("🗑️ 清空当前会话"):
        st.session_state.chat_display = []
        st.rerun()

# ======== 主区域 =========
st.markdown("# 💬 Chatbot")
if not api_key:
    st.warning("请先在左侧输入 DashScope API Key 才能开始对话。")
else:
    backend = ChatBackend(api_key=api_key, model_name=model_name, temperature=temperature)

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    if not st.session_state.chat_display:
        st.info("开始对话吧！")
    else:
        for role, content in st.session_state.chat_display:
            css_class = "human" if role == "user" else "ai"
            avatar = "🧑‍💻" if role == "user" else "🤖"
            st.markdown(f"<div class='msg {css_class}'><span class='small'>{avatar} {role}</span><br/>{content}</div>", unsafe_allow_html=True)

    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_area("输入你的问题/指令：", height=100, placeholder="比如：帮我写一个二分查找函数。")
        submitted = st.form_submit_button("发送 ➤")

    if submitted and user_input.strip():
        st.session_state.chat_display.append(("user", user_input))
        with st.spinner("思考中..."):
            ai_text = backend.chat(user_input, st.session_state.thread_id)
            st.session_state.chat_display.append(("assistant", ai_text))
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='footer-note'>Session: <code>{st.session_state.thread_id}</code></div>", unsafe_allow_html=True)
