import os
import sqlite3
import uuid
from typing import Annotated, List

import streamlit as st

# LangChain / LangGraph 基础
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# ============ LangGraph 状态定义 ============
class ChatState(TypedDict):
    """对话图的状态。messages 使用 add_messages 作为 reducer，自动累积历史。"""
    messages: Annotated[List[AnyMessage], add_messages]


# ============ LangGraph 节点：调用模型 ============

def make_call_model(llm: ChatTongyi):
    """闭包：返回一个节点函数，读取 state -> 调用 LLM -> 追加到 state.messages。"""

    def call_model(state: ChatState) -> ChatState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    return call_model

# ============ Streamlit UI ============
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

st.markdown("""
<style>
/* 主容器内边距 */
.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* 消息卡片 */
.chat-card {
    background: rgba(245,245,245,0.85);
    backdrop-filter: blur(6px);
    border-radius: 20px;
    padding: 1.5rem;
    max-height: 70vh;
    overflow-y: auto;
}

/* 消息气泡 */
.msg {
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin: 0.35rem 0;
    line-height: 1.5;
    transition: all 0.3s ease;
    opacity: 0;
    animation: fadeIn 0.3s forwards;
    max-width: 75%;
}

/* 用户消息 */
.human {
    background: #dbeafe;
    float: left;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* AI 消息 */
.ai {
    background: #cffafe;
    float: right;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* 清除浮动 */
.msg::after {
    content: "";
    display: block;
    clear: both;
}

/* 小字体 */
.small {font-size: 0.86rem; color: #6b7280;}

/* 标签样式 */
.tag {display:inline-block; padding: .25rem .6rem; border-radius: 9999px; border: 1px solid #e5e7eb; margin-right:.4rem; cursor:pointer;}
.tag:hover {background:#f3f4f6}

/* 页脚信息 */
.footer-note {color:#9ca3af; font-size:.85rem}

/* 侧边栏按钮美化 */
.stButton>button {
    background-color:#3b82f6;
    color:white;
    border-radius:8px;
    padding:0.5rem 1rem;
}
.stButton>button:hover {background-color:#2563eb;}

/* 滚动条美化 */
.chat-card::-webkit-scrollbar {width:6px;}
.chat-card::-webkit-scrollbar-thumb {background:#9ca3af; border-radius:3px;}

/* 消息淡入动画 */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(5px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# 侧边栏优化
with st.sidebar:
    st.markdown("## ⚡ 模型参数")
    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))
    model_name = st.selectbox("选择模型", ["qwen-plus","qwen-turbo","qwen-max"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    st.markdown("---")
    st.markdown("### 🧠 记忆控制")
    if st.button("➕ 新建会话 (保留旧记忆)"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_display = []
        st.rerun()
    if st.button("🗑️ 清空当前会话消息 (不删数据库)"):
        st.session_state.chat_display = []
        st.rerun()

# 主区域标题
st.markdown("# 💬 Chatbot\n轻量却专业的对话机器人示例")

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# 消息显示容器
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
if len(st.session_state.chat_display) == 0:
    st.info("开始对话吧！")
else:
    for role, content in st.session_state.chat_display:
        css_class = "human" if role == "user" else "ai"
        avatar = "🧑‍💻" if role == "user" else "🤖"
        st.markdown(
            f"<div class='msg {css_class}'><span class='small'>{avatar} {role}</span><br/>{content}</div>",
            unsafe_allow_html=True,
        )
st.markdown('</div>', unsafe_allow_html=True)

# 输入框与发送按钮
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_area("输入你的问题/指令：", height=80, placeholder="比如：帮我写一个带注释的二分查找函数。")
    submitted = st.form_submit_button("发送 ➤")

if submitted and user_input.strip():
    st.session_state.chat_display.append(("user", st.markdown(user_input).markdown))
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("思考中..."):
        final_state = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        ai_text = final_state["messages"][-1].content
        st.session_state.chat_display.append(("assistant", ai_text))
        st.rerun()

# 页脚
st.markdown(f"<div class='footer-note'>Session: <code>{st.session_state.thread_id}</code> · DB: <code>memory.sqlite</code></div>", unsafe_allow_html=True)


