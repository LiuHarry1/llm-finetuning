# -*- coding: utf-8 -*-
"""
一个用 LangGraph 实现的带记忆(chat memory) 的对话机器人，配套 Streamlit UI。
默认使用通义千问 qwen-plus。
"""

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

# 侧边栏：配置与控制
with st.sidebar:
    st.markdown("## ⚙️ 配置")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # API Key 优先级：侧边栏输入 > 环境变量 DASHSCOPE_API_KEY
    api_key = st.text_input("DashScope API Key", type="password", value=os.getenv("DASHSCOPE_API_KEY", ""))

    model_name = st.selectbox(
        "选择模型",
        [
            "qwen-plus",
            "qwen-turbo",
            "qwen-max",
        ],
        index=0,
    )

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

    st.markdown(
        """
        - 每个会话拥有独立的 thread_id，与数据库中的记忆绑定。
        - 若想**彻底删除**数据库文件，请停止应用并删除 `memory.sqlite`。
        """
    )

# 主区域标题
st.markdown("""
# 💬Chatbot
轻量却专业的对话机器人示例
""")

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

conn_str = "memory.sqlite"
# checkpointer = SqliteSaver.from_conn_string(conn_str)

if not api_key:
    st.warning("请先在左侧输入 DashScope API Key 才能开始对话。")
else:
    llm = ChatTongyi(model=model_name, temperature=temperature, api_key=api_key)

    builder = StateGraph(ChatState)
    builder.add_node("model", make_call_model(llm))
    builder.set_entry_point("model")
    builder.add_edge("model", END)

    if "checkpointer" not in st.session_state:
        conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
        st.session_state.checkpointer = SqliteSaver(conn)

    app = builder.compile(checkpointer=st.session_state.checkpointer)

    col1, col2 = st.columns([3, 1])
    # col1 = st.columns([2, 1])

    with col1:
        st.markdown("### 对话")
        st.markdown('<div class="chat-card">', unsafe_allow_html=True)

        if len(st.session_state.chat_display) == 0:
            st.info("开始对话吧！右侧有一些快捷提示。")
        else:
            for role, content in st.session_state.chat_display:
                css_class = "human" if role == "user" else "ai"
                avatar = "🧑‍💻" if role == "user" else "🤖"
                st.markdown(
                    f"<div class='msg {css_class}'><span class='small'>{avatar} {role}</span><br/>{content}</div>",
                    unsafe_allow_html=True,
                )

        with st.form("chat-form", clear_on_submit=True):
            user_input = st.text_area("输入你的问题/指令：", height=100, placeholder="比如：帮我写一个带注释的二分查找函数。")
            submitted = st.form_submit_button("发送")

        if submitted and user_input.strip():
            st.session_state.chat_display.append(("user", st.markdown(user_input).markdown))
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # final_state = None
            # with st.spinner("思考中..."):
            #     for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
            #         if "messages" in event:
            #             final_state = event
            #
            # if final_state is not None and "messages" in final_state:
            #     print("debug",final_state["messages"])
            #     ai_text = final_state["messages"][-1].content
            #     st.session_state.chat_display.append(("assistant", ai_text))
            #     st.rerun()
            with st.spinner("思考中..."):
                final_state = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
                ai_text = final_state["messages"][-1].content
                st.session_state.chat_display.append(("assistant", ai_text))
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='footer-note'>Session: <code>{st.session_state.thread_id}</code> · DB: <code>{conn_str}</code></div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### ⚡️ 快捷提示")
        tips = [
            "解释下面这段报错，并给出修复建议：...",
            "把这段文本要点总结成 5 条 bullet：...",
            "把我写的邮件润色成商务正式风格：...",
            "把这个函数加上类型标注和单元测试：...",
            "把这段英文翻译成自然的中文，并保留技术术语：...",
        ]
        for i, t in enumerate(tips):
            if st.button(t, key=f"tip-{i}"):
                st.session_state.chat_display.append(("user", t))
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                final_state = app.invoke({"messages": [HumanMessage(content=t)]}, config)
                ai_text = final_state["messages"][-1].content
                st.session_state.chat_display.append(("assistant", ai_text))
                st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ 说明")
        st.markdown(
            """
            - 默认使用 **通义 Qwen (qwen-plus)**。
            - 通过 `SqliteSaver` 在本地数据库里持久化记忆。
            - 不同 `thread_id` 互不干扰，可在侧边栏一键新建会话。
            - API Key 请使用阿里云 DashScope 控制台获取的 `DASHSCOPE_API_KEY`。
            """
        )
