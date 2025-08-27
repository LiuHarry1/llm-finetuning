# -*- coding: utf-8 -*-
"""
一个用 LangGraph 实现的带记忆(chat memory) 的对话机器人，配套 Streamlit UI。

特性：
- LangGraph 状态机 + SQLite Checkpointer（跨会话持久化记忆）
- Chat history 自动累积（State 中通过 add_messages 归约器）
- Streamlit 漂亮 UI（左右布局、头像、消息气泡、输入区、快捷提示、重置会话/清空记忆）
- 可选择模型（OpenAI，或兼容 OpenAI 接口的服务）

运行方法：
1) 安装依赖：
   pip install -U streamlit langgraph langchain-core langchain-openai typing_extensions

2) 启动：
   streamlit run app.py

3) 在侧边栏填入 API Key（或使用环境变量 OPENAI_API_KEY），选择模型，然后聊天即可。

提示：
- 默认使用 gpt-4o-mini，可切换为其他兼容 OpenAI Chat Completions 的模型。
- 记忆是基于 LangGraph 的 SqliteSaver（memory.sqlite），按 session/thread 隔离，可跨重启保留。
"""

import os
import uuid
from typing import Annotated, List

import streamlit as st

# LangChain / LangGraph 基础
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# ============ LangGraph 状态定义 ============
class ChatState(TypedDict):
    """对话图的状态。messages 使用 add_messages 作为 reducer，自动累积历史。"""
    messages: Annotated[List[AnyMessage], add_messages]


# ============ LangGraph 节点：调用模型 ============

def make_call_model(llm: ChatOpenAI):
    """闭包：返回一个节点函数，读取 state -> 调用 LLM -> 追加到 state.messages。"""

    def call_model(state: ChatState) -> ChatState:
        response = llm.invoke(state["messages"])  # 直接把所有消息喂给模型
        return {"messages": [response]}

    return call_model


# ============ Streamlit UI ============
# 页面配置
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="wide")

# 自定义样式：简单的玻璃拟态卡片 + 圆角消息气泡
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
    </style>
    """,
    unsafe_allow_html=True,
)

# 侧边栏：配置与控制
with st.sidebar:
    st.markdown("## ⚙️ 配置")

    # Session / Thread ID，用于区分不同会话（也是记忆的 key）
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # API Key 优先级：侧边栏输入 > 环境变量 OPENAI_API_KEY
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    # 模型选择（可改成你可用的任何 OpenAI 兼容模型）
    model_name = st.selectbox(
        "选择模型",
        [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-3.5-turbo",
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
# 💬 LangGraph Chatbot
轻量却专业的对话机器人示例（LangGraph + 持久化记忆 + 漂亮 UI）
""")

# 初始化显示用的简易内存（仅用于前端回显，不是 LangGraph 的记忆）
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# LangGraph：构建与编译（带 SQLite Checkpointer 实现跨会话记忆）
# 注意：Checkpointer 的责任是将 State 保存到 SQLite，并按 thread_id 进行区分。
conn_str = "memory.sqlite"  # SQLite 存储文件
checkpointer = SqliteSaver.from_conn_string(conn_str)

# 延迟创建 LLM（依赖用户在侧边栏输入的 API Key）
if not api_key:
    st.warning("请先在左侧输入 OpenAI API Key 才能开始对话。")
else:
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

    # 组图（只有 1 个节点：model）
    builder = StateGraph(ChatState)
    builder.add_node("model", make_call_model(llm))
    builder.set_entry_point("model")
    builder.add_edge("model", END)

    # 编译图并接入 checkpointer
    app = builder.compile(checkpointer=checkpointer)

    # 左右两列布局：左显示聊天，右显示快捷提示 / 系统信息
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 对话")
        st.markdown('<div class="chat-card">', unsafe_allow_html=True)

        # 把 session_state.chat_display 渲染出来
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

        # 输入区
        with st.form("chat-form", clear_on_submit=True):
            user_input = st.text_area("输入你的问题/指令：", height=100, placeholder="比如：帮我写一个带注释的二分查找函数。")
            submitted = st.form_submit_button("发送")

        if submitted and user_input.strip():
            # 1) 先回显到本地 UI
            st.session_state.chat_display.append(("user", st.markdown(user_input).markdown))

            # 2) 通过 LangGraph 发送消息；利用 thread_id 做记忆隔离
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # 使用 stream 便于未来拓展（你也可以直接 app.invoke）
            final_state = None
            with st.spinner("思考中..."):
                for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config):
                    # 这里可逐步处理事件；简单起见只取最后的状态
                    final_state = event

            if final_state is not None and "messages" in final_state:
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
            - **LangGraph 记忆**：通过 `add_messages` 把每次对话自动合并进 `state.messages`，并使用 `SqliteSaver` 在本地数据库里持久化。
            - **会话隔离**：不同 `thread_id` 互不干扰，可在侧边栏一键新建会话。
            - **可扩展**：你可以在图中加入工具节点（检索、函数调用等），或改用企业内部模型。
            - **安全**：不要把敏感 Key 写进代码，请使用环境变量或侧边栏输入。
            """
        )

        with st.expander("🔧 进阶：如何接入自建/代理的 OpenAI 兼容服务？"):
            st.markdown(
                """
                1. 在 `ChatOpenAI` 初始化时添加 `base_url` 参数：
                   ```python
                   llm = ChatOpenAI(
                       model=model_name,
                       temperature=temperature,
                       api_key=api_key,
                       base_url="https://你的代理域名/v1"
                   )
                   ```
                2. 保持接口协议兼容即可。
                """
            )

