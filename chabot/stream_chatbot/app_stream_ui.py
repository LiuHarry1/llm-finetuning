# app_stream_ui_no_thinking.py
import os
import time
import uuid
import streamlit as st
from backend_stream import ChatBackend  # 请确保 backend_stream.chat_stream 返回逐 token 字符串

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# ======== 原有 CSS（保持不变） =========
st.markdown(
    """
    <style>
      .main {padding: 2rem 2rem;}
      .chat-card {background: rgba(255,255,255,0.65); backdrop-filter: blur(8px); border-radius: 20px; padding: 1.25rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08);} 
      .msg {border-radius: 16px; padding: 0.8rem 1rem; margin: 0.35rem 0; line-height: 1.5;}
      .human {background: #eef2ff;}
      .ai {background: #ecfeff;}
      .small {font-size: 0.86rem; color: #6b7280;}
      .footer-note {color:#9ca3af; font-size:.85rem}
      [data-testid="stSidebarHeader"] { margin-bottom: 0px }
      .block-container { padding-top: 1.5rem !important; }
      
      
              /* 全局字体稍微缩小 */
        html, body, [class*="css"] {
            font-size: 15px;   /* 默认是 16px，这里改小一点 */
        }
        
        /* 聊天气泡字体 */
        .msg {
            font-size: 14px;
        }
        
        /* 小字标签 */
        .small {
            font-size: 12.5px;
            color: #6b7280;
        }
        
        /* 页面主标题再缩小一点 */
        h1 {
            font-size: 1.6rem !important;  /* 默认 2rem+ */
        }
        
        /* 输入框里的字体 */
        textarea {
            font-size: 14px !important;
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
        st.experimental_rerun()

    if st.button("🗑️ 清空当前会话"):
        st.session_state.chat_display = []
        st.experimental_rerun()

# ======== 主区域 =========
st.markdown("# 💬 Chatbot")

if not api_key:
    st.warning("请先在左侧输入 DashScope API Key 才能开始对话。")
else:
    # 可以缓存 backend 到 session_state，但为简单起见这里每次实例化（如需优化我可以帮你加缓存）
    backend = ChatBackend(api_key=api_key, model_name=model_name, temperature=temperature)

    # 聊天历史占位（位于输入上方）
    chat_area = st.container()
    history_ph = chat_area.empty()

    def render_history(ph):
        """渲染 st.session_state.chat_display（使用你原来的 HTML + CSS 样式）"""
        html = '<div class="chat-card">'
        for role, content in st.session_state.chat_display:
            css_class = "human" if role == "user" else "ai"
            avatar = "🧑‍💻" if role == "user" else "🤖"
            html += f"<div class='msg {css_class}'><span class='small'>{avatar} {role}</span><br/>{content}</div>"
        html += "</div>"
        ph.markdown(html, unsafe_allow_html=True)

    # 初始渲染历史
    render_history(history_ph)

    # 用户输入表单（表单在历史下方，因此输出始终在上面）
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_area("输入你的问题/指令：", height=100, placeholder="比如：帮我写一个带注释的二分查找函数。")
        submitted = st.form_submit_button("发送 ➤")

    if submitted and user_input.strip():
        # 1) 把用户消息写入历史（立即可见）
        st.session_state.chat_display.append(("user", user_input))

        # 2) 插入 assistant 占位（空字符串），保证后续输出显示在上方历史
        st.session_state.chat_display.append(("assistant", ""))
        ai_index = len(st.session_state.chat_display) - 1

        # 立刻渲染一次，使用户看到自己的消息和空 assistant 气泡（输出会填充此气泡）
        render_history(history_ph)

        # 3) 开始流式生成并实时写回历史
        full_text = ""
        try:
            for token in backend.chat_stream(user_input, st.session_state.thread_id):
                # token: 每次 yield 的字符串（可能是字符或片段）
                full_text += token

                # 更新 session 中的 assistant 占位内容（注意不要添加"思考中"）
                st.session_state.chat_display[ai_index] = ("assistant", full_text)

                # 重新渲染历史，保证输出始终在输入上方
                render_history(history_ph)

                # 小睡短暂时间帮助 Streamlit 刷新（按需调整或删除）
                time.sleep(0.01)

        except Exception as e:
            # 如果生成出错，把错误消息放到 assistant 气泡里（替代原逻辑）
            st.session_state.chat_display[ai_index] = ("assistant", f"[生成出错] {e}")
            render_history(history_ph)
        else:
            # 生成完成（full_text 已包含最终结果），已在循环中写回，无需额外操作
            pass

    # 页脚（session id）
    st.markdown(f"<div class='footer-note'>Session: <code>{st.session_state.thread_id}</code></div>", unsafe_allow_html=True)
