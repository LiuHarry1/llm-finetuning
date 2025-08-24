import os
import uuid
import asyncio
import signal
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from zep_cloud import Message
from zep_cloud.client import AsyncZep, Zep
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.chat_models import ChatTongyi

# ================== 环境变量 ==================
load_dotenv(override=True)
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

if not ZEP_API_KEY or not TONGYI_API_KEY:
    raise RuntimeError("缺少 API Key，请在 .env 文件中设置。")

# 初始化
llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY, temperature=0)
zep = AsyncZep(api_key=ZEP_API_KEY)


# ================== 工具：搜索记忆 ==================
@tool
async def search_facts(state: MessagesState, query: str, limit: int = 5):
    """在用户的所有对话中搜索记忆（facts）"""
    results = await zep.graph.search(
        user_id=state["human_id"],
        query=query,
        limit=limit,
    )
    return [edge.fact for edge in results.edges]


tools = [search_facts]
tool_node = ToolNode(tools)
llm = llm.bind_tools(tools)


# ================== LLM 节点 ==================
async def call_llm(state: MessagesState):
    """调用 LLM 生成回复"""
    response = await llm.ainvoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# ================== 构建 LangGraph ==================
graph = StateGraph(MessagesState)
graph.add_node("agent", call_llm)
graph.add_node("tools", tool_node)

graph.add_edge("agent", "tools")
graph.add_edge("tools", "agent")

graph.set_entry_point("agent")
graph.set_finish_point("agent")
app = graph.compile()


# ================== Zep 辅助函数 ==================
async def ensure_user(user_id: str) -> str:
    """保证用户存在"""
    # await zep.user.add(user_id=user_id)
    await  zep.user.get(user_id)
    return user_id


def new_session(user_id) -> str:
    # thread_id = "76b3af0a348d4d55a741afc14da7ae41"
    client = Zep(api_key=ZEP_API_KEY)
    thread_id = uuid.uuid4().hex
    client.thread.create(
        thread_id=thread_id,
        user_id=user_id,
    )
    print(thread_id)
    return thread_id


async def write_message(user_id: str, session_id: str, role: str, content: str):
    """写入单条消息到 Zep"""
    messages = [
        Message(
            name=user_id,
            role=role,
            content=content,
        ),
    ]
    print("thread id", session_id )
    await zep.thread.add_messages(thread_id=session_id, messages = messages)
    print("Finished to write messages", session_id)

# ================== 对话循环 ==================
async def chat_loop():
    human_id = await ensure_user("user_human")
    agent_id = await ensure_user("agent_bot")
    session_id = new_session("user_human")

    state = MessagesState(messages=[], human_id=human_id, agent_id=agent_id, session_id=session_id)

    print("🤖 Zep 记忆机器人启动！输入消息开始对话，Ctrl+C 退出。\n")

    while True:
        try:
            user_text = input("👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_text:
            continue

        # 用户说的话 -> 存入 Zep
        await write_message(human_id, session_id, "user", user_text)
        state["messages"].append({"role": "user", "content": user_text})

        # 调用 Agent
        result_state = await app.ainvoke(state)

        # 取出助手的回复
        assistant_msg = result_state["messages"][-1]
        assistant_text = assistant_msg["content"]

        print(f"🤖 助手：{assistant_text}")

        # 助手回复 -> 存入 Zep
        await write_message(agent_id, session_id, "assistant", assistant_text)

        # 更新状态
        state = result_state


def _install_sigint_handler(loop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    _install_sigint_handler(loop)
    try:
        loop.run_until_complete(chat_loop())
    finally:
        loop.close()
