import os
import asyncio
import uuid
from dotenv import load_dotenv
from zep_cloud import Message

from zep_cloud.client import AsyncZep
from langchain_community.chat_models import ChatTongyi
from langchain.agents import Tool, initialize_agent, AgentType

# ==================== 初始化 ====================
load_dotenv()
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

zep = AsyncZep(api_key=ZEP_API_KEY)
llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY, temperature=0)

HUMAN_ID = "user_human"
AGENT_ID = "agent_bot"


# ==================== 工具 ====================
async def search_facts(user_id: str, query: str):
    """从 Zep 知识图检索用户过去的事实"""
    results = await zep.graph.search(user_id=user_id, query=query, limit=5)
    return [edge.fact for edge in results.edges]


# 将 async 工具包装为同步调用给 LangChain Agent 使用
def search_facts_sync(query: str):
    return asyncio.run(search_facts(HUMAN_ID, query))


tools = [
    Tool(
        name="SearchFacts",
        # name="Intermediate Answer",
        func=search_facts_sync,
        description="根据用户问题，从历史对话中检索相关事实。"
    )
]


# ==================== Zep 消息写入 ====================
async def write_message(user_id: str, session_id: str, role: str, content: str):
    """写入单条消息到 Zep"""
    print(user_id, role, session_id, content)
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


# ==================== ReAct Agent 初始化 ====================
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # ReAct Agent
    verbose=True
)


# ==================== 对话循环 ====================
async def chat_loop():
    # await zep.user.add(user_id=HUMAN_ID)
    # await zep.user.add(user_id=AGENT_ID)
    # session_id = f"session_{uuid.uuid4().hex}"
    session_id = "942f1c89f04a4220ac81bf9418912fe2"

    print("🤖 Zep ReAct 机器人启动！Ctrl+C 退出。\n")

    chat_history = []
    while True:
        try:
            user_text = input("👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_text:
            continue

        # 1️⃣ 写入用户消息到 Zep
        await write_message(HUMAN_ID, session_id, "user", user_text)

        # 2️⃣ Agent 生成回复（自动判断是否调用 SearchFacts）
        reply_text = agent.run({
            "input": user_text,
            "chat_history": chat_history
        })

        # 3️⃣ 输出并写入 Zep
        print(f"🤖 助手：{reply_text}")
        await write_message(AGENT_ID, session_id, "assistant", reply_text)
        chat_history.append((user_text, reply_text))


if __name__ == "__main__":
    asyncio.run(chat_loop())
