from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Dict
import sqlite3
import re

from langchain_community.chat_models import ChatTongyi


# -----------------------------
# 定义状态结构
# -----------------------------
class ConversationState(TypedDict):
    history: List[str]
    entities: Dict[str, str]
    last_output: str


# -----------------------------
# 初始化 LLM
# -----------------------------
llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")


# -----------------------------
# 定义节点：实体抽取
# -----------------------------
def extract_entities(state: ConversationState):
    """调用 LLM 抽取用户输入中的实体（项目/模块/优先级）"""
    user_input = state["history"][-1]
    prompt = f"""
    从下面的用户输入中，提取关键信息（项目名, 模块名, 优先级）。
    如果没有提到则忽略，返回 JSON 格式。

    用户输入: {user_input}
    """

    resp = llm.invoke(prompt).content

    # 去掉 ```json ``` 包裹
    cleaned = re.sub(r"^```json\s*|\s*```$", "", resp.strip(), flags=re.MULTILINE)


    # 简单起见：假设 LLM 输出就是 JSON 格式
    try:
        import json
        new_entities = json.loads(cleaned)
    except:
        new_entities = {}

    # 更新已有实体
    state["entities"].update(new_entities)
    return state


# -----------------------------
# 定义节点：生成回答
# -----------------------------
def generate_answer(state: ConversationState):
    user_input = state["history"][-1]
    context = state["entities"]

    prompt = f"""
    用户的问题: {user_input}
    已知的上下文实体: {context}

    根据实体和历史对话，给用户一个合适的回答。
    """
    answer = llm.invoke(prompt).content
    state["last_output"] = answer
    return state


# -----------------------------
# 构建对话图
# -----------------------------
workflow = StateGraph(ConversationState)

workflow.add_node("extract_entities", extract_entities)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("extract_entities")
workflow.add_edge("extract_entities", "generate_answer")
workflow.add_edge("generate_answer", END)

# -----------------------------
# 配置持久化存储（SQLite）
# -----------------------------
# checkpointer = SqliteSaver.from_conn_string(":memory:")
# 建立 SQLite 连接
conn = sqlite3.connect(":memory:", check_same_thread=False)

# 直接实例化 SqliteSaver
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

# -----------------------------
# 模拟对话
# -----------------------------
config = {"configurable": {"thread_id": "user-123"}}


def ask(question):
    events = app.stream(
        {"history": [question], "entities": {}, "last_output": ""},
        config,
        stream_mode="values"
    )
    for ev in events:
        pass
    return ev["last_output"], ev["entities"]


# 多轮对话
print(ask("帮我看看项目 Alpha 最近有没有 Bug"))
print(ask("我主要关心支付模块"))
print(ask("那有没有高优先级的？"))
