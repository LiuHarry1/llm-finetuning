import sqlite3
from typing import Annotated, List

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict


# ============ 状态定义 ============
class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# ============ 节点函数 ============
def make_call_model(llm: ChatTongyi):
    """返回一个节点函数，读取 state -> 调用 LLM -> 更新 state.messages"""
    def call_model(state: ChatState) -> ChatState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    return call_model


# ============ 后端核心类 ============
class ChatBackend:
    def __init__(self, api_key: str, model_name: str = "qwen-plus", temperature: float = 0.7, db_path: str = "memory.sqlite"):
        self.llm = ChatTongyi(model=model_name, temperature=temperature, api_key=api_key)
        self.builder = StateGraph(ChatState)
        self.builder.add_node("model", make_call_model(self.llm))
        self.builder.set_entry_point("model")
        self.builder.add_edge("model", END)

        # SQLite 检查点
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)
        self.app = self.builder.compile(checkpointer=self.checkpointer)

    def chat(self, user_input: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        ai_text = final_state["messages"][-1].content
        return ai_text


if __name__ == '__main__':
    from backend import ChatBackend

    backend = ChatBackend(api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    print(backend.chat("你好", "thread1"))

