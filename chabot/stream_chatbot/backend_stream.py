import json
import os
import sqlite3
from typing import Annotated, List, Dict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict

load_dotenv()
# ===== 状态定义 =====
class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# ===== 节点函数（流式调用） =====
def make_call_model(llm: ChatTongyi):
    def call_model(state: ChatState) -> ChatState:
        writer = get_stream_writer()  # LangGraph 流式输出
        full_response = ""
        for chunk in llm.stream(state["messages"]):
            writer({"llm_chunk": chunk.content})  # 实时流式输出
            full_response += chunk.content

        # 返回更新后的状态，保存完整生成内容
        return {"messages": [AIMessage(content=full_response)]}

    return call_model


# ===== 后端核心 =====
class ChatBackend:
    def __init__(self, api_key: str, model_name: str = "qwen-plus", temperature: float = 0.7, db_path: str = "memory.sqlite"):
        self.llm = ChatTongyi(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            streaming=True,
        )

        # 构建 LangGraph 状态图
        self.builder = StateGraph(ChatState)
        self.builder.add_node("model", make_call_model(self.llm))
        self.builder.add_edge(START, "model")
        self.builder.add_edge("model", END)

        self.checkpointer = MemorySaver()

        # 编译图
        self.app = self.builder.compile(checkpointer=self.checkpointer)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            state TEXT
        )
        """)
        self.conn.commit()

    def save_thread(self, thread_id: str, state: Dict):
        """保存完整状态到 SQLite"""
        print("save thread", thread_id, state)
        cursor = self.conn.cursor()
        state_json = json.dumps(state, ensure_ascii=False)
        cursor.execute("""
        INSERT INTO threads(thread_id, state)
        VALUES(?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET state=excluded.state
        """, (thread_id, state_json))
        self.conn.commit()

    def load_thread(self, thread_id: str) -> Dict:
        """从 SQLite 加载历史对话"""
        print("load_thread", thread_id)
        cursor = self.conn.cursor()
        cursor.execute("SELECT state FROM threads WHERE thread_id=?", (thread_id,))
        row = cursor.fetchone()
        if not row:
            return {"messages": []}
        state_json = row[0]
        state = json.loads(state_json)
        print("load thread state", state)
        return state

    def get_all_thread(self) -> List:
        cursor = self.conn.cursor()
        cursor.execute("SELECT thread_id FROM threads ORDER BY rowid DESC")
        all_threads = [row[0] for row in cursor.fetchall()]
        return all_threads

    def chat_stream(self, user_input: str, thread_id: str):
        print("thread_id", thread_id)
        state = self.load_thread(thread_id)
        if not state:
            state = {"messages":[{"role": "user", "content": user_input}]}
        else:
            state.get("messages").append({"role": "user", "content": user_input})
        messages: List[Dict] = state.get("messages", [])
        # messages.append({"role": "user", "content": user_input})

        # 2️⃣ 流式输出
        full_text = ""
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.app.stream(state, stream_mode="custom", config=config):
            if "llm_chunk" in chunk:
                full_text += chunk["llm_chunk"]
                yield chunk["llm_chunk"]

        print("state", state)

        messages.append({"role": "assistant", "content": full_text})
        self.save_thread(thread_id, {"messages": messages})


if __name__ == "__main__":

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请在环境变量中设置 DASHSCOPE_API_KEY")

    backend = ChatBackend(api_key=api_key)
    thread_id = "test_thread_1"
    user_input = "请写一个 Python 的 hello world 示例"

    print(f"=== 测试发送消息 ===\nThread ID: {thread_id}\n用户输入: {user_input}\n")

    # 流式输出 AI 回复
    full_response = ""
    for token in backend.chat_stream(user_input, thread_id):
        print(token, end="", flush=True)
        full_response += token

    print("\n\n=== 历史读取 ===")
    state = backend.load_thread(thread_id)
    for msg in state["messages"]:
        role = msg["role"]
        content = msg["content"]
        print(f"{role}: {content}")