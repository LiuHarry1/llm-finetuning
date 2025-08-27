from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

class ChatState(TypedDict):
    messages: List[HumanMessage]

def call_model(state: ChatState):
    writer = get_stream_writer()  # LangGraph 流式输出

    llm = ChatTongyi(
        model="qwen-plus",
        api_key="sk-f256c03643e9491fb1ebc278dd958c2d",
        temperature=0.7,
        streaming=True,
        stream_options={"include_usage": True}
    )

    # 关键改动：使用 stream_messages，保证每个 token 调用 writer
    for chunk in llm.stream(state["messages"]):
        writer({"llm_chunk": chunk.content})  # 每个 token 都输出

    return {"messages": state["messages"] + [HumanMessage(content="(流式完成)")]}

graph = (
    StateGraph(ChatState)
    .add_node("model_node", call_model)
    .add_edge(START, "model_node")
    .add_edge("model_node", END)
    .compile()
)

initial_state = {"messages":[HumanMessage(content="给我写一个Python hello world示例")]}
for chunk in graph.stream(initial_state, stream_mode="custom", configurable={"thread_id":"thread1"}):
    print(chunk.get("llm_chunk", ""))
