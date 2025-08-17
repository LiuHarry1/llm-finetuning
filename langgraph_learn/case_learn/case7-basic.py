from langchain_community.chat_models import ChatTongyi
from typing import Annotated

from rich.panel import Panel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

from rich.console import Console
from rich.markdown import Markdown

console = Console()

from rich.table import Table
# rich 控制台
console = Console()

# 流式输出
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            msg = value["messages"][-1].content
            print(f"🤖 Assistant: {msg}")



# 主循环
while True:
    try:
        user_input = input("🧑 User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("[red]Goodbye![/red]")
            break
        stream_graph_updates(user_input)
    except:
        # fallback: 在不支持 input() 的环境下自动执行一次
        user_input = "What do you know about LangGraph?"
        console.print(f"User: {user_input}")
        stream_graph_updates(user_input)
        break