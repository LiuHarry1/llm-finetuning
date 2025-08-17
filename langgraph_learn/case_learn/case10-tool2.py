from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from langgraph.constants import START
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.chat_models import ChatTongyi
import os

llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

os.environ["TAVILY_API_KEY"] = "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"

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 打印图结构
print(graph.get_graph().draw_ascii())

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    last_msg = None
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                              config, stream_mode="values"):
        for value in event.values():
            msg = value["messages"][-1]
            # print(msg)
            last_msg = msg.content
    if last_msg:
        print(f"🤖 Assistant: {last_msg}")

while True:
    try:
        user_input = input("🧑 User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break