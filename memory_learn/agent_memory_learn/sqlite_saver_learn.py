import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
saver = SqliteSaver(conn)

builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke(3, config)
print(result)
