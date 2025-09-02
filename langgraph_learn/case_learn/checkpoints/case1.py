from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# 定义状态结构
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

# 定义节点函数
def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# 构建图
workflow = StateGraph(State)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 配置 checkpointer
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# png_data = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_data)


# === 执行 & 保存状态 ===
print("=== 首次执行 ===")
graph.invoke({"foo": "", "bar": []}, {"configurable": {"thread_id": "1"}})

# 获取最新状态快照
state = graph.get_state({"configurable": {"thread_id": "1"}})
print("当前 foo:", state.values.get("foo"))
print("当前 bar:", state.values.get("bar"))

print("=== 二次执行 ===")
graph.invoke({"foo": "", "bar": ['c']}, {"configurable": {"thread_id": "1"}})

# 获取最新状态快照
state = graph.get_state({"configurable": {"thread_id": "1"}})
print("当前 foo:", state.values.get("foo"))
print("当前 bar:", state.values.get("bar"))

# === 查看历史 checkpoint ===
print("\n=== 历史 checkpoint ===")
history = list(graph.get_state_history({"configurable": {"thread_id": "1"}}))
history.reverse()  # 反转，确保从最早到最新

for i, step in enumerate(history, start=1):
    foo_val = step.values.get("foo")
    bar_val = step.values.get("bar")
    print(f"Step {i}: foo={foo_val} | bar={bar_val}")

print("\n=== checkpint 历史 每一个步骤里的状态快照 ===")
for i, step in enumerate(history, start=1):
    print(step)