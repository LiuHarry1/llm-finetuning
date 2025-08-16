from typing import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph, END


def node1(state):
    print("执行节点1")
    return {"result": state["input"] + " 经过节点1处理"}

def node2(state):
    print("执行节点2")
    return {"result": state["result"] + " 然后经过节点2处理"}

class AgentState(TypedDict, total=False):
    input: str
    result: str

workflow = StateGraph(AgentState)

workflow.add_node("node1", node1)

# 设置入口点
workflow.set_entry_point("node1")

# 设置节点间的流转
workflow.add_edge(START, "node1")
workflow.add_edge("node1", END)

# 设置出口点
# workflow.set_exit_point("node2")

app = workflow.compile()

# 运行工作流
result = app.invoke({"input": "初始输入"})
print(result)

print(app.get_graph().draw_ascii())