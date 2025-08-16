from typing import TypedDict

from langgraph.graph import StateGraph, END

def node1(state):
    return {"value": state["input"] + " processed"}

def node2(state):
    return {"value": state["value"] + " by node2"}

def node3(state):
    return {"value": state["value"] + " by node3"}

def router(state):
    if len(state["value"]) > 20:
        return "node2"
    else:
        return "node3"

class AgentState(TypedDict, total=False):
    input: str
    result: str
    value: str

workflow = StateGraph(AgentState)
workflow.add_node("node1", node1)
workflow.add_node("node2", node2)
workflow.add_node("node3", node3)

workflow.set_entry_point("node1")
workflow.add_conditional_edges("node1", router, {"node2": "node2", "node3": "node3"})
# workflow.add_edge("node2", "node3")
# workflow.set_exit_point("node3")

app = workflow.compile()
result = app.invoke({"input": "short"})  # 会走node3
print(result)

result = app.invoke({"input": "a very long input string"})  # 会走node2
print(result)
print(app.get_graph().draw_ascii())
print(app.get_graph().draw_png("test"))