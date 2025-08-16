from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from typing import TypedDict

from langgraph.graph._node import StateNode


class State(TypedDict):
    x: float

def node1(state: State, config: RunnableConfig) -> State:
    return {"x": state["x"] + 1}

def node2(state: State, config: RunnableConfig) -> State:
    return {"x": state["x"] * 2}

graph = StateGraph(State)
# 创建 START 节点
start_node = StateNode(lambda state, config: state)
graph.add_node("START", start_node)

graph.add_sequence([("node1", node1), ("node2", node2)] )

compiled_graph = graph.compile()
result = compiled_graph.invoke({"x": 1.0})
print(result)  # 输出：{'x': 4.0}
