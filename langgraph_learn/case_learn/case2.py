from typing import TypedDict
from langgraph.graph import StateGraph

# 定义节点1
def node1(state):
    print("执行节点1")
    return {"mid": state["input"] + " → 节点1处理"}

# 定义节点2
def node2(state):
    print("执行节点2")
    return {"result": state["mid"] + " → 节点2处理"}

# 定义状态
class AgentState(TypedDict, total=False):
    input: str
    mid: str
    result: str

# 构建图
graph_builder = StateGraph(AgentState)

graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)

# 设置入口和出口
graph_builder.set_entry_point("node1")
graph_builder.set_finish_point("node2")

# 定义节点之间的流转
graph_builder.add_edge("node1", "node2")

# 编译
app = graph_builder.compile()

# 运行
result = app.invoke({"input": "初始输入"})
print(result)


png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

# 打印图结构
print(app.get_graph().draw_ascii())
