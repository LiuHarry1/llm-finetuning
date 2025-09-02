from typing import TypedDict
from langgraph.graph import StateGraph

# 定义节点
def node_start(state):
    print("进入起始节点")
    return {"base": state["input"]}

def nodeA(state):
    print("执行节点A")
    return {"a": state["base"] + " → 经过A"}

def nodeB(state):
    print("执行节点B")
    return {"b": state["base"] + " → 经过B"}

def merge_node(state):
    print("合并结果")
    return {"result": state["a"] + " | " + state["b"]}

# 定义状态
class AgentState(TypedDict, total=False):
    input: str
    base: str
    a: str
    b: str
    result: str

# 构建图
graph_builder = StateGraph(AgentState)

graph_builder.add_node("node_start", node_start)
graph_builder.add_node("nodeA", nodeA)
graph_builder.add_node("nodeB", nodeB)
graph_builder.add_node("merge_node", merge_node)

# 设置入口和出口
graph_builder.set_entry_point("node_start")
graph_builder.set_finish_point("merge_node")

# 定义并行流转
graph_builder.add_edge("node_start", "nodeA")
graph_builder.add_edge("node_start", "nodeB")

# 两个分支最后合并到 merge_node
graph_builder.add_edge("nodeA", "merge_node")
graph_builder.add_edge("nodeB", "merge_node")

# 编译
app = graph_builder.compile()

# 运行
result = app.invoke({"input": "初始输入"})
print("\n最终结果：", result)

# 打印图结构
print(app.get_graph().draw_ascii())

png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
