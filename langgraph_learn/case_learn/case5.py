from typing import TypedDict
from langgraph.graph import StateGraph

# 定义节点
def node1(state):
    print("执行节点1")
    return {"step1": state["input"] + " → 节点1处理"}

def node2(state):
    print("执行节点2")
    return {"step2": state["step1"] + " → 节点2处理"}

def node3(state):
    print("执行节点3")
    return {"result": state["step2"] + " → 节点3处理"}

# 定义状态
class AgentState(TypedDict, total=False):
    input: str
    step1: str
    step2: str
    result: str

# 构建图
graph_builder = StateGraph(AgentState)

# 用 add_sequence 定义并添加节点，同时建立顺序关系
graph_builder.add_sequence([
    ("node1", node1),
    ("node2", node2),
    ("node3", node3),
])

# 设置入口和出口
graph_builder.set_entry_point("node1")
graph_builder.set_finish_point("node3")

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
