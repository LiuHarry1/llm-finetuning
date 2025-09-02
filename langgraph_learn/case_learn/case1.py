from typing import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# 定义节点函数
def node1(state):
    print("执行节点1")
    return {"result": state["input"] + " 经过节点1处理"}

# 定义状态
class AgentState(TypedDict, total=False):
    input: str
    result: str

# 构建图
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("node1", node1)

# 设置入口和出口
graph_builder.set_entry_point("node1")
graph_builder.set_finish_point("node1")

# 上面的入口和出口的代码代码等价下面的
# graph_builder.add_edge(START, "node1")
# graph_builder.add_edge("node1", END)

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

# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
