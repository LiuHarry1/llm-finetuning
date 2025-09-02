from typing import TypedDict
from langgraph.graph import StateGraph

# 定义节点
def check_input(state):
    print("检查输入内容")
    return {}

def success_node(state):
    print("进入成功节点")
    return {"result": state["input"] + " → 成功处理"}

def error_node(state):
    print("进入错误节点")
    return {"result": state["input"] + " → 错误处理"}

# 定义状态
class AgentState(TypedDict, total=False):
    input: str
    result: str

# 构建图
graph_builder = StateGraph(AgentState)

graph_builder.add_node("check_input", check_input)
graph_builder.add_node("success", success_node)
graph_builder.add_node("error", error_node)

# 设置入口
graph_builder.set_entry_point("check_input")

# 条件分支逻辑
def route_condition(state):
    if "错误" in state["input"]:
        return "error"
    else:
        return "success"

graph_builder.add_conditional_edges("check_input", route_condition, {"success": "success", "error": "error"})

# 设置出口
graph_builder.set_finish_point("success")
graph_builder.set_finish_point("error")

# 编译
app = graph_builder.compile()

# 测试
print("\n=== 测试1: 正常输入 ===")
print(app.invoke({"input": "这是一次正常请求"}))

print("\n=== 测试2: 错误输入 ===")
print(app.invoke({"input": "这是一次错误请求"}))

# 打印图结构
print(app.get_graph().draw_ascii())

png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
