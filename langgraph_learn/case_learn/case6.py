from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义状态
class State(TypedDict):
    # messages 是一个列表，用 add_messages 说明合并时是“追加”
    messages: Annotated[list, add_messages]

# 2. 定义节点
def user_node(state: State):
    print("用户节点运行")
    return {"messages": ["用户：你好"]}

def assistant_node(state: State):
    print("助手节点运行")
    return {"messages": ["助手：你好呀，我是 LangGraph Bot"]}

def second_user_node(state: State):
    print("第二个用户节点运行")
    return {"messages": ["用户：再见"]}

# 3. 构建图
graph = StateGraph(State)

graph.add_node("user_node", user_node)
graph.add_node("assistant_node", assistant_node)
graph.add_node("second_user_node", second_user_node)

# 流程：用户 → 助手 → 用户
graph.add_edge("user_node", "assistant_node")
graph.add_edge("assistant_node", "second_user_node")

graph.set_entry_point("user_node")
graph.set_finish_point("second_user_node")

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"messages": []})

print("\n最终结果：", result)
