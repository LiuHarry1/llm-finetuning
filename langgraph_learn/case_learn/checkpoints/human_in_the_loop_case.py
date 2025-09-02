from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 定义状态结构
class State(TypedDict):
    user_query: str
    to_search: bool
    search_result: str

# 第一步：获取用户查询
def ask_user(state: State):
    # 提供提示给用户，等待其确认/修正 query
    revised = interrupt({"prompt": "请确认或修改你的查询内容", "current_query": state["user_query"]})
    return {"user_query": revised, "to_search": True}

# 第二步：可选的工具调用（如搜索）
def call_search(state: State):
    if not state["to_search"]:
        return {"search_result": "Skipped search by user decision."}
    # 示例工具调用
    result = f"搜索结果：你查询的是 '{state['user_query']}'"
    return {"search_result": result}

# 第三步：最终输出
def final_response(state: State):
    return {"search_result": state["search_result"]}

# 构建图
workflow = StateGraph(State)
workflow.add_node("ask_user", ask_user)
workflow.add_node("call_search", call_search)
workflow.add_node("final_response", final_response)
workflow.add_edge(START, "ask_user")
workflow.add_edge("ask_user", "call_search")
workflow.add_edge("call_search", "final_response")
workflow.add_edge("final_response", END)

# 使用内存存储器（实验用），生产请换成 SqliteSaver/PostgresSaver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 启动图执行：初始输入来自用户
config = {"configurable": {"thread_id": "thread_human"}}
result = graph.invoke({"user_query": "LangGraph 是什么？", "to_search": False, "search_result": ""}, config=config)

# 检查是否被中断
if "__interrupt__" in result:
    print("图执行已中断，等待人类输入…")
    interrupt_info = result["__interrupt__"]
    print("Interrupt payload:", interrupt_info)

    # 假设我们从 UI 提供用户修改后的文本
    human_input = "请告诉我 LangGraph 支持哪些功能？"


    # 继续执行图
    resumed = graph.invoke(Command(resume=human_input), config=config)
    print("图恢复后结果：", resumed)
else:
    print("未触发中断，最终结果：", result)
