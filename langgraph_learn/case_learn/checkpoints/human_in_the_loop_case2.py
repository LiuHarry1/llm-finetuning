from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 定义状态
class State(TypedDict):
    user_query: str
    approved: bool
    result: str

# 第一步：接收用户输入
def receive_query(state: State):
    return {"user_query": state["user_query"], "approved": False}


# 第二步：请求人类审批
def request_approval(state: State):
    approval = interrupt({
        "prompt": f"AI 想要调用数据库查询工具，问题是：{state['user_query']}\n是否允许？(yes/no): "
    })
    # 人类输入 yes/no
    return {"approved": approval.lower().startswith("y")}


# 第三步：根据审批结果执行
def maybe_execute_tool(state: State):
    if state["approved"]:
        return {"result": f"✅ 工具执行成功，查询结果：'{state['user_query']}' 的数据在这里..."}
    else:
        return {"result": "❌ 工具调用被拒绝，未执行。"}


# 构建图
workflow = StateGraph(State)
workflow.add_node("receive_query", receive_query)
workflow.add_node("request_approval", request_approval)
workflow.add_node("maybe_execute_tool", maybe_execute_tool)

workflow.add_edge(START, "receive_query")
workflow.add_edge("receive_query", "request_approval")
workflow.add_edge("request_approval", "maybe_execute_tool")
workflow.add_edge("maybe_execute_tool", END)

# 使用内存存储器（实验用）
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


# === 首次执行 ===
config = {"configurable": {"thread_id": "thread_approval"}}
result = graph.invoke({"user_query": "查询客户订单信息", "approved": False, "result": ""}, config=config)

import langgraph_learn.case_learn.checkpoints.graph_logger as graph_logger

# 检查是否中断
if "__interrupt__" in result:
    print("执行暂停，等待人工审批...")
    interrupts = result["__interrupt__"]

    for interrupt_info in interrupts:
        # print("中断信息：", interrupt_info)
        prompt = interrupt_info.value.get("prompt")

    print("\n=== checkpint 历史 每一个步骤里的状态快照 ===")

    graph_logger.print_state_history(graph, "thread_approval")
    # 人类输入模拟（批准或拒绝）
    # human_enter = "yes"   # 改成 "no" 就能拒绝
    human_enter= input(prompt)

    # 恢复执行
    resumed = graph.invoke(Command(resume=human_enter), config=config)

    print("最终结果：", resumed["result"])
    graph_logger.print_state_history(graph, "thread_approval")
else:
    print("未触发中断，最终结果：", result["result"])
