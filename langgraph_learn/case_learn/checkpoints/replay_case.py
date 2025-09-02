from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 定义状态
class State(TypedDict):
    step_msg: str
    counter: int

# 定义节点函数
def step_a(state: State):
    return {"step_msg": "执行 A 节点", "counter": state["counter"] + 1}

def step_b(state: State):
    return {"step_msg": "执行 B 节点", "counter": state["counter"] + 1}

# 构建图
workflow = StateGraph(State)
workflow.add_node("step_a", step_a)
workflow.add_node("step_b", step_b)
workflow.add_edge(START, "step_a")
workflow.add_edge("step_a", "step_b")
workflow.add_edge("step_b", END)

# 使用内存存储器
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 执行图
config = {"configurable": {"thread_id": "replay_thread"}}
graph.invoke({"step_msg": "", "counter": 0}, config=config)

# 获取历史 checkpoint
history = list(graph.get_state_history(config))
print("执行历史回放：")
for i, state in enumerate(reversed(history), start=1):  # 从最早到最新
    # print(f"Step {i}: {state.values}, checkpoint_id : {state.config}")
    print(f"Step {i}: {state.values}")


# Replay：从第一个 checkpoint 开始重新执行
first_checkpoint = history[-1]  # 最早的 checkpoint
# print("\n从第一个 checkpoint 开始回放：", first_checkpoint)
# print(first_checkpoint.config)
replay_state = graph.invoke(None, config=first_checkpoint.config)
# replay_state = graph.update_state(first_checkpoint.config, values={"step_msg": "replay node", "counter": 6})
print("回放状态：", replay_state)

# 获取历史 checkpoint
history = list(graph.get_state_history(config))
print("执行历史回放后：")
for i, state in enumerate(reversed(history), start=1):  # 从最早到最新
    print(f"Step {i}: {state.values}")