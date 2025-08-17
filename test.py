from typing import Annotated, get_args, get_origin

# 定义一个合并函数（类似 langgraph.graph.message.add_messages）
def add_messages(old: list, new: list) -> list:
    return old + new

# 定义带 Annotated 的类型
MessagesType = Annotated[list, add_messages]

# 假设我们在 TypedDict 里定义过类似：
class State:
   messages: MessagesType

# ---- 模拟 LangGraph 内部逻辑 ----

# 1. 取出注解
anno = State.__annotations__["messages"]
print("原始注解：", anno)

# 2. 拆解 Annotated
origin = get_origin(anno)   # 原始类型
args = get_args(anno)       # (原始类型参数, 附加元数据...)

print("origin:", origin)
print("args:", args)

# 3. 取出合并函数
merge_fn = args[1]
print("合并函数:", merge_fn)

# 4. 模拟两次状态更新
state = {"messages": ["用户：你好"]}
update1 = {"messages": ["助手：你好呀"]}
update2 = {"messages": ["用户：再见"]}

# 每次合并时调用 merge_fn
state["messages"] = merge_fn(state["messages"], update1["messages"])
state["messages"] = merge_fn(state["messages"], update2["messages"])

print("最终状态：", state)
