# reflection_agent_example.py
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import MessageGraph
from langchain.schema import BaseMessage, HumanMessage
import os
from typing import List

load_dotenv()

llm = ChatTongyi( model="qwen-plus", api_key=os.getenv("TONGYI_API_KEY"))

# ---------- 节点函数 ----------

def generation_node(state: List[BaseMessage]) -> str:
    """生成节点：根据历史对话生成回答"""
    if state:
        prompt = "继续回答问题：" + " ".join([m.content for m in state])
    else:
        prompt = "请给我一个简短有趣的回答"
    response = llm([HumanMessage(content=prompt)])
    print(f"[生成节点] {response.content}")
    return response.content


def reflection_node(state: List[BaseMessage]) -> str:
    """反思节点：对生成节点的回答进行批评与改进建议"""
    last_message = state[-1].content if state else ""
    prompt = (
        f"请分析以下回答是否有问题或可以改进，并给出建议：\n"
        f"{last_message}"
    )
    response = llm([HumanMessage(content=prompt)])
    print(f"[反思节点] {response.content}")
    return response.content


# ---------- 定义是否继续的函数 ----------

def should_continue(state: List[BaseMessage]):
    """控制循环次数，超过6轮停止"""
    if len(state) >= 6:
        return "END"
    return "reflect"


# ---------- 构建图 ----------

builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

# ---------- 测试运行 ----------

if __name__ == "__main__":
    print("===== 测试反思型智能体 =====")
    # 初始对话状态
    state: List[BaseMessage] = []

    # 执行图
    result = graph.invoke(state)

    print("\n===== 最终结果 =====")
    for i, msg in enumerate(result):
        print(f"轮次 {i + 1}: {msg.content}")
