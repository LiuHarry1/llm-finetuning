# multi_agent_langgraph.py
"""
一个最小可运行的 LangGraph 多 agent 示例：
- flight_assistant: 负责订机票
- hotel_assistant: 负责订酒店
两者可以通过 handoff 工具相互转接。
此示例使用 LangGraph 的 React agent helper（示例风格），并使用内存状态。
"""


from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, MessagesState

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    from langchain_core.tools import tool, InjectedToolCallId
    from langgraph.types import Command, Send
    from typing import Annotated
    from langgraph.prebuilt import InjectedState

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str, "Description for next agent"],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_msg = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_msg]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )
    return handoff_tool



# ---- 简单的“工具”函数（模拟真实后端） ----
def book_flight(origin: str, dest: str, date: str):
    return f"Flight booked: {origin} -> {dest} on {date} (PNR: FL-{hash((origin,dest,date))%10000})"

def book_hotel(hotel_name: str, nights: int):
    return f"Hotel booked: {hotel_name} for {nights} nights (CONF: HT-{hash((hotel_name,nights))%10000})"

# ---- Handoff 工具：将对话或任务转给另一个 agent ----
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)

transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

# ---- 定义 agent（示例使用 create_react_agent） ----
# 注意：model 字段值视你可用的模型而定（openai:..., anthropic:..., local:...）
flight_assistant = create_react_agent(
    model="openai:gpt-4o",       # 按需替换为你有权限/可用的模型标识
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a helpful flight booking assistant. Ask for origin, destination and date.",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a helpful hotel booking assistant. Ask for hotel name and nights.",
    name="hotel_assistant"
)

# ---- 可选：一个 supervisor（管理/分派任务），这里简单演示不强制使用 ----
from langgraph.prebuilt import create_supervisor
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model="openai:gpt-4o",
    prompt="You manage a flight and hotel assistant. Assign tasks or ask for clarifications.",
    name="supervisor"
)

# ---- 构建图（graph） ----
# 使用 MessagesState（用于保存对话消息的简单状态类型）
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_node(supervisor)   # 如果不需要 supervisor，可以不加
)

# ---- 执行示例任务 ----
# Graph 的执行方式可以按官方示例 compile() -> run()
compiled = multi_agent_graph.compile()


# 举例：从 START 开始，给出用户一句话：我想订机票和酒店，优先订机票
task_input = {
    "messages": [
        {"role": "user", "content": "我想订机票和酒店，先帮我订机票：北京 到 上海，5月20日。"}
    ]
}

# run 执行（同步示例）
result = compiled.run(task_input)
print("Graph run result:", result)


