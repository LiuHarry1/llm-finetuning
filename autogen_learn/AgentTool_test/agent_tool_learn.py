import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.ui import Console
import autogen_learn.llm_client as llm_client


async def main() -> None:
    model_client = llm_client.model_client

    math_agent = AssistantAgent(
        "math_expert",
        model_client=model_client,
        system_message="You are a math expert.",
        description="A math expert assistant.",
        model_client_stream=False,
    )
    math_agent_tool = AgentTool(math_agent, return_value_as_last_message=True)

    chemistry_agent = AssistantAgent(
        "chemistry_expert",
        model_client=model_client,
        system_message="You are a chemistry expert.",
        description="A chemistry expert assistant.",
        model_client_stream=False,
    )
    chemistry_agent_tool = AgentTool(chemistry_agent, return_value_as_last_message=True)

    agent = AssistantAgent(
        "general_assistant",
        system_message="You are a general assistant. Use expert tools when needed.",
        model_client=model_client,
        model_client_stream=False,
        tools=[math_agent_tool, chemistry_agent_tool],
        max_tool_iterations=10,
    )

    async for message in agent.run_stream(task = "What is the integral of x^2?"):
        if isinstance(message, TaskResult):
            print("停止原因", message.stop_reason)
        else:
           print(f"[{message.source}]: {message.content}")

    # await Console(agent.run_stream(task="What is the integral of x^2?"))
    # await Console(agent.run_stream(task="What is the molecular weight of water?"))


asyncio.run(main())