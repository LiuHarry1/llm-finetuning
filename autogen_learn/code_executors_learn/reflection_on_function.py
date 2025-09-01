import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

import autogen_learn.llm_client as llm_client

# Global counter state
counter = 0


def increment_counter() -> str:
    """Increment the counter by 1 and return the current value."""
    global counter
    counter += 1
    return f"Counter incremented to: {counter}"


def get_counter() -> str:
    """Get the current counter value."""
    global counter
    return f"Current counter value: {counter}"


async def main() -> None:
    model_client = llm_client.model_client

    # Create agent with max_tool_iterations=5 to allow multiple tool calls
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[increment_counter, get_counter],
        max_tool_iterations=5,  # Allow up to 5 tool call iterations
        reflect_on_tool_use=False,  # Get a final summary after tool calls
    )

    await Console(agent.run_stream(task="Increment the counter 3 times and then tell me the final value."))


asyncio.run(main())
