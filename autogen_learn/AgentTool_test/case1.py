import asyncio
import json
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from pydantic import BaseModel
import autogen_learn.llm_client as llm_client

# Define the structured output format.
class AgentResponse(BaseModel):
    thoughts: str|None = None
    response: Literal["happy", "sad", "neutral"]


# Define the function to be called as a tool.
def sentiment_analysis(text: str) -> str:
    """Given a text, return the sentiment."""
    print("sentiment_analysis here")
    result = "happy" if "happy" in text else "sad" if "sad" in text else "neutral"
    # s = 1/0
    # return result
    print(result)
    return result


# Create a FunctionTool instance with `strict=True`,
# which is required for structured output mode.
tool = FunctionTool(sentiment_analysis, description="Sentiment Analysis", strict=True)

# Create an OpenAIChatCompletionClient instance that supports structured output.
model_client = llm_client.model_client

# Create an AssistantAgent instance that uses the tool and model client.
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[tool],
    system_message="Use the tool to analyze sentiment only.",
    # output_content_type=AgentResponse,
    # reflect_on_tool_use=True

)


async def main() -> None:
    stream = agent.run_stream(task="I am happy today!")
    await Console(stream)


asyncio.run(main())
asyncio.get_running_loop()
