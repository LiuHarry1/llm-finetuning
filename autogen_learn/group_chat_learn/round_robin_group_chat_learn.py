import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


# max_messages_termination = MaxMessageTermination(max_messages=25)

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
print(TONGYI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=TONGYI_API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
    },
)

primary_agent = AssistantAgent("primary", model_client=model_client, system_message="You are a helpful AI assistant")

critic_agent = AssistantAgent("critic", model_client=model_client,
                              system_message="Provide constructive feedback, Respond with 'APPROVE' when your feedbacks are addressed.")
text_termination = TextMentionTermination("APPROVE")
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)


async def main():

    async for message in team.run_stream(task = "写一首关于秋天的诗"):
        if isinstance(message, TaskResult):
            print("停止原因", message.stop_reason)
        else:
           print(f"[{message.source}]: {message.content}")

if __name__ == '__main__':
    asyncio.run(main())



