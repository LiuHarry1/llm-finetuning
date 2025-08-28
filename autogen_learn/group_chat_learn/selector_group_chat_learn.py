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

researcher = AssistantAgent("researcher",
                            description="负责研究和收集信息的专家， 应该在需要搜集背景信息被选择",
                            model_client=model_client,
                            system_message="""你是一个专业的研究员，
                                        你的任务是收集，整理和分析相关信息，
                                        当其他智能体需要背景信息或数据支持时，你应该提供帮助。
                                """)

analyzer = AssistantAgent("analyzer",
                          description="负责分析数据和提供见解的专家，应该在需要深入分析是被选择",
                            model_client=model_client,
                            system_message="""你是一个数据分析专家，
                                        你的任务是分析数据，识别趋势，得到洞察，
                                        基于研究员提供的信息进行深入分析。""")
writer = AssistantAgent("writer",
                        description="负责撰写报告和总结的专家，应该在需要整理成文档时被选择",
                            model_client=model_client,
                            system_message="""你是一个专业的撰写员
                                        你的任务是将研究和分析结果真理成清晰，结构话的报告
                                        当需要最终总结时发言并说‘完成报告’结束对话。""")
text_termination = TextMentionTermination("完成报告")
max_message_termination = MaxMessageTermination(max_messages=10)
termination = text_termination | max_message_termination
team = SelectorGroupChat([researcher, analyzer, writer],
                        model_client=model_client,
                         termination_condition=termination,
                         allow_repeated_speaker=False)

task = "请分析人工智能在教育领域的应用现代和发展趋势，并撰写一份报告"
async def main():

    async for message in team.run_stream(task = task):
        if isinstance(message, TaskResult):
            print("停止原因", message.stop_reason)
        else:
           print(f"[{message.source}]: {message.content}")

if __name__ == '__main__':
    asyncio.run(main())



