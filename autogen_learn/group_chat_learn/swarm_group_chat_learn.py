import asyncio
import os
from typing import Dict, Any

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Create the agents.

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
print(TONGYI_API_KEY)


def create_client():
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
    return model_client


class ContentCreationSwarm:

    def __init__(self):
        self.planner = None
        self.researcher = None
        self.writer = None
        self.editor=None
        self.model_client = create_client()
        self.setup_agents()
        self.setup_team()

    @staticmethod
    def research_topic(topic:str) -> Dict[str, Any]:
        # todo
        research_data = {
            "topic":topic,
            "sources":["学术论文", "行业报告", "专家访谈"],
            "statistics": {"市场管理": "100亿", "增长率":"15%"}
        }
        return research_data

    @staticmethod
    def check_grammar(text:str)  -> Dict[str, Any]:

        return {
            "original_text":text[:100] +"...",
            "issues_found": 2,
            "suggestions":["建议使用更简洁的句式","注意段落之间的连贯性"],
            "overall_score":85
        }

    def setup_agents(self):
        self.planner =AssistantAgent("planner",
                            model_client=self.model_client,
                                     handoffs=["researcher", "writer", "editor"],
                            system_message="""你是内容策划总结，
                                    职责：
                                    1. 制订内容创作计划
                                    2. 协调个专业团队成员
                                    3. 分配具体任务给，研究员，写作者，编辑
                                    4。 确保内容质量和进度
                                    5. 完成所有任务后使用 TERMINATE
                                    工作流程 研究 -> 写作 -> 编辑
                                """)
        # todo
        self.researcher = AssistantAgent("researcher",
                            model_client=self.model_client,
                                     handoffs=["planner"],
                                    tools=[self.research_topic],
                            system_message="""你是专业研究员，
                                    职责：
                                    1. 使用 research_topic 工具收集资料
                                    
                                """)

        self.writer = AssistantAgent("writer",
                                         model_client=self.model_client,
                                         handoffs=["planner"],
                                         tools=[self.research_topic],
                                         system_message="""你是专业写作者，
                                    职责：
                                    1. 风格适合目标读者 - 基于研究资料创作内容
                                    2. 确保内容结构清晰， 逻辑合理
                                    3. 完成写作后转回策划者，
                                """)

        self.editor = AssistantAgent("editor",
                                         model_client=self.model_client,
                                         handoffs=["planner"],
                                         tools=[self.check_grammar],
                                         system_message="""你是专业编辑，
                                    职责：
                                    1. 使用 check_grammar 工具检查文本
                                    2. 优化内容结构和表达
                                    3. 确保内容质量符合发布标准
                                    4. 完成编辑后转会策划者
                                """)

    def setup_team(self):
        text_termination = TextMentionTermination("完成报告")
        max_message_termination = MaxMessageTermination(max_messages=10)
        termination = text_termination | max_message_termination

        self.team = Swarm([self.planner, self.researcher, self.writer, self.editor]
                          , termination_condition = termination)

    async def create_content(self, topic: str):

        print("=== 内容创作团队启动")
        task = f"为猪蹄 ‘{topic}’ 创作一篇高质量的文章"
        result = await Console(self.team.run_stream(task=task))

        print("内容制作完成")
        return result

async def demo_content_creation():

    print("-"*30)
    content_team = ContentCreationSwarm()
    result = await content_team.create_content("人工智能在线教育中的应用")
    print("内容创作演示完成")
    print(result)

if __name__ == '__main__':
    asyncio.run(demo_content_creation())
