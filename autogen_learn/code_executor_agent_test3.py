import asyncio
import os

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

from autogen_agentchat.agents import CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

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


termination_condition = TextMessageTermination("code_executor_agent")



async def main() -> None:

    async def model_client_approval_func(request: ApprovalRequest) -> ApprovalResponse:
        instruction = "Approve or reject the code in the last message based on whether it is dangerous or not. Use the following JSON format for your response: {approved: true/false, reason: 'your reason here'}"
        response = await model_client.create(
            messages=[SystemMessage(content=instruction)]
            + request.context
            + [UserMessage(content=request.code, source="user")],
            json_output=ApprovalResponse,
        )
        assert isinstance(response.content, str)
        return ApprovalResponse.model_validate_json(response.content)

    # define the Docker CLI Code Executor
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

    # start the execution container
    await code_executor.start()

    code_executor_agent = CodeExecutorAgent(
        "code_executor_agent",
        code_executor=code_executor,
        model_client=model_client,
        approval_func=model_client_approval_func,
    )

    task = "你可以python调用oc命令，因为你以及登录到openshift 里了，你帮我查一下 announcement这个service 下面log 里面有哪些runtime Exception"
    await Console(code_executor_agent.run_stream(task=task))

    # stop the execution container
    await code_executor.stop()


asyncio.run(main())
