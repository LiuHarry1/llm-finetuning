import asyncio
import os

from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

termination_condition = MaxMessageTermination(3)

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

def group_chat_approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Approval function for group chat that allows basic Python operations."""
    # Allow common safe operations
    safe_operations = ["print(", "import ", "def ", "class ", "if ", "for ", "while "]
    if any(op in request.code for op in safe_operations):
        return ApprovalResponse(approved=True, reason='Safe Python operation')

    # Deny file system operations in group chat
    dangerous_operations = ["open(", "file(", "os.", "subprocess", "eval(", "exec("]
    if any(op in request.code for op in dangerous_operations):
        return ApprovalResponse(approved=False, reason='File system or dangerous operation not allowed')

    return ApprovalResponse(approved=True, reason='Operation approved')


async def main() -> None:

    # define the Docker CLI Code Executor
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

    # start the execution container
    await code_executor.start()

    code_executor_agent = CodeExecutorAgent(
        "code_executor_agent",
        code_executor=code_executor,
        approval_func=group_chat_approval_func
    )
    coder_agent = AssistantAgent("coder_agent", model_client=model_client)

    groupchat = RoundRobinGroupChat(
        participants=[coder_agent, code_executor_agent], termination_condition=termination_condition
    )

    task = "Write python code to print Hello World!"
    await Console(groupchat.run_stream(task=task))

    # stop the execution container
    await code_executor.stop()


asyncio.run(main())
