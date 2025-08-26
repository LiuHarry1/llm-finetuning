import asyncio
from autogen_agentchat.agents import CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.messages import TextMessage
# from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

from autogen_core import CancellationToken
# from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


def simple_approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Simple approval function that requests user input for code execution approval."""
    print("Code execution approval requested:")
    print("=" * 50)
    print(request.code)
    print("=" * 50)

    while True:
        user_input = input("Do you want to execute this code? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            return ApprovalResponse(approved=True, reason='Approved by user')
        elif user_input in ['n', 'no']:
            return ApprovalResponse(approved=False, reason='Denied by user')
        else:
            print("Please enter 'y' for yes or 'n' for no.")


async def run_code_executor_agent() -> None:
    # Create a code executor agent that uses a Docker container to execute code.
    # code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")
    await code_executor.start()
    code_executor_agent = CodeExecutorAgent(
        "code_executor",
        code_executor=code_executor,
        approval_func=simple_approval_func
    )

    # Run the agent with a given code snippet.
    task = TextMessage(
        content='''Here is some code
```python
print('Hello world')
```
''',
        source="user",
    )
    response = await code_executor_agent.on_messages([task], CancellationToken())
    print(response.chat_message)

    # Stop the code executor.
    await code_executor.stop()


asyncio.run(run_code_executor_agent())
