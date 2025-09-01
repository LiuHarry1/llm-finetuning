import subprocess

from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import TypedDict

from langgraph.types import interrupt, Command
# https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/#pause-using-interrupt


# ------------------------------
# 状态定义
# ------------------------------
class AgentState(TypedDict):
    user_input: str
    generated_code: str
    approval: str
    exec_result: str

# ------------------------------
# 工具定义：OpenShift 命令执行
# ------------------------------
def run_openshift_command(code: str) -> str:
    try:
        # result = subprocess.check_output(
        #     code, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10
        # )
        result = "执行成功"
        return f"[执行成功]\n{result}"
    except subprocess.CalledProcessError as e:
        return f"[执行失败]\n{e.output}"
    except Exception as e:
        return f"[错误] {str(e)}"

# ------------------------------
# 节点定义
# ------------------------------
llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

def generate_code(state: AgentState):
    """根据用户输入生成 OpenShift 代码"""
    prompt = f"""
    用户请求: {state['user_input']}
    你是一个可以生成python 代码 和oc cli 的助手。
    如果用户描述需要操作 OpenShift 集群，请通过生成oc 命令,或者如果有必要也可以用python 代码来完成用户请求。
    只输出代码，不要解释。
    """
    response = llm.invoke(prompt)
    return {"generated_code": response.content.strip()}

def human_review(state: AgentState):
    """人工审核：确认是否执行代码"""
    code = state["generated_code"]
    print("\n=== AI 生成的代码 ===")
    print(code)
    # approval = input("是否执行这段代码？(approve/reject): ").strip().lower()

    result = interrupt(
        {
            "question": "是否执行这段代码？(approve/reject): ",
            # Surface the output that should be
            # reviewed and approved by the human.
            "generated_code": state["generated_code"]
        }
    )
    print("human_review", result)
    return {"approval": result["approval"]}

def execute_code(state: AgentState):
    """执行代码（仅在审批通过时）"""
    if state["approval"] == "approve":
        result = run_openshift_command(state["generated_code"])
    else:
        result = "[未执行] 人工拒绝执行代码。"
    print("\n=== 执行结果 ===")
    print(result)
    return {"exec_result": result}

# ------------------------------
# 构建 LangGraph 流程
# ------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("generate", generate_code)
workflow.add_node("review", human_review)
workflow.add_node("execute", execute_code)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_edge("review", "execute")
workflow.add_edge("execute", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ------------------------------
# 测试运行
# ------------------------------
if __name__ == "__main__":
    print("=== OpenShift Agent (LangGraph HITL Demo) ===")
    user_query = input("请输入操作描述: ")

    # 运行 Agent，增加 configurable 参数
    result = app.invoke(
        {"user_input": user_query},
        config={"configurable": {"thread_id": "demo"}}
    )

    print("\n=== 最终状态 ===")
    print(result)

    if result['__interrupt__']:
        value = result['__interrupt__'][0].value
        question = value.get("question", "")

        is_approve = input(question+": ")
        result = app.invoke(
            Command(resume={"approval":is_approve}),
            config={"configurable": {"thread_id": "demo"}}
        )




