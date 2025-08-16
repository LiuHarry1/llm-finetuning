import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import requests
import pandas as pd
from langchain_community.chat_models import ChatTongyi
from langgraph.types import Send

llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d"
)


# ========== 已知报表 API ==========
REPORT_APIS = {
    "sales_report": "https://api.example.com/reports/sales",
    "inventory_report": "https://api.example.com/reports/inventory",
    "customer_report": "https://api.example.com/reports/customers"
}

# ========== 工具函数 ==========
# def call_report_api(report_type: str, params: dict = None):
#     url = REPORT_APIS.get(report_type)
#     if not url:
#         raise ValueError(f"未知报表类型: {report_type}")
#     response = requests.get(url, params=params or {})
#     response.raise_for_status()
#     return response.json()

def call_report_api(report_type: str, params: dict = None):
    """Mock 报表 API"""
    if report_type == "sales_report":
        return [
            {"order_id": "SO-1001", "date": "2025-07-01", "customer": "Alice", "amount": 1200.50, "status": "paid"},
            {"order_id": "SO-1002", "date": "2025-07-02", "customer": "Bob", "amount": 850.00, "status": "pending"},
            {"order_id": "SO-1003", "date": "2025-07-05", "customer": "Charlie", "amount": 450.75, "status": "paid"}
        ]
    elif report_type == "inventory_report":
        return [
            {"item_id": "SKU-001", "name": "Laptop", "stock": 34, "warehouse": "Shanghai"},
            {"item_id": "SKU-002", "name": "Mouse", "stock": 120, "warehouse": "Beijing"},
            {"item_id": "SKU-003", "name": "Keyboard", "stock": 75, "warehouse": "Shanghai"}
        ]
    elif report_type == "customer_report":
        return [
            {"customer_id": "C-1001", "name": "Alice", "email": "alice@example.com", "orders": 12, "total_spent": 5800.50},
            {"customer_id": "C-1002", "name": "Bob", "email": "bob@example.com", "orders": 8, "total_spent": 3200.00},
            {"customer_id": "C-1003", "name": "Charlie", "email": "charlie@example.com", "orders": 5, "total_spent": 1500.75}
        ]
    else:
        raise ValueError(f"未知报表类型: {report_type}")

def export_to_csv(data, filename="report.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

# ========== Agent 状态 ==========
class AgentState(TypedDict, total=False):
    user_input: str
    report_type: str
    params: dict
    report_data: list
    report_file: str
    message: str


# ========== 定义 LLM ==========
# llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# ========== 定义图中的节点 ==========
def parse_request(state: AgentState):
    """1. LLM 解析用户输入 → 生成 report_type + params"""
    prompt = f"""
    你是报表助手。
    用户输入: "{state['user_input']}"。
    请判断用户是否提供了完整参数生成报表。
    - 如果参数不全，只输出 JSON: {{ "message": "需要以下参数: ..."}} 
    - 如果参数齐全，输出 JSON: {{
        "message": "可以生成报表",
        "report_type": "...",
        "params": {{"参数": "值"}}
    }}
    """
    response = llm.invoke(prompt).content
    # 去掉 ```json ``` 包裹
    response = re.sub(r"^```json\s*|\s*```$", "", response.strip(), flags=re.MULTILINE)
    import json
    parsed = json.loads(response)
    state["report_type"] = parsed.get("report_type", "")
    state["params"] = parsed.get("params",  {})
    state["message"] = parsed.get("message", {})
    return state

def workflow_decision(state: AgentState):
    if state.get("report_type"):
        # 下一步节点在 dict 中指定
        state["_next_node"] = "fetch_report"
        # state["message"] = "参数完整，准备获取报表"
        return state
    else:
        state["_next_node"] = "finish"  # 或者 END
        # state["message"] = "参数不全，需要补充报表参数"
        return state


def condition_fetch_report(state):
    return bool(state.get("report_type"))

def condition_finish(state):
    return not bool(state.get("report_type"))

def fetch_report(state: AgentState):
    """2. 调用 API 获取数据"""
    data = call_report_api(state["report_type"], state["params"])
    state["report_data"] = data
    return state

def export_report(state: AgentState):
    """3. 导出报表"""
    filename = export_to_csv(state["report_data"], f"{state['report_type']}.csv")
    state["report_file"] = filename
    return state

def finish(state: AgentState):
    """4. 返回结果"""
    if "report_file" in state:
        state["message"] = f"✅ 报表已生成: {state['report_file']}"
    # message 已经在 workflow_decision 设置好了，不用修改
    return state

# ========== 构建 LangGraph ==========
workflow = StateGraph(AgentState)

workflow.add_node("workflow_decision", workflow_decision)
workflow.add_node("parse_request", parse_request)
workflow.add_node("fetch_report", fetch_report)
workflow.add_node("export_report", export_report)
workflow.add_node("finish", finish)

workflow.set_entry_point("parse_request")
workflow.add_edge("parse_request", "workflow_decision")

workflow.add_conditional_edges( "workflow_decision", workflow_decision)


workflow.add_edge("fetch_report", "export_report")
workflow.add_edge("export_report", "finish")
workflow.add_edge("finish", END)

app = workflow.compile()


def show_workflow_pic(workflow):
    from graphviz import Digraph

    dot = Digraph(comment="Report Agent Workflow")

    # 添加节点
    for node in workflow.nodes:
        dot.node(node)

    # 添加边
    for start, end in workflow.edges:
        dot.edge(start, end)

    dot.render("report_agent_workflow", format="png", view=True)

# ========== 示例运行 ==========
if __name__ == "__main__":
    user_input = "帮我导出上个月的销售报表"
    state = AgentState(user_input=user_input)  # 用 AgentState 初始化
    result = app.invoke(state)
    print(result["message"])
    # show_workflow_pic(workflow)