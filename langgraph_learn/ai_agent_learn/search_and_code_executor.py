from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.chat_models import ChatTongyi
from langchain_experimental.tools import PythonREPLTool
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, SystemMessage

load_dotenv()

system_prompt = SystemMessage(content="""
你是一个智能助手，擅长搜索信息和执行Python代码。
回答用户问题时：
1. 如果需要搜索或执行代码，请调用对应工具。
2. 执行代码：请调用对应工具， 如果是生成图片，请把图片保存到当前目录，并且用print的方式输出图片地址, 禁止调用plt.show()
3. Matplotlib 可以切换到 非 GUI 后端，如 Agg，只生成图片文件，不弹窗：
""")


# 工具
python_tool = PythonREPLTool()
search_tool = TavilySearch(max_results=2)
tools = [search_tool, python_tool]
tool_node = ToolNode(tools)

# LLM
llm = ChatTongyi(model="qwen-plus").bind_tools(tools)

# agent 节点（只返回“增量”消息）
def agent(state: MessagesState):
    messages = [system_prompt] + state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}

# 路由：如果没有 tool_calls 就结束；有就去 tools
def route_after_agent(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

# 图
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

graph = workflow.compile()  # ✅ 不需要 merge=add_messages

# 测试
if __name__ == "__main__":
    # user_input = "给我总结一下今天的top5国内新闻, 并且生成一个表格图片给我"
    # user_input = "给我总结一下今天的top2国内新闻, 并且用代码把新闻总结生成一个表格图片给我"
    user_input = "给我随机生成一个五行五列的表格图片"
    events = graph.stream({"messages": [("user", user_input)]})

    ai_summaries = []
    tool_results = []

    for ev in events:
        print(ev)
        for node_name, node_data in ev.items():
            msgs = node_data.get("messages", [])
            for msg in msgs:
                # AIMessage 的总结内容
                if hasattr(msg, "content") and msg.content.strip():
                    ai_summaries.append(msg.content.strip())

                # ToolMessage 的搜索结果
                if getattr(msg, "__class__", None).__name__ == "ToolMessage":
                    import json

                    try:
                        data = json.loads(msg.content)
                        results = data.get("results", [])
                        for r in results:
                            tool_results.append({
                                "title": r.get("title"),
                                "url": r.get("url"),
                                "date": r.get("published_date"),
                                "score": r.get("score"),
                                "content": r.get("content") or r.get("raw_content")
                            })
                    except Exception as e:
                        continue

    # 输出友好摘要
    print("===== 今日国内新闻 AI 总结 =====")
    for idx, text in enumerate(ai_summaries, 1):
        print(f"{idx}. {text}\n")

    # 输出每条详细搜索结果
    print("===== 搜索结果详细信息 =====")
    for idx, r in enumerate(tool_results, 1):
        print(f"{idx}. 标题: {r['title']}")
        print(f"   链接: {r['url']}")
        print(f"   日期: {r['date']}")
        print(f"   相关内容: {r['content']}\n")

