# backend/main.py
import re
import os
import uuid
import traceback
import matplotlib.pyplot as plt
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatTongyi
from langchain_experimental.tools import PythonREPLTool
from langchain_tavily import TavilySearch
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

# 工具定义
python_repl = PythonREPLTool()
search = TavilySearch()

llm = ChatTongyi(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True
)

def sanitize_input(query: str) -> str:
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"^(\s|`)*(?i:py)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    query = query.replace("plt.show()", "")
    return query

# 改造 Python 工具：自动保存图表
def python_repl_tool(code: str) -> str:
    try:
        img_filename = f"{uuid.uuid4().hex}.png"
        img_path = os.path.join("static", img_filename)
        exec_globals = {"plt": plt, "os": os, "img_path": img_path}
        code = sanitize_input(code)
        print(code)
        exec(code, exec_globals)
        if plt.get_fignums():
            plt.savefig(img_path)
            plt.close()
            print(f"图表已生成: /static/{img_filename}")
            if os.path.exists(img_path):
                # 👇 返回绝对明确的 JSON-like 文本，避免 LLM 自己幻想
                return f"图表已生成，请在此路径访问: /static/{img_filename}"
            return "代码执行完成，无图表生成。"
        return "代码执行完成，无图表生成。"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Execution failed with error: {e!r}\nTraceback:\n{tb}"

tools = [
    Tool(
        name="python_repl",
        func=python_repl_tool,
        description=  "执行Python代码并返回结果。"
        "如果生成了图表，结果会包含 `图表已生成，请在此路径访问: /static/xxx.png`，"
        "请在最终回答中引用这个路径作为图片地址。"
    ),
    Tool(
        name="search",
        func=search.run,
        description="搜索网络信息。"
    ),
]

class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

callback_manager = CallbackManager([StreamHandler()])
# llm.callbacks = callback_manager

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# ========================
# FastAPI 应用
# ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端 HTML 可以直接访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())



@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    response = agent.run(query)
    print(response)
    return JSONResponse({"response": response})



def main():
    uvicorn.run("fastapi_ui:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
