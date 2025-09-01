import os
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI
import uvicorn

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 简单对话历史
chat_history = []

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        msg = await websocket.receive_text()
        data = json.loads(msg)
        user_message = data.get("message", "")

        # 将用户消息加入历史
        chat_history.append({"role": "user", "content": user_message})

        # Human-in-the-Loop Node: 等待人工确认
        await websocket.send_text(json.dumps({"type": "human_review", "content": user_message}))

        # 这里前端可以选择修改或确认，然后发送 "confirm" 消息
        confirm_msg = await websocket.receive_text()
        confirm_data = json.loads(confirm_msg)
        final_user_message = confirm_data.get("final_message", user_message)

        # 调用大模型流式生成
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "system", "content": "You are a helpful assistant."}] + chat_history[:-1] + [{"role":"user","content":final_user_message}],
            stream=True,
            stream_options={"include_usage": True}
        )

        bot_reply = ""
        for chunk in completion:
            if chunk.choices:
                delta_obj = chunk.choices[0].delta
                delta = delta_obj.content if hasattr(delta_obj, "content") else ""
                if delta:
                    bot_reply += delta
                    await websocket.send_text(json.dumps({"type": "bot_stream", "content": delta}))

        # 完成后把机器人回复加入历史
        chat_history.append({"role": "assistant", "content": bot_reply})
        await websocket.send_text(json.dumps({"type": "done", "full_content": bot_reply}))


def main():
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
