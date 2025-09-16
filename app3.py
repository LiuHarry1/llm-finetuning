# main.py
import os
import threading
import time

import uvicorn
import asyncio
from fastapi import FastAPI

# 设置 uvloop

app = FastAPI()

@app.get("/sync")
def sync_endpoint():
    t = threading.current_thread()
    print(f"[sync] thread={t.name}, ident={t.ident}")
    time.sleep(2)  # 阻塞 2 秒
    return {"msg": "sync done"}

@app.get("/async")
async def async_endpoint():
    t = threading.current_thread()
    print(f"[async] thread={t.name}, ident={t.ident}")
    await asyncio.sleep(2)  # 异步挂起 2 秒
    return {"msg": "async done"}
# 通常使用 uvicorn 运行，见下文

def main():
    uvicorn.run("app3:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
