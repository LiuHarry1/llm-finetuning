import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# 用百炼 API 兼容 OpenAI SDK
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 或直接写成 api_key="sk-xxx"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = "qwen-plus"   # 可以换成 "qwen-max"

SYSTEM_PROMPT = """你是一个问题澄清与改写助手，负责把用户的问题转换成适合知识库检索的**独立问题**。
不要回答问题，只做澄清和改写。

工作流程：
1) 判断用户的问题是否清晰完整：
   - 若清晰：直接改写成完整、独立的问题（rewrite）。
   - 若不清晰：输出一个简洁的澄清问题向用户确认（clarify）。
2) 一旦得到用户补充，再生成最终独立问题（rewrite）。
3) 严格输出 JSON，不要多余文字、不要代码块标记。

约束：
- 澄清问题必须简短具体（不超过 1 句话）。
- 最终问题必须独立，不依赖上下文即可理解。
- 不要回答事实内容，不要引用或生成出处。
- 若你无法判断，请优先澄清。

输出 JSON 模式：
{
  "status": "clarify" 或 "rewrite",
  "message": "若为 clarify：这里是向用户追问的澄清问题；若为 rewrite：这里是改写后的独立问题"
}
"""

def call_llm(messages) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()

    # 清理 ```json ... ``` 包裹
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

def interactive_loop():
    print("=" * 60)
    print("澄清+改写机器人（带对话历史）")
    print("输入你的问题，模型会决定是否澄清。输入 /exit 退出")
    print("=" * 60)

    while True:
        user_q = input("\n你的问题： ").strip()
        if user_q.lower() in ("/exit", "exit", "quit"):
            break

        # 对话历史初始化
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_q},
        ]

        result = call_llm(messages)

        if result["status"] == "clarify":
            print(f"[需要澄清] {result['message']}")
            add = input("你的补充： ").strip()
            if not add:
                print("[提示] 未提供补充，将直接用原问题。")
                continue

            # 把澄清与补充加入历史
            messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
            messages.append({"role": "user", "content": add})

            final = call_llm(messages)
            print("✅ 最终结果：", final)
        else:
            print("✅ 改写结果：", result)

if __name__ == "__main__":
    interactive_loop()
