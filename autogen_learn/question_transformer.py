import asyncio
import os
import json
import re
from typing import Tuple

from autogen_agentchat.agents import AssistantAgent

import autogen_learn.llm_client as llm_client

SYSTEM_PROMPT = """你是一个问题澄清与改写助手，负责把用户的问题转换成适合知识库检索的**独立问题**。
不要回答问题，只做澄清和改写。

工作流程：
1) 先判断用户的问题是否清晰完整：
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

示例 1（多轮消解指代）：
输入：
原始问题：乔布斯什么时候去世的？
当前问题：他去世后谁接任的？
输出：
{"status":"rewrite","message":"乔布斯去世后，谁接任了苹果公司的 CEO？"}

示例 2（需要补充约束）：
输入：
原始问题：苹果笔记本能换电池吗？
当前问题：苹果笔记本能换电池吗？
输出：
{"status":"clarify","message":"您是指所有苹果笔记本，还是特定型号或年份（例如 2020 款 MacBook Pro）？"}
"""

def make_agent() -> AssistantAgent:

    agent = AssistantAgent(
        name="clarify_rewriter",
        system_message=SYSTEM_PROMPT,
        model_client=llm_client.model_client,
    )
    return agent

def call_agent(agent: AssistantAgent, original_question: str, current_question: str) -> str:
    """
    调用 LLM，返回原始文本回复（可能是 JSON 或带代码块的 JSON）。
    """
    user_payload = (
        f"输入：\n原始问题：{original_question}\n当前问题：{current_question}\n"
        "请按上述规范输出严格的 JSON。"
    )
    reply = asyncio.run(agent.run(task= user_payload))

    # autogen 版本差异兼容：可能返回 str，也可能返回 dict
    return reply.messages[-1].content

def extract_json(text: str) -> dict:
    """
    从模型输出中鲁棒地提取 JSON（去掉 ```json ... ``` 代码块、杂质）。
    """
    # 去掉代码块围栏
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：提取第一个 {...} 片段
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法解析为 JSON：\n{text}")

def interactive_loop():
    print("=" * 70)
    print("Clarify & Rewrite Agent (AutoGen)  - 控制台实验版")
    print("说明：输入你的问题，Agent 会判断是否需要澄清；如需澄清，会向你追问。\n"
          "当问题足够清晰时，会输出独立改写问题（standalone question）。")
    print("命令：/exit 退出\n")
    print("=" * 70)

    agent = make_agent()

    while True:
        user_q = input("\n你的问题： ").strip()
        if not user_q:
            continue
        if user_q.lower() in ("/exit", "exit", "quit", "/q"):
            print("已退出。")
            break

        original = user_q
        current = user_q

        try:
            raw = call_agent(agent, original, current)
            print(raw)
            data = extract_json(raw)
        except Exception as e:
            print(f"[错误] 首次判定失败：{e}")
            continue

        if data.get("status") == "clarify":
            print(f"\n[需要澄清] {data.get('message')}")
            add = input("你的补充： ").strip()
            if not add:
                print("[提示] 未提供补充信息，将使用原问题继续（可能不够精确）。")

            # 将补充纳入“当前问题”再请求一次
            current = f"{current}（补充：{add}）" if add else current

            try:
                raw2 = call_agent(agent, original, current)
                data2 = extract_json(raw2)
            except Exception as e:
                print(f"[错误] 二次改写失败：{e}")
                continue

            if data2.get("status") == "rewrite":
                print("\n✅ 最终独立问题（standalone）：")
                print(data2.get("message"))
                print("\nJSON 输出：")
                print(json.dumps(data2, ensure_ascii=False, indent=2))
            else:
                # 如模型仍返回 clarify，则直接给出给用户（避免死循环）
                print("\n[再次澄清] 模型认为仍需补充信息：")
                print(data2.get("message"))
                print("（请重新发起一次对话以继续。）")
        elif data.get("status") == "rewrite":
            print("\n✅ 独立问题（standalone）：")
            print(data.get("message"))
            print("\nJSON 输出：")
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print("\n[提示] 未识别的状态：", data)

if __name__ == "__main__":
    interactive_loop()
