import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "城市名称"}
                            },
                            "required": ["city"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "计算数学表达式",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expr": {"type": "string", "description": "数学表达式"}
                            },
                            "required": ["expr"]
                        }
                    }
                }
        ]

def tool_call_stream_test(user_query= "今天绥德县的天气怎么样"):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ],
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )

    print("流式输出内容为：")
    full_content = ""
    current_function_call = {"name": None, "arguments": ""}

    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 1️⃣ 普通内容流式输出
        if delta.content:
            print(delta.content, end="", flush=True)
            full_content += delta.content

        # 2️⃣ 工具调用的流式输出
        if delta.tool_calls and len(delta.tool_calls) > 0:
            tool_call = delta.tool_calls[0]  # ⚠️ 这是个 pydantic 对象
            func = tool_call.function        # ✅ 直接用属性访问

            if func.name:
                current_function_call["name"] = func.name

            if func.arguments:
                current_function_call["arguments"] += func.arguments

        # 3️⃣ 工具调用完成时输出完整 JSON
        if chunk.choices[0].finish_reason == "tool_calls":
            print("\n🧩 检测到完整 function call:")
            try:
                args = json.loads(current_function_call["arguments"])
            except json.JSONDecodeError:
                args = current_function_call["arguments"]

            print(json.dumps({
                "name": current_function_call["name"],
                "arguments": args
            }, ensure_ascii=False, indent=2))

            current_function_call = {"name": None, "arguments": ""}

        elif chunk.choices[0].finish_reason == "stop":
            print("\n✅ 输出完成。")

    print(f"\n完整自然语言输出：{full_content}")


def stream_test():

    completion = client.chat.completions.create(
        # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ],
        stream=True,
        stream_options={
            "include_usage": True
        },
        # 使用Qwen3开源版模型时，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )

    print(completion)
    full_content = ""
    print("流式输出内容为：")

    for chunk in completion:

        if chunk.choices:
            full_content += chunk.choices[0].delta.content
            delta = chunk.choices[0].delta
            print(delta)
            print(chunk.choices[0].delta.content)
    print(f"完整内容为：{full_content}")
    print(f"Token 使用量：{chunk.usage.model_dump_json()}")


def stream_with_tool_detection(user_query = "帮我查一下绥德县今天的天气"):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ],
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_content = ""
    current_function_call = {"name": None, "arguments": ""}
    is_tool_calling = False

    print("💬 模型输出：", end="", flush=True)

    for chunk in completion:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # ---- 普通文字内容 ----
        if delta.content:
            # 如果之前是 function_call，就先输出它
            if is_tool_calling:
                # 输出完整 function call
                print("\n🧩 检测到完整 function call:")
                try:
                    args = json.loads(current_function_call["arguments"])
                except json.JSONDecodeError:
                    args = current_function_call["arguments"]
                print(json.dumps({
                    "name": current_function_call["name"],
                    "arguments": args
                }, ensure_ascii=False, indent=2))
                # 重置状态
                current_function_call = {"name": None, "arguments": ""}
                is_tool_calling = False
                print("\n💬 模型继续输出：", end="", flush=True)

            print(delta.content, end="", flush=True)
            full_content += delta.content

        # ---- 工具调用流 ----
        if delta.tool_calls and len(delta.tool_calls) > 0:
            tool_call = delta.tool_calls[0]
            func = tool_call.function

            if func.name:
                current_function_call["name"] = func.name
            if func.arguments:
                current_function_call["arguments"] += func.arguments

            is_tool_calling = True  # 标记进入 function_call 模式

        # ---- 结束标记 ----
        if chunk.choices[0].finish_reason in ["stop", "tool_calls"]:
            # 如果输出以 function_call 结束
            if is_tool_calling:
                print("\n🧩 检测到完整 function call:")
                try:
                    args = json.loads(current_function_call["arguments"])
                except json.JSONDecodeError:
                    args = current_function_call["arguments"]
                print(json.dumps({
                    "name": current_function_call["name"],
                    "arguments": args
                }, ensure_ascii=False, indent=2))
                is_tool_calling = False
            print("\n✅ 输出完成。")
            break

    print(f"\n\n完整自然语言输出：{full_content}")

def run_test():

    completion = client.chat.completions.create(
        # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ],
        stream=False,

        # 使用Qwen3开源版模型时，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )

    print(completion)
    full_content = ""

    print(completion.model_dump_json())

    for  choice in completion.choices:
        print(choice.message.role)
        print(choice.message.content)

    print(f"Token 使用量：{completion.usage.model_dump_json()}")

if __name__ == '__main__':
    stream_with_tool_detection("今天上海的天气怎么样")