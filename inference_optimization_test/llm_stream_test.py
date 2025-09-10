import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

def stream_test():

    completion = client.chat.completions.create(
        # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="llama-1b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "我想学习python, 怎么学呢？"}
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
            # full_content += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.content)
            # print(token, end="", flush=True)

            token =  chunk.choices[0].delta.content
            if token:  # 有时候 chunk 里可能为空
                full_content += token
                print(token, end="", flush=True)
    print(f"完整内容为：{full_content}")
    print(f"Token 使用量：{chunk.usage.model_dump_json()}")


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
    # run_test()
    stream_test()