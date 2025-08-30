import requests

import requests
import json

def run_stream():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [
            {"role": "user", "content": "请写一段含中性逻辑推理的中文段落"}
        ]

    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)

        if response.status_code == 200:
            # 按行解析流式 JSON
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        content = data.get("message", {}).get("content")
                        if content:
                            # print("模型输出:\n", content)
                            print(content)
                    except json.JSONDecodeError:
                        # 非 JSON 行直接忽略
                        pass
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("返回内容:", response.text)

    except requests.exceptions.RequestException as e:
        print("请求异常:", e)

def run_test():
    import requests
    import json

    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [
            {"role": "user", "content": "<think></think> 讲个冷笑话"}
        ],
        "stream": True  # 依然使用流式输出
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)

        final_content = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        content = data.get("message", {}).get("content")
                        if content:
                            final_content += content  # 更新最终输出
                    except json.JSONDecodeError:
                        pass

            print("模型最终输出:\n", final_content)
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("返回内容:", response.text)

    except requests.exceptions.RequestException as e:
        print("请求异常:", e)


def ollama_run():
    import requests, json

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "请写一段含中性逻辑推理的中文段落"}],
        "stream": True
    }

    response = requests.post(url, json=payload, stream=True)
    final_text = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode())
                text = data.get("message", {}).get("content")
                if text:
                    print(text, end="", flush=True)
                    final_text += text
            except json.JSONDecodeError:
                continue
    print("\n\n最终输出：", final_text)


if __name__ == '__main__':
    ollama_run()
