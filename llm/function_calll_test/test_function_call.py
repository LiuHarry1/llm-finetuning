import os

from dotenv import load_dotenv
from openai import OpenAI

"""
OPENAI_LOG=debug
[2025-09-14 14:20:23 - openai._base_client:482 - DEBUG] Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-1a634a1a-9dc5-4fe8-9ec2-4da8c434236a', 'json_data': {'messages': [{'role': 'user', 'content': "How's the weather in Hangzhou?"}], 'model': 'qwen-plus', 'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather of a location, the user should supply a location first.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}}, 'required': ['location']}}}]}}
[2025-09-14 14:20:23 - openai._base_client:978 - DEBUG] Sending HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
[2025-09-14 14:20:23 - httpx:1025 - INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
[2025-09-14 14:20:23 - openai._base_client:1016 - DEBUG] HTTP Response: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "200 OK" Headers({'vary': 'Origin,Access-Control-Request-Method,Access-Control-Request-Headers, Accept-Encoding', 'x-request-id': 'a1774a93-7a4f-4b12-bfc4-bba8a9689827', 'x-dashscope-call-gateway': 'true', 'content-type': 'application/json', 'req-cost-time': '432', 'req-arrive-time': '1757830823340', 'resp-start-time': '1757830823772', 'x-envoy-upstream-service-time': '431', 'set-cookie': 'acw_tc=a1774a93-7a4f-4b12-bfc4-bba8a96898276b26e38ba1562d39ba500f95eef797b7;path=/;HttpOnly;Max-Age=1800', 'content-encoding': 'gzip', 'date': 'Sun, 14 Sep 2025 06:20:23 GMT', 'server': 'istio-envoy', 'transfer-encoding': 'chunked'})
[2025-09-14 14:20:23 - openai._base_client:1024 - DEBUG] request_id: a1774a93-7a4f-4b12-bfc4-bba8a9689827
User>	 How's the weather in Hangzhou?
[2025-09-14 14:20:23 - openai._base_client:482 - DEBUG] Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-a14adb4a-82e1-4e64-a541-cf34da5b53b9', 'json_data': {'messages': [{'role': 'user', 'content': "How's the weather in Hangzhou?"}, {'content': '', 'role': 'assistant', 'tool_calls': [{'id': 'call_5c06437c803a4fc99c0544', 'function': {'arguments': '{"location": "Hangzhou"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0}]}, {'role': 'tool', 'tool_call_id': 'call_5c06437c803a4fc99c0544', 'content': '24℃'}], 'model': 'qwen-plus', 'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather of a location, the user should supply a location first.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}}, 'required': ['location']}}}]}}
[2025-09-14 14:20:23 - openai._base_client:978 - DEBUG] Sending HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions   
[2025-09-14 14:20:24 - httpx:1025 - INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
[2025-09-14 14:20:24 - openai._base_client:1016 - DEBUG] HTTP Response: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "200 OK" Headers({'vary': 'Origin,Access-Control-Request-Method,Access-Control-Request-Headers, Accept-Encoding', 'x-request-id': '70833da4-f6cc-447c-8fca-4db623579058', 'x-dashscope-call-gateway': 'true', 'content-type': 'application/json', 'req-cost-time': '401', 'req-arrive-time': '1757830823939', 'resp-start-time': '1757830824341', 'x-envoy-upstream-service-time': '400', 'content-encoding': 'gzip', 'date': 'Sun, 14 Sep 2025 06:20:24 GMT', 'server': 'istio-envoy', 'transfer-encoding': 'chunked'})
[2025-09-14 14:20:24 - openai._base_client:1024 - DEBUG] request_id: 70833da4-f6cc-447c-8fca-4db623579058
Model>	 The current temperature in Hangzhou is 24℃.

"""


load_dotenv()

def send_messages(messages):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message


client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")