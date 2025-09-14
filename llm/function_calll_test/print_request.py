# print_request.py
from mitmproxy import http

host_name = "datascope.aliyuncs.com"

def request(flow: http.HTTPFlow) -> None:
    # 只打印目标主机（按需改成 api.openai.com 或你的服务主机）
    if "openai" in flow.request.host or host_name in flow.request.host:
        print("=== REQUEST ===")
        print(flow.request.method, flow.request.pretty_url)
        # headers（注意：会打印 Authorization，包含 API key）
        print("Headers:")
        for k,v in flow.request.headers.items():
            print(f"  {k}: {v}")
        # 打印 body 文本（如果是 JSON 会是 JSON 字符串）
        try:
            print("Body:")
            print(flow.request.get_text())
        except Exception as e:
            print("Failed to get text:", e)
        print("================\n")
