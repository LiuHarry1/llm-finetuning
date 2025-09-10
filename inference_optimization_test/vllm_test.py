
"""
python -m vllm.entrypoints.openai.api_server --model /mnt/c/apps/ml_model/Llama-3.2-1B-Instruct --served-model-name llama-1b --swap-space 4 --max-model-len 8192 --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8000


curl http://localhost:8000/v1/completions  -H "Content-Type: application/json"
  -d '{
        "model": "llama3-8b",
        "prompt": "用一句话解释 LCEL 是什么",
        "max_tokens": 50
      }'

pip install vllm

(vllm-env) harry@DESKTOP-T4TU7JE:/mnt/c/Users/Harry/PycharmProjects/llm-finetuning/inference_optimization_test$ hostname -I
172.26.227.205

"""


from openai import OpenAI

# 连接到本地 vLLM API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 发送请求
response = client.chat.completions.create(
    model="llama-1b",   # 注意这里要和 --served-model-name 一致
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用中文介绍一下Python的优点。"}
    ],
    max_tokens=200
)

# 输出结果
print(response.choices[0].message.content)

