
"""
python -m vllm.entrypoints.openai.api_server --model C:/apps/ml_model/Llama-3.2-1B-Instruct --swap-space


curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "llama3-8b",
        "prompt": "用一句话解释 LCEL 是什么",
        "max_tokens": 50
      }'

pip install vllm

"""


from vllm import LLM, SamplingParams

# 加载 LLaMA-3-2-1B-Instruct 模型
llm = LLM(model="C:/apps/ml_model/Llama-3.2-1B-Instruct")

params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

prompt = "用一句话解释什么是 LCEL？"
outputs = llm.generate([prompt], params)

for out in outputs:
    print(out.outputs[0].text)
