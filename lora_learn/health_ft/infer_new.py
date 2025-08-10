from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_path = r"C:\apps\ml_model\Llama-3.2-3B-Instruct"
lora_model_path = "lora-llama3-mental-health1"


tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 用这里传入
    llm_int8_threshold=6.0,
)

# 微调后的模型（基座+LoRA）
ft_model = AutoModelForCausalLM.from_pretrained(
    # r"C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined",
    base_model_path,
    device_map="auto",
    # quantization_config=bnb_config,
    torch_dtype=torch.float16
)


# ft_model = PeftModel.from_pretrained(ft_model, lora_model_path)
# ft_model = ft_model.half()
# ft_model = ft_model.to(torch.bfloat16)  # 或 .half()

def chat_with_model(model, user_input, max_new_tokens=256):
    # prompt = f"User: {user_input}\nAssistant: "
    prompt = f"<|user|> {user_input}\n<|assistant|>"
    # prompt = f"用户: {user_input}\n助手: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # 解码，去掉prompt部分，只返回模型生成的回答
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉prompt，保留Assistant回答部分
    return output_text[len(prompt):].strip()


# test_cases = [
#     "我最近压力特别大，经常失眠，你能帮我分析一下原因吗？",
#     "我总是害怕和别人交流，怎么办？",
#     "我在工作中遇到很大挫折，情绪很低落。"
# ]

test_cases = [
    "I've been under a lot of stress lately and often have trouble sleeping. Can you help me analyze the reasons?",
    "I'm always afraid of communicating with others. What should I do?",
    "I've encountered major setbacks at work and feel very down."
]



for query in test_cases:
    print(f"用户: {query}")
    print("\n--- 基座模型 ---")
    print(chat_with_model(base_model, query))
    print("\n--- 微调模型 ---")
    print(chat_with_model(ft_model, query))
    print("=" * 50)
