from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def download():
    model_id = "meta-llama/Meta-Llama-3-8B"
    save_path = "./llama3-8b-fp16"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def quantize_q4():


    model_path = "~/Documents/apps/ml/llama3-8B-HF"
    quant_path = "./llama3-8b-gptq-4bit"

    quant_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )

    # 加载并量化
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quant_config,
        use_safetensors=True,
        trust_remote_code=True,
    )

    # 保存量化后的模型
    model.save_quantized(quant_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(quant_path)

def load_q4_model():

    model_path = "./llama3-8b-gptq-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_path, device="cuda:0", use_safetensors=True)

    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    quantize_q4()
