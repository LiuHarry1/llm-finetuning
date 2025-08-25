import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path, lora_model_path, output_path):
    """
    合并 LoRA adapter 权重到 base 模型，并导出为全精度 fp16 模型
    """
    print(f"加载 base 模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 全精度 fp16
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"加载 LoRA 权重: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    print("合并 LoRA 权重到 base 模型...")
    merged_model = model.merge_and_unload()

    print(f"保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("✅ 合并完成，可以直接用 `AutoModelForCausalLM.from_pretrained(output_path)` 推理")

if __name__ == "__main__":
    # 修改成你自己的路径
    base_model_path = r"C:\apps\ml_model\Llama-3.2-3B-Instruct"
    lora_model_path = "lora-llama3-mental-health1"
    output_path = "./llama3-8b-lora-merged"       # 导出目录

    merge_lora(base_model_path, lora_model_path, output_path)
