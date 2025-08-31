
# how to convert huggingface model to gguf format

git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert.py -h
python convert.py C:\Users\Harry\PycharmProjects\llm-finetuning\llama2-finetuned-combined --outfile C:\Users\Harry\PycharmProjects\llm-finetuning\llama2-finetuned-combined\llama2-7b-chat_f16.gguf --outtype f16

# quanization

mkdir build
cd build
cmake ..
cmake --build . --config Release


cd llama.cpp/build/bin && \
   ./quantize /Users/harry/Documents/apps/ml/llama-2-7b-chat/llama2-7b.gguf /Users/harry/Documents/apps/ml/llama-2-7b-chat/llama2-7b-q4_0.gguf q4_0

quantize.exe C:\Users\Harry\PycharmProjects\llm-finetuning\llama2-finetuned-combined\llama2-7b-chat_f16.gguf C:\Users\Harry\PycharmProjects\llm-finetuning\llama2-finetuned-combined\llama2-7b-chat_f16-q4_0.gguf q4_0

main -m C:\Users\Harry\PycharmProjects\llm-finetuning\llama2-finetuned-combined\llama2-7b-chat_f16-q4_0.gguf --color --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 8


main -m /Users/harry/PycharmProjects/llama.cpp/models3/Meta-Llama-3-8B-Instruct.Q4_0.gguf --color --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 8



python convert.py C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined --outfile C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_f16.gguf --outtype f16

python convert_hf_to_gguf.py C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined --outfile C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_f16.gguf --outtype f16

quantize.exe C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_f16.gguf C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_q4.gguf q4_0

main -m C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_q4.gguf --color --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 8


llama-cli -m C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_q4.gguf

llama-cli -m C:\Users\Harry\PycharmProjects\llm-finetuning\lora_learn\health_ft\llama3.2-3b-finetuned-combined\llama3-3b-chat_q4.gguf  -i -n 512 -p "你好，请用中文介绍你自己。"
./llama-cli \
  -m models/llama-7b.Q4_K_M.gguf \
  -i \
  -n 512


https://bailian.console.aliyun.com/?tab=model#/model-market



启动 deekseek r1:1.5b . 
ollama run deepseek-r1:1.5b

https://github.com/open-webui/open-webui
pip install open-webui
open-webui serve

启动 autogenstadio
pip install -U autogenstudio
autogenstudio ui --port 8081

https://langchain-ai.github.io/langgraph/concepts/memory/#episodic-memory
https://langchain-ai.github.io/langgraph/concepts/memory/#procedural-memory
https://arxiv.org/abs/2303.11366?utm_source=chatgpt.com
https://huggingface.co/blog/Kseniase/reflection?utm_source=chatgpt.com
https://arxiv.org/abs/2310.11511?ref=blog.langchain.com
https://blog.langchain.com/agentic-rag-with-langgraph/?utm_source=chatgpt.com