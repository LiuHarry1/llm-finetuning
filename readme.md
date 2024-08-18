
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
