from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Token 计数函数（用 cl100k_base 适配 OpenAI embedding/gpt-4）
encoding = tiktoken.get_encoding("cl100k_base")

text = """
Dr. Smith went to Washington. He arrived at 10 a.m. It was raining. He had a meeting with Mr. Johnson, who said: "Let's delay the launch." The decision was unexpected. However, the team quickly adapted.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,              # 每段不超过 400 tokens
    chunk_overlap=5,            # 保持 80 tokens 的上下文重叠
    length_function=lambda x: len(encoding.encode(x)),  # 用 token 数量计算长度
    separators=["\n\n", "\n", ".", " ", ""]             # 递归级切分
)



chunks = splitter.split_text(text)

print("\n✅ 最终切分后的 Chunk：")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}", len(chunk))