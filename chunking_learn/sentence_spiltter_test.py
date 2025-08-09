from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
import nltk

# 下载 punkt 分词模型（第一次运行需要）
nltk.download('punkt')
nltk.download('punkt_tab')

# 示例文本（包含缩写、时间等复杂语句）
text = """
Dr. Smith went to Washington. He arrived at 10 a.m. It was raining. He had a meeting with Mr. Johnson, who said: "Let's delay the launch." The decision was unexpected. However, the team quickly adapted.
"""

# 1️⃣ 使用 nltk 做精准句子分割
sentences = sent_tokenize(text)
print("🔹 NLTK 分句结果：")
for s in sentences:
    print("-", s)

# 2️⃣ 拼接句子，交给 LangChain 的 RecursiveCharacterTextSplitter 切分
# 注意：LangChain 切分的是字符串，因此我们先把句子合并
joined_text = "\n".join(sentences)

# 初始化 Recursive splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,        # 每个chunk最大长度
    chunk_overlap=30,      # 重叠部分
    separators=["\n"]      # 使用我们分句后的换行作为语义分隔
    # separators= ["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(joined_text)

print("\n✅ 最终切分后的 Chunk：")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}", len(chunk))
