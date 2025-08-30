# 1. 安装依赖
# pip install markitdown langchain faiss-cpu

from markitdown import convert_file
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import os

# ---------------------------
# Step 1: 将上传文件转换为 Markdown
# ---------------------------

input_file = "example.docx"  # 你要上传的文件路径
output_md_file = "example.md"

# 使用 MarkItDown 转换文件
md_content = convert_file(input_file)
with open(output_md_file, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[INFO] 文件已转换为 Markdown：{output_md_file}")

# ---------------------------
# Step 2: 分块（Splitter）
# ---------------------------

# 使用 LangChain 的 MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=["#","##","###"],  # 按标题拆分
    chunk_size=500,                        # 每块最大 500 token
    chunk_overlap=100                      # 100 token 重叠
)

chunks = splitter.split_text(md_content)
print(f"[INFO] 总共拆分为 {len(chunks)} 个块")

# ---------------------------
# Step 3: 构建向量数据库（FAISS 示例）
# ---------------------------

# 这里使用 OpenAI Embeddings 作为示例，你也可以用其他 embedding 模型
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

vector_db = FAISS.from_texts(chunks, embeddings)

print("[INFO] 向量数据库已创建，可以进行检索和 RAG")

# ---------------------------
# Step 4: 简单检索示例
# ---------------------------

query = "文档中关于合同条款的内容有哪些？"
docs = vector_db.similarity_search(query, k=3)

print("[INFO] 检索结果：")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:500])  # 显示前 500 个字符
    print("---------------------\n")
