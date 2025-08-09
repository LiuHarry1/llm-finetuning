
from langchain.text_splitter import RecursiveCharacterTextSplitter



text = """Title: Meeting Summary

- Discussed Q3 Goals
- Reviewed Budget
- Next steps planned

Thank you."""

text = """text = "Dr. Smith went to Washington. He arrived at 10 a.m. It was raining heavily that morning. He arrived at 10 a.m. It was raining heavily that morning. He arrived at 10 a.m. It was raining heavily that morning."
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=10,
    # separators=["."]
)

chunks = splitter.split_text(text)


print("\n✅ 最终切分后的 Chunk：")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}", len(chunk))
