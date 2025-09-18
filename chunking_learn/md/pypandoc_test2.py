import pypandoc

# output = pypandoc.convert_text("Hello **world**!", "plain", format="md")
# print(output)  # 输出: Hello world!


# DOCX → Markdown
# md_text = pypandoc.convert_file("../data/example.docx", "md")
# print(md_text[:500])  # 打印前 500 字符


import pypandoc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Step 1: 转换文档为纯文本
text = pypandoc.convert_file("../data/example.docx", "md")

# Step 2: 分块
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
# docs = splitter.split_text(text)

headers_to_split_on = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on  # 按标题拆分
    # chunk_size=200,                        # 每块最大 500 token
    # chunk_overlap=30                      # 100 token 重叠
)

header_docs = splitter.split_text(text)

# Step 3: 再对每个标题块进行递归字符切分
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,   # 每块最大字数
    chunk_overlap=20  # 重叠字数
)

final_docs = []
for d in header_docs:
    sub_docs = recursive_splitter.split_text(d.page_content)
    for sub in sub_docs:
        # 保留 metadata（标题信息）
        final_docs.append(
            d.copy(update={"page_content": sub})
        )

for index , doc in enumerate(final_docs):
    print(f"--------------{index}---------------")
    print(doc.metadata)  # 会保存标题结构
    print(doc.page_content)

# PDF → TXT
#
# import fitz  # PyMuPDF
#
# doc = fitz.open("../data/test.pdf")
# text = ""
# for page in doc:
#     text += page.get_text()
# print(text[:500])


