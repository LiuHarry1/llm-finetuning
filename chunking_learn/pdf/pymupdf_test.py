import re

import fitz  # PyMuPDF
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


def test1():
    # 打开 PDF
    doc = fitz.open("../data/test.pdf")

    all_text = ""
    for page in doc:
        text = page.get_text("text")  # 提取纯文本
        all_text += text + "\n"

    print(all_text[:500])  # 打印前500字符

def clean_text(text):
    text = text.replace("\xa0", " ")     # 去掉不间断空格
    text = re.sub(r"-\n", "", text)      # 处理换行断词 (like "exam-\nple")
    text = re.sub(r"\n+", "\n", text)    # 合并多余的换行
    text = re.sub(r"\s+", " ", text)     # 多空格合一
    return text.strip()

def test2():

    doc = fitz.open("../data/test.pdf")

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = clean_text(text)
        pages.append({
            "page_number": i + 1,
            "content": text.strip()
        })
        print(f"Page {i + 1}: {text.strip()}")
    # print(pages[0])  # 看第一页内容

def test3():
    doc = fitz.open("../data/test.pdf")
    page = doc[2]

    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        print(f"Block {block_no} ({block_type}): {x0},{y0} -> {x1},{y1}")
        print("内容:", text)
        print("-" * 50)

def test4():
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown("../data/test.pdf", write_images=True, image_path="./parsed_output/images")

    # now work with the markdown text, e.g. store as a UTF8-encoded file
    import pathlib
    pathlib.Path("output.md").write_bytes(md_text.encode())

def test5():
    md_read = pymupdf4llm.LlamaMarkdownReader()
    data = md_read.load_data("../data/test.pdf")

    for page in data:
        print(page.doc_id,page.metadata)
        print(page.text)
        # print(page)


def test6():
    data = pymupdf4llm.to_markdown("../data/test.pdf", page_chunks=True)

    print(data[0].keys())
    # 输出: dict_keys(['metadata', 'text'])
    print(data[0]['metadata'])
    print(data[0]['text'][:200])  # 打印前 200 个字符

def test7():
    import pymupdf4llm
    from langchain.text_splitter import MarkdownTextSplitter

    # Get the MD text
    md_text = pymupdf4llm.to_markdown("../data/test.pdf")  # get markdown for all pages

    splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
    # splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=["#", "##", "###"],  # 按标题拆分
    # )

    documents = splitter.create_documents([md_text])

    for doc in documents:
        print(doc.metadata)
        print(doc.page_content)



if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test6()
    test7()
