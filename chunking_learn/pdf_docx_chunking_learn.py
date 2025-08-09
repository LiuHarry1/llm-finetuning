import os
from pathlib import Path
from typing import List, Dict, Any
from docx import Document as DocxDocument
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import tiktoken
# import pytesseract
import re

def docx_clean_text(text: str) -> str:
    # 你之前那个通用清理函数，保持不变
    import re
    text = ''.join(ch for ch in text if ch.isprintable() or ch in '\n\r\t')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not re.search(r'(第\s*\d+\s*页|Page\s*\d+)', line, re.I)]
    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = text.replace('，', ',').replace('。', '.').replace('：', ':').replace('；', ';')
    return text.strip()


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    # 按单词切分，模拟token数，避免太短
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


def get_paragraph_level(style_name: str) -> int:
    """
    根据样式名返回标题层级，默认正文返回0
    例如: Heading 1 -> 1, Heading 2 -> 2
    """
    match = re.match(r'Heading (\d+)', style_name)
    if match:
        return int(match.group(1))
    return 0


def parse_docx_improved(file_path: str) -> List[Dict[str, Any]]:
    doc = DocxDocument(file_path)
    chunks = []

    # 用来保存当前的标题层级路径，比如 [Chapter 1, Section 1.1]
    current_headings = []

    # 用来累积非标题文本，等待分片
    paragraph_buffer = []

    def flush_buffer():
        if not paragraph_buffer:
            return
        text_block = "\n".join(paragraph_buffer)
        text_block = docx_clean_text(text_block)
        for chunk in chunk_text(text_block):
            chunks.append({
                "text": chunk,
                "tags": [f"Heading Level {i+1}: {h}" for i, h in enumerate(current_headings)],
                "source": Path(file_path).name
            })
        paragraph_buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        level = get_paragraph_level(para.style.name)
        if level > 0:
            # 碰到标题，先flush之前的段落
            flush_buffer()

            # 更新current_headings路径
            # 如果是更高层级标题，覆盖后面所有层级
            if len(current_headings) < level:
                # 层级增加，扩展列表
                current_headings += [""] * (level - len(current_headings))
            current_headings = current_headings[:level]
            current_headings[level - 1] = text

        else:
            # 普通段落，累积文本
            paragraph_buffer.append(text)

    # 文档结尾，flush剩余段落
    flush_buffer()

    return chunks

# ===== Tokenizer for chunk size control =====
def count_tokens(text: str, model_name="gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def clean_text(text: str) -> str:
    """
    通用文本清理方法，适合从pdf、docx等提取后的文本。

    功能：
    - 去除多余空白字符（包括制表符、空格）
    - 合并多余换行（多于两个换行归为两个换行）
    - 去除页眉页脚常见格式（可根据需要自定义）
    - 过滤非打印字符
    - 标准化全角半角符号（这里简单示例）

    Args:
        text (str): 原始文本

    Returns:
        str: 清理后的文本
    """

    # 1. 去除非打印字符
    text = ''.join(ch for ch in text if ch.isprintable() or ch in '\n\r\t')

    # 2. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 3. 去除页眉页脚示例（简单示范，根据你的文档格式自定义）
    # 例：去除含“第xx页” 或 “Page xx” 的行
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if re.search(r'(第\s*\d+\s*页|Page\s*\d+)', line, re.IGNORECASE):
            continue
        filtered_lines.append(line)
    text = '\n'.join(filtered_lines)

    # 4. 合并多余空白行，超过2个换行只保留2个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 去除行首尾多余空白
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 6. 标准化符号（示例：全角逗号、句号转半角）
    text = text.replace('，', ',').replace('。', '.').replace('：', ':').replace('；', ';')

    return text.strip()

# ===== Chunking function =====
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ===== DOCX parser =====
def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    doc = DocxDocument(file_path)
    chunks = []
    current_heading = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            current_heading = text
            continue
        text = clean_text(text)
        for chunk in chunk_text(text):
            chunks.append({
                "text": chunk,
                "tags": [f"Heading: {current_heading}"] if current_heading else [],
                "source": Path(file_path).name
            })
    return chunks

# ===== 智能 OCR 判断 =====
def needs_ocr(page) -> bool:
    """如果页面文字少于20个字符且有图片，则认为需要 OCR"""
    text = page.get_text("text").strip()
    image_list = page.get_images(full=True)
    return len(text) < 20 and len(image_list) > 0

# ===== OCR 提取文字 =====
def ocr_page(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    text = pytesseract.image_to_string(images[0], lang="chi_sim+eng")
    return text

# ===== PDF parser =====
def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(file_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        if needs_ocr(page):
            text = ocr_page(file_path, page_num)
            tag_info = [f"Page: {page_num}", "OCR: True"]
        else:
            text = page.get_text("text")
            tag_info = [f"Page: {page_num}", "OCR: False"]
        cleaned_text = clean_text(text)
        if cleaned_text:
            for chunk in chunk_text(cleaned_text):
                chunks.append({
                    "text": chunk,
                    "tags": tag_info,
                    "source": Path(file_path).name
                })
    return chunks

# ===== General loader =====
def load_and_chunk_documents(folder_path: str) -> List[Dict[str, Any]]:
    all_chunks = []
    for file in Path(folder_path).glob("*"):
        if file.suffix.lower() == ".docx":
            # all_chunks.extend(parse_docx(str(file)))
            all_chunks.extend(parse_docx_improved(str(file)))

            print("size of docs", len(all_chunks))
        # elif file.suffix.lower() == ".pdf":
        #     all_chunks.extend(parse_pdf(str(file)))
        #     print("size of pdf", len(all_chunks))
        elif file.suffix.lower() in [".txt", ".md"]:
            text = file.read_text(encoding="utf-8")
            for chunk in chunk_text(text):
                all_chunks.append({
                    "text": chunk,
                    "tags": [],
                    "source": file.name
                })
            print("size of file", len(all_chunks))
    return all_chunks

# ===== Example usage =====
if __name__ == "__main__":
    folder = "./data"  # 存放用户上传的所有文档
    chunks = load_and_chunk_documents(folder)
    print(f"总分片数: {len(chunks)}")
    for i in range(50):
        print(chunks[i])  # 预览一个chunk
