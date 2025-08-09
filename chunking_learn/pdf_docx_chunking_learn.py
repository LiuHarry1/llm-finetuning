import os
from pathlib import Path
from typing import List, Dict, Any
from docx import Document as DocxDocument
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import tiktoken
# import pytesseract

# ===== Tokenizer for chunk size control =====
def count_tokens(text: str, model_name="gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def clean_text(text: str) -> str:
    # 去除多余空白、连续换行替换为一个换行
    import re
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 多个换行替换成两个换行
    text = text.strip()
    # 这里可以加更多清洗规则，比如过滤页眉页脚（需要自定义规则）
    return text

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
            all_chunks.extend(parse_docx(str(file)))
            print("size of docs", len(all_chunks))
        elif file.suffix.lower() == ".pdf":
            all_chunks.extend(parse_pdf(str(file)))
            print("size of pdf", len(all_chunks))
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
    print(chunks[0])  # 预览一个chunk
