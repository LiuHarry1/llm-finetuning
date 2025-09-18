import fitz
import re
import os
import json
from pathlib import Path

# ========= 基础清洗函数 =========
def clean_text(text):
    text = text.replace("\xa0", " ")        # 去掉不间断空格
    text = re.sub(r"-\n", "", text)         # 处理换行断词
    text = re.sub(r"\n+", "\n", text)       # 合并多余的换行
    text = re.sub(r"\s+", " ", text)        # 多空格合一
    return text.strip()

# ========= 分片函数 =========
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ========= 表格转 Markdown =========
def table_to_markdown(table_text):
    rows = table_text.strip().split("\n")
    rows = [r.split("\t") for r in rows]
    md = "\n".join([" | ".join(r) for r in rows])
    return md

def is_table_like(text):
    lines = text.split("\n")
    count_table_like = 0
    for line in lines:
        # 用连续两个以上空格或制表符作为分隔
        cols = re.split(r"\s{2,}|\t", line.strip())
        if len([c for c in cols if c]) >= 2:  # 至少两列
            count_table_like += 1
    return count_table_like >= 2  # 至少两行符合条件，才判定为表格

# ========= 主函数 =========
def parse_pdf(pdf_path, output_dir="output", chunk_size=500):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    image_dir = Path(output_dir) / "images"
    image_dir.mkdir(exist_ok=True)

    rag_data = []

    for page_num, page in enumerate(doc, start=1):
        # --- 1. 提取文本块 ---
        blocks = page.get_text("blocks")
        page_text = []
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b

            # 文本块
            if block_type == 0 and text.strip():
                cleaned = clean_text(text)
                # 判断是否像表格
                if is_table_like(text):
                    md_table = table_to_markdown(cleaned)
                    rag_data.append({
                        "id": f"table_{page_num}_{block_no}",
                        "type": "table",
                        "page": page_num,
                        "content": md_table
                    })
                else:
                    page_text.append(cleaned)

            # 图片块
            elif block_type == 1:
                # 提取图片
                images = page.get_images(full=True)
                for img_index, img in enumerate(images, start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_filename = image_dir / f"page{page_num}_img{img_index}.{image_ext}"
                    with open(image_filename, "wb") as f:
                        f.write(image_bytes)

                    rag_data.append({
                        "id": f"image_{page_num}_{img_index}",
                        "type": "image",
                        "page": page_num,
                        "path": str(image_filename)
                    })

        # --- 2. 文本分片 ---
        if page_text:
            full_page_text = " ".join(page_text)
            chunks = chunk_text(full_page_text, chunk_size=chunk_size, overlap=50)
            for i, ch in enumerate(chunks):
                rag_data.append({
                    "id": f"text_{page_num}_{i}",
                    "type": "text",
                    "page": page_num,
                    "content": ch
                })

    # --- 3. 保存 JSON ---
    output_json = Path(output_dir) / "rag_data.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=2)

    print(f"解析完成 ✅, 数据保存在: {output_json}")




# ========= 调用示例 =========
if __name__ == "__main__":
    parse_pdf("../data/test.pdf", output_dir="parsed_output", chunk_size=500)
