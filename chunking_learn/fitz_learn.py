import fitz

def pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    md = []
    for page in doc:
        blocks = page.get_text("blocks")  # [(x0, y0, x1, y1, text, block_no, block_type), ...]
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            # 简单标题判断：字体大于 12pt
            # 这里简化处理，你可以用 page.get_text("dict") 分析字体信息
            if len(text) < 60 and text.isupper():
                md.append(f"# {text}")
            else:
                md.append(text)
        md.append("")  # 空行分段
    return "\n".join(md)

markdown = pdf_to_markdown("your.pdf")
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
