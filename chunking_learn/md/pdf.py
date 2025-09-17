import fitz  # PyMuPDF

# 打开 PDF
pdf_path = "example.pdf"
doc = fitz.open(pdf_path)

# 硬编码 S3 链接示例
image_s3_map = {
    0: "https://bucket.s3.amazonaws.com/images/pdf_image1.png",
    1: "https://bucket.s3.amazonaws.com/images/pdf_image2.png"
}

md_lines = []
image_counter = 0

for page_index in range(len(doc)):
    page = doc[page_index]

    # 先提取文本
    text = page.get_text("text").strip()
    if text:
        md_lines.append(text)

    # 提取图片
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        s3_url = image_s3_map.get(image_counter, "")
        if s3_url:
            md_lines.append(f"![image{image_counter + 1}]({s3_url})")
        image_counter += 1

# 输出 Markdown
md_text = "\n\n".join(md_lines)
with open("example.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("Markdown 转换完成，内容如下：")
print(md_text)
