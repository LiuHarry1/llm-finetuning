from pdf2docx import Converter
import pypandoc
import os
import base64
import re

# -------------------------------
# 1️⃣ PDF → Word (.docx)
# -------------------------------
pdf_path = "../data/test.pdf"
docx_path = "../data/test.docx"

cv = Converter(pdf_path)
cv.convert(docx_path, start=0, end=None)
cv.close()
print(f"✅ PDF 已转换成 Word：{docx_path}")

# -------------------------------
# 2️⃣ Word → Markdown（先导出图片到 media/）
# -------------------------------
md_path = "../data/test.md"
media_dir = "media"

# 保证 media 目录存在
os.makedirs(media_dir, exist_ok=True)

pypandoc.download_pandoc()  # 如果没有 Pandoc 自动下载

# extra_args 指定导出图片目录
pypandoc.convert_file(
    docx_path,
    "md",
    outputfile=md_path,
    extra_args=[f"--extract-media={media_dir}"]
)
print(f"✅ Word 已转换成 Markdown：{md_path}，图片导出到 {media_dir}/")

# -------------------------------
# 3️⃣ Markdown 图片路径 → Base64 内嵌
# -------------------------------
with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# 匹配 Markdown 图片语法 ![alt](media/xxx)
def img_to_base64(match):
    img_path = match.group(1)
    if not os.path.isfile(img_path):
        return match.group(0)
    ext = os.path.splitext(img_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"![image](data:{mime};base64,{encoded})"

md_content = re.sub(r"!\[.*?\]\((media/.*?)\)", img_to_base64, md_content)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ 图片已内嵌 Base64，最终 Markdown 完整生成：{md_path}")
