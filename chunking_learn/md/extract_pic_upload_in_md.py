from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert("../data/test3.docx")
print(result.text_content)



import re, base64

pattern = re.compile(r"!\[.*?\]\(data:image/(.*?);base64,(.*?)\)")

for match in pattern.finditer(result.text_content):
    ext = match.group(1)       # 图片类型，比如 png, jpeg
    b64_data = match.group(2)  # Base64 内容
    img_bytes = base64.b64decode(b64_data)
    print(img_bytes)
    with open("temp_image.png", "wb") as f:
        f.write(img_bytes)



