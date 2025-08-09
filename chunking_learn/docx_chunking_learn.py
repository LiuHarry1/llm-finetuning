from docx import Document

def split_docx(path, chunk_size=1000):
    doc = Document(path)
    chunks = []
    current_chunk = ""
    current_tags = []

    for para in doc.paragraphs:
        style = para.style.name
        text = para.text.strip()
        if not text:
            continue

        # 记录标题标签
        if style.startswith("Heading"):
            if current_chunk:
                chunks.append({"text": current_chunk.strip(), "tags": current_tags})
                current_chunk, current_tags = "", []
            current_tags.append(f"Heading: {text}")
        else:
            if len(current_chunk) + len(text) > chunk_size:
                chunks.append({"text": current_chunk.strip(), "tags": current_tags})
                current_chunk = text
                current_tags = []
            else:
                current_chunk += " " + text

    if current_chunk:
        chunks.append({"text": current_chunk.strip(), "tags": current_tags})

    return chunks

chunks = split_docx("data/test2.docx")
for c in chunks:
    print(c)
