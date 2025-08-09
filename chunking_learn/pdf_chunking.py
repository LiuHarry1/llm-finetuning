import subprocess
import os
import re
import json

# 1. 用 marker-pdf 转 PDF 为 Markdown（不启用 LLM）
def convert_pdf_with_marker(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "marker_single",
        pdf_path,
        "--output_dir",
        output_dir,
        "--strip_existing_ocr"
    ]
    subprocess.run(cmd, check=True)
    basename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
    return os.path.join(output_dir, basename)


# 2. Markdown 预处理（去空行等）
def preprocess_markdown(md_text):
    lines = md_text.splitlines()
    cleaned = [line for line in lines if line.strip()]  # 去掉空行
    return "\n".join(cleaned)

# 3. 按标题拆分并加 tags
def split_and_tag(md_text):
    chunks = []
    current = {"text": "", "tags": []}
    current_tags = []

    for line in md_text.splitlines():
        m = re.match(r'^(#{1,6})\s*(.+)', line)
        if m:
            # 遇到标题，先保存前一个 chunk
            if current["text"]:
                chunks.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            current_tags = [f"h{level}:{title}"]
            current = {"text": line + "\n", "tags": current_tags}
        else:
            current["text"] += line + "\n"

    if current["text"]:
        chunks.append(current)

    return chunks

# 主流程
def main():
    pdf_path = "./data/test.pdf"       # 输入 PDF
    output_dir = "./marker_output"       # 输出目录

    # Step 1: PDF -> Markdown
    md_path = convert_pdf_with_marker(pdf_path, output_dir)

    # Step 2: 读取并预处理 Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    md_content = preprocess_markdown(md_content)

    # Step 3: 拆分并加 tags
    chunks = split_and_tag(md_content)

    # Step 4: 保存 JSON
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"完成：生成 {len(chunks)} 个 chunk，已保存到 {output_dir}")

if __name__ == "__main__":
    main()
