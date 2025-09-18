import os
import pypandoc
import fitz  # PyMuPDF

def convert_pdf_to_text(pdf_path: str) -> str:
    """提取 PDF 文本"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"[WARN] PDF 提取失败: {pdf_path}, 错误: {e}")
    return text


def convert_with_pypandoc(file_path: str, to_format: str = "plain") -> str:
    """用 pypandoc 转换其他文档"""
    try:
        return pypandoc.convert_file(file_path, to_format)
    except Exception as e:
        print(f"[WARN] pypandoc 转换失败: {file_path}, 错误: {e}")
        return ""


def convert_file(file_path: str, output_dir: str = "converted") -> str:
    """统一接口：根据文件类型转换文本"""
    ext = os.path.splitext(file_path)[-1].lower()
    os.makedirs(output_dir, exist_ok=True)

    if ext == ".pdf":
        text = convert_pdf_to_text(file_path)
    elif ext in [".docx", ".odt", ".md", ".html", ".htm", ".rtf"]:
        text = convert_with_pypandoc(file_path, "plain")
    else:
        print(f"[SKIP] 不支持的文件类型: {file_path}")
        return ""

    if text.strip():
        output_path = os.path.join(output_dir, os.path.basename(file_path) + ".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] 已转换: {file_path} → {output_path}")
        return output_path
    else:
        print(f"[FAIL] 内容为空: {file_path}")
        return ""


def batch_convert(input_dir: str, output_dir: str = "converted"):
    """批量转换文件夹下所有文件"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            convert_file(file_path, output_dir)


if __name__ == "__main__":
    # 输入你存放文档的目录
    batch_convert("documents", "converted_texts")
