
import pypandoc
# pypandoc.download_pandoc()

def convert_to_markdown(word_file):
    # output = pypandoc.convert_file("../data/test3.docx", "md", outputfile="../data/test3.md")



    output = pypandoc.convert_file(
        "../data/test3.docx",
        "md",
        outputfile="../data/test3.md",
        extra_args=["--extract-media=../data/media"]
    )

    print("✅ 转换完成：output.md")

def convert_from_pdf_to_markdown(pdf_file, markdown_file):
    from pdf2docx import Converter

    cv = Converter(pdf_file)
    cv.convert("../data/temp.docx")
    cv.close()

    # DOCX → Markdown
    # import pypandoc
    # pypandoc.download_pandoc()
    # pypandoc.convert_file("temp.docx", "md", outputfile="output.md")


def get_plain_text_from_pdf(pdf_file):
    from pypdf import PdfReader

    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    page = reader.pages[0]

    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        print(text)
        # print("page: ", number_of_pages)

        # print("first page", page)
        # print("---------------------")


def pdf2markdown(pdf_file):
    import fitz

    doc = fitz.open(pdf_file)
    for page in doc:

        text = page.get_text("text")  # 可选 "blocks", "words" 精细控制
        print(text)


if __name__ == '__main__':
    # convert_to_markdown()
    # convert_from_pdf_to_markdown("../data/test.pdf", "../data/test.md")
    get_plain_text_from_pdf("../data/test.pdf")
    # pdf2markdown("../data/test.pdf")




