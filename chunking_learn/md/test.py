import os
import base64
import pytesseract
from pdf2image import convert_from_path
from bs4 import BeautifulSoup

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, fmt='tiff')

# Function to convert image to hOCR using pytesseract
def image_to_hocr(image):
    return pytesseract.image_to_pdf_or_hocr(image, extension='hocr')

# Function to convert hOCR to markdown
def hocr_to_markdown(hocr):
    soup = BeautifulSoup(hocr, 'html.parser')
    markdown_text = ""

    for line in soup.find_all('span', class_='ocr_line'):
        line_text = " ".join([word.get_text() for word in line.find_all('span', class_='ocrx_word')])
        markdown_text += f"{line_text}\n"

    return markdown_text

# Main function to convert PDF to Markdown
def pdf_to_markdown(pdf_path):
    images = pdf_to_images(pdf_path)
    markdown_text = ""

    for image in images:
        hocr = image_to_hocr(image)
        markdown_text += hocr_to_markdown(hocr) + "\n\n"

    return markdown_text

# Example usage
if __name__ == "__main__":
    pdf_path = "../data/test.pdf"  # Path to your PDF file
    markdown_output = pdf_to_markdown(pdf_path)
    with open("../data/test.md", "w") as file:
        file.write(markdown_output)