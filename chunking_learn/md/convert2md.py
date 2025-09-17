from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
# result = md.convert("../data/example.docx")
# result = md.convert("../data/test3.docx")
result = md.convert("../data/test.pdf")
print(result.text_content)





