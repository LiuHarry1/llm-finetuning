import pdfplumber

with pdfplumber.open("./data/test.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print("文本:", text)

        tables = page.extract_tables()
        print("table.....")
        for table in tables:
            for row in table:
                print("row:..", row)

