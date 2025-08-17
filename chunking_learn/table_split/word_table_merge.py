from docx import Document

doc = Document("data/example.docx")
table = doc.tables[0]

# 先读取表格内容
data = []
for row in table.rows:
    data.append([cell.text.strip() if cell.text.strip() else None for cell in row.cells])

# 填补合并单元格（垂直填充）
for col in range(len(data[0])):
    last_val = None
    for row in data:
        if row[col]:
            last_val = row[col]
        else:
            row[col] = last_val

for row in data:
    print(row)
