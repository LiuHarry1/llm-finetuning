# 原始表格数据，None 表示空单元格
# 每个单元格也记录 (rowspan, colspan)
table = [
    [{"value": "张三", "rowspan": 2, "colspan": 1},
     {"value": "数学+语文", "rowspan": 1, "colspan": 2}],
    [{"value": None, "rowspan": 1, "colspan": 1},
     {"value": "英语", "rowspan": 1, "colspan": 1},
     {"value": 85, "rowspan": 1, "colspan": 1}],
    [{"value": "李四", "rowspan": 1, "colspan": 1},
     {"value": "英语", "rowspan": 1, "colspan": 1},
     {"value": 88, "rowspan": 1, "colspan": 1}],
]

rows = len(table)
cols = len(table[0])

# 垂直填充
for r in range(rows):
    for c in range(cols):
        cell = table[r][c]
        if cell["value"] is not None and cell["rowspan"] > 1:
            for i in range(1, cell["rowspan"]):
                if r + i < rows and table[r+i][c]["value"] is None:
                    table[r+i][c]["value"] = cell["value"]

# 水平填充
for r in range(rows):
    for c in range(cols):
        cell = table[r][c]
        if cell["value"] is not None and cell["colspan"] > 1:
            for i in range(1, cell["colspan"]):
                if c + i < cols and table[r][c+i]["value"] is None:
                    table[r][c+i]["value"] = cell["value"]

# 打印填补后的表格
for r in table:
    print([cell["value"] for cell in r])
