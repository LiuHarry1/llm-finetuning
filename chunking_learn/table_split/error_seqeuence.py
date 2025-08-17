# 原始表格，None 表示空单元格
# 张三 rowspan=2, 科目 "数学+语文" colspan=2
table = [
    ["张三", "数学+语文", None],
    [None, "英语", 85],
    ["李四", "英语", 88]
]

# 错误做法：先水平填充（处理 colspan）
def horizontal_fill(table):
    for row in table:
        for i in range(len(row)-1):
            if row[i] is not None and row[i+1] is None:
                row[i+1] = row[i]

# 再垂直填充（处理 rowspan）
def vertical_fill(table):
    for col in range(len(table[0])):
        for row in range(1, len(table)):
            if table[row][col] is None:
                table[row][col] = table[row-1][col]

# 执行错误顺序
vertical_fill(table)
horizontal_fill(table)


# 输出结果
for r in table:
    print(r)
