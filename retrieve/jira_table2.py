description = """
这是普通的描述文本。

| 列1  | 列2  |
| ---- | ---- |
| 数据1 | 数据2 |
| 数据3 | 数据4 |

这里是表格后面的文本内容。
"""

lines = description.strip().splitlines()

text_lines = []
table_lines = []
in_table = False

for line in lines:
    if line.strip().startswith("|") and line.strip().endswith("|"):
        table_lines.append(line)
        in_table = True
    else:
        if in_table and line.strip() == "":
            # 空行，认为表格结束
            in_table = False
        if not in_table:
            text_lines.append(line)

# 打印普通文本
print("文本内容：")
print("\n".join(text_lines))

# 解析表格
print("\n表格内容：")
if table_lines:
    for row in table_lines:
        # 去掉首尾|，按|分割，并去除空白
        cells = [cell.strip() for cell in row.strip().strip('|').split('|')]
        print(cells)
else:
    print("没有检测到表格")
