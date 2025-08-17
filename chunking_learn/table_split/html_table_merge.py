from bs4 import BeautifulSoup

html = """
<table>
  <tr>
    <td rowspan="2">张三</td>
    <td>数学</td>
    <td>90</td>
  </tr>
  <tr>
    <td>语文</td>
    <td>85</td>
  </tr>
  <tr>
    <td rowspan="2">李四</td>
    <td>数学</td>
    <td>92</td>
  </tr>
  <tr>
    <td>语文</td>
    <td>80</td>
  </tr>
</table>

"""

soup = BeautifulSoup(html, "html.parser")
rows = soup.find_all("tr")

# 构建空表格
table = []
for r, row in enumerate(rows):
    cols = row.find_all("td")
    current_row = []
    for col in cols:
        value = col.get_text()
        rowspan = int(col.get("rowspan", 1))
        colspan = int(col.get("colspan", 1))
        current_row.append({"value": value, "rowspan": rowspan, "colspan": colspan})
    table.append(current_row)

# 计算总列数
max_cols = max(sum(cell["colspan"] for cell in row) for row in table)

# 初始化填补后的表格
filled_table = [[None]*max_cols for _ in range(len(table))]

# 填充逻辑
for r, row in enumerate(table):
    c_idx = 0
    for cell in row:
        # 找到当前行可用位置
        while filled_table[r][c_idx] is not None:
            c_idx += 1
        # 填充 rowspan 和 colspan
        for i in range(cell["rowspan"]):
            for j in range(cell["colspan"]):
                filled_table[r+i][c_idx+j] = cell["value"]
        c_idx += cell["colspan"]

# 打印结果
for r in filled_table:
    print(r)
