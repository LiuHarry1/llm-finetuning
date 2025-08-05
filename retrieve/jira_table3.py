from bs4 import BeautifulSoup

def parse_description(html_desc):
    soup = BeautifulSoup(html_desc, "html.parser")
    content = []

    for elem in soup.body or soup.contents:
        # 如果是文本段落
        if elem.name in ["p", "div"]:
            text = elem.get_text(strip=True)
            if text:
                content.append({"type": "text", "content": text})

        # 如果是表格
        elif elem.name == "table":
            table_data = parse_html_table(str(elem))
            content.append({"type": "table", "content": table_data})

        # 其他标签也可能是文本或表格，可以根据需要扩展

    return content

def parse_html_table(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = table.find_all("tr")
    result = []
    occupied = {}

    for r_idx, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        result.append([])
        c_idx = 0
        for col in cols:
            while occupied.get((r_idx, c_idx)):
                c_idx += 1

            rowspan = int(col.get("rowspan", 1))
            colspan = int(col.get("colspan", 1))
            text = col.get_text(strip=True)

            result[r_idx].append(text)

            for i in range(rowspan):
                for j in range(colspan):
                    if i == 0 and j == 0:
                        continue
                    occupied[(r_idx + i, c_idx + j)] = True

            for _ in range(colspan - 1):
                result[r_idx].append(None)

            c_idx += colspan

    return result

# 假设从Jira API获得的html_description如下：
html_description = """
<body>
<p>这是描述开头的文本。</p>
<table border="1">
<tr><th rowspan="2">姓名</th><th colspan="2">成绩</th></tr>
<tr><th>数学</th><th>英语</th></tr>
<tr><td>小明</td><td>90</td><td>85</td></tr>
<tr><td>小红</td><td>95</td><td>88</td></tr>
</table>
<p>这是表格后的文本。</p>
</body>
"""

content = parse_description(html_description)
for block in content:
    print(block["type"])
    print(block["content"])
    print("----")


import pandas as pd
from io import StringIO

html = html_description
soup = BeautifulSoup(html, 'html.parser')
table = pd.read_html(StringIO(str(soup)))[0]
markdown = table.to_markdown(index=False)
print(table.to_json())
print(markdown)
