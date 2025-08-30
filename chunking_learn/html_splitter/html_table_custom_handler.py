from langchain_text_splitters import HTMLSemanticPreservingSplitter
from bs4 import Tag
import pandas as pd


def table_handler(element: Tag, max_rows: int = 20) -> str:
    """
    将 <table> 转换为 Markdown 表格。
    - 自动处理 rowspan / colspan
    - 大表格按行切分
    """
    rows = []
    # 遍历 tr
    for tr in element.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            text = cell.get_text(strip=True).replace("\n", " ")
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 填充 colspan
            for _ in range(colspan):
                row.append(text if text else "")
            # 这里简单处理 rowspan: 先重复内容，保证表格形状完整
            if rowspan > 1:
                for _ in range(rowspan - 1):
                    rows.append(["" for _ in row])  # 占位符行
        rows.append(row)

    # 转为 DataFrame 方便处理
    df = pd.DataFrame(rows).fillna("")

    # 大表格拆分
    chunks = []
    for start in range(0, len(df), max_rows):
        sub_df = df.iloc[start:start + max_rows]
        chunks.append(sub_df.to_markdown(index=False))

    return "\n\n".join(chunks)


# 示例 HTML
html_string = """
<html>
  <body>
    <h1>表格章节</h1>
    <p>这是一个有合并单元格的大表格：</p>
    <table>
      <tr><th>姓名</th><th>年龄</th><th colspan="2">联系方式</th></tr>
      <tr><td rowspan="2">张三</td><td>25</td><td>电话</td><td>123456</td></tr>
      <tr><td>25</td><td>邮箱</td><td>zhangsan@example.com</td></tr>
      <tr><td>李四</td><td>30</td><td>电话</td><td>987654</td></tr>
    </table>
  </body>
</html>
"""

# 使用 HTMLSemanticPreservingSplitter
splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=[("h1", "Header 1")],
    max_chunk_size=500,
    elements_to_preserve=["table"],
    custom_handlers={"table": table_handler}
)

documents = splitter.split_text(html_string)

for i, doc in enumerate(documents):
    print(f"\n---- 块 {i + 1} ----")
    print("metadata:", doc.metadata)
    print("content:\n", doc.page_content)
