from langchain_text_splitters import HTMLSemanticPreservingSplitter

html_string = """
<html>
  <body>
    <h1>第一章：介绍</h1>
    <p>这是第一章的内容。</p>
    <ul>
      <li>要点一</li>
      <li>要点二</li>
    </ul>

    <h2>第一章 - 小节 A</h2>
    <p>这是小节 A 的段落，里面有一张图片。</p>
    <img src="image.png" alt="示例图片"/>

    <h2>第一章 - 小节 B</h2>
    <p>这是小节 B 的段落，下面是一个表格：</p>
    <table>
      <tr><th>姓名</th><th>年龄</th></tr>
      <tr><td>张三</td><td>25</td></tr>
      <tr><td>李四</td><td>30</td></tr>
    </table>
  </body>
</html>
"""

# 定义拆分器
splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
    max_chunk_size=300,
    chunk_overlap=20,
    preserve_images=True,   # 转换 <img> 为 markdown
    elements_to_preserve=["table", "ul", "ol"]  # 表格和列表整体保留
)

# 拆分
documents = splitter.split_text(html_string)

# 打印结果
for i, doc in enumerate(documents):
    print(f"\n---- 块 {i+1} ----")
    print("metadata:", doc.metadata)
    print("content:\n", doc.page_content)
