from langchain_text_splitters import HTMLSemanticPreservingSplitter
from bs4 import Tag

# 定义 HTML 测试数据
html_string = """
<html>
  <body>
    <h1>代码示例章节</h1>
    <p>下面是一个 Python 代码片段：</p>
    <code data-lang="python">
def add(a, b):
    return a + b
    </code>

    <h2>另一种代码</h2>
    <p>这是 JavaScript 的例子：</p>
    <code data-lang="javascript">
function add(a, b) {
  return a + b;
}
    </code>
  </body>
</html>
"""

# 自定义处理函数
def code_handler(element: Tag) -> str:
    """将 <code> 标签内容转为自定义格式"""
    lang = element.get("data-lang", "text")
    code_text = element.get_text().strip()
    return f"\n<<<{lang}>>>\n{code_text}\n<<<END>>>\n"

# 定义拆分器
splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
    max_chunk_size=300,
    elements_to_preserve=["code"],  # 保留 code 块
    custom_handlers={"code": code_handler}  # 指定自定义处理器
)

# 拆分
documents = splitter.split_text(html_string)

# 打印结果
for i, doc in enumerate(documents):
    print(f"\n---- 块 {i+1} ----")
    print("metadata:", doc.metadata)
    print("content:\n", doc.page_content)
