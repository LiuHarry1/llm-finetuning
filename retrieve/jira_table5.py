from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def parse_jira_description(html: str, rows_per_table_chunk=5):
    soup = BeautifulSoup(html, 'html.parser')
    chunks = []
    current_text = ""

    for tag in soup.children:
        if tag.name in ["p", "ul", "ol", "h1", "h2", "div"]:
            current_text += tag.get_text(separator="\n") + "\n\n"
        elif tag.name == "table":
            # 如果前面有文本，先存为一个 chunk
            if current_text.strip():
                chunks.append(current_text.strip())
                current_text = ""

            # 表格转成 markdown
            table = pd.read_html(StringIO(str(tag)))[0]
            markdown_table = table.to_markdown(index=False)

            # 表格行数较多时，进行拆分
            if len(table) > rows_per_table_chunk:
                n_chunks = (len(table) + rows_per_table_chunk - 1) // rows_per_table_chunk
                for i in range(n_chunks):
                    sub = table.iloc[i*rows_per_table_chunk:(i+1)*rows_per_table_chunk]
                    sub_md = sub.to_markdown(index=False)
                    chunks.append(f"参数变更表（第{i+1}段）\n\n{sub_md}")
            else:
                chunks.append("参数变更表：\n\n" + markdown_table)

    # 最后剩下的文本
    if current_text.strip():
        chunks.append(current_text.strip())

    return chunks

description_html = """<p>该功能旨在提升系统稳定性，主要更新内容如下：</p>
<ul>
  <li>增加了连接池大小</li>
  <li>优化了错误重试机制</li>
</ul>
<p>以下是参数变更表：</p>
<table>
  <tr><th>参数</th><th>旧值</th><th>新值</th></tr>
  <tr><td>timeout</td><td>300</td><td>500</td></tr>
  <tr><td>retries</td><td>3</td><td>5</td></tr>
</table>
<p>此外，还修复了若干历史兼容性问题。</p>
"""

chunks = parse_jira_description(description_html)

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---\n{chunk}")

