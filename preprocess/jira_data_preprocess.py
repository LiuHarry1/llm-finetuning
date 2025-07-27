import re
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pprint import pprint

# 示例Jira数据（包含表格、列表和无格式文本）
sample_jira_issue = {
    "key": "PROJ-123",
    "fields": {
        "summary": "产品发布计划",
        "description": """
            h1. 发布概述

            本次发布包含以下主要功能:

            * 用户管理模块升级
            * 支付系统集成
            * 性能优化

            h2. 时间安排

            ||开始日期||结束日期||负责人||
            |2023-11-01|2023-11-10|张三|
            |2023-11-11|2023-11-20|李四|

            h2. 资源需求

            # 开发人员: 3名
            # 测试人员: 2名
            # 服务器: AWS c5.xlarge
        """,
        "comment": {
            "comments": [
                {
                    "body": "注意: 测试环境已经准备好\n* 测试服务器IP: 192.168.1.100\n* 数据库版本: MySQL 8.0"
                }
            ]
        },
        "created": "2023-10-15T09:00:00.000+0800",
        "status": {"name": "In Progress"},
        "priority": {"name": "High"}
    }
}


def jira_to_html(text):
    """将Jira格式转换为HTML"""
    # 标题转换
    text = re.sub(r'^h([1-6])\.(.*)$', r'<h\1>\2</h\1>', text, flags=re.MULTILINE)

    # 无序列表
    text = re.sub(r'^\*\s(.*)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>', text)

    # 有序列表
    text = re.sub(r'^#\s(.*)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*</li>\n)+', r'<ol>\g<0></ol>', text)

    # 表格转换
    text = re.sub(r'^\|\|(.+?)\|\|$', r'<table>\n<tr><th>\1</th></tr>', text, flags=re.MULTILINE)
    text = re.sub(r'^\|(.+?)\|$', r'<tr><td>\1</td></tr>', text, flags=re.MULTILINE)
    text = text.replace('</tr>', '</tr>\n</table>', 1)
    text = text.replace('<td>', '<td>').replace('|', '</td><td>')

    return f"<div>{text}</div>"


def process_jira_content(content):
    """处理Jira内容中的各种格式"""
    if not content:
        return ""

    # 转换为HTML
    html_content = jira_to_html(content)

    soup = BeautifulSoup(html_content, 'html.parser')

    # 处理表格
    for table in soup.find_all('table'):
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = []

        for tr in table.find_all('tr')[1:]:  # 跳过表头
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cells:
                row_desc = "; ".join([f"{headers[i]}: {cells[i]}" for i in range(min(len(headers), len(cells)))])
                rows.append(row_desc)

        if rows:
            table.replace_with(f"[TABLE] {', '.join(rows)}")

    # 处理列表
    for list_type in ['ul', 'ol']:
        for lst in soup.find_all(list_type):
            items = [li.get_text(strip=True) for li in lst.find_all('li')]
            prefix = "* " if list_type == 'ul' else "1. "
            list_text = prefix + ("\n" + prefix).join(items)
            lst.replace_with(f"[LIST] {list_type.upper()}\n{list_text}")

    # 清理多余空白
    text = ' '.join(soup.stripped_strings)

    # 恢复换行符
    text = text.replace('[LIST]', '\n[LIST]').replace('[TABLE]', '\n[TABLE]')

    return text.strip()


def smart_chunking(text, chunk_size=300, chunk_overlap=50):
    """智能分块处理"""
    # 特殊内容保持完整（表格和列表）
    special_blocks = []

    # 提取特殊内容块
    def extract_special(match):
        special_blocks.append(match.group(0))
        return f"||SPECIAL_BLOCK_{len(special_blocks) - 1}||"

    text = re.sub(r'(\[TABLE\].*?)(?=\n\[|$)', extract_special, text, flags=re.DOTALL)
    text = re.sub(r'(\[LIST\].*?)(?=\n\[|$)', extract_special, text, flags=re.DOTALL)

    # 常规文本分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ";", ",", " "]
    )
    chunks = splitter.split_text(text)

    # 将特殊内容块插回
    final_chunks = []
    for chunk in chunks:
        while '||SPECIAL_BLOCK_' in chunk:
            chunk = re.sub(r'\|\|SPECIAL_BLOCK_(\d+)\|\|',
                           lambda m: special_blocks[int(m.group(1))], chunk)
        final_chunks.append(chunk)

    return final_chunks


# 处理示例数据
processed_description = process_jira_content(sample_jira_issue['fields']['description'])
processed_comment = process_jira_content(sample_jira_issue['fields']['comment']['comments'][0]['body'])

print("=== 处理后的描述 ===")
print(processed_description)
print("\n=== 处理后的评论 ===")
print(processed_comment)

# 执行分块
print("\n=== 描述分块结果 ===")
desc_chunks = smart_chunking(processed_description)
for i, chunk in enumerate(desc_chunks, 1):
    print(f"\n--- 块 {i} ---")
    print(chunk)

print("\n=== 评论分块结果 ===")
comment_chunks = smart_chunking(processed_comment)
for i, chunk in enumerate(comment_chunks, 1):
    print(f"\n--- 块 {i} ---")
    print(chunk)

# 构建完整的文档结构
document = {
    "metadata": {
        "issue_key": sample_jira_issue["key"],
        "created": sample_jira_issue["fields"]["created"],
        "status": sample_jira_issue["fields"]["status"]["name"],
        "priority": sample_jira_issue["fields"]["priority"]["name"]
    },
    "content": {
        "summary": sample_jira_issue["fields"]["summary"],
        "description_chunks": desc_chunks,
        "comment_chunks": comment_chunks
    }
}

print("\n=== 最终文档结构 ===")
pprint(document)