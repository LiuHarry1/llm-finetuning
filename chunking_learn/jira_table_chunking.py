import re
import pandas as pd
from io import StringIO
import nltk

nltk.download('punkt')

jira_markdown = """
系统支持用户登录和权限管理。

| 功能模块 | 状态  | 负责人 |
| -------- | ----- | ------ |
| 登录     | 完成  | 张三   |
| 注册     | 进行中 | 李四   |
| 重置密码 | 未开始 | 王五   |
| 用户资料 | 完成  | 赵六   |
| 权限管理 | 进行中 | 孙七   |
| 通知系统 | 未开始 | 周八   |
| 数据备份 | 完成  | 吴九   |
| 报表     | 进行中 | 郑十   |
| 审计日志 | 未开始 | 钱十一 |
| 系统监控 | 完成  | 孙十二 |
| 性能优化 | 进行中 | 李十三 |
| 安全加固 | 完成  | 张十四 |
| 日志分析 | 未开始 | 王十五 |
| 任务调度 | 进行中 | 赵十六 |
| 数据清理 | 完成  | 孙十七 |
| 备份恢复 | 进行中 | 周十八 |
| 用户行为 | 未开始 | 吴十九 |
| 访问控制 | 完成  | 郑二十 |

系统还需提供报表生成和系统监控功能。
"""

def is_table_line(line):
    return bool(re.match(r'^\s*\|.*\|\s*$', line))

# 1. 按行拆分并区分文本块和表格块
lines = jira_markdown.strip().split('\n')
blocks = []
current_block = []
in_table = False

for line in lines:
    if is_table_line(line):
        if not in_table:
            # 文本块结束，保存
            if current_block:
                blocks.append({'type': 'text', 'content': '\n'.join(current_block)})
                current_block = []
            in_table = True
        current_block.append(line)
    else:
        if in_table:
            # 表格块结束，保存
            blocks.append({'type': 'table', 'content': '\n'.join(current_block)})
            current_block = []
            in_table = False
        current_block.append(line)
if current_block:
    blocks.append({'type': 'table' if in_table else 'text', 'content': '\n'.join(current_block)})

# 2. 文本拆句，表格按固定窗口切分
window_size = 5   # 表格切片行数
overlap = 2       # 重叠行数

text_slices = []
table_slices = []

def split_table(df, window_size, overlap):
    slices = []
    step = window_size - overlap
    for start in range(0, len(df), step):
        end = start + window_size
        slice_df = df.iloc[start:end]
        slices.append(slice_df)
        if end >= len(df):
            break
    return slices

import nltk

for block in blocks:
    if block['type'] == 'text':
        sentences = nltk.sent_tokenize(block['content'])
        text_slices.extend(sentences)
    else:
        # 解析表格
        df = pd.read_csv(StringIO(block['content']), sep='|', engine='python', skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df.columns = [col.strip() for col in df.columns]

        # 切分大表格
        small_tables = split_table(df, window_size, overlap)
        for small_df in small_tables:
            lines = [', '.join(small_df.columns)]
            for _, row in small_df.iterrows():
                lines.append(', '.join(str(x).strip() for x in row.values))
            slice_text = '\n'.join(lines)
            table_slices.append(slice_text)

print("文本切片示例：")
for t in text_slices[:3]:
    print("-", t)

print("\n表格切片示例：")
for i, t in enumerate(table_slices[:2]):
    print(f"切片 {i+1}:\n{t}\n")
