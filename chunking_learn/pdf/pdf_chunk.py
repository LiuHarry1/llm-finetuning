import re
from markdown_it import MarkdownIt
import tiktoken

ENCODING = "cl100k_base"
MAX_TOKENS = 50
OVERLAP_TOKENS = 15
MIN_TOKENS = 15

encoding = tiktoken.get_encoding(ENCODING)

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunk(node, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    chunks = []

    # 对于 inline 类型的 token（段落文本、列表文本）
    if node.type == 'inline' and node.content.strip():
        text = node.content.strip()
        sentences = split_sentences(text)
        current_chunk = ""
        current_tokens = 0
        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if current_tokens + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append({'text': current_chunk.strip()})
                    # overlap
                    if overlap > 0:
                        overlap_text = " ".join(current_chunk.split()[-overlap:])
                        current_chunk = overlap_text + " "
                        current_tokens = count_tokens(current_chunk)
                    else:
                        current_chunk = ""
                        current_tokens = 0
            current_chunk += sent + " "
            current_tokens += sent_tokens
        if current_chunk.strip():
            chunks.append({'text': current_chunk.strip()})
        return chunks

    # 对于 code_block, fence 等
    elif node.type in ['fence', 'code_block', 'blockquote', 'list_item']:
        text = node.content.strip()
        if text:
            chunks.append({'text': text})
        return chunks

    # 容器 token，有 children
    elif hasattr(node, 'children') and node.children:
        for child in node.children:
            child_chunks = recursive_chunk(child, max_tokens, overlap)
            chunks.extend(child_chunks)
    return chunks

def markdown_to_chunks(md_text):
    md = MarkdownIt()
    tokens = md.parse(md_text)
    all_chunks = []
    for node in tokens:
        node_chunks = recursive_chunk(node)
        all_chunks.extend(node_chunks)

    # 合并太小的 chunk
    merged_chunks = []
    for c in all_chunks:
        if merged_chunks and count_tokens(c['text']) < MIN_TOKENS:
            merged_chunks[-1]['text'] += " " + c['text']
        else:
            merged_chunks.append(c)
    return merged_chunks


from typing import List
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document


def structured_markdown_chunk(
    markdown: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    1) 先用 MarkdownNodeParser 拆分成结构节点
    2) 再用 TokenTextSplitter 对过大 node 做拆分
    3) 合并特别小的 nodes
    """

    # -------- Step 1: MarkdownNodeParser -> structural nodes --------
    parser = MarkdownNodeParser(include_metadata=True)
    docs = [Document(text=markdown)]
    nodes = parser.get_nodes_from_documents(docs)

    # -------- Step 2: 对每个 node 按大小规范化 --------
    normalized_nodes = []
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    buffer_node = None
    for n in nodes:
        # 如果 node 很大，则拆分
        if len(n.text) > chunk_size:
            split_docs = text_splitter.get_nodes_from_documents([n])
            for sd in split_docs:
                normalized_nodes.append(sd)
            continue

        # 如果 node 很小，暂存到 buffer
        if buffer_node is None:
            buffer_node = n
        else:
            # 判断合并后大小是否超过阈值
            combined_text = buffer_node.text + "\n" + n.text
            if len(combined_text) <= chunk_size:
                # 合并
                buffer_node = Document(
                    text=combined_text,
                    metadata={**buffer_node.metadata}
                )
            else:
                # push 当前 buffer, buffer = this
                normalized_nodes.append(buffer_node)
                buffer_node = n

    # 最后一份 buffer 若不空也 push
    if buffer_node:
        normalized_nodes.append(buffer_node)

    return normalized_nodes


if __name__ == "__main__":


    sample_md = """
# 一级标题

这是第一段内容，它比较长，需要被分成多个 chunk。它包括一些示例句子。再加一句。


## 列表示例

- 第一项
- 第二项
  - 子项 a
  - 子项 b
- 第三项

1. 有序项 1
2. 有序项 2
3. 有序项 3

## 代码块示例
```python
print("Hello World")
```

## 图片示例

这里插入一张图片：

![示例图片](https://example.com/image.png)

图片周围的文本描述可以帮助 LLM 理解图片内容。

---

## 大表格示例

| 姓名  | 年龄 | 城市 | 职业    | 爱好     |
| --- | -- | -- | ----- | ------ |
| 张三  | 28 | 北京 | 工程师   | 足球, 阅读 |
| 李四  | 35 | 上海 | 设计师   | 绘画, 旅行 |
| 王五  | 42 | 广州 | 教师    | 写作, 游泳 |
| 赵六  | 30 | 深圳 | 数据分析师 | 游戏, 健身 |
| 孙七  | 27 | 成都 | 产品经理  | 音乐, 美食 |
| 周八  | 33 | 杭州 | 程序员   | 编程, 跑步 |
| 吴九  | 29 | 南京 | 医生    | 摄影, 旅行 |
| 郑十  | 40 | 重庆 | 律师    | 阅读, 游泳 |
| 冯十一 | 31 | 苏州 | 会计    | 旅行, 音乐 |
| 朱十二 | 26 | 天津 | 教师    | 写作, 美食 |


这个示例中：

1. 图片、代码块、列表、表格都包含了。  
2. 大表格列数和行数足够测试 chunk 分片。  
3. 你可以直接复制到 Markdown 文件里，然后用你之前的 `MarkdownNodeParser + tiktoken` 分片逻辑测试。

如果你需要，我可以再帮你做一个**带图片 alt 描述的更复杂 Markdown 示例**，更贴近实际 RAG 系统的 PDF 转 Markdown 场景。  

你希望我帮你生成吗？


"""



    # chunks = markdown_to_chunks(sample_md)
    # for i, c in enumerate(chunks):
    #     print(f"Chunk {i + 1} ({count_tokens(c['text'])} tokens):")
    #     print(c['text'])
    #     print("-" * 50)


    chunks = structured_markdown_chunk(sample_md, chunk_size=40, chunk_overlap=10)
    for i, c in enumerate(chunks):
        print(f"=== chunk {i}: size={len(c.text)}")
        print(c.text[:200].replace("\n", " "), "\n---\n")



