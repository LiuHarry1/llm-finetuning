from rstlite import Segmenter
import nltk

# 如果是第一次用 nltk 的句法工具，需要下载 punkt
nltk.download('punkt')

# 示例英文文本（可替换为任意长文本）
text = """
Artificial intelligence is transforming industries. Machines are now capable of performing complex tasks such as diagnosing diseases or driving cars.
However, with great power comes great responsibility. Many experts warn about ethical concerns and the potential for misuse.
Education systems need to adapt quickly. Training the next generation to work with AI is critical.
"""

# 初始化 segmenter（使用 nltk 分句器）
segmenter = Segmenter()

# EDU 分割（结果就是一个个语义单元）
edus = segmenter.segment_text(text)

# 输出 EDU 结果
print("Discourse Units (EDUs):\n")
for i, edu in enumerate(edus, 1):
    print(f"EDU {i}: {edu}")
