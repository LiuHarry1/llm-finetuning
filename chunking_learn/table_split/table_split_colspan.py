import pandas as pd
import numpy as np

data = {
    "姓名": ["张三", "李四"],
    "数学": ["优秀", None],  # 张三的数学和语文水平合并
    "语文": [None, "良好"]
}

df = pd.DataFrame(data)
print("原始表格：\n", df)

# 按行填充空值
df = df.apply(lambda row: row.ffill(), axis=1)
print("水平填充后：\n", df)

