import pandas as pd
import numpy as np

# 模拟读取后，部门列合并产生空值
data = {
    "部门": ["销售", np.nan, "技术", np.nan],
    "姓名": ["张三", "李四", "王五", "赵六"],
    "年龄": [28, 32, 25, 30]
}

df = pd.DataFrame(data)
print("原始表格：\n", df)

# 向下填充
df['部门'] = df['部门'].ffill()
print("垂直填充后：\n", df)
