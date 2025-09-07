import pandas as pd

# 创建示例数据
data = pd.DataFrame({
    "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "Glucose": [90, 95, 100, 110, 105, 115, 120, 125, 130, 135],
    "HbA1c": [5.2, 5.4, 5.5, 5.8, 5.7, 6.0, 6.2, 6.4, 6.5, 6.8]
})

# 保存为 CSV 文件
data.to_csv("13100326.csv", index=False)

print("CSV 文件已生成：13100326.csv")
