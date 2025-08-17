import pandas as pd
import numpy as np

def fill_merged_cells(df, vertical_cols=None, horizontal_rows=None):
    """
    通用函数：填充表格中合并单元格产生的空值
    vertical_cols: 需要垂直填充的列名列表（rowspan）
    horizontal_rows: 需要水平填充的行索引或行切片（colspan）
    """
    df_filled = df.copy()

    # 1. 垂直合并填充（列方向空值）
    if vertical_cols is not None:
        for col in vertical_cols:
            df_filled[col] = df_filled[col].ffill()  # 向下填充

    # 2. 水平合并填充（行方向空值）
    if horizontal_rows is not None:
        for row_idx in horizontal_rows:
            df_filled.loc[row_idx] = df_filled.loc[row_idx].ffill()


    return df_filled

# ============================
# 示例表格
data = {
    "部门": ["销售", np.nan, "技术", np.nan],
    "姓名": ["张三", "李四", "王五", "赵六"],
    "数学": ["优秀", None, "良好", None],
    "语文": [None, None, "中等", "中等"]
}
df = pd.DataFrame(data)
print("原始表格：\n", df)

# 调用函数：垂直填充部门列，水平填充第0行和第2行（示例）
df_filled = fill_merged_cells(df, vertical_cols=['部门',"数学", "语文"], horizontal_rows=[0,1,2,3])
print("\n填充后表格：\n", df_filled)
