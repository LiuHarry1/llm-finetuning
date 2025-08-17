import pandas as pd

df = pd.read_excel("data/table.xlsx")
df['姓名'] = df['姓名'].ffill()
print(df)
