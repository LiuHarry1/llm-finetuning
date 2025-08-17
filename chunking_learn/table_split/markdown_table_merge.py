import pandas as pd
from io import StringIO

markdown_table = """
姓名|科目|分数
张三|数学|90
|语文|85
张三|英语|88
李四|数学|92
|语文|80
"""

df = pd.read_csv(StringIO(markdown_table), sep="|")
df['姓名'] = df['姓名'].ffill()
print(df)
