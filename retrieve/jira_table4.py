import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

def html_table_to_markdown_chunks(html: str, rows_per_chunk=5):
    soup = BeautifulSoup(html, 'html.parser')
    df = pd.read_html(StringIO(str(soup)))[0]

    chunks = []
    total = len(df)
    n_chunks = (total + rows_per_chunk - 1) // rows_per_chunk

    for i in range(n_chunks):
        start = i * rows_per_chunk
        end = min((i + 1) * rows_per_chunk, total)
        sub_df = df.iloc[start:end]
        header = f"这是学生成绩表（第{i+1}组，共{n_chunks}组）：\n\n"
        chunk_md = header + sub_df.to_markdown(index=False)
        chunks.append(chunk_md)

    return chunks


html = """<table>
<tr><th>姓名</th><th>数学</th><th>英语</th></tr>
<tr><td>学生1</td><td>90</td><td>80</td></tr>
<tr><td>学生2</td><td>91</td><td>81</td></tr>
<tr><td>学生3</td><td>92</td><td>82</td></tr>
<tr><td>学生4</td><td>93</td><td>83</td></tr>
<tr><td>学生5</td><td>94</td><td>84</td></tr>
<tr><td>学生6</td><td>90</td><td>80</td></tr>
<tr><td>学生7</td><td>91</td><td>81</td></tr>
<tr><td>学生8</td><td>92</td><td>82</td></tr>
<tr><td>学生9</td><td>93</td><td>83</td></tr>
<tr><td>学生10</td><td>94</td><td>84</td></tr>
<tr><td>学生11</td><td>95</td><td>85</td></tr>
<tr><td>学生12</td><td>96</td><td>86</td></tr>
</table>"""

html = """<table>
<tr><th>姓名</th><th>数学</th><th>英语</th></tr>
<tr><td>学生1</td><td>90</td><td>80</td></tr>
<tr><td>学生2</td><td>91</td><td>81</td></tr>
<tr><td>学生3</td><td>92</td><td>82</td></tr>
<tr><td>学生4</td><td>93</td><td>83</td></tr>
<tr><td>学生5</td><td>94</td><td>84</td></tr>
<tr><td>学生6</td><td>90</td><td>80</td></tr>
<tr><td>学生7</td><td>91</td><td>81</td></tr>
<tr><td>学生8</td><td>92</td><td>82</td></tr>
<tr><td>学生9</td><td>93</td><td>83</td></tr>
<tr><td>学生10</td><td>94</td><td>84</td></tr>
<tr><td>学生11</td><td>95</td><td>85</td></tr>
<tr><td>学生12</td><td>96</td><td>86</td></tr>
</table>"""

chunks = html_table_to_markdown_chunks(html, rows_per_chunk=5)

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---\n")
    print(chunk)
