jira_table = """
|| 类别 || 产品 || 销售额（万元） || 利润率 ||
| 饮料 | 可乐   | 500           | 20%     |
|      | 果汁   | 300           | 18%     |
| 食品 | 面包   | 450           | 22%     |
|      | 蛋糕   | 350           | 25%     |
"""


def parse_jira_table(raw_table: str):
    lines = raw_table.strip().splitlines()
    rows = []

    for line in lines:
        if not line.strip().startswith('|'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        rows.append(cells)

    return rows

def fill_merged_like_cells(table_rows):
    if not table_rows:
        return []
    filled_rows = []
    last_vals = [''] * len(table_rows[0])

    for row in table_rows:
        new_row = []
        for i, cell in enumerate(row):
            if cell:
                last_vals[i] = cell
            new_row.append(last_vals[i])
        filled_rows.append(new_row)

    return filled_rows


parsed = parse_jira_table(jira_table)
cleaned = fill_merged_like_cells(parsed)

for row in cleaned:
    print(row)


