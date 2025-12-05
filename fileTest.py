from openpyxl import Workbook, load_workbook
from statistics import mean
import os

file = 'score.xlsx'
path= './大一上 工程学导论/AI组学习资料/test projects/output'
path_to_file = os.path.join(path, file)

# ---------- 1. 写 ----------
wb = Workbook()
ws = wb.active
ws.title = '成绩'
# 表头
ws.append(['姓名', '文', '数', '英'])
# 数据
rows = [
    ['张三', 90, 88, 76],
    ['李四', 95, 92, 85],
    ['王五', 78, 86, 92],
]
for r in rows:
    ws.append(r)
wb.save(path_to_file)