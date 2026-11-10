# -*- coding: utf-8 -*-
"""提取论文B工作稿全文（段落+表格），供修改方案分析。"""
import sys
from docx import Document

path = r'd:\competitionresearch\cognitivesafety\实验2\智能系统学报\论文B_工作稿.docx'
doc = Document(path)

out = []
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if not t:
        continue
    style = p.style.name if p.style else ''
    out.append(f'[{i}|{style}] {t}')

for ti, tbl in enumerate(doc.tables):
    out.append(f'===== 表格{ti+1} ({len(tbl.rows)}行×{len(tbl.columns)}列) =====')
    for r in tbl.rows:
        cells = [c.text.strip().replace('\n', ' ') for c in r.cells]
        out.append(' | '.join(cells))

text = '\n'.join(out)
with open(r'C:\Users\22676\AppData\Local\Temp\paperB_dump.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'共{len(out)}个非空段落/表格行，已写入 paperB_dump.txt')
print(f'总字符数: {len(text)}')
