# -*- coding: utf-8 -*-
"""step21: 用户三处修改（2026-11-07）。
1) 摘要核心问题句→候选C："但分数的成分构成与维度结构的成立性均缺乏定量检验。"
2) 图2（含题注2段）从3.1末尾移入3.2开头+新增引导句；3.1末段删"图2给出…"句
3) 删4.1.1小节，[104]压缩并入4.1引言段；4.1.2~4.1.4重新编号为4.1.1~4.1.3
用法: python analysis/step21_docx_fix3.py
"""
import copy
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "智能系统学报" / "论文B_工作稿.docx"

d = docx.Document(DOCX)
ps = d.paragraphs


def set_text(p, text):
    """整段替换文本，保留段落样式与第一个run格式。"""
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)
    if p.runs:
        p.runs[0].text = text
    else:
        p.add_run(text)


# ---------- 1) 摘要句（候选C） ----------
abstract = ps[3]
old = "但分数变异由哪些成分构成缺乏定量刻画"
new = "但分数的成分构成与维度结构的成立性均缺乏定量检验"
assert old in abstract.text, "摘要旧句未找到"
full = abstract.text.replace(old, new)
set_text(abstract, full)
print("[1] 摘要句已替换")

# ---------- 2) 图2移入3.2 ----------
# 定位：[60]=3.1末段（含"图2给出…"），[61]=空段, [62]/[63]=图2题注, [64]=Ward段, [65]=3.2标题
i60 = next(i for i, p in enumerate(ps) if "图2给出交互残差的题目间相关矩阵" in p.text)
i_h32 = next(i for i, p in enumerate(ps)
             if p.text.strip() == "3.2 结构检验")
i_cap = [i for i in range(i60, i_h32) if ps[i].text.strip().startswith("图2")]
i_figcap = [i for i in range(i60, i_h32) if ps[i].text.strip().startswith("Fig.2")]
assert len(i_cap) == 1 and len(i_figcap) == 1, "图2题注定位失败"

# 2a. 3.1末段删"图2给出…"句
t = ps[i60].text
old_sent = "图2给出交互残差的题目间相关矩阵，题目按预设维度分组排列。"
assert old_sent in t
new_t60 = t.replace(old_sent, "").rstrip()
set_text(ps[i60], new_t60)
print("[2a] 3.1末段已删图2句")

# 2b. 图2题注两段（XML元素）移到3.2标题之后
cap_els = [ps[i_cap[0]]._p, ps[i_figcap[0]]._p]
anchor = ps[i_h32]._p
for el in cap_els:
    el.getparent().remove(el)
    anchor.addnext(el)  # 依次插入标题后（第二个会插在第一个后面）
print("[2b] 图2题注已移至3.2标题后")

# 2c. 3.2标题后（题注前）插入引导句段——放在题注之前更合理：先引导句再看图
# 实际顺序：3.2标题 → 引导句 → 图2题注 → 原第一层检验段
guide = ("若七维结构成立，交互残差按预设维度分组排列应呈现七个块状组。图2给出交互残差的题目间相关矩阵，"
         "题目按预设维度分组排列，直观上未见块状结构，以下两层检验对此作统计确认。")
# 复制一个Body Text段落作为模板
tmpl = ps[i_h32 + 4]  # 原[64]Ward段，Body Text样式
new_p = copy.deepcopy(tmpl._p)
# 清空复制段落的runs后写引导句
tmp_para = docx.text.paragraph.Paragraph(new_p, tmpl._parent)
set_text(tmp_para, guide)
anchor2 = ps[i_cap[0]]._p  # 图2中文题注
anchor2.addprevious(new_p)
print("[2c] 3.2引导句已插入")

# ---------- 3) 4.1.1删除并入 ----------
# 重新枚举（XML已变动）
ps = d.paragraphs
i_411 = next(i for i, p in enumerate(ps) if p.text.strip() == "4.1.1 交互残差提取")
i_412 = next(i for i, p in enumerate(ps) if p.text.strip() == "4.1.2 奇异方向分解")
i_413 = next(i for i, p in enumerate(ps) if p.text.strip() == "4.1.3 列洗牌置换零分布")
i_414 = next(i for i, p in enumerate(ps) if p.text.strip() == "4.1.4 显著方向的效应判定")
i_intro = next(i for i, p in enumerate(ps)
               if "方法分三步：交互残差提取、奇异方向分解、列洗牌置换检验" in p.text)
# 3a. 4.1引言段改写（并入残差提取信息，方法改为两步）
old_intro = ps[i_intro].text
assert "方法分三步" in old_intro
new_intro = old_intro.replace(
    "方法分三步：交互残差提取、奇异方向分解、列洗牌置换检验。",
    "本章对1198道活跃题的计分矩阵开展解耦分析，输入为2.4节式(4)的双重中心化交互残差，"
    "即承载分数变异中占23.4%交互成分（3.3.3节）的矩阵，方法分奇异方向分解与列洗牌置换检验两步。")
set_text(ps[i_intro], new_intro)
print("[3a] 4.1引言段已改写")

# 3b. 删除4.1.1标题段与其正文段
for idx in sorted([i_411, i_411 + 1], reverse=True):
    el = ps[idx]._p
    el.getparent().remove(el)
print("[3b] 4.1.1标题+正文已删除")

# 3c. 重新编号
ps = d.paragraphs
set_text(next(p for p in ps if p.text.strip() == "4.1.2 奇异方向分解"), "4.1.1 奇异方向分解")
set_text(next(p for p in ps if p.text.strip() == "4.1.3 列洗牌置换零分布"), "4.1.2 列洗牌置换零分布")
set_text(next(p for p in ps if p.text.strip() == "4.1.4 显著方向的效应判定"), "4.1.3 显著方向的效应判定")
print("[3c] 4.1.x已重新编号")

d.save(DOCX)
print("保存完成:", DOCX.name)
