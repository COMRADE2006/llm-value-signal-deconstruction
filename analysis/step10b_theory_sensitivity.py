# -*- coding: utf-8 -*-
"""5.1理论定向划分的敏感性分析：改划/剔除争议维度后的方向检验。
V0=现行理论版(个人:自由/敬业/诚信)；
V1=诚信改归社会(个人:自由/敬业)；
V2=剔除诚信与民主(10维)；
V3=仅剔除诚信(11维)。
统计量：Spearman(标签置换精确p,枚举全部组合)+Mann-Whitney U单尾精确p(枚举)。
锚点级不受V1~V3影响(锚点为自由/民主/爱国，均不含诚信)。
输出 data/results/agree/theory_sensitivity.json。
"""
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RES = Path(config.DATA_DIR) / "results" / "agree"
tm = json.loads((RES / "theory_mapping.json").read_text(encoding="utf-8"))
load = tm["axis_vs_theory_理论版(主分析)"]["axis_loadings"]
dims = list(load.keys())


def spearman(x, y):
    """Pearson on ranks with tie handling (与scipy.spearmanr一致)。"""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sp = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return sp / (sx * sy)


def perm_p(x, y):
    """标签置换精确p：枚举全部标签组合。"""
    obs = abs(spearman(x, y))
    n_pos = sum(1 for v in x if v > 0)
    idx = range(len(x))
    cnt = tot = 0
    for pos in combinations(idx, n_pos):
        xp = [1 if i in pos else -1 for i in idx]
        tot += 1
        if abs(spearman(xp, y)) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def mw_one_tail(coords, vals):
    """U=个人定向取值低于社会定向的配对数；精确p=P(U>=u)，枚举组合。"""
    keys = list(vals.keys())
    pos = [vals[d] for d in keys if coords[d] > 0]
    neg = [vals[d] for d in keys if coords[d] < 0]
    u = sum(1 for nv in neg for pv in pos if nv < pv)
    n_neg, n_pos = len(neg), len(pos)
    # 零分布：从全部维度中选n_neg个给个人组，枚举每种组合的U
    order = sorted(range(len(keys)), key=lambda i: vals[keys[i]])
    rank_of = {order[k]: k + 1 for k in range(len(keys))}
    n_tot = len(keys)
    cnt = tot = 0
    for grp in combinations(range(n_tot), n_neg):
        rk = sorted(rank_of[g] for g in grp)
        ug = sum(r - 1 - k for k, r in enumerate(rk))  # 每个个人项之前的社会项数
        tot += 1
        if ug >= u:
            cnt += 1
    return u, n_neg, n_pos, cnt / tot


variants = {
    "V0_现行(个人:自由/敬业/诚信)": {"自由": -1, "敬业": -1, "诚信": -1},
    "V1_诚信改归社会": {"自由": -1, "敬业": -1},
    "V2_剔除诚信与民主": {"自由": -1, "敬业": -1},
    "V3_仅剔除诚信": {"自由": -1, "敬业": -1},
}
drop = {"V2_剔除诚信与民主": ["诚信", "民主"], "V3_仅剔除诚信": ["诚信"]}

out = {}
for name, personal in variants.items():
    dims_v = [d for d in dims if d not in drop.get(name, [])]
    coords = {d: (-1 if d in personal else 1) for d in dims_v}
    vals = {d: load[d] for d in dims_v}
    rho = spearman([coords[d] for d in dims_v], [vals[d] for d in dims_v])
    p_perm = perm_p([coords[d] for d in dims_v], [vals[d] for d in dims_v])
    u, n_neg, n_pos, p_mw = mw_one_tail(coords, vals)
    out[name] = {
        "n_dims": len(dims_v), "n_personal": n_neg, "n_social": n_pos,
        "spearman_rho": round(rho, 3), "perm_p_two_sided": p_perm,
        "mw_U": u, "mw_p_one_tail": round(p_mw, 4),
        "combos": comb(len(dims_v), n_neg),
    }

out["_anchors_note"] = "V1~V3均不改变锚点集合(自由/民主/爱国)，锚点级3/3一致不受影响"
(RES / "theory_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
