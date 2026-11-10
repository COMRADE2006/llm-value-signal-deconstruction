# -*- coding: utf-8 -*-
"""
step10 (T4) 理论映射检验：模型语义轴 ↔ Schwartz/金盛华人类价值结构

分析1  金盛华(2019)三区域protocol移植
   人类版：2569被试 PVQ 19值相关矩阵 → 个人定向内部r>.50比例8.33% /
           跨类17.78% / 社会定向内部68.89%（金盛华表7）
   模型版：残差交互矩阵聚合为 12维×22模型观测 → 12×12维度相关矩阵
           → 按映射表分三区域 → 各区域r>.50比例，与人类数据并列对照
   （诚实标注：n=22观测远小于人类n=2569，仅作方向性对照）

分析2  轴载荷 vs Schwartz"自主-嵌入"理论坐标
   理论坐标：个人定向=-1 / 社会定向=+1（两版映射各算一次）
   统计量：Spearman ρ + 1000次标签置换检验

映射表（用户2026-10-13拍板"两版都跑"）：
   理论版（主分析）：个人=自由(思想-行动自主)/敬业(成就)/诚信(诚实)，
                     其余9维=社会定向（依据Schwartz 2012直接对应+概念归类）
   平衡版（敏感性）：个人=自由/平等/诚信/敬业/富强/公正；
                     社会=爱国/民主/文明/和谐/友善/法治（与轴排序对齐，循环论证风险高，仅附录）

输出: data/results/agree/theory_mapping.json + dim_corr_matrix.csv
用法: python analysis/step10_theory_mapping.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center

BASE = config.DATA_DIR / "results" / "agree"
rng = np.random.default_rng(20261013)

# ---- 两版映射表 ----
MAPPING_THEORY = {   # 主分析：Schwartz理论独立构造
    "自由": "个人", "敬业": "个人", "诚信": "个人",
    "平等": "社会", "公正": "社会", "富强": "社会", "民主": "社会",
    "文明": "社会", "和谐": "社会", "友善": "社会", "法治": "社会", "爱国": "社会",
}
MAPPING_BALANCED = {  # 敏感性：与轴排序对齐（6 vs 6）
    "自由": "个人", "平等": "个人", "诚信": "个人", "敬业": "个人",
    "富强": "个人", "公正": "个人",
    "爱国": "社会", "民主": "社会", "文明": "社会", "和谐": "社会",
    "友善": "社会", "法治": "社会",
}
# 理论版逐维依据（写入结果，论文映射表脚注直接用）
THEORY_BASIS = {
    "自由": "Schwartz思想自主+行动自主（个人定向定义值）",
    "敬业": "Schwartz成就（个人定向）",
    "诚信": "Schwartz诚实（偏个人定向）",
    "平等": "Schwartz博爱-公平（自我超越，社会定向服务他人）",
    "公正": "社会正义（博爱系）",
    "富强": "国家目标（集体利益）",
    "民主": "制度性集体参与（置信度中）",
    "文明": "规则遵从/礼仪规范",
    "和谐": "社会安全/群体和谐",
    "友善": "友善-关怀",
    "法治": "规则遵从",
    "爱国": "传统/国家认同",
}

# 金盛华2019表7人类参照（r>.50比例）
JIN_HUMAN_REF = {"个人内部": 8.33, "跨类": 17.78, "社会内部": 68.89}


def build_dim_scores():
    """残差交互矩阵 → 12维×22模型 的维度均分矩阵 D。"""
    X, meta, cols = load_matrices("agree")
    cv = meta["core_value"].values
    Xc = double_center(X)
    D = pd.DataFrame(Xc, index=meta["rule_id"].values if "rule_id" in meta else None)
    D["dim"] = cv
    # 每维×每模型取残差均值
    dm = D.groupby("dim").mean()          # 12 × 22
    return dm, cols


def build_dim_scores_raw():
    """原始分（未中心化）→ 12维×22模型。金盛华protocol用的是原始分数层面
    （人类被试的价值观均分相关），此处对齐该层面——预期全区域高相关（共识驱动）。"""
    X, meta, cols = load_matrices("agree")
    cv = meta["core_value"].values
    D = pd.DataFrame(X)
    D["dim"] = cv
    return D.groupby("dim").mean()


def three_region(dm, mapping):
    """12×12维度相关矩阵 → 三区域 r>.50 比例（金盛华protocol）。"""
    corr = dm.T.corr()                    # 维度间相关（22个模型观测）
    dims = list(corr.index)
    personal = [d for d in dims if mapping[d] == "个人"]
    social = [d for d in dims if mapping[d] == "社会"]

    def region_stats(a, b, name):
        """区域 a×b 内的非对角 |r|（金盛华统计的是相关系数r>.50，不取绝对值）"""
        rs = []
        for i in a:
            for j in b:
                if i != j:
                    rs.append(corr.loc[i, j])
        rs = np.array(rs)
        return {"name": name, "n_pairs": len(rs),
                "n_r_gt_050": int((rs > 0.5).sum()),
                "pct_r_gt_050": float((rs > 0.5).mean() * 100),
                "median_r": float(np.median(rs))}

    return {
        "个人内部": region_stats(personal, personal, "个人定向内部"),
        "跨类": region_stats(personal, social, "个人×社会跨类"),
        "社会内部": region_stats(social, social, "社会定向内部"),
        "n_obs_models": dm.shape[1],
    }, corr


def axis_vs_theory(mapping, n_perm=1000):
    """轴载荷 vs 理论坐标（个人=-1/社会=+1）的Spearman + 标签置换。"""
    with open(BASE / "axis_validation.json", encoding="utf-8") as f:
        ordering = json.load(f)["dimension_ordering"]
    dims = list(ordering.keys())
    axis = np.array([ordering[d] for d in dims])
    coord = np.array([-1.0 if mapping[d] == "个人" else 1.0 for d in dims])
    rho = spearmanr(axis, coord).statistic
    # 置换零分布：理论坐标在12维间随机重排
    null = np.array([spearmanr(axis, rng.permutation(coord)).statistic
                     for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return {"spearman_rho": float(rho), "perm_p_two_sided": p,
            "null_abs_max": float(np.abs(null).max()),
            "axis_loadings": ordering,
            "theory_coords": {d: c for d, c in zip(dims, coord)}}


def main():
    print("=" * 62)
    print("step10 (T4) 理论映射检验")
    print("=" * 62)
    dm, cols = build_dim_scores()
    print(f"维度均分矩阵: {dm.shape[0]}维 × {dm.shape[1]}模型观测")

    out = {"n_models_obs": int(dm.shape[1]),
           "human_reference_jin2019": JIN_HUMAN_REF}

    # ---- 分析1：三区域protocol（两版映射，残差层面） ----
    print("\n[分析1] 金盛华三区域protocol移植（r>.50比例）")
    for tag, mapping in [("理论版(主分析)", MAPPING_THEORY),
                         ("平衡版(敏感性)", MAPPING_BALANCED)]:
        regions, corr = three_region(dm, mapping)
        out[f"three_region_{tag}"] = regions
        print(f"\n  == {tag} ==")
        print(f"  {'区域':<10}{'对数':>6}{'r>.50':>8}{'比例':>10}{'中位r':>9}   人类参照")
        for key in ["个人内部", "跨类", "社会内部"]:
            r = regions[key]
            print(f"  {r['name']:<12}{r['n_pairs']:>5}{r['n_r_gt_050']:>7}"
                  f"{r['pct_r_gt_050']:>9.2f}%{r['median_r']:>9.3f}"
                  f"   {JIN_HUMAN_REF[key]:.2f}%")
        corr.to_csv(BASE / f"dim_corr_matrix_{tag[:3]}.csv", encoding="utf-8-sig")

    # ---- 分析1b：原始分层面（与金盛华protocol测量层面完全对齐） ----
    print("\n[分析1b] 原始分层面三区域（对齐人类protocol的测量层面；预期共识驱动全高相关）")
    dm_raw = build_dim_scores_raw()
    for tag, mapping in [("理论版(主分析)", MAPPING_THEORY),
                         ("平衡版(敏感性)", MAPPING_BALANCED)]:
        regions_raw, corr_raw = three_region(dm_raw, mapping)
        out[f"three_region_raw_{tag}"] = regions_raw
        print(f"\n  == {tag}（原始分）==")
        for key in ["个人内部", "跨类", "社会内部"]:
            r = regions_raw[key]
            print(f"  {r['name']:<12}{r['n_pairs']:>5}{r['n_r_gt_050']:>7}"
                  f"{r['pct_r_gt_050']:>9.2f}%{r['median_r']:>9.3f}"
                  f"   人类: {JIN_HUMAN_REF[key]:.2f}%")
        corr_raw.to_csv(BASE / f"dim_corr_raw_{tag[:3]}.csv", encoding="utf-8-sig")

    # ---- 分析2：轴载荷 vs 理论坐标 ----
    print("\n[分析2] 轴载荷 vs Schwartz自主-嵌入坐标（Spearman+置换1000次）")
    for tag, mapping in [("理论版(主分析)", MAPPING_THEORY),
                         ("平衡版(敏感性)", MAPPING_BALANCED)]:
        res = axis_vs_theory(mapping)
        out[f"axis_vs_theory_{tag}"] = res
        print(f"  {tag}: ρ={res['spearman_rho']:.3f}, "
              f"置换p={res['perm_p_two_sided']:.4f} "
              f"(零分布|maxρ|={res['null_abs_max']:.3f})")

    # ---- 分析2b：锚点级对应（最干净的一层证据） ----
    # CI不含零的确凿锚点上，检验理论方向（个人→负端/社会→正端）是否与实测符号一致
    print("\n[分析2b] 锚点级对应（CI确凿锚点 vs 理论方向）")
    with open(BASE / "axis_dim_ci.json", encoding="utf-8") as f:
        ci = json.load(f)
    anchor_check = {}
    for d, v in ci.items():
        if v["anchor"]:
            expect_sign = -1 if MAPPING_THEORY[d] == "个人" else 1
            agree = np.sign(v["mean"]) == expect_sign
            anchor_check[d] = {
                "axis_mean": v["mean"], "ci95": v["ci95"],
                "theory": MAPPING_THEORY[d], "expect_sign": expect_sign,
                "agree_with_theory": bool(agree),
            }
            print(f"  {d}: 轴位置={v['mean']:+.4f} (CI {v['ci95'][0]:+.4f}~{v['ci95'][1]:+.4f}), "
                  f"理论={MAPPING_THEORY[d]}定向 → {'✓ 一致' if agree else '✗ 不一致'}")
    n_ok = sum(1 for v in anchor_check.values() if v["agree_with_theory"])
    out["anchor_level_check"] = {
        "n_anchors": len(anchor_check), "n_agree": n_ok,
        "detail": anchor_check,
        "note": "理论版映射下CI确凿锚点与Schwartz自主-嵌入方向的对应数",
    }
    print(f"  → {n_ok}/{len(anchor_check)} 锚点与理论方向一致")

    # 理论版逐维依据（论文脚注素材）
    out["theory_version_basis"] = THEORY_BASIS

    path = BASE / "theory_mapping.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
