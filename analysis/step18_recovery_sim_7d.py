# -*- coding: utf-8 -*-
"""step18 结构恢复仿真（七维口径，3.3.1"检验管线分辨率"论证重跑）。

问题：聚类检验未检出七维结构，是否因为检验管线本身分辨率不足？
协议：在1198题×22观测的同等规模下植入块结构——同一维度内题目对的总体相关为ρ、
      跨维度为0（块大小=真实活跃题分布）；走与真实数据完全相同的管线
      （双重中心化→1-相关距离→Ward强设K=7），计算与植入标签的ARI。
      ρ=0.5为旧文引用的强结构标定水平；另跑ρ=0.2/0.1给出梯度参考。
对照：真实数据同维-异维平均相关差Δr=0.018（95%CI[0.016,0.031]），
      仅为ρ=0.5的约1/28——"可检出水平远强于实际信号"的论证保持成立。

输出: data/results/schwartz/recovery_sim_7d.json
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import dist_1_minus_corr
from step6_mp_spectrum import double_center

rng = np.random.default_rng(20261105)
# 块大小 = 七维活跃题真实分布（普世284/仁慈283/安全217/成就125/自我106/从众94/传统89）
SIZES = [284, 283, 217, 125, 106, 94, 89]
DIM_NAMES = ['普世主义', '仁慈', '安全', '成就', '自我导向', '从众', '传统']
N_OBS = 22
REPS = 20
N_ITEMS = sum(SIZES)


def plant(rho):
    """块内等相关ρ、跨块独立的Gauss矩阵（题目×观测），块内共同因子结构。"""
    blocks = np.repeat(np.arange(len(SIZES)), SIZES)
    X = np.zeros((N_ITEMS, N_OBS))
    for b in range(len(SIZES)):
        idx = np.where(blocks == b)[0]
        f = rng.standard_normal(N_OBS)                       # 块共同因子
        E = rng.standard_normal((len(idx), N_OBS))           # 题目特异噪声
        X[idx] = np.sqrt(rho) * f + np.sqrt(1 - rho) * E
    return X, blocks


rows = []
for rho in (0.5, 0.2, 0.1, 0.05, 0.02):
    aris = []
    for _ in range(REPS):
        X, blocks = plant(rho)
        Xc = double_center(X)
        D = dist_1_minus_corr(Xc)
        Z = linkage(squareform(D, checks=False), method="ward")
        labels = fcluster(Z, t=len(SIZES), criterion="maxclust")
        aris.append(adjusted_rand_score(blocks, labels))
    aris = np.array(aris)
    row = {"rho_planted": rho, "ari_median": float(np.median(aris)),
           "ari_min": float(aris.min()), "full_recovery_rate": float((aris > 0.99).mean())}
    rows.append(row)
    print(f"植入ρ={rho:.2f}: ARI中位={row['ari_median']:.3f} 最小={row['ari_min']:.3f} "
          f"完全恢复(ARI>0.99)比例={row['full_recovery_rate']:.0%}")

out = {
    "protocol": {
        "n_items": N_ITEMS, "n_obs": N_OBS, "n_clusters": len(SIZES),
        "block_sizes": dict(zip(DIM_NAMES, SIZES)),
        "planted_structure": "块内等相关ρ、跨块独立（共同因子构造），双重中心化→1-相关距离→Ward强设K=7",
        "n_reps": REPS, "seed": 20261105},
    "results": rows,
    "reference": {"delta_r_observed_7d": 0.018,
                  "delta_r_ci95": [0.016, 0.031],
                  "note": "真实同维-异维相关差Δr=0.018，约为ρ=0.5的1/28——"
                          "管线可恢复的强度远强于实际存在的信号"},
}
BASE = config.DATA_DIR / "results" / "schwartz"
with open(BASE / "recovery_sim_7d.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"已保存: {BASE / 'recovery_sim_7d.json'}")
