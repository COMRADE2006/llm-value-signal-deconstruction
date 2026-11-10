# -*- coding: utf-8 -*-
"""
step15 七维主框架轮廓系数K扫描（论文图3曲线数据）。

对七维主框架活跃题（1198×22，BIG7=仁慈/普世主义/安全/成就/从众/自我导向/传统）
的双重中心化矩阵跑Ward聚类 K=2~100，落盘逐K平均轮廓系数。
协议与旧框架 check_sil_k100_orig.csv 完全一致（dist_1_minus_corr + ward_best_k），
仅题集从十维1227换为七维1198（权力/享乐/刺激已于题库构造阶段剔除）。

用法: python analysis/step15_sil_k100_schwartz.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import dist_1_minus_corr, ward_best_k
from step5_ega import load_matrices
from step6_mp_spectrum import double_center

RES = Path(config.DATA_DIR) / "results" / "schwartz"
BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']

# 输入：七维主框架活跃题（先子集后中心化，与step14/16/17同口径）
X, meta, cols = load_matrices("schwartz")
mask = meta["core_value"].isin(BIG7).values
X7 = X[mask]
print(f"输入: {X7.shape[0]}题 × {X7.shape[1]}观测（七维主框架活跃题）")

Xc = np.nan_to_num(double_center(X7).astype(float))
labels, sil, k_star = ward_best_k(dist_1_minus_corr(Xc), 100)
sil.to_csv(RES / "check_sil_k100_orig.csv", index=False)

s7 = sil.loc[sil["k"] == 7, "silhouette"].iloc[0]
peak_k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
print(f"K*={k_star} | K=7处轮廓={s7:.4f} | 全局最大轮廓在K={peak_k}")
print(f"曲线端点: K=2 → {sil['silhouette'].iloc[0]:.3f}, K=100 → {sil['silhouette'].iloc[-1]:.3f}")
print(f"单调性: 2~100内相邻差分最大值 = {sil['silhouette'].diff().max():.4f} "
      f"（>0 表示持续上升，无内部峰）")
print(f"已保存: {RES / 'check_sil_k100_orig.csv'}")
