# -*- coding: utf-8 -*-
"""
一键运行 阶跃星辰 Step-3.7-Flash（thinking on + off）。
用法：python run_models/run_step_3_7_flash.py
断点续跑：重复运行会跳过已完成的题目。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from api_caller import runner

if __name__ == "__main__":
    config.ensure_dirs()
    print("=" * 60)
    print("开始运行: Step-3.7-Flash (thinking on + off, StepFun原生)")
    print(f"题目文件: {config.FINAL_ITEMS_PATH}")
    print(f"输出目录: {config.RAW_RESP_DIR}")
    print(f"重复次数: {config.REPEAT}")
    print("=" * 60)
    runner.run(
        items_path=config.FINAL_ITEMS_PATH,
        repeat=config.REPEAT,
        only_model_alias="step-3.7-flash",
    )
    print("\n完成！可以关闭电脑。")
