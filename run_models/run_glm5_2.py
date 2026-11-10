# -*- coding: utf-8 -*-
"""
一键运行 GLM-5.2（thinking on + off）。
用法：python run_models/run_glm5_2.py
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
    print("开始运行: GLM-5.2 (thinking on + off)")
    print(f"题目文件: {config.FINAL_ITEMS_PATH}")
    print(f"输出目录: {config.RAW_RESP_DIR}")
    print(f"重复次数: {config.REPEAT}")
    print("=" * 60)
    runner.run(
        items_path=config.FINAL_ITEMS_PATH,
        repeat=config.REPEAT,
        only_model_alias="glm-5.2",
    )
    print("\n完成！可以关闭电脑。")
