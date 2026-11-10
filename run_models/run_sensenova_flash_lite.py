# -*- coding: utf-8 -*-
"""
一键运行 商汤 SenseNova 6.8 Flash Lite（thinking on + off）。
用法：python run_models/run_sensenova_flash_lite.py
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
    print("开始运行: SenseNova 6.8 Flash Lite (thinking on + off, 商汤)")
    print(f"题目文件: {config.FINAL_ITEMS_PATH}")
    print(f"输出目录: {config.RAW_RESP_DIR}")
    print(f"重复次数: {config.REPEAT}")
    print("=" * 60)
    runner.run(
        items_path=config.FINAL_ITEMS_PATH,
        repeat=config.REPEAT,
        only_model_alias="sensenova-flash-lite",
    )
    print("\n完成！可以关闭电脑。")
