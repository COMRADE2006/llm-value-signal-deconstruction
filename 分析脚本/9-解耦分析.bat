@echo off
title 解耦分析（v3第四步）
echo ============================================
echo   [9] 解耦分析：方差预算审计 + 通道定位（D/IP主+Dstrict/M/E敏感性）
echo   + 统计解耦（剥PC1残差重聚类+对半复制+置换检验）+ 计分解耦（7通道对比）
echo   预注册结局判定：A结构恢复 / B部分恢复 / C不恢复
echo   运行约20~35分钟（子抽样300次×多通道），请耐心等待
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\step4_decouple.py" --source old
echo.
echo 运行完毕：查看 data\results\old\decouple_result.json 的 verdict 字段为预注册结局判定。
pause