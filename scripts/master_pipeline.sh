#!/bin/bash

LOG_FILE="./auto_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 重新启动训练与导出总线"
echo "================================================================="

echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] 启动 ms-swift 3.x LoRA 微调..."
if ! ./run_stage2_sft.sh > ./train_stage2.log 2>&1; then
    echo "❌ 模型训练发生异常！请检查 ./train_stage2.log"
    exit 1
fi

echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] 微调收敛，执行 LoRA 权重合并..."
if ! ./merge_stage2_lora.sh >> ./auto_pipeline.log 2>&1; then
    echo "❌ 权重合并异常！"
    exit 1
fi

echo "================================================================="
echo "🎊 [$(date '+%Y-%m-%d %H:%M:%S')] 全流程闭环完成！生产级 PRM 判定模型已就绪。"
echo "================================================================="
