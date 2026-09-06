#!/bin/bash
set -e

BASE_OUTPUT="/hy-tmp/output/Qwen3-VL-8B-RPRM-Stage2"
LATEST_CKPT=$(find "$BASE_OUTPUT" -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ 未检测到训练生成的 checkpoint 目录！"
    exit 1
fi

EXPORT_DIR="/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"

echo "🚀 [Merge] 正在使用 Swift 3.x 导出并合并权重..."
echo "• 适配层权重: $LATEST_CKPT"
echo "• 导出目标路径: $EXPORT_DIR"

swift export \
    --adapters "$LATEST_CKPT" \
    --output_dir "$EXPORT_DIR" \
    --merge_lora true

echo "✅ 模型合并完成！生产级模型就绪: $EXPORT_DIR"
