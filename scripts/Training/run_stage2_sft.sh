#!/bin/bash
set -e

BASE_MODEL="/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
if [ ! -d "$BASE_MODEL" ]; then
    BASE_MODEL=$(ls -d /hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/* | head -n 1)
fi

TRAIN_DATA="./multimodal_data/train_stage2.jsonl"
VAL_DATA="./multimodal_data/val_stage2.jsonl"
OUTPUT_DIR="/hy-tmp/output/Qwen3-VL-8B-RPRM-Stage2"

echo "--------------------------------------------------"
echo "🚀 启动 ms-swift 3.x Stage 2 全量 LoRA 训练"
echo "• 基础模型: $BASE_MODEL"
echo "• 训练集规模: $(wc -l < $TRAIN_DATA) 条"
echo "• 验证集规模: $(wc -l < $VAL_DATA) 条"
echo "• 目标输出: $OUTPUT_DIR"
echo "--------------------------------------------------"

swift sft \
    --model "$BASE_MODEL" \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --max_length 3072 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --gradient_checkpointing true \
    --eval_steps 150 \
    --save_steps 150 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --metric_for_best_model loss \
    --greater_is_better false

echo "🎉 Stage 2 微调完成！"
