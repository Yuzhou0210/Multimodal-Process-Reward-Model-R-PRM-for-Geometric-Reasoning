#!/bin/bash
set -e

# ==============================================================================
# 1. 数据质检与切分脚本 (prepare_stage2_data.py)
# ==============================================================================
cat << 'SCRIPT_DATA' > prepare_stage2_data.py
import os
import json
import random

RAW_FILE = "./multimodal_data/sft_distilled_train.jsonl"
OUTPUT_DIR = "./multimodal_data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_stage2.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_stage2.jsonl")

def main():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"未找到源文件: {RAW_FILE}")

    valid_samples = []
    yes_cnt, no_cnt = 0, 0

    print("🔍 [Step 1/2] 正在质检蒸馏数据格式与图片路径...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" not in data or "images" not in data:
                    continue
                img_path = data["images"][0]
                if not os.path.exists(img_path):
                    continue

                content = data["messages"][-1]["content"]
                if "Verification: Is the step correct (Yes/No)? Yes" in content or content.strip().endswith("Yes"):
                    yes_cnt += 1
                else:
                    no_cnt += 1

                valid_samples.append(json.dumps(data, ensure_ascii=False))
            except Exception:
                continue

    total = len(valid_samples)
    print(f"📊 质检完成: 共 {total} 条有效样本 | Yes: {yes_cnt} ({(yes_cnt/max(1,total))*100:.1f}%) | No: {no_cnt} ({(no_cnt/max(1,total))*100:.1f}%)")

    if total < 1000:
        raise ValueError(f"有效数据量过低 ({total} 条)，可能蒸馏出现异常，终止训练！")

    random.seed(42)
    random.shuffle(valid_samples)

    # 验证集切分：5%，最少 300 条，最多 1200 条
    val_size = min(1200, max(300, int(total * 0.05)))
    val_data = valid_samples[:val_size]
    train_data = valid_samples[val_size:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")

    print(f"✅ [Step 2/2] 数据集切分成功：")
    print(f"   • 训练集 (train_stage2.jsonl): {len(train_data)} 条")
    print(f"   • 验证集 (val_stage2.jsonl)  : {len(val_data)} 条")

if __name__ == "__main__":
    main()
SCRIPT_DATA

# ==============================================================================
# 2. 生产级 SFT 训练脚本 (run_stage2_sft.sh)
# ==============================================================================
cat << 'SCRIPT_SFT' > run_stage2_sft.sh
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
echo "🚀 启动 ms-swift Stage 2 全量 LoRA 训练"
echo "• 基础模型: $BASE_MODEL"
echo "• 训练集规模: $(wc -l < $TRAIN_DATA) 条"
echo "• 验证集规模: $(wc -l < $VAL_DATA) 条"
echo "• 目标产物: $OUTPUT_DIR"
echo "--------------------------------------------------"

swift sft \
    --model_type qwen2_5_vl \
    --model_id_or_path "$BASE_MODEL" \
    --train_type lora \
    --sft_type lora \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --min_lr 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --max_length 3072 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --eval_steps 150 \
    --save_steps 150 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --metric_for_best_model loss \
    --greater_is_better false

echo "🎉 Stage 2 微调成功收敛！"
SCRIPT_SFT
chmod +x run_stage2_sft.sh

# ==============================================================================
# 3. 权重合并与独立导出脚本 (merge_stage2_lora.py)
# ==============================================================================
cat << 'SCRIPT_MERGE' > merge_stage2_lora.py
import os
import glob
from swift.llm import merge_lora

base_output = "/hy-tmp/output/Qwen3-VL-8B-RPRM-Stage2"
checkpoints = glob.glob(f"{base_output}/*/checkpoint-*")
if not checkpoints:
    print("❌ 未检测到训练生成的 checkpoint 目录！")
    exit(1)

# 自动选取最新/最好的检查点
latest_ckpt = max(checkpoints, key=os.path.getmtime)
export_dir = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"

print(f"🚀 [Merge] 正在合并 LoRA 权重: {latest_ckpt}")
print(f"📦 [Export] 导出独立生产模型: {export_dir}")

merge_lora(
    checkpoint_dir=latest_ckpt,
    output_dir=export_dir
)

print(f"✅ 模型合并完成！已成功产出独立就绪模型: {export_dir}")
SCRIPT_MERGE

# ==============================================================================
# 4. 主控监听与串联调度进程 (master_pipeline.sh)
# ==============================================================================
cat << 'SCRIPT_MASTER' > master_pipeline.sh
#!/bin/bash

LOG_FILE="./auto_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动无人值守数据与训练总线监听"
echo "================================================================="

# 阶段一：等待蒸馏进程自然结束
while pgrep -f "distill_production.py" > /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 蒸馏任务运行中，每 5 分钟轮询一次..."
    sleep 300
done

echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] 阶段 1 完成：数据蒸馏已结束，等待 15 秒释放 CUDA 显存..."
sleep 15

# 阶段二：数据质检与验证集切分
echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] 阶段 2 启动：执行数据质量检验与 95:5 切分..."
if ! python prepare_stage2_data.py; then
    echo "❌ 数据质检或切分失败，流水线中止！请检查日志。"
    exit 1
fi

# 阶段三：启动 SFT 训练
echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] 阶段 3 启动：拉起 ms-swift 生产级 LoRA 微调..."
if ! ./run_stage2_sft.sh > ./train_stage2.log 2>&1; then
    echo "❌ 模型训练发生异常！请检查 ./train_stage2.log 获取报错信息。"
    exit 1
fi

# 阶段四：合并 LoRA 导出终版权重
echo "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] 阶段 4 启动：合并 LoRA 权重，导出 Qwen3-VL-8B-RPRM-v2..."
python merge_stage2_lora.py

echo "================================================================="
echo "🎊 [$(date '+%Y-%m-%d %H:%M:%S')] 全流程闭环完成！生产级 PRM 判定模型已就绪。"
echo "================================================================="
SCRIPT_MASTER
chmod +x master_pipeline.sh
