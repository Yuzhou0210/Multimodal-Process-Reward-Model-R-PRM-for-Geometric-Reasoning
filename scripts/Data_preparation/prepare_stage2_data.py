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
