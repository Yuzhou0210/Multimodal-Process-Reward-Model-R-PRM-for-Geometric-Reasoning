import json
import os
import random

# ================= 路径配置 =================
INPUT_FILE = "./multimodal_data/filtered_mm_seed_data.jsonl"
IMAGES_DIR = "./multimodal_data/images"  # 当前服务器上存放图片的目录
OUTPUT_TRAIN = "./multimodal_data/sft_train.jsonl"
OUTPUT_VAL = "./multimodal_data/sft_val.jsonl"

MM_SYSTEM_PROMPT = "You are an expert multimodal mathematics teacher. Your task is to verify the correctness of the \"Now Step\" based on the provided Image, Question, and context."

def resolve_image_path(raw_path):
    """
    自适应定位图片：优先判断原路径，原路径不存在则按文件名在当前 IMAGES_DIR 中查找
    """
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)

    # 提取文件名（如 geo_12.png）
    filename = os.path.basename(raw_path)
    current_server_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(current_server_path):
        return os.path.abspath(current_server_path)

    return None

def convert_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到标注文件: {INPUT_FILE}")
        return

    # 检查图片文件夹是否存在
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ 找不到图片目录: {IMAGES_DIR}，请确认是否已将 images 文件夹上传至服务器。")
        return

    raw_data = []
    corrupted_lines = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                raw_data.append(item)
            except json.JSONDecodeError:
                corrupted_lines += 1

    print(f"[*] 读取有效 JSON 行: {len(raw_data)} 条 (跳过损坏行: {corrupted_lines})")

    formatted_data = []
    missing_images = 0

    for item in raw_data:
        raw_img_path = item.get("image_path", "")
        real_img_path = resolve_image_path(raw_img_path)

        if not real_img_path:
            missing_images += 1
            continue

        user_content = f"""Question: {item['question']}
Previous Steps: {item.get('previous_steps', 'None')}
Now Step: {item['now_step']}

Conclusion:
Conclude your evaluation. At the very end, strictly output your verdict in the following format:
Verification: Is the step correct (Yes/No)? X
(where X is either Yes or No).
Reply:"""

        sample = {
            "messages": [
                {"role": "system", "content": MM_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>{user_content}"}
            ],
            "images": [real_img_path],
            "response": item["generated_cot"]
        }
        formatted_data.append(sample)

    if not formatted_data:
        print(f"❌ 依然未能匹配到任何图片！")
        print(f"  - 示例原路径: {raw_data[0].get('image_path') if raw_data else 'None'}")
        print(f"  - 当前服务器尝试查找路径: {os.path.abspath(IMAGES_DIR)}")
        return

    random.seed(42)
    random.shuffle(formatted_data)
    split_idx = int(len(formatted_data) * 0.9)
    train_set = formatted_data[:split_idx]
    val_set = formatted_data[split_idx:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 转换与切分成功！")
    print(f"  - 成功转换样本: {len(formatted_data)} 条")
    print(f"  - 训练集: {len(train_set)} 条 -> {OUTPUT_TRAIN}")
    print(f"  - 验证集: {len(val_set)} 条 -> {OUTPUT_VAL}")
    if missing_images > 0:
        print(f"  - 仍缺失图片: {missing_images} 条")

if __name__ == "__main__":
    convert_data()