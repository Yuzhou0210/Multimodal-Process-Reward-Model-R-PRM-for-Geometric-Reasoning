import os
import json
import random
import re
from PIL import Image
from tqdm import tqdm

# ================= 路径配置 =================
GEO170K_DIR = "./geo170k_raw"
IMG_DIR = os.path.join(GEO170K_DIR, "images")
JSON_PATH = os.path.join(GEO170K_DIR, "qa_tuning.json")

OUTPUT_DIR = "./multimodal_data"
OUTPUT_CANDIDATE_POOL = os.path.join(OUTPUT_DIR, "unlabeled_candidate_pool_50k.jsonl")
TARGET_NUM_SAMPLES = 40000  # 目标生成 4 万条图文对

os.makedirs(OUTPUT_DIR, exist_ok=True)


def perturb_number(step_text):
    """
    提取步骤中的数字并随机扰动制造逻辑错误步骤
    增加对特殊格式的处理，避免死循环或异常
    """
    if not step_text or len(step_text) > 2000:
        return step_text + " (incorrect inference)"
    
    # 匹配整数、小数、分数、科学计数法
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", step_text, re.IGNORECASE)
    if not numbers:
        # 如果没有数字，尝试替换关键词制造错误
        keywords = ["等于", "大于", "小于", "垂直", "平行", "相等"]
        for kw in keywords:
            if kw in step_text:
                return step_text.replace(kw, "错误地判断为" + kw, 1)
        return step_text + " (contradicting step)"
    
    # 随机选一个数字扰动
    target_num = random.choice(numbers)
    try:
        val = float(target_num)
        delta = random.choice([-3, -2, -1, 1, 2, 3])
        new_val = val + delta
        if new_val <= 0 and val > 0:
            new_val = val + 1
        new_val_str = str(int(new_val)) if new_val.is_integer() else f"{new_val:.2f}"
        return step_text.replace(target_num, new_val_str, 1)
    except Exception:
        return step_text + " (contradicting step)"


def split_into_steps(solution_text):
    """
    将长解题答案拆分为独立步骤
    支持多种分隔符：换行、句号、分号、数字序号（1. 2. 等）
    """
    if not solution_text:
        return []
    
    # 先按换行拆分
    lines = [s.strip() for s in solution_text.split('\n') if s.strip()]
    
    # 如果换行拆分后步骤太少，尝试按句号/分号/问号拆分
    if len(lines) < 2:
        sentences = re.split(r'[。；；！？!?;]', solution_text)
        lines = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    # 如果仍然太少，尝试按数字序号拆分（如 1. 2.）
    if len(lines) < 2:
        numbered = re.split(r'\d+\.\s*', solution_text)
        lines = [s.strip() for s in numbered if len(s.strip()) > 5]
    
    # 过滤太短的片段（可能是噪音）
    lines = [s for s in lines if len(s) > 5]
    return lines


def clean_and_build_pool():
    """主处理函数"""
    if not os.path.exists(JSON_PATH):
        print(f"❌ 找不到标注文件: {JSON_PATH}，请确认 Geo170K 已正确解压。")
        return

    print("1. 正在读取原始标注文件...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始条目总数: {len(data)}")

    random.seed(42)
    random.shuffle(data)

    records = []
    skipped_img = 0
    skipped_no_steps = 0
    skipped_no_question = 0
    error_count = 0

    print("2. 正在清洗、切分步骤并构造正负候选对...")
    
    for idx, item in enumerate(tqdm(data, desc="处理进度")):
        # 达到目标数量则停止
        if len(records) >= TARGET_NUM_SAMPLES:
            break
        
        # 每处理500条打印一次进度
        if idx % 500 == 0 and idx > 0:
            print(f"  已处理 {idx} 条，当前记录数: {len(records)}")
        
        try:
            # ----- 校验图像 -----
            img_name = item.get("image")
            if not img_name:
                skipped_img += 1
                continue
            img_full_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_full_path):
                skipped_img += 1
                continue

            # ----- 提取题干 -----
            question_text = item.get("problem", "")
            if not question_text:
                # 尝试从 conversations 中提取
                convs = item.get("conversations", [])
                if convs and isinstance(convs, list):
                    question_text = convs[0].get("value", "")
            question_text = question_text.replace("<image>", "").strip()
            if not question_text:
                skipped_no_question += 1
                continue

            # ----- 提取解题过程 -----
            solution_text = item.get("solution", "")
            if not solution_text:
                convs = item.get("conversations", [])
                if convs and isinstance(convs, list) and len(convs) > 1:
                    solution_text = convs[1].get("value", "")
            if not solution_text:
                skipped_no_steps += 1
                continue

            # ----- 拆分解题步骤 -----
            raw_steps = split_into_steps(solution_text)
            if len(raw_steps) < 1:
                skipped_no_steps += 1
                continue

            # ----- 选取当前步，构造 Previous Steps -----
            step_idx = random.randint(0, len(raw_steps) - 1)
            prev_steps_list = raw_steps[:step_idx]
            prev_steps = " ".join(prev_steps_list) if prev_steps_list else "None"
            true_now_step = raw_steps[step_idx]

            base_id = item.get("id", f"geo_{idx}")

            # ----- 构造正样本 -----
            records.append({
                "id": f"geo170k_{base_id}_pos",
                "image_path": os.path.abspath(img_full_path),
                "question": question_text,
                "previous_steps": prev_steps,
                "now_step": true_now_step,
                "ground_truth_label": "Yes"
            })

            # ----- 构造负样本（数字扰动）-----
            wrong_now_step = perturb_number(true_now_step)
            records.append({
                "id": f"geo170k_{base_id}_neg",
                "image_path": os.path.abspath(img_full_path),
                "question": question_text,
                "previous_steps": prev_steps,
                "now_step": wrong_now_step,
                "ground_truth_label": "No"
            })

        except Exception as e:
            error_count += 1
            if error_count <= 10:  # 只打印前10个错误
                print(f"  ⚠️ 处理条目 {item.get('id', 'N/A')} 时出错: {e}")
            continue

    # ----- 写入结果 -----
    print(f"\n3. 正在写入大规模候选池文件...")
    with open(OUTPUT_CANDIDATE_POOL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- 统计报告 -----
    print(f"\n🎉 候选池构建完成！")
    print(f"有效生成条目: {len(records)} 条")
    print(f"跳过失效图片: {skipped_img} 张")
    print(f"跳过无解题步骤: {skipped_no_steps} 条")
    print(f"跳过无题干: {skipped_no_question} 条")
    print(f"处理异常数: {error_count} 条")
    print(f"文件保存至: {os.path.abspath(OUTPUT_CANDIDATE_POOL)}")


if __name__ == "__main__":
    clean_and_build_pool()