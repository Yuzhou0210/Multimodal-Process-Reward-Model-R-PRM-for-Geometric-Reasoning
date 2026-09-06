import os
import json
import re
import glob
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_50k.jsonl"
OUTPUT_FILE = "./multimodal_data/distilled_sft_sample.jsonl"
TEST_LIMIT = 50

SYSTEM_PROMPT = (
    "You are an expert multimodal mathematics teacher. "
    "Your task is to carefully analyze the Image and evaluate the correctness of the 'Now Step' "
    "by providing a rigorous, step-by-step mathematical and visual reasoning chain."
)

# 强制约束 4 段式结构化 CoT 输出
USER_PROMPT_TEMPLATE = """<image>Question: {question}
Previous Steps: {previous_steps}
Now Step: {now_step}

Please evaluate this step thoroughly. Your response MUST strictly follow this structure:
Analysis:
1. Visual & Geometric Analysis: (Analyze key elements, shapes, measurements, or relations shown in the image)
2. Now Step Analysis: (Evaluate whether the logic and claims in the Now Step are mathematically sound)
3. Data Source Analysis: (Check if values, formulas, or theorems cited originate correctly from the problem/image)
4. Calculation Analysis: (Verify the arithmetic, algebraic, or trigonometric calculations)

Conclusion:
Summarize the findings.

Verification: Is the step correct (Yes/No)? [Yes/No]"""

def extract_verdict(text):
    match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).capitalize() if match else None

def is_valid_cot(text):
    """质检门禁：必须包含完整 CoT 分析，且字符数充分"""
    if len(text.strip()) < 180:
        return False
    # 必须至少包含两个核心分析板块
    has_analysis = bool(re.search(r"(Visual|Geometric|Now Step|Calculation|Data Source)", text, re.IGNORECASE))
    return has_analysis

def main():
    base_model_cache = glob.glob("/hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/*")
    base_model_path = base_model_cache[0] if base_model_cache else "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
    model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

    print("🚀 正在载入蒸馏推理引擎...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    candidates = []
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= TEST_LIMIT:
                break
            if line.strip():
                candidates.append(json.loads(line))

    print(f"[*] 成功读取 {len(candidates)} 条候选样本，开始执行多模态蒸馏与质检...\n")

    retained_records = []
    mismatch_count = 0
    short_cot_count = 0
    format_error_count = 0

    for item in tqdm(candidates, desc="Distilling"):
        img_path = item["image_path"]
        expected_label = item["ground_truth_label"]
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=item["question"],
            previous_steps=item["previous_steps"],
            now_step=item["now_step"]
        )

        formatted_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_prompt.replace("<image>", "").strip()}
            ]}
        ]

        text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(formatted_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda:0")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            cot_response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        # 1. 提取判决
        pred_label = extract_verdict(cot_response)
        if not pred_label:
            format_error_count += 1
            continue

        # 2. 检查 CoT 深度与长度
        if not is_valid_cot(cot_response):
            short_cot_count += 1
            continue

        # 3. 标签一致性过滤
        if pred_label != expected_label:
            mismatch_count += 1
            continue

        # 4. 组装合规样本
        retained_records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>{user_prompt.replace('<image>', '').strip()}"},
                {"role": "assistant", "content": cot_response}
            ],
            "images": [img_path]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in retained_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("📊 蒸馏与质检测试报告")
    print("=" * 60)
    print(f"• 处理总数          : {len(candidates)}")
    print(f"• 质检通过并留存    : {len(retained_records)} (留存率: {len(retained_records)/len(candidates)*100:.1f}%)")
    print(f"• 标签冲突丢弃      : {mismatch_count}")
    print(f"• CoT 过短/跳步丢弃 : {short_cot_count}")
    print(f"• 格式未识别丢弃    : {format_error_count}")
    print(f"• 输出文件保存至    : {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
