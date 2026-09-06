import os
import json
import re
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_50k.jsonl"
OUTPUT_FILE = "./multimodal_data/sft_distilled_train.jsonl"
PROGRESS_FILE = "./multimodal_data/distilled_done.ids"
BATCH_SIZE = 128  # A800 80GB 显存充裕，一次并发处理 4 个多模态样本

SYSTEM_PROMPT = (
    "You are an expert multimodal mathematics teacher. "
    "Your task is to carefully analyze the Image and evaluate the correctness of the 'Now Step' "
    "by providing a rigorous, step-by-step mathematical and visual reasoning chain."
)

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
    if len(text.strip()) < 180:
        return False
    return bool(re.search(r"(Visual|Geometric|Now Step|Calculation|Data Source)", text, re.IGNORECASE))

def load_processed_ids():
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if item:
                    processed.add(item)
    return processed

def main():
    base_model_cache = glob.glob("/hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/*")
    base_model_path = base_model_cache[0] if base_model_cache else "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
    model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

    print("🚀 正在载入多模态推理模型 (bfloat16 原生加速)...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # Batch 推理必须采用左侧填充

    model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    done_ids = load_processed_ids()
    print(f"[*] 读取进度：已有 {len(done_ids)} 条样本完成记录。")

    candidates = []
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sid = str(item.get("id", item.get("image_path")))
                if sid not in done_ids:
                    candidates.append(item)

    print(f"[*] 待处理全量候选样本数: {len(candidates)} 条\n")
    if not candidates:
        print("🎉 全部数据已蒸馏质检完毕！")
        return

    # 按 BATCH_SIZE 进行批处理推理
    for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Batch Distilling"):
        batch_items = candidates[i : i + BATCH_SIZE]
        batch_messages = []
        batch_prompts = []

        for item in batch_items:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=item["question"],
                previous_steps=item["previous_steps"],
                now_step=item["now_step"]
            )
            batch_prompts.append(user_prompt)
            batch_messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": item["image_path"]},
                    {"type": "text", "text": user_prompt.replace("<image>", "").strip()}
                ]}
            ])

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)

        inputs = processor(
            text=texts,
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
            cot_responses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # 质检过滤与实时落盘
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, open(PROGRESS_FILE, "a", encoding="utf-8") as id_f:
            for item, user_prompt, cot_resp in zip(batch_items, batch_prompts, cot_responses):
                sid = str(item.get("id", item.get("image_path")))
                id_f.write(sid + "\n")

                cot_resp = cot_resp.strip()
                pred_label = extract_verdict(cot_resp)
                if not pred_label or pred_label != item["ground_truth_label"]:
                    continue
                if not is_valid_cot(cot_resp):
                    continue

                clean_record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>{user_prompt.replace('<image>', '').strip()}"},
                        {"role": "assistant", "content": cot_resp}
                    ],
                    "images": [item["image_path"]]
                }
                out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

            out_f.flush()
            id_f.flush()

if __name__ == "__main__":
    main()
