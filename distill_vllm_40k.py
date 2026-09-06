
import os
import json
import re
import gc
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_50k.jsonl"
OUTPUT_FILE = "./multimodal_data/sft_distilled_train.jsonl"
PROGRESS_FILE = "./multimodal_data/distill_vllm_done.ids"

BATCH_CHUNK_SIZE = 500  # 每批送入 vLLM 调度器并发生成的规模
MAX_MODEL_LEN = 3072

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
    processed_ids = load_processed_ids()
    print(f"[*] 检测到历史进度，已完成样本: {len(processed_ids)} 条。")

    # 1. 过滤待处理样本
    all_candidates = []
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sample_id = data.get("id", data.get("image_path"))
                if sample_id not in processed_ids:
                    all_candidates.append(data)

    print(f"[*] 待处理全量候选样本: {len(all_candidates)} 条")
    if not all_candidates:
        print("🎉 所有候选样本已全部蒸馏完成！")
        return

    # 2. 初始化 Processor 与 vLLM 引擎
    print("🚀 正在载入 vLLM 多模态高并发引擎...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 1}
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=1024
    )

    # 3. 分批 Chunk 处理，防止一次性预载所有 PIL 图像耗尽内存
    total_chunks = (len(all_candidates) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
    
    for chunk_idx in range(total_chunks):
        chunk = all_candidates[chunk_idx * BATCH_CHUNK_SIZE : (chunk_idx + 1) * BATCH_CHUNK_SIZE]
        inputs = []
        valid_items = []

        # 组装 prompt 与图像输入
        for item in chunk:
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                continue
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

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

            prompt_text = processor.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": pil_img}
            })
            valid_items.append((item, user_prompt))

        if not inputs:
            continue

        print(f"\n[Chunk {chunk_idx + 1}/{total_chunks}] 正在并发推理 {len(inputs)} 条样本...")
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        retained_count = 0
        newly_done_ids = []

        # 4. 质检对齐与实时落盘
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, open(PROGRESS_FILE, "a", encoding="utf-8") as id_f:
            for (item, user_prompt), out in zip(valid_items, outputs):
                sample_id = item.get("id", item.get("image_path"))
                newly_done_ids.append(sample_id)

                cot_response = out.outputs[0].text.strip()
                pred_label = extract_verdict(cot_response)

                # 质检过滤门禁
                if not pred_label or pred_label != item["ground_truth_label"]:
                    continue
                if not is_valid_cot(cot_response):
                    continue

                clean_record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>{user_prompt.replace('<image>', '').strip()}"},
                        {"role": "assistant", "content": cot_response}
                    ],
                    "images": [item["image_path"]]
                }
                out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                retained_count += 1

            for s_id in newly_done_ids:
                id_f.write(s_id + "\n")
            out_f.flush()
            id_f.flush()

        print(f"[Chunk {chunk_idx + 1}] 完成。本批合格留存: {retained_count}/{len(inputs)} 条")
        gc.collect()

if __name__ == "__main__":
    main()
