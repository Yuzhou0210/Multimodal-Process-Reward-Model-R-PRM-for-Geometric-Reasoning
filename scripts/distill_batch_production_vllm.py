import os
import json
import re
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_50k.jsonl"
OUTPUT_FILE = "./multimodal_data/sft_distilled_train.jsonl"
PROGRESS_FILE = "./multimodal_data/distilled_done.ids"

# ============ A800 优化配置 ============
BATCH_SIZE = 128
MAX_NUM_SEQS = 256
GPU_MEMORY_UTILIZATION = 0.95
MAX_TOKENS = 1024
TEMPERATURE = 0.2

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
    print("🚀 正在载入多模态推理模型 (vLLM A800 优化)...")
    
    model_path = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
    
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 10},
        tensor_parallel_size=1,
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stop_token_ids=None,
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    
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
    
    total_processed = 0
    start_time = time.time()
    
    for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Batch Distilling"):
        batch_items = candidates[i:i + BATCH_SIZE]
        
        batch_messages = []
        for item in batch_items:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=item["question"],
                previous_steps=item["previous_steps"],
                now_step=item["now_step"]
            )
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
        
        vllm_inputs = []
        for text, images in zip(texts, image_inputs):
            vllm_inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": images}
            })
        
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        cot_responses = [output.outputs[0].text for output in outputs]
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, open(PROGRESS_FILE, "a", encoding="utf-8") as id_f:
            for item, cot_resp in zip(batch_items, cot_responses):
                sid = str(item.get("id", item.get("image_path")))
                id_f.write(sid + "\n")
                
                cot_resp = cot_resp.strip()
                pred_label = extract_verdict(cot_resp)
                if not pred_label or pred_label != item["ground_truth_label"]:
                    continue
                if not is_valid_cot(cot_resp):
                    continue
                
                # 修复：先构建 user_prompt
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    question=item['question'],
                    previous_steps=item['previous_steps'],
                    now_step=item['now_step']
                )
                
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
        
        total_processed += len(batch_items)
        
        if total_processed % 500 == 0:
            elapsed = time.time() - start_time
            speed = total_processed / elapsed
            eta = (len(candidates) - total_processed) / speed
            print(f"\n📊 速度: {speed:.2f} 条/秒, 预计剩余: {eta/3600:.2f} 小时")

if __name__ == "__main__":
    main()