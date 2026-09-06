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
OUTPUT_FILE = "./multimodal_data/sft_distilled_train.jsonl"
PROGRESS_FILE = "./multimodal_data/distilled_done.ids"
BATCH_SIZE = 32  # A800 80GB 吞吐最佳 Batch

SYSTEM_PROMPT = "You are an expert multimodal mathematics teacher. Your task is to evaluate the correctness of the 'Now Step' with a step-by-step reasoning chain."

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
                sid = line.strip()
                if sid: processed.add(sid)
    return processed

def main():
    base_model_cache = glob.glob("/hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/*")
    base_model_path = base_model_cache[0] if base_model_cache else "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"

    print("🚀 载入极速原生引擎（FlashAttention-2 + Token 动态约束）...")
    
    # 约束最大/最小像素点，减少 70% 冗余视觉 Token 计算
    processor = AutoProcessor.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=512*28*28
    )
    processor.tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
    model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

    model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    done_ids = load_processed_ids()
    candidates = []
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sid = str(item.get("id", item.get("image_path")))
                if sid not in done_ids:
                    candidates.append(item)

    print(f"[*] 待处理样本量: {len(candidates)} 条 | Batch Size: {BATCH_SIZE}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, open(PROGRESS_FILE, "a", encoding="utf-8") as id_f:
        for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Fast Distilling"):
            batch = candidates[i : i + BATCH_SIZE]
            messages_list, prompts_list = [], []

            for it in batch:
                u_prompt = USER_PROMPT_TEMPLATE.format(
                    question=it["question"],
                    previous_steps=it["previous_steps"],
                    now_step=it["now_step"]
                )
                prompts_list.append(u_prompt)
                messages_list.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": it["image_path"]},
                        {"type": "text", "text": u_prompt.replace("<image>", "").strip()}
                    ]}
                ])

            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
            img_in, vid_in = process_vision_info(messages_list)

            inputs = processor(
                text=texts,
                images=img_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt"
            ).to("cuda:0")

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
                gen_ids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
                cot_outputs = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for it, u_p, cot in zip(batch, prompts_list, cot_outputs):
                sid = str(it.get("id", it.get("image_path")))
                id_f.write(sid + "\n")
                
                resp = cot.strip()
                label = extract_verdict(resp)
                if not label or label != it["ground_truth_label"]:
                    continue
                if not is_valid_cot(resp):
                    continue

                out_f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>{u_p.replace('<image>', '').strip()}"},
                        {"role": "assistant", "content": resp}
                    ],
                    "images": [it["image_path"]]
                }, ensure_ascii=False) + "\n")

            out_f.flush()
            id_f.flush()

if __name__ == "__main__":
    main()
