import os
import json
import re
import glob
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

# ==================== 路径与超参配置 ====================
RAW_POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_50k.jsonl"
OUTPUT_FILE = "./multimodal_data/sft_distilled_train.jsonl"
PROGRESS_FILE = "./multimodal_data/distilled_done.ids"
BATCH_SIZE = 16          
MAX_NEW_TOKENS = 768     # 充足预算，彻底杜绝尾部截断

SYSTEM_PROMPT = (
    "You are an expert multimodal mathematics teacher. "
    "Your task is to evaluate the correctness of the 'Now Step' with a concise, rigorous reasoning chain."
)

# 增加篇幅约束，保证在 300~450 tokens 内高效闭环
USER_PROMPT_TEMPLATE = """<image>Question: {question}
Previous Steps: {previous_steps}
Now Step: {now_step}

Evaluate this step concisely (1-2 sentences per point). Your response MUST strictly follow this structure:
Analysis:
1. Visual & Geometric Analysis: (Key elements, shapes, or measurements shown in the image)
2. Now Step Analysis: (Whether the step's logic is mathematically sound)
3. Data Source Analysis: (Whether numbers, formulas, or theorems cited are accurate)
4. Calculation Analysis: (Verification of arithmetic or algebraic calculations)

Conclusion:
Summary of findings.

Verification: Is the step correct (Yes/No)? [Yes/No]"""

def is_pure_answer_step(step_text):
    """过滤掉终局选项步 (如 Answer: C, The answer is B)"""
    s = step_text.strip()
    if len(s) <= 12:
        return True
    if re.search(r"^(?:Answer\s*:\s*[A-D]|(?:The\s+)?answer\s+is\s+[A-D]\.?)$", s, re.IGNORECASE):
        return True
    if re.match(r"^Answer\s*:\s*[A-D]\s*\(", s, re.IGNORECASE):
        return True
    return False

def extract_verdict(text):
    """多层级鲁棒正则提取 Yes / No"""
    match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize()
    tail_words = text.strip().split()[-4:]
    for w in tail_words:
        clean = re.sub(r"[^a-zA-Z]", "", w).capitalize()
        if clean in ["Yes", "No"]:
            return clean
    return None

def is_valid_cot(text):
    """质检四段式结构与基本长度"""
    if len(text.strip()) < 150:
        return False
    return bool(re.search(r"(Visual|Geometric|Now Step|Calculation|Data Source)", text, re.IGNORECASE))

def load_processed_ids():
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    processed.add(sid)
    return processed

def main():
    base_model_cache = glob.glob("/hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/*")
    base_model_path = base_model_cache[0] if base_model_cache else "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"

    print("🚀 正在初始化多模态模型与视觉预处理器...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28
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
    
    print("📋 正在筛选过滤纯推导步骤...")
    with open(RAW_POOL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            step = item.get("now_step", "")
            
            # 自动剔除纯答案与无效短句
            if is_pure_answer_step(step):
                continue
                
            sid = str(item.get("id", item.get("image_path")))
            if sid not in done_ids:
                candidates.append(item)

    print(f"[*] 待处理纯推导候选样本: {len(candidates)} 条 | Batch Size: {BATCH_SIZE}")

    if not candidates:
        print("🎉 全部纯中间步骤已处理完毕！")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, open(PROGRESS_FILE, "a", encoding="utf-8") as id_f:
        for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Distilling Production"):
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
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.2
                )
                gen_ids_trimmed = [o[len(in_ids):] for in_ids, o in zip(inputs.input_ids, gen_ids)]
                cot_outputs = processor.batch_decode(
                    gen_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            # 质检过滤与落盘
            for it, u_p, cot in zip(batch, prompts_list, cot_outputs):
                sid = str(it.get("id", it.get("image_path")))
                id_f.write(sid + "\n")

                resp = cot.strip()
                pred_label = extract_verdict(resp)
                
                # 判定不匹配或非四段式结构直接丢弃
                if not pred_label or pred_label != it["ground_truth_label"]:
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
