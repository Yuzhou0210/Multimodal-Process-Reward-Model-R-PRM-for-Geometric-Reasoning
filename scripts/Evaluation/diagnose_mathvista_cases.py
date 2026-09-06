import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import json
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"

print("🚀 载入模型...")
policy_proc = AutoProcessor.from_pretrained(POLICY_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
policy_model = AutoModelForImageTextToText.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
).eval()

prm_proc = AutoProcessor.from_pretrained(PRM_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
prm_model = AutoModelForImageTextToText.from_pretrained(
    PRM_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
).eval()

prm_tok = prm_proc.tokenizer
yes_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["Yes", " Yes", "yes", " yes"]]))
no_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["No", " No", "no", " no"]]))

print("📥 加载 MathVista Geometry 数据...")
dataset = load_dataset("AI4Math/MathVista", split="testmini")
geo_data = []
for item in dataset:
    m = item.get("metadata", {})
    if any("geometry" in str(x).lower() for x in [m.get("subject"), m.get("task"), item.get("category")]):
        if item.get("decoded_image") is not None:
            geo_data.append(item)

# 仅抽查前 5 道题
sample_cases = geo_data[:5]

def generate_solution(image, question, temp=0.7):
    prompt_text = f"{question}\nSolve this geometry problem step by step. Conclude with 'Answer: <choice or value>'."
    messages = [
        {"role": "system", "content": "You are an expert geometry solver."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
    ]
    text = policy_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = policy_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = policy_model.generate(**inputs, max_new_tokens=600, temperature=temp, do_sample=(temp > 0))
    return policy_proc.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def score_step_with_cot(image, question, prev_steps, now_step):
    user_prompt = f"""Question: {question}
Previous Steps: {prev_steps}
Now Step: {now_step}

Evaluate this step concisely (1-2 sentences per point). Your response MUST strictly follow this structure:
Analysis:
1. Visual & Geometric Analysis: (Key elements, shapes, or measurements shown in the image)
2. Now Step Analysis: (Whether the step's logic is mathematically sound)
3. Data Source Analysis: (Whether numbers, formulas, or theorems cited are accurate)
4. Calculation Analysis: (Verification of arithmetic or algebraic calculations)

Conclusion:
Summary of findings.

Verification: Is the step correct (Yes/No)? ["""

    messages = [
        {"role": "system", "content": "You are an expert mathematics verifier."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
    ]
    text = prm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = prm_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = prm_model.generate(**inputs, max_new_tokens=250, return_dict_in_generate=True, output_scores=True)
        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        prm_text = prm_proc.decode(gen_tokens, skip_special_tokens=True)
        
        decision_idx = next((i for i in range(len(gen_tokens)-1, -1, -1) if gen_tokens[i].item() in yes_ids or gen_tokens[i].item() in no_ids), max(0, len(outputs.scores)-2))
        logits = outputs.scores[decision_idx][0]
        p_yes = F.softmax(torch.tensor([max([logits[i].item() for i in no_ids]), max([logits[i].item() for i in yes_ids])]), dim=-1)[1].item()
    return p_yes, prm_text

def split_into_steps(text):
    cleaned = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    blocks = re.split(r"(?=(?:\*\*Step\s*\d+|Step\s*\d+:))", cleaned) if re.search(r"(?:\*\*Step\s*\d+|Step\s*\d+:)", cleaned) else re.split(r"\n\s*\n", cleaned)
    steps = [b.strip() for b in blocks if len(b.strip()) > 10 and not re.match(r"^(Let's\s+solve|Here\s+is)", b.strip(), re.I)]
    return steps if steps else [text.strip()]

def extract_answer(text):
    match = re.search(r"Answer\s*:\s*([A-Za-z0-9\.\-\_\/]+)", text, re.IGNORECASE)
    if match: return match.group(1).strip().strip(".").upper()
    tail = text.strip().split("\n")[-1]
    m = re.search(r"\b([A-E])\b", tail)
    return m.group(1) if m else "UNKNOWN"

for idx, item in enumerate(sample_cases, 1):
    q = item["question"]
    gt = item["answer"]
    img = item["decoded_image"]
    
    print("\n" + "#" * 80)
    print(f"📌 [Case {idx}/5] 题干: {q}")
    print(f"🎯 标准答案 (GT): {gt}")
    print("#" * 80)
    
    # 观察 1 条 Greedy 和 1 条采样解
    for mode, temp in [("Greedy (T=0)", 0.0), ("Sampled (T=0.7)", 0.7)]:
        sol = generate_solution(img, q, temp=temp)
        pred = extract_answer(sol)
        print(f"\n--- 模式: {mode} | 抽取答案: [{pred}] | 是否吻合GT: {pred == str(gt).strip().upper()} ---")
        print(f"【生成全文】:\n{sol}\n")
        
        steps = split_into_steps(sol)
        print(f"【切分步骤数】: {len(steps)} 步")
        
        # 审计前 2 步作为代表观察 PRM 的判断
        prevs = []
        for s_i, st in enumerate(steps[:2], 1):
            p_val, cot = score_step_with_cot(img, q, " ".join(prevs) if prevs else "None", st)
            print(f"  👉 Step {s_i} (P(Yes)={p_val:.4f}): {st[:60]}...")
            print(f"     PRM 审核理由: {cot.replace(chr(10), ' ')[:120]}...")
            prevs.append(st)
