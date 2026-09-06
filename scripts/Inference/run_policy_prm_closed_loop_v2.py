import re
import json
import math
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
VAL_FILE = "./multimodal_data/val_stage2.jsonl"

print("=" * 70)
print("🚀 [1/3] 正在载入 Policy 策略模型 (SFT)...")
print("=" * 70)
policy_proc = AutoProcessor.from_pretrained(POLICY_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
policy_model = AutoModelForImageTextToText.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
)
policy_model.eval()

print("=" * 70)
print("🚀 [2/3] 正在载入 PRM 判定模型 (v2)...")
print("=" * 70)
prm_proc = AutoProcessor.from_pretrained(PRM_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
prm_model = AutoModelForImageTextToText.from_pretrained(
    PRM_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
)
prm_model.eval()

prm_tok = prm_proc.tokenizer
yes_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["Yes", " Yes", "yes", " yes"]]))
no_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["No", " No", "no", " no"]]))

def generate_solution(image_path, question, temperature=0.7):
    prompt_text = f"{question}\nSolve this geometry problem step by step clearly. Conclude with 'Answer: <choice>'."
    messages = [
        {"role": "system", "content": "You are a professional geometry solver. Solve problems step by step with rigorous mathematical proof."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    text = policy_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = policy_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    do_sample = temperature > 0
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, 
            max_new_tokens=1024, # 扩展到 1024，杜绝截断
            temperature=temperature if do_sample else 1.0, 
            do_sample=do_sample,
            top_p=0.9 if do_sample else 1.0
        )
    gen_tokens = out[0, inputs.input_ids.shape[1]:]
    return policy_proc.decode(gen_tokens, skip_special_tokens=True)

def score_step_prm(image_path, question, prev_steps, now_step):
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
        {"role": "system", "content": "You are an expert multimodal mathematics teacher. Your task is to evaluate the correctness of the 'Now Step' with a concise, rigorous reasoning chain."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    text = prm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = prm_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = prm_model.generate(**inputs, max_new_tokens=300, return_dict_in_generate=True, output_scores=True)
        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        
        decision_idx = None
        for idx in range(len(gen_tokens) - 1, -1, -1):
            tok_id = gen_tokens[idx].item()
            if tok_id in yes_ids or tok_id in no_ids:
                decision_idx = idx
                break
        if decision_idx is None:
            decision_idx = max(0, len(outputs.scores) - 2)

        decision_logits = outputs.scores[decision_idx][0]
        max_yes = max([decision_logits[idx].item() for idx in yes_ids])
        max_no = max([decision_logits[idx].item() for idx in no_ids])
        probs = F.softmax(torch.tensor([max_no, max_yes]), dim=-1)
        p_yes = probs[1].item()

    return p_yes

def split_into_steps(text):
    # 按照换行和编号拆解
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    steps = []
    for l in raw_lines:
        cleaned = re.sub(r"^(Step\s*\d+[:.]?|\d+[\.)]|-|\*)\s*", "", l).strip()
        # 过滤掉无实质推导意义的短语
        if len(cleaned) > 10 and not cleaned.lower().startswith("let's solve"):
            steps.append(cleaned)
    return steps if len(steps) >= 2 else [text]

with open(VAL_FILE, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

img = sample["images"][0]
prompt_raw = sample["messages"][1]["content"]
q_match = re.search(r"Question:\s*(.*?)(?:\nPrevious Steps:|$)", prompt_raw, re.DOTALL)
question = q_match.group(1).strip() if q_match else "Find the required geometric measurement from the figure."

print(f"\n🖼️ 评测图像: {img}")
print(f"❓ 几何题目: {question}\n")

# 并发采样 N=3 条解答
N = 3
candidates = []
print(f"👉 [Step 1] 正在由策略模型并发采样 N={N} 条完整解题分支 (max_new_tokens=1024)...")
for k in range(N):
    sol = generate_solution(img, question, temperature=0.7)
    candidates.append(sol)
    print(f"   • 候选路径 #{k+1} 采样完毕 (长度: {len(sol)} 字符)")

print("\n👉 [Step 2] PRM 正在逐步评估每条路径的逻辑与计算...")
ranked_results = []
for idx, sol in enumerate(candidates, 1):
    steps = split_into_steps(sol)
    print(f"\n🔍 评估路径 #{idx} (共解析出 {len(steps)} 个有效推导步):")
    step_scores = []
    prev_accum = []
    for s_idx, st in enumerate(steps, 1):
        prev_str = " ".join([f"({i}) {s}" for i, s in enumerate(prev_accum, 1)]) if prev_accum else "None"
        p_val = score_step_prm(img, question, prev_str, st)
        if math.isnan(p_val):
            p_val = 0.0
        step_scores.append(p_val)
        prev_accum.append(st)
        mark = "✅" if p_val >= 0.8 else ("⚠️" if p_val >= 0.4 else "❌")
        print(f"   [{mark}] 步 {s_idx}: P(Yes)={p_val:.4f} | {st[:55]}...")
    
    min_s = min(step_scores) if step_scores else 0.0
    avg_s = sum(step_scores) / max(1, len(step_scores))
    ranked_results.append({
        "id": idx,
        "solution": sol,
        "min_score": min_s,
        "avg_score": avg_s,
        "step_count": len(steps)
    })

# 安全排序：优先看木桶短板，再看均分
ranked_results.sort(key=lambda x: (round(x["min_score"], 4), round(x["avg_score"], 4)), reverse=True)

print("\n" + "=" * 80)
print(f"{'🏆 策略闭环重排序结果 V2 (Policy-PRM Closed Loop)':^70}")
print("=" * 80)
print(f"{'排名':<4} | {'路径编号':<10} | {'步数':<6} | {'Min-Score (短板分)':<18} | {'Avg-Score (均分)':<16}")
print("-" * 80)
for rank, r in enumerate(ranked_results, 1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else f"#{rank}")
    print(f"{medal:<4} | Path #{r['id']:<7} | {r['step_count']:<6} | {r['min_score']:<18.4f} | {r['avg_score']:<16.4f}")
print("=" * 80)

winner = ranked_results[0]
print(f"\n🎉 【PRM 最终选出的胜出解法 (Path #{winner['id']})】:")
print("-" * 80)
print(winner["solution"])
print("-" * 80)
