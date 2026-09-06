import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import math
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
NUM_SAMPLES = 30  # 评测几何题量
N_CANDIDATES = 4  # 每题采样候选数

print("=" * 75)
print("🚀 [1/3] 加载 Policy 解题策略模型与 PRM 过程奖励判定模型...")
print("=" * 75)

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

print("📥 [2/3] 加载 MathVista Geometry 题域数据集...")
dataset = load_dataset("AI4Math/MathVista", split="testmini")
geo_data = []
for item in dataset:
    m = item.get("metadata", {})
    if any("geometry" in str(x).lower() for x in [m.get("subject"), m.get("task"), item.get("category")]):
        if item.get("decoded_image") is not None:
            geo_data.append(item)

eval_subset = geo_data[:NUM_SAMPLES]
print(f"✅ 成功加载 {len(geo_data)} 道几何题目，本次评测前 {len(eval_subset)} 道题\n")

# 1. 鲁棒的答案提取函数（兼容中英文、Markdown 加粗与选项/数值）
def extract_answer_robust(text):
    patterns = [
        r"(?:Answer|答案|故选|应选|结论为)\s*[:：]\s*[*_`]*([A-Za-z0-9\.\-\_\/°度cm]+)",
        r"[*_]{2}(?:Answer|答案)[:：]\s*([A-Za-z0-9\.\-\_\/°度cm]+)[*_]{2}",
        r"(?:等于|=)\s*([0-9\.\-\_\/]+)\s*$"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip(".").upper()
            
    tail_lines = [l.strip() for l in text.strip().split("\n")[-3:] if l.strip()]
    for line in reversed(tail_lines):
        m_choice = re.search(r"\b([A-E])\b", line)
        if m_choice:
            return m_choice.group(1)
        m_num = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m_num:
            return m_num.group(1)
            
    return "UNKNOWN"

# 2. 鲁棒的答案核验函数（去除度数、几何单位、LaTeX 括号，并支持浮点容差）
def check_correctness_robust(pred, gt):
    def clean(s):
        s = str(s).strip()
        s = re.sub(r"\\[a-zA-Z]+", "", s)
        s = re.sub(r"(°|度|cm|mm|m|km|rad|\$|\{|\}|\(|\)|\[|\]|\*)", "", s)
        return s.strip().upper()

    p, g = clean(pred), clean(gt)
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        p_f, g_f = float(p), float(g)
        return math.isclose(p_f, g_f, rel_tol=1e-2) or abs(p_f - g_f) < 1e-3
    except:
        return False

# 3. 策略模型生成解答
def generate_solution(image, question, temp=0.7):
    prompt_text = f"{question}\nSolve this geometry problem step by step clearly. Conclude with 'Answer: <choice or value>'."
    messages = [
        {"role": "system", "content": "You are an expert geometry solver. Solve problems step by step with mathematical rigor."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    text = policy_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = policy_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    do_sample = temp > 0
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, 
            max_new_tokens=800, 
            temperature=temp if do_sample else 1.0, 
            do_sample=do_sample, 
            top_p=0.9 if do_sample else 1.0
        )
    return policy_proc.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

# 4. PRM 单步打分（Log-Sum-Exp 数值防下溢与防 NaN 保护）
def score_step_prm(image, question, prev_steps, now_step):
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
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    text = prm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = prm_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = prm_model.generate(**inputs, max_new_tokens=250, return_dict_in_generate=True, output_scores=True)
        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        
        decision_idx = next(
            (i for i in range(len(gen_tokens)-1, -1, -1) if gen_tokens[i].item() in yes_ids or gen_tokens[i].item() in no_ids),
            max(0, len(outputs.scores)-2)
        )
        logits = outputs.scores[decision_idx][0]
        max_yes = max([logits[idx].item() for idx in yes_ids])
        max_no = max([logits[idx].item() for idx in no_ids])
        
        # 平移 Softmax，杜绝指数爆炸与下溢成 NaN
        shift = max(max_yes, max_no)
        e_yes = math.exp(max_yes - shift)
        e_no = math.exp(max_no - shift)
        denom = e_yes + e_no
        p_yes = e_yes / denom if denom > 0 else 0.5

    if math.isnan(p_yes):
        p_yes = 0.0
    return p_yes

# 5. 步骤切分（自动过滤无实质推导的客套开场白）
def split_into_steps(text):
    cleaned = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    cleaned = re.sub(r"(?m)^[ \t]*>[ \t]*", "", cleaned)
    
    # 优先按段落或 Markdown 粗体标号切分
    if re.search(r"(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步)", cleaned):
        blocks = re.split(r"(?=(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步))", cleaned)
    else:
        blocks = re.split(r"\n\s*\n", cleaned)
        
    steps = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        # 过滤客套句与复读开场白
        lines = [l for l in lines if not re.match(r"^(Let's\s+solve|Here\s+is|Sure|Okay|我们来|题目描述|题目分析)", l, re.I)]
        if lines:
            s_str = " ".join(lines)
            if len(s_str) > 10:
                steps.append(s_str)

    # 吸收末尾孤立的 Answer 标签
    if len(steps) >= 2 and len(steps[-1]) < 40 and re.search(r"(Answer|答案)\s*[:：]", steps[-1], re.I):
        steps[-2] = steps[-2] + " " + steps[-1]
        steps.pop()

    return steps if steps else [text.strip()]

# 6. 核心重排序闭环评测循环
print("=" * 75)
print(f"📊 [3/3] 启动修复后的 MathVista Best-of-{N_CANDIDATES} 自动化重排序评测...")
print("=" * 75)

stats = {
    "total": len(eval_subset),
    "greedy_correct": 0,
    "prm_avg_correct": 0,
    "prm_min_correct": 0,
    "random_n_correct": 0,
    "oracle_correct": 0
}

for i, item in enumerate(eval_subset, 1):
    q = item["question"]
    gt = item["answer"]
    img = item["decoded_image"]
    
    print(f"\n[{i}/{len(eval_subset)}] 题干: {q[:65]}...")
    print(f"     标准答案 (GT): '{gt}'")

    # A. 评估 Greedy 基线
    g_sol = generate_solution(img, q, temp=0.0)
    g_pred = extract_answer_robust(g_sol)
    g_ok = check_correctness_robust(g_pred, gt)
    stats["greedy_correct"] += int(g_ok)
    print(f"     -> Greedy 预测: '{g_pred}' [{'✅' if g_ok else '❌'}]")

    # B. 采样 N 条解题分支并由 PRM 打分
    cands = []
    for c_idx in range(N_CANDIDATES):
        cand_sol = generate_solution(img, q, temp=0.7)
        cand_pred = extract_answer_robust(cand_sol)
        c_ok = check_correctness_robust(cand_pred, gt)
        if c_ok:
            stats["random_n_correct"] += 1.0 / N_CANDIDATES

        steps = split_into_steps(cand_sol)
        scores = []
        prevs = []
        for st in steps:
            prev_str = " ".join(prevs) if prevs else "None"
            score = score_step_prm(img, q, prev_str, st)
            scores.append(score)
            prevs.append(st)

        avg_s = sum(scores) / max(1, len(scores))
        min_s = min(scores) if scores else 0.0
        cands.append({
            "pred": cand_pred,
            "is_correct": c_ok,
            "avg_s": avg_s,
            "min_s": min_s
        })
        print(f"        候选 #{c_idx+1}: 预测='{cand_pred:<8}' [{'✅' if c_ok else '❌'}] | PRM 均分={avg_s:.4f} | 短板={min_s:.4f}")

    if any(c["is_correct"] for c in cands):
        stats["oracle_correct"] += 1

    # C. PRM 选拔最优路径
    top_avg = sorted(cands, key=lambda x: x["avg_s"], reverse=True)[0]
    top_min = sorted(cands, key=lambda x: (round(x["min_s"], 4), round(x["avg_s"], 4)), reverse=True)[0]

    stats["prm_avg_correct"] += int(top_avg["is_correct"])
    stats["prm_min_correct"] += int(top_min["is_correct"])
    print(f"     -> PRM 选定: Avg选出='{top_avg['pred']}' [{'✅' if top_avg['is_correct'] else '❌'}] | Min选出='{top_min['pred']}'")

# 7. 最终指标报表输出
T = max(1, stats["total"])
greedy_acc = (stats["greedy_correct"] / T) * 100
random_acc = (stats["random_n_correct"] / T) * 100
prm_avg_acc = (stats["prm_avg_correct"] / T) * 100
prm_min_acc = (stats["prm_min_correct"] / T) * 100
oracle_acc = (stats["oracle_correct"] / T) * 100
gain = prm_avg_acc - greedy_acc

print("\n" + "=" * 80)
print(f"{'🏆 MathVista (Geometry OOD) 修正后重排序泛化评测报表':^70}")
print("=" * 80)
print(f"• 评测总题数 (Evaluated Questions) : {stats['total']}")
print(f"• 候选采样数 (Candidates per Q)   : N = {N_CANDIDATES}")
print("-" * 80)
print(f"{'评测策略模式':<35} | {'Pass@1 准确率':<15} | {'相对基线增益':<12}")
print("-" * 80)
print(f"{'1. Baseline (Greedy Search)':<35} | {greedy_acc:>6.2f}%         | {'-':^10}")
print(f"{'2. Random Pick @ N':<35} | {random_acc:>6.2f}%         | {random_acc - greedy_acc:>+6.2f}%")
print(f"{'3. PRM Best-of-N (Min-Score 短板)':<35} | {prm_min_acc:>6.2f}%         | {prm_min_acc - greedy_acc:>+6.2f}%")
print(f"{'4. PRM Best-of-N (Avg-Score 均分)':<35} | {prm_avg_acc:>6.2f}%         | {gain:>+6.2f}%")
print(f"{'5. Oracle Upper Bound (Pass@N)':<35} | {oracle_acc:>6.2f}%         | {oracle_acc - greedy_acc:>+6.2f}%")
print("=" * 80)
