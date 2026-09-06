import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import re
import math
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
NUM_SAMPLES = 30
N_CANDIDATES = 8
LOG_FILE = "/hy-tmp/eval_mathvista_n8.log"

# 1. 精确解析已跑完题目的战果
stats = {
    "total": NUM_SAMPLES,
    "greedy_corr": 0,
    "random_n_corr": 0.0,
    "prm_pure_n_corr": 0,
    "prm_n_plus_1_corr": 0,
    "oracle_pure_n": 0,
    "oracle_n_plus_1": 0
}

completed_count = 0
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    blocks = re.split(r"\n\[(\d+)/30\]", log_text)
    for idx in range(1, len(blocks), 2):
        b_id = int(blocks[idx])
        b_body = blocks[idx+1]
        if "选优结论: Pure-N选定" in b_body:
            g_m = re.search(r"\[候选#0 基准Greedy\]:.*?\[(✅|❌)\]", b_body)
            g_ok = (g_m.group(1) == "✅") if g_m else False

            c_oks = [m == "✅" for m in re.findall(r"候选 #\d+:.*?\[(✅|❌)\]", b_body)]
            
            pure_m = re.search(r"Pure-N选定\[.*?\]\s*\[(✅|❌)\]", b_body)
            pure_ok = (pure_m.group(1) == "✅") if pure_m else False

            plus_m = re.search(r"\(N\+1\)选定\[.*?\]\s*\[(✅|❌)\]", b_body)
            plus_ok = (plus_m.group(1) == "✅") if plus_m else False

            stats["greedy_corr"] += int(g_ok)
            stats["random_n_corr"] += (sum(c_oks) / len(c_oks)) if c_oks else 0.0
            stats["prm_pure_n_corr"] += int(pure_ok)
            stats["prm_n_plus_1_corr"] += int(plus_ok)
            stats["oracle_pure_n"] += int(any(c_oks))
            stats["oracle_n_plus_1"] += int(any(c_oks) or g_ok)
            completed_count = max(completed_count, b_id)

print(f"\n==================================================================")
print(f"🔄 成功识别前 {completed_count} 道题已完成！自动从第 {completed_count + 1} 题续跑。")
print(f"==================================================================\n")

print("🚀 正在加载模型权重...")
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

try:
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
except Exception:
    dataset = load_dataset("AI4Math/MathVista", split="testmini", local_files_only=True)

geo_data = []
for item in dataset:
    m = item.get("metadata", {})
    if any("geometry" in str(x).lower() for x in [m.get("subject"), m.get("task"), item.get("category")]):
        if item.get("decoded_image") is not None:
            geo_data.append(item)

eval_subset = geo_data[:NUM_SAMPLES]

def extract_answer_robust(text):
    patterns = [
        r"(?:Answer|答案|故选|应选|结论为)\s*[:：]\s*[*_`]*([A-Za-z0-9\.\-\_\/°度cm]+)",
        r"[*_]{2}(?:Answer|答案)[:：]\s*([A-Za-z0-9\.\-\_\/°度cm]+)[*_]{2}",
        r"(?:等于|=)\s*([0-9\.\-\_\/]+)\s*$"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1).strip().strip(".").upper()
    tail_lines = [l.strip() for l in text.strip().split("\n")[-3:] if l.strip()]
    for line in reversed(tail_lines):
        m_choice = re.search(r"\b([A-E])\b", line)
        if m_choice: return m_choice.group(1)
        m_num = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m_num: return m_num.group(1)
    return "UNKNOWN"

def clean_val(s):
    s = str(s).strip()
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"(°|度|cm|mm|m|km|rad|inch|inches|米|\$|\{|\}|\(|\)|\[|\]|\*|\\circ)", "", s, flags=re.I)
    return s.strip().upper()

def check_correctness_ultimate(pred, gt, choices=None):
    p, g = clean_val(pred), clean_val(gt)
    if not p or not g: return False
    if p == g: return True
    try:
        p_val, g_val = float(p), float(g)
        if math.isclose(p_val, g_val, rel_tol=1e-2) or abs(p_val - g_val) < 1e-3:
            return True
    except: pass
    if choices and isinstance(choices, list) and len(choices) > 0:
        letters = [chr(ord('A') + idx) for idx in range(len(choices))]
        choice_cleans = [clean_val(c) for c in choices]
        if p in letters:
            idx = letters.index(p)
            if idx < len(choice_cleans) and (choice_cleans[idx] == g or check_correctness_ultimate(choice_cleans[idx], g)):
                return True
        if g in letters:
            idx = letters.index(g)
            if idx < len(choice_cleans) and (choice_cleans[idx] == p or check_correctness_ultimate(p, choice_cleans[idx])):
                return True
    return False

def generate_solution(image, question, temp=0.7):
    prompt_text = f"{question}\nSolve this geometry problem step by step clearly. Conclude with 'Answer: <choice or value>'."
    messages = [
        {"role": "system", "content": "You are an expert geometry solver. Solve problems step by step with mathematical rigor."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
    ]
    text = policy_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = policy_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")
    do_sample = temp > 0
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, max_new_tokens=800,
            temperature=temp if do_sample else 1.0,
            do_sample=do_sample, top_p=0.9 if do_sample else 1.0
        )
    return policy_proc.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

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
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
    ]
    text = prm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = prm_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = prm_model.generate(**inputs, max_new_tokens=250, return_dict_in_generate=True, output_scores=True)
        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        decision_idx = next((i for i in range(len(gen_tokens)-1, -1, -1) if gen_tokens[i].item() in yes_ids or gen_tokens[i].item() in no_ids), max(0, len(outputs.scores)-2))
        logits = outputs.scores[decision_idx][0]
        max_yes = max([logits[idx].item() for idx in yes_ids])
        max_no = max([logits[idx].item() for idx in no_ids])
        shift = max(max_yes, max_no)
        e_yes = math.exp(max_yes - shift)
        e_no = math.exp(max_no - shift)
        denom = e_yes + e_no
        p_yes = e_yes / denom if denom > 0 else 0.5
    return 0.0 if math.isnan(p_yes) else p_yes

def split_into_steps(text):
    cleaned = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    cleaned = re.sub(r"(?m)^[ \t]*>[ \t]*", "", cleaned)
    if re.search(r"(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步)", cleaned):
        blocks = re.split(r"(?=(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步))", cleaned)
    else:
        blocks = re.split(r"\n\s*\n", cleaned)
    steps = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        lines = [l for l in lines if not re.match(r"^(Let's\s+solve|Here\s+is|Sure|Okay|我们来|题目描述|题目分析)", l, re.I)]
        if lines:
            s_str = " ".join(lines)
            if len(s_str) > 10:
                steps.append(s_str)
    if len(steps) >= 2 and len(steps[-1]) < 40 and re.search(r"(Answer|答案)\s*[:：]", steps[-1], re.I):
        steps[-2] = steps[-2] + " " + steps[-1]
        steps.pop()
    return steps if steps else [text.strip()]

# 从第 19 题（索引 18）开始续跑
for i in range(completed_count, NUM_SAMPLES):
    item = eval_subset[i]
    q = item["question"]
    gt = item["answer"]
    img = item["decoded_image"]
    choices = item.get("choices") or []  # 核心防崩保护
    
    print(f"\n[{i+1}/{len(eval_subset)}] 题干: {q[:60]}...")
    print(f"     标准答案 (GT): '{gt}' | 选项数: {len(choices)}")

    g_sol = generate_solution(img, q, temp=0.0)
    g_pred = extract_answer_robust(g_sol)
    g_ok = check_correctness_ultimate(g_pred, gt, choices)
    stats["greedy_corr"] += int(g_ok)
    
    g_steps = split_into_steps(g_sol)
    g_scores, g_prevs = [], []
    for st in g_steps:
        s = score_step_prm(img, q, " ".join(g_prevs) if g_prevs else "None", st)
        g_scores.append(s)
        g_prevs.append(st)
    g_avg = sum(g_scores) / max(1, len(g_scores))
    g_min = min(g_scores) if g_scores else 0.0
    print(f"     -> [候选#0 基准Greedy]: 预测='{g_pred:<8}' [{'✅' if g_ok else '❌'}] | PRM均分={g_avg:.4f} | 短板={g_min:.4f}")

    sampled_cands = []
    for c_idx in range(1, N_CANDIDATES + 1):
        cand_sol = generate_solution(img, q, temp=0.7)
        cand_pred = extract_answer_robust(cand_sol)
        c_ok = check_correctness_ultimate(cand_pred, gt, choices)
        if c_ok:
            stats["random_n_corr"] += 1.0 / N_CANDIDATES

        c_steps = split_into_steps(cand_sol)
        c_scores, c_prevs = [], []
        for st in c_steps:
            s = score_step_prm(img, q, " ".join(c_prevs) if c_prevs else "None", st)
            c_scores.append(s)
            c_prevs.append(st)

        avg_s = sum(c_scores) / max(1, len(c_scores))
        min_s = min(c_scores) if c_scores else 0.0
        sampled_cands.append({
            "idx": c_idx, "pred": cand_pred, "is_correct": c_ok,
            "avg_s": avg_s, "min_s": min_s
        })
        print(f"        候选 #{c_idx}: 预测='{cand_pred:<8}' [{'✅' if c_ok else '❌'}] | PRM均分={avg_s:.4f} | 短板={min_s:.4f}")

    any_sampled_ok = any(c["is_correct"] for c in sampled_cands)
    stats["oracle_pure_n"] += int(any_sampled_ok)
    stats["oracle_n_plus_1"] += int(any_sampled_ok or g_ok)

    best_pure = sorted(sampled_cands, key=lambda x: x["avg_s"], reverse=True)[0]
    stats["prm_pure_n_corr"] += int(best_pure["is_correct"])

    all_cands = [{"idx": 0, "pred": g_pred, "is_correct": g_ok, "avg_s": g_avg, "min_s": g_min}] + sampled_cands
    best_plus_one = sorted(all_cands, key=lambda x: x["avg_s"], reverse=True)[0]
    stats["prm_n_plus_1_corr"] += int(best_plus_one["is_correct"])

    print(f"     -> 选优结论: Pure-N选定[#{best_pure['idx']}:{best_pure['pred']}] [{'✅' if best_pure['is_correct'] else '❌'}] | (N+1)选定[#{best_plus_one['idx']}:{best_plus_one['pred']}] [{'✅' if best_plus_one['is_correct'] else '❌'}]")

# 4. 打印最终报表
T = max(1, stats["total"])
greedy_acc = (stats["greedy_corr"] / T) * 100
random_acc = (stats["random_n_corr"] / T) * 100
prm_pure_acc = (stats["prm_pure_n_corr"] / T) * 100
prm_plus_acc = (stats["prm_n_plus_1_corr"] / T) * 100
oracle_pure_acc = (stats["oracle_pure_n"] / T) * 100
oracle_plus_acc = (stats["oracle_n_plus_1"] / T) * 100

print("\n" + "=" * 85)
print(f"{'🏆 MathVista (Geometry OOD) N=8 扩展重排序深度评测报表':^75}")
print("=" * 85)
print(f"• 评测题数 (Evaluated Questions) : {stats['total']}")
print(f"• 采样规模 (Candidate Budget)   : N = {N_CANDIDATES} (+1 Greedy Guard)")
print("-" * 85)
print(f"{'评测策略模式':<35} | {'Pass@1 准确率':<15} | {'相对基线净增益':<15}")
print("-" * 85)
print(f"{'1. Baseline (Greedy Search)':<35} | {greedy_acc:>6.2f}%         | {'-':^12}")
print(f"{'2. Random Pick @ N=8':<35} | {random_acc:>6.2f}%         | {random_acc - greedy_acc:>+6.2f}%")
print(f"{'3. PRM Best-of-8 (Pure Sample)':<35} | {prm_pure_acc:>6.2f}%         | {prm_pure_acc - greedy_acc:>+6.2f}%")
print(f"{'4. PRM Best-of-(8+1) [Greedy Guard]':<35} | {prm_plus_acc:>6.2f}%         | {prm_plus_acc - greedy_acc:>+6.2f}%")
print(f"{'5. Oracle Upper Bound (Pure N=8)':<35} | {oracle_pure_acc:>6.2f}%         | {oracle_pure_acc - greedy_acc:>+6.2f}%")
print(f"{'6. Oracle Upper Bound (N+1 Guard)':<35} | {oracle_plus_acc:>6.2f}%         | {oracle_plus_acc - greedy_acc:>+6.2f}%")
print("=" * 85)
