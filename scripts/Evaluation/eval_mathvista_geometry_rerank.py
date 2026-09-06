import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
import math
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# 模型与路径配置
POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
NUM_SAMPLES = 50  # 评测几何题数量（可调整，建议 30-50 题验证泛化性）
N_CANDIDATES = 4  # 每道题采样的候选解答数

print("=" * 75)
print("🚀 [1/3] 正在加载策略解题模型 (Policy) 与过程奖励模型 (PRM)...")
print("=" * 75)

policy_proc = AutoProcessor.from_pretrained(POLICY_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
policy_model = AutoModelForImageTextToText.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
)
policy_model.eval()

prm_proc = AutoProcessor.from_pretrained(PRM_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
prm_model = AutoModelForImageTextToText.from_pretrained(
    PRM_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
)
prm_model.eval()

prm_tok = prm_proc.tokenizer
yes_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["Yes", " Yes", "yes", " yes"]]))
no_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["No", " No", "no", " no"]]))

# 1. MathVista 数据加载与几何子集筛选
print("\n" + "=" * 75)
print("📥 [2/3] 正在下载/加载 MathVista 公开基准并筛选 Geometry 几何题域...")
print("=" * 75)

try:
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
except Exception:
    # 针对国内网络设置镜像备选
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    dataset = load_dataset("AI4Math/MathVista", split="testmini")

geo_data = []
for item in dataset:
    # 筛选几何相关题目与包含图像的样本
    metadata = item.get("metadata", {})
    subject = metadata.get("subject", "").lower()
    task = metadata.get("task", "").lower()
    category = item.get("category", "").lower()
    
    if "geometry" in subject or "geometry" in task or "geometry" in category:
        if item.get("decoded_image") is not None:
            geo_data.append(item)

print(f"✅ 成功从 MathVista 中检索出 {len(geo_data)} 道几何大题，本次抽取评测: {min(NUM_SAMPLES, len(geo_data))} 道")
eval_subset = geo_data[:NUM_SAMPLES]

# 2. 核心辅助函数
def generate_solution(image, question, temperature=0.7):
    prompt_text = f"{question}\nSolve this geometry problem step by step clearly. Conclude your answer at the very end in the format: 'Answer: <choice or value>'."
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

    do_sample = temperature > 0
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, 
            max_new_tokens=800, 
            temperature=temperature if do_sample else 1.0, 
            do_sample=do_sample,
            top_p=0.9 if do_sample else 1.0
        )
    gen_tokens = out[0, inputs.input_ids.shape[1]:]
    return policy_proc.decode(gen_tokens, skip_special_tokens=True)

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
    cleaned = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    cleaned = re.sub(r"(?m)^[ \t]*>[ \t]*", "", cleaned)
    if re.search(r"(?:\*\*Step\s*\d+|Step\s*\d+:)", cleaned):
        blocks = re.split(r"(?=(?:\*\*Step\s*\d+|Step\s*\d+:))", cleaned)
    else:
        blocks = re.split(r"\n\s*\n", cleaned)
    steps = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        lines = [l for l in lines if not re.match(r"^(Let's\s+solve|Here\s+is|Sure|Okay)", l, re.I)]
        if lines:
            s_str = " ".join(lines)
            if len(s_str) > 10:
                steps.append(s_str)
    if len(steps) >= 2 and len(steps[-1]) < 40 and re.search(r"Answer\s*:", steps[-1], re.I):
        steps[-2] = steps[-2] + " " + steps[-1]
        steps.pop()
    return steps if steps else [text.strip()]

def extract_answer(solution_text):
    # 优先抽取 "Answer: X" 格式
    match = re.search(r"Answer\s*:\s*([A-Za-z0-9\.\-\_\/]+)", solution_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip(".").upper()
    # 提取最后一行中的大写选项
    tail = solution_text.strip().split("\n")[-1]
    choice_match = re.search(r"\b([A-E])\b", tail)
    if choice_match:
        return choice_match.group(1)
    return "UNKNOWN"

def check_correctness(pred, gt):
    pred = str(pred).strip().upper().rstrip(".")
    gt = str(gt).strip().upper().rstrip(".")
    if pred == gt:
        return True
    try:
        # 支持数值浮点容差比较
        return math.isclose(float(pred), float(gt), rel_tol=1e-2)
    except Exception:
        return False

# 3. 开始端到端重排序评测
print("\n" + "=" * 75)
print(f"📊 [3/3] 启动 MathVista Best-of-{N_CANDIDATES} 自动化重排序评测...")
print("=" * 75)

stats = {
    "total": len(eval_subset),
    "greedy_correct": 0,
    "prm_avg_correct": 0,
    "prm_min_correct": 0,
    "random_n_correct": 0,
    "oracle_correct": 0
}

for q_idx, item in enumerate(tqdm(eval_subset, desc="Evaluating OOD Geometry")):
    image = item["decoded_image"]
    question = item["question"]
    gt_answer = item["answer"]
    
    # A. 基线：Greedy 解答 (Temp=0)
    greedy_sol = generate_solution(image, question, temperature=0.0)
    greedy_pred = extract_answer(greedy_sol)
    is_greedy_corr = check_correctness(greedy_pred, gt_answer)
    if is_greedy_corr:
        stats["greedy_correct"] += 1

    # B. 采样 N 条不同路径 (Temp=0.7)
    cands = []
    for _ in range(N_CANDIDATES):
        cand_sol = generate_solution(image, question, temperature=0.7)
        cands.append(cand_sol)

    # C. PRM 逐步审计与打分
    cand_records = []
    any_correct = False
    for c_idx, sol in enumerate(cands):
        pred_ans = extract_answer(sol)
        corr = check_correctness(pred_ans, gt_answer)
        if corr:
            any_correct = True
            stats["random_n_correct"] += 1.0 / N_CANDIDATES

        steps = split_into_steps(sol)
        scores = []
        prevs = []
        for st in steps:
            prev_str = " ".join([f"({i}) {s}" for i, s in enumerate(prevs, 1)]) if prevs else "None"
            p_val = score_step_prm(image, question, prev_str, st)
            scores.append(p_val)
            prevs.append(st)

        min_s = min(scores) if scores else 0.0
        avg_s = sum(scores) / max(1, len(scores))
        cand_records.append({
            "sol": sol,
            "pred": pred_ans,
            "is_correct": corr,
            "min_score": min_s,
            "avg_score": avg_s
        })

    if any_correct:
        stats["oracle_correct"] += 1

    # D. PRM 依据 Avg-Score 和 Min-Score 选出 Top-1
    top_avg_cand = sorted(cand_records, key=lambda x: x["avg_score"], reverse=True)[0]
    top_min_cand = sorted(cand_records, key=lambda x: (x["min_score"], x["avg_score"]), reverse=True)[0]

    if top_avg_cand["is_correct"]:
        stats["prm_avg_correct"] += 1
    if top_min_cand["is_correct"]:
        stats["prm_min_correct"] += 1

# 4. 打印最终学术级对比报表
T = max(1, stats["total"])
greedy_acc = (stats["greedy_correct"] / T) * 100
random_acc = (stats["random_n_correct"] / T) * 100
prm_avg_acc = (stats["prm_avg_correct"] / T) * 100
prm_min_acc = (stats["prm_min_correct"] / T) * 100
oracle_acc = (stats["oracle_correct"] / T) * 100
gain = prm_avg_acc - greedy_acc

print("\n" + "=" * 80)
print(f"{'🏆 MathVista (Geometry OOD) PRM 重排序泛化评测报告':^70}")
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
if gain > 0:
    print(f"🎉 结论：R-PRM 在完全独立的 MathVista 几何基准上取得了 +{gain:.2f}% 的净增益！")
    print("证明模型具备真实的跨域几何图元感知、防跳步与抗伪证泛化能力。")
else:
    print("⚠️ 结论：重排序增益未拉开，需排查复杂题型的切分粒度或采样多样性。")
print("=" * 80)
