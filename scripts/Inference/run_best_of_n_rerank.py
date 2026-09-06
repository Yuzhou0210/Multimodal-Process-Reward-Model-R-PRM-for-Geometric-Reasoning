import re
import json
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
VAL_FILE = "./multimodal_data/val_stage2.jsonl"

print("=" * 70)
print("🚀 初始化 Qwen3-VL-8B-RPRM-v2 生产级 Best-of-N 重排序引擎...")
print("=" * 70)

# 1. 载入处理器与模型
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

model = model_cls.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="sdpa",
    device_map="cuda:0", 
    trust_remote_code=True
)
model.eval()

tokenizer = processor.tokenizer
yes_ids = list(set([tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["Yes", " Yes", "yes", " yes"]]))
no_ids = list(set([tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["No", " No", "no", " no"]]))

# 2. 单步 Soft Reward 打分内核
def score_step_raw(image_path, question, prev_steps, now_step):
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
        {
            "role": "system",
            "content": "You are an expert multimodal mathematics teacher. Your task is to evaluate the correctness of the 'Now Step' with a concise, rigorous reasoning chain."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            return_dict_in_generate=True,
            output_scores=True
        )

        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        cot_text = processor.decode(gen_tokens, skip_special_tokens=True)

        # 定位 Yes/No 决策 Token 位点
        decision_idx = None
        for idx in range(len(gen_tokens) - 1, -1, -1):
            tok_id = gen_tokens[idx].item()
            if tok_id in yes_ids or tok_id in no_ids:
                decision_idx = idx
                break

        if decision_idx is None:
            decision_idx = max(0, len(outputs.scores) - 2)

        decision_logits = outputs.scores[decision_idx][0]
        max_yes_logit = max([decision_logits[idx].item() for idx in yes_ids])
        max_no_logit = max([decision_logits[idx].item() for idx in no_ids])

        probs = F.softmax(torch.tensor([max_no_logit, max_yes_logit]), dim=-1)
        p_yes = probs[1].item()

    # 提取简要结论
    verdict = "Yes" if p_yes >= 0.5 else "No"
    return p_yes, verdict, cot_text

# 3. 轨迹步进解析与评分器
def evaluate_trajectory(image_path, question, candidate_name, steps):
    print(f"\n👉 正在评测路径 [{candidate_name}] (共 {len(steps)} 步):")
    step_records = []
    prev_steps_accumulator = []

    for idx, step_content in enumerate(steps, 1):
        prev_str = " ".join([f"({i}) {s}" for i, s in enumerate(prev_steps_accumulator, 1)]) if prev_steps_accumulator else "None"
        p_yes, verdict, cot = score_step_raw(image_path, question, prev_str, step_content)
        
        step_records.append({
            "step_num": idx,
            "step_text": step_content,
            "p_yes": p_yes,
            "verdict": verdict,
            "cot": cot
        })
        prev_steps_accumulator.append(step_content)

        # 实时打印单步判决
        status_symbol = "✅" if p_yes >= 0.8 else ("⚠️" if p_yes >= 0.4 else "❌")
        print(f"   Step {idx}: {status_symbol} P(Yes)={p_yes:.4f} | {step_content[:55]}...")

    # 计算轨迹级打分
    scores = [r["p_yes"] for r in step_records]
    min_score = min(scores)
    # 几何平均分，避免过长路径惩罚
    geom_mean = 1.0
    for s in scores:
        geom_mean *= max(1e-4, s)
    geom_mean = geom_mean ** (1.0 / len(scores))

    return {
        "name": candidate_name,
        "steps": step_records,
        "min_score": min_score,
        "geom_mean": geom_mean,
        "final_answer": steps[-1]
    }

# ==============================================================================
# 4. 实战评测：从验证集提取真实几何真题，构建 4 条典型的解题轨迹 (Best-of-4)
# ==============================================================================
# 从独立验证集读取一张真实图像与题干
with open(VAL_FILE, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

test_image = sample["images"][0]
prompt_raw = sample["messages"][1]["content"]

# 提取题目内容
q_match = re.search(r"Question:\s*(.*?)(?:\nPrevious Steps:|$)", prompt_raw, re.DOTALL)
test_question = q_match.group(1).strip() if q_match else "Refer to the image to find the required measurement."

print(f"\n🖼️  评测样本图像: {test_image}")
print(f"❓ 评测几何题干: {test_question}\n")

# 构造 4 条不同质量的候选路径 (Best-of-4 Candidates)
candidates = {
    "Candidate 1 (严谨正解)": [
        "From the given figure, identify the geometric shape as a square ABCD with side length AB = 55, and a circle with radius R = 25 tangent to the boundaries.",
        "Set up a Cartesian coordinate system with A as origin (0,0), giving the center of the circle at (25, 25).",
        "Line DE is tangent to the circle; using the tangent distance formula and circle equation (x-25)^2 + (y-25)^2 = 25^2, solve for the coordinates of E on CD.",
        "By applying the Pythagorean theorem on triangle CDE, we calculate DE = 30.0.",
        "Therefore, the final answer is 30.0."
    ],
    "Candidate 2 (中间代数算错)": [
        "From the given figure, identify the square ABCD with AB = 55, and the inscribed tangent circle with radius 25.",
        "Set up the distance equation from center (25,25) to tangent line DE.",
        "Solve the linear system: 55 - 25 = 35 (Arithmetic mistake here).",
        "Substitute the value into the equation to get DE = 38.5.",
        "Therefore, the final answer is 38.5."
    ],
    "Candidate 3 (前期几何前提幻觉)": [
        "Assume triangle ADE is an equilateral triangle based on visual symmetry.",
        "Since all angles in an equilateral triangle are 60 degrees, angle ADE = 60°.",
        "Calculate DE = AD * cos(60°) = 55 * 0.5 = 27.5.",
        "Therefore, the final answer is 27.5."
    ],
    "Candidate 4 (无推导直接猜答案)": [
        "Looking at the four options provided, option B (30.0) appears visually closest.",
        "Therefore, the final answer is B."
    ]
}

# 运行 Best-of-N 评分
results = []
for name, step_list in candidates.items():
    res = evaluate_trajectory(test_image, test_question, name, step_list)
    results.append(res)

# 按照木桶原则 (Min-Score) 降序排序，若同分则按几何平均分决胜
results.sort(key=lambda x: (x["min_score"], x["geom_mean"]), reverse=True)

# ==============================================================================
# 5. 打印天梯榜与最优胜出报告
# ==============================================================================
print("\n" + "=" * 80)
print(f"{'🏆 Best-of-N 综合重排序天梯榜 (R-PRM Rerank Leaderboard)':^75}")
print("=" * 80)
print(f"{'排名':<4} | {'候选路径名称':<25} | {'Min-Score (瓶颈分)':<18} | {'Geom-Mean (均分)':<16} | {'最终结论':<10}")
print("-" * 80)

for rank, item in enumerate(results, 1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
    print(f"{medal:<4} | {item['name']:<25} | {item['min_score']:<18.4f} | {item['geom_mean']:<16.4f} | {item['final_answer'][:15]}")

print("=" * 80)

winner = results[0]
print(f"\n🎉 【R-PRM 判决胜出路径】: {winner['name']}")
print(f"• 胜出瓶颈置信度: {winner['min_score']:.4f}")
print(f"• 胜出完整推导链:")
for s in winner['steps']:
    print(f"   [Step {s['step_num']}] ({s['verdict']} - {s['p_yes']:.4f}): {s['step_text']}")

# 打印典型失败路径的检错原因（可解释性展示）
print("\n🔍 【典型淘汰路径归因剖析 (Failure Analysis)】:")
for item in results[1:]:
    # 找到第一个挂掉的步骤
    failed_step = next((s for s in item["steps"] if s["p_yes"] < 0.5), item["steps"][0])
    print(f"\n❌ 被淘汰解: {item['name']}")
    print(f"   • 首次崩溃于 Step {failed_step['step_num']}: \"{failed_step['step_text']}\"")
    print(f"   • 该步 P(Yes): {failed_step['p_yes']:.4f}")
    print(f"   • R-PRM 给出的审核理由前瞻:")
    # 提取 Conclusion 部分
    concl = re.search(r"Conclusion:\s*(.*?)(?:\nVerification|$)", failed_step["cot"], re.DOTALL)
    concl_str = concl.group(1).strip() if concl else failed_step["cot"][-150:]
    print(f"     \"{concl_str}\"")

print("\n" + "=" * 80)
print("✅ Best-of-N 端到端重排序流程顺利完成！")
print("=" * 80)
