import json, re, torch
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_30k_steps.jsonl"

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
    # 兼容换行、空格与方括号
    match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize()
    # 兜底捕获尾部的 Yes/No
    tail = text.strip().split()[-3:]
    for w in tail:
        w_clean = re.sub(r"[^a-zA-Z]", "", w).capitalize()
        if w_clean in ["Yes", "No"]:
            return w_clean
    return None

# 抓取 2 个真正的中间推导步
samples = []
with open(POOL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        step = item.get("now_step", "")
        # 跳过带 answer 的尾部残留
        if re.search(r"Answer\s*:", step, re.IGNORECASE):
            continue
        if item.get("ground_truth_label") == "Yes" and len([x for x in samples if x["ground_truth_label"] == "Yes"]) < 1:
            samples.append(item)
        elif item.get("ground_truth_label") == "No" and len([x for x in samples if x["ground_truth_label"] == "No"]) < 1:
            samples.append(item)
        if len(samples) >= 2:
            break

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

model = model_cls.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
model.eval()

for idx, item in enumerate(samples):
    print(f"\n================ 测试步骤 {idx + 1} ================")
    print(f"GT: {item['ground_truth_label']} | 待评步骤: {item['now_step'][:100]}...")
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=item["question"],
        previous_steps=item["previous_steps"],
        now_step=item["now_step"]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": item["image_path"]},
            {"type": "text", "text": user_prompt.replace("<image>", "").strip()}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        resp = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    pred = extract_verdict(resp)
    print("\n--- 模型原始输出尾部 (后 300 字符) ---")
    print(resp[-300:] if len(resp) > 300 else resp)
    print(f"\n正则捕获: {pred} | 匹配结果: {pred == item['ground_truth_label']}")
