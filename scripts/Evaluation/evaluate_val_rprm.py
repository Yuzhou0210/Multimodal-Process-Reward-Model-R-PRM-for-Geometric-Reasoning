import json, re, torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
VAL_FILE = "./multimodal_data/val_stage2.jsonl"
BATCH_SIZE = 16

def extract_verdict(text):
    match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize()
    tail = text.strip().split()[-4:]
    for w in tail:
        c = re.sub(r"[^a-zA-Z]", "", w).capitalize()
        if c in ["Yes", "No"]:
            return c
    return "Unknown"

print("🚀 正在加载独立生产模型 Qwen3-VL-8B-RPRM-v2...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
processor.tokenizer.padding_side = "left"

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

model = model_cls.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True)
model.eval()

# 读取验证集
samples = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"📊 待评测验证样本: {len(samples)} 条")

tp, fp, tn, fn, unknown = 0, 0, 0, 0, 0

for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Evaluating PRM"):
    batch = samples[i : i + BATCH_SIZE]
    messages_list = []
    gt_labels = []

    for item in batch:
        # 提取 GT 标签
        assistant_gt = item["messages"][-1]["content"]
        gt_verdict = extract_verdict(assistant_gt)
        gt_labels.append(gt_verdict)

        # 构建纯 Prompt 输入（不包含 assistant 答案）
        messages_list.append([
            item["messages"][0],
            {"role": "user", "content": [
                {"type": "image", "image": item["images"][0]},
                {"type": "text", "text": item["messages"][1]["content"].replace("<image>", "").strip()}
            ]}
        ])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    img_in, vid_in = process_vision_info(messages_list)
    inputs = processor(text=texts, images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.01)
        trimmed = [o[len(in_ids):] for in_ids, o in zip(inputs.input_ids, gen_ids)]
        preds = processor.batch_decode(trimmed, skip_special_tokens=True)

    for pred_text, gt in zip(preds, gt_labels):
        pred_label = extract_verdict(pred_text)
        if pred_label == "Unknown":
            unknown += 1
        elif gt == "Yes" and pred_label == "Yes":
            tp += 1
        elif gt == "No" and pred_label == "Yes":
            fp += 1
        elif gt == "No" and pred_label == "No":
            tn += 1
        elif gt == "Yes" and pred_label == "No":
            fn += 1

total_valid = tp + fp + tn + fn
acc = (tp + tn) / max(1, total_valid) * 100
prec = tp / max(1, tp + fp) * 100
rec = tp / max(1, tp + fn) * 100
f1 = 2 * prec * rec / max(1e-5, (prec + rec))

print("\n" + "="*50)
print(f"🎯 Qwen3-VL-8B-RPRM-v2 验证集评测指标报告")
print("="*50)
print(f"• 评估总数: {len(samples)} | 格式合规率: {(total_valid / len(samples))*100:.2f}% (未识别: {unknown})")
print(f"• 整体准确率 (Accuracy) : {acc:.2f}%")
print(f"• 正类精确率 (Precision): {prec:.2f}%")
print(f"• 正类召回率 (Recall)   : {rec:.2f}%")
print(f"• 综合 F1 分数 (F1 Score): {f1:.2f}%")
print("-" * 50)
print(f"混淆矩阵 (Confusion Matrix):")
print(f"  [实际 Yes] 预测 Yes (TP): {tp:<4} | 预测 No (FN): {fn:<4}")
print(f"  [实际 No ] 预测 Yes (FP): {fp:<4} | 预测 No (TN): {tn:<4}")
print("="*50)
