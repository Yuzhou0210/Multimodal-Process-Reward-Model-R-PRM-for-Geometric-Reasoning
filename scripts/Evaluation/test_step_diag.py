import json, re, torch
from transformers import AutoConfig, AutoProcessor
import transformers
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
POOL_FILE = "./multimodal_data/unlabeled_candidate_pool_30k_steps.jsonl"

def extract_verdict(text):
    match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).capitalize() if match else None

samples = []
with open(POOL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("ground_truth_label") == "Yes" and len([x for x in samples if x["ground_truth_label"] == "Yes"]) < 2:
            samples.append(item)
        elif item.get("ground_truth_label") == "No" and len([x for x in samples if x["ground_truth_label"] == "No"]) < 1:
            samples.append(item)
        if len(samples) >= 3:
            break

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None
model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForImageTextToText

model = model_cls.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
model.eval()

for idx, item in enumerate(samples):
    print(f"\n================ 测试步骤 {idx + 1} ================")
    print(f"GT: {item['ground_truth_label']} | Now Step: {item['now_step']}")
    user_prompt = f"<image>Question: {item['question']}\nPrevious Steps: {item['previous_steps']}\nNow Step: {item['now_step']}\nPlease evaluate this step thoroughly."
    messages = [
        {"role": "system", "content": "You are an expert multimodal mathematics teacher."},
        {"role": "user", "content": [{"type": "image", "image": item["image_path"]}, {"type": "text", "text": user_prompt.replace("<image>", "")}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.1)
        resp = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    pred = extract_verdict(resp)
    print(f"预测结果: {pred} | 匹配保留: {pred == item['ground_truth_label']}")
