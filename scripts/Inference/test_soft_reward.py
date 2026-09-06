import re
import json
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"
VAL_FILE = "./multimodal_data/val_stage2.jsonl"

print("🚀 正在加载生产级模型以测试 Soft Reward 提取...")
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

# 获取 Yes 和 No 的所有常见 token 表达形式
tokenizer = processor.tokenizer
yes_ids = set()
no_ids = set()
for word in ["Yes", " Yes", "yes", " yes"]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids:
        yes_ids.add(ids[-1])
for word in ["No", " No", "no", " no"]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids:
        no_ids.add(ids[-1])

yes_ids = list(yes_ids)
no_ids = list(no_ids)

def score_step(image_path, user_prompt_text):
    messages = [
        {
            "role": "system", 
            "content": "You are an expert multimodal mathematics teacher. Your task is to evaluate the correctness of the 'Now Step' with a concise, rigorous reasoning chain."
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt_text.replace("<image>", "").strip()}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_scores=True
        )

        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        resp_text = processor.decode(gen_tokens, skip_special_tokens=True)

        # 逆序搜索确定 Yes/No 决策 token 的生成位点
        decision_idx = None
        for idx in range(len(gen_tokens) - 1, -1, -1):
            tok_id = gen_tokens[idx].item()
            if tok_id in yes_ids or tok_id in no_ids:
                decision_idx = idx
                break

        # 若未精确定位，回退至倒数第二个 token（通常在闭合符 ] 前）
        if decision_idx is None:
            decision_idx = max(0, len(outputs.scores) - 2)

        decision_logits = outputs.scores[decision_idx][0]
        max_yes_logit = max([decision_logits[idx].item() for idx in yes_ids])
        max_no_logit = max([decision_logits[idx].item() for idx in no_ids])

        probs = F.softmax(torch.tensor([max_no_logit, max_yes_logit]), dim=-1)
        p_yes = probs[1].item()

    return p_yes, resp_text

# 从验证集中抽取一条真实样本进行核验
with open(VAL_FILE, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

img = sample["images"][0]
full_user_prompt = sample["messages"][1]["content"]
gt_cot = sample["messages"][2]["content"]

step_match = re.search(r"Now Step:\s*(.*?)(?:\nPlease|\nEvaluate|$)", full_user_prompt, re.DOTALL)
now_step_display = step_match.group(1).strip() if step_match else "(未能解析出单独步骤)"

print("\n" + "=" * 60)
print(f"📝 待核验步骤 (Now Step):\n{now_step_display}")
print("=" * 60)

p_yes, cot_output = score_step(img, full_user_prompt)

print(f"\n🎯 提取到的 Soft Reward 分数: P(Yes) = {p_yes:.4f}  (P(No) = {1 - p_yes:.4f})")
print(f"\n📋 模型生成的四段式分析链:\n{cot_output}")
print("\n" + "=" * 60)
