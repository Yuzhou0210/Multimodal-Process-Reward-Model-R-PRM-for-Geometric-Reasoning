import os
import json
import glob
import re
import torch
import transformers
from transformers import AutoConfig, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

VAL_FILE = "./multimodal_data/sft_val.jsonl"
OUTPUT_DIR = "./output_rprm_sft"

def find_best_checkpoint():
    """查找训练产出的最优或最新 checkpoint"""
    ckpts_200 = glob.glob(os.path.join(OUTPUT_DIR, "**", "checkpoint-200"), recursive=True)
    if ckpts_200:
        return ckpts_200[-1]
    
    all_ckpts = glob.glob(os.path.join(OUTPUT_DIR, "**", "checkpoint-*"), recursive=True)
    if not all_ckpts:
        raise FileNotFoundError(f"在 {OUTPUT_DIR} 下未找到任何 Checkpoint")
    all_ckpts.sort(key=os.path.getmtime, reverse=True)
    return all_ckpts[0]

def verify_cot_structure(text):
    """检测 CoT 推理结构及末尾判决（兼容长短句式）"""
    has_analysis = bool(re.search(r"(Analysis|Step|Geometric|Visual|Calculation)", text, re.IGNORECASE))
    # 匹配 Verification: No 或 Verification: Is the step correct...? No
    verdict_match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    
    passed = has_analysis and (verdict_match is not None)
    verdict = verdict_match.group(1).capitalize() if verdict_match else "未识别到标准判决"
    return passed, verdict

def main():
    adapter_path = find_best_checkpoint()
    print("=" * 60)
    print(f"🚀 正在加载 LoRA 权重: {adapter_path}")
    print("=" * 60)

    # 1. 自动定位基座权重路径
    base_model_cache = glob.glob("/hy-tmp/cache/modelscope/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/*")
    base_model_path = base_model_cache[0] if base_model_cache else "Qwen/Qwen3-VL-8B-Instruct"

    # 2. 动态自适应获取多模态模型架构类
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    arch = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None

    if arch and hasattr(transformers, arch):
        model_cls = getattr(transformers, arch)
    else:
        try:
            from transformers import AutoModelForImageTextToText as model_cls
        except ImportError:
            from transformers import AutoModelForCausalLM as model_cls

    print(f"[*] 采用模型类加载: {model_cls.__name__}")

    # 3. 初始化 Processor 与模型
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 4. 读取验证集样本
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:3]

    print(f"[*] 抽取 {len(samples)} 条验证样本进行推理评估...\n")

    # 5. 执行推理核验
    for idx, item in enumerate(samples, start=1):
        raw_msgs = item["messages"]
        images = item.get("images", [])

        # 提取真实标注
        gt_text = ""
        for m in raw_msgs:
            if m["role"] == "assistant":
                gt_text = m["content"]
        gt_verdict_match = re.search(r"Verification:(?:.*?)(Yes|No)", gt_text, re.IGNORECASE | re.DOTALL)
        gt_verdict = gt_verdict_match.group(1).capitalize() if gt_verdict_match else "Unknown"

        # 规范化消息结构
        chat_content = []
        for img in images:
            chat_content.append({"type": "image", "image": img})
        
        user_prompt = ""
        system_prompt = "You are an expert multimodal mathematics teacher."
        for m in raw_msgs:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                user_prompt = m["content"].replace("<image>", "").strip()
        chat_content.append({"type": "text", "text": user_prompt})

        formatted_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_content}
        ]

        text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(formatted_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda:0")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        print(f"------------ [测试样本 #{idx}] ------------")
        print(f"🖼️ 图像: {images[0] if images else '无'}")
        print("\n📝 [模型预测的 CoT 思维链与判决]:")
        print(output_text.strip())
        print("-" * 50)

        passed, pred_verdict = verify_cot_structure(output_text)
        print("🔍 [评估结论]:")
        status_label = "✅ PASS" if passed else "❌ FAIL"
        print(f"  • 格式规范合格: {status_label}")
        print(f"  • 模型预测结果: {pred_verdict}")
        print(f"  • Ground Truth: {gt_verdict}")
        match_label = "🎯 完全一致" if pred_verdict == gt_verdict else "⚠️ 存在差异"
        print(f"  • 结果一致性  : {match_label}\n")

if __name__ == "__main__":
    main()
