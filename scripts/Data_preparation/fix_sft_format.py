import json
import os

def fix_file(filepath):
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return

    fixed_count = 0
    total_count = 0
    new_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            data = json.loads(line)
            
            messages = data.get("messages", [])
            
            # 1. 统一旧版/非标角色名
            for msg in messages:
                if msg.get("role") in ["gpt", "bot", "model"]:
                    msg["role"] = "assistant"
                elif msg.get("role") in ["human"]:
                    msg["role"] = "user"

            # 2. 如果 messages 缺少 assistant，但外层有 response，则自动追加
            has_assistant = any(m.get("role") == "assistant" for m in messages)
            if not has_assistant and "response" in data and data["response"]:
                messages.append({
                    "role": "assistant",
                    "content": str(data["response"])
                })
                fixed_count += 1

            # 3. 规范化结构，移除多余的外层冗余键
            clean_item = {
                "messages": messages,
                "images": data.get("images", [])
            }
            new_lines.append(clean_item)

    # 备份并覆盖
    os.system(f"cp {filepath} {filepath}.bak")
    with open(filepath, "w", encoding="utf-8") as f:
        for item in new_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 处理完成 {filepath}: 共 {total_count} 条样本，修复/补全了 {fixed_count} 条 assistant 节点。")

fix_file("./multimodal_data/sft_train.jsonl")
fix_file("./multimodal_data/sft_val.jsonl")
