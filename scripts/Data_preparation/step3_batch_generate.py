import os
import json
import base64
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 1. 配置参数 =================
API_KEY = ""  
URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

MAX_WORKERS = 5  
INPUT_FILE = "./multimodal_data/raw_input_multimodal.jsonl"
OUTPUT_FILE = "./multimodal_data/filtered_mm_seed_data.jsonl"

# ================= 2. 多模态 Prompt =================
MM_PROMPT = """You are an expert multimodal mathematics teacher. Your task is to verify the correctness of the "Now Step" based on the provided Image, Question, and context.

Analysis Dimensions:
1. **Visual & Geometric Analysis**: Check if conditions derived from the image are accurate.
2. **Now Step Analysis**: Explain what the current step aims to solve.
3. **Data Source Analysis**: Verify values/variables coming from the problem or image.
4. **Calculation Analysis**: Check if the calculations are correct.

Conclusion:
Conclude your evaluation. At the very end, strictly output your verdict in the following format:
"Verification: Is the step correct (Yes/No)? X" (where X is either Yes or No).

Question: {question}
Previous Steps: {previous_steps}
Now Step: {now_step}
Reply:"""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ================= 3. 单条数据请求与校验 =================
def process_item(item):
    image_path = item.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return None, "图片不存在"

    try:
        base64_img = encode_image(image_path)
    except Exception as e:
        return None, f"图片读取失败: {e}"

    prompt_text = MM_PROMPT.format(
        question=item["question"],
        previous_steps=item.get("previous_steps", "None"),
        now_step=item["now_step"]
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        res = requests.post(URL, headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            return None, f"API 状态码异常: {res.status_code} - {res.text[:100]}"
        
        cot_text = res.json()["choices"][0]["message"]["content"]

        # 增强版正则匹配：更宽松地捕获 Yes 或 No
        match = re.search(r"Verification:.*?Is the step correct.*?\?\s*(Yes|No)", cot_text, re.IGNORECASE | re.DOTALL)
        if not match:
            # 备用匹配：如果模型直接在结尾说了 Yes/No
            match = re.search(r"\b(Yes|No)\b\s*$", cot_text.strip(), re.IGNORECASE)

        if match:
            pred = match.group(1).capitalize()
            # 硬过滤：预测与原始标签一致时保留
            if pred == item["human_label"]:
                return {
                    "id": item.get("id"),
                    "image_path": item["image_path"],
                    "question": item["question"],
                    "previous_steps": item.get("previous_steps", "None"),
                    "now_step": item["now_step"],
                    "human_label": item["human_label"],
                    "generated_cot": cot_text
                }, "成功"
            else:
                return None, f"标签不符丢弃 (预测: {pred}, 实际: {item['human_label']})"
        else:
            return None, "正则未匹配到结论"

    except Exception as e:
        return None, f"网络请求异常: {e}"

# ================= 4. 主控运行 =================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"共读取到 {len(items)} 条数据，开始多线程批量处理...")

    success_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in items}
            
            pbar = tqdm(as_completed(future_to_item), total=len(items))
            for future in pbar:
                result, msg = future.result()
                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success_count += 1
                
                pbar.set_description(f"已成功写入: {success_count} 条 (最新状态: {msg[:15]})")

    print(f"\n全部处理完毕！成功生成并过滤出 {success_count} 条数据至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
