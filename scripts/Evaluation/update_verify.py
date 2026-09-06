with open("test_rprm_infer.py", "r", encoding="utf-8") as f:
    code = f.read()

# 替换为更宽容、更鲁棒的判定提取正则
old_func = '''def verify_cot_structure(text):
    """检测 CoT 推理结构及末尾判决"""
    has_analysis = bool(re.search(r"(Analysis|Step|Geometric|Visual|Calculation)", text, re.IGNORECASE))
    verdict_match = re.search(r"Verification:\s*Is the step correct\s*\(Yes/No\)\?\s*(Yes|No)", text, re.IGNORECASE)
    
    passed = has_analysis and (verdict_match is not None)
    verdict = verdict_match.group(1).capitalize() if verdict_match else "未识别到标准判决"
    return passed, verdict'''

new_func = '''def verify_cot_structure(text):
    """检测 CoT 推理结构及末尾判决（兼容长短句式）"""
    has_analysis = bool(re.search(r"(Analysis|Step|Geometric|Visual|Calculation)", text, re.IGNORECASE))
    # 匹配 Verification: No 或 Verification: Is the step correct...? No
    verdict_match = re.search(r"Verification:(?:.*?)(Yes|No)", text, re.IGNORECASE | re.DOTALL)
    
    passed = has_analysis and (verdict_match is not None)
    verdict = verdict_match.group(1).capitalize() if verdict_match else "未识别到标准判决"
    return passed, verdict'''

code = code.replace(old_func, new_func)

# 针对 gt 提取的正则同步修正
code = code.replace(
    'gt_verdict_match = re.search(r"Verification:\\s*Is the step correct\\s*\\(Yes/No\\)\\?\\s*(Yes|No)", gt_text, re.IGNORECASE)',
    'gt_verdict_match = re.search(r"Verification:(?:.*?)(Yes|No)", gt_text, re.IGNORECASE | re.DOTALL)'
)

with open("test_rprm_infer.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ 判决正则匹配逻辑已更新！")
