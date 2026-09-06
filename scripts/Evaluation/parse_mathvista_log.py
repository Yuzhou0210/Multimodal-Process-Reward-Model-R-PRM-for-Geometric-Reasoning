import re

log_file = "eval_mathvista.log"

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# 按题目切分块
blocks = re.split(r"={5,}\s*\[(\d+)/50\]", content)

parsed_cases = []
for i in range(1, len(blocks), 2):
    q_id = blocks[i]
    body = blocks[i+1]
    
    gt_match = re.search(r"标准答案 \(GT\):\s*(.+)", body)
    gt = gt_match.group(1).strip() if gt_match else ""
    
    greedy_match = re.search(r"Greedy 基准预测:\s*(.+?)\s*\[(✅|❌)\]", body)
    g_pred = greedy_match.group(1).strip() if greedy_match else ""
    g_corr = (greedy_match.group(2) == "✅") if greedy_match else False
    
    cands = []
    cand_matches = re.findall(r"候选 #\d+:\s*预测=(.+?)\s*\[(✅|❌)\]\s*\|\s*PRM 均分=([0-9\.]+)\s*\|\s*短板=([0-9\.]+)", body)
    for c in cand_matches:
        cands.append({
            "pred": c[0].strip(),
            "corr": (c[1] == "✅"),
            "avg_s": float(c[2]),
            "min_s": float(c[3])
        })
        
    prm_match = re.search(r"PRM 选优结果:\s*Avg选定\[(.+?)\]\s*\[(✅|❌)\]", body)
    prm_pred = prm_match.group(1).strip() if prm_match else ""
    prm_corr = (prm_match.group(2) == "✅") if prm_match else False
    
    oracle = any(c["corr"] for c in cands)
    unknown_count = sum(1 for c in cands if "UNKNOWN" in c["pred"]) + (1 if "UNKNOWN" in g_pred else 0)
    
    parsed_cases.append({
        "id": q_id,
        "gt": gt,
        "g_pred": g_pred,
        "g_corr": g_corr,
        "prm_pred": prm_pred,
        "prm_corr": prm_corr,
        "oracle": oracle,
        "cands": cands,
        "has_unknown": unknown_count > 0,
        "body": body[:500]
    })

# 归类统计
total = len(parsed_cases)
all_wrong_pool = [c for c in parsed_cases if not c["oracle"]] # 候选池全错
prm_rescued = [c for c in parsed_cases if not c["g_corr"] and c["prm_corr"]] # PRM 纠错成功
prm_degraded = [c for c in parsed_cases if c["g_corr"] and not c["prm_corr"]] # Greedy对但PRM挑错
prm_missed_oracle = [c for c in parsed_cases if c["oracle"] and not c["prm_corr"]] # 池子里有对的，但PRM没挑出来
unknown_cases = [c for c in parsed_cases if c["has_unknown"]] # 涉及 UNKNOWN 提取失败

print("=" * 70)
print(f"{'📊 MathVista 评测日志深挖归因统计表':^60}")
print("=" * 70)
print(f"• 成功解析题目总数         : {total}")
print(f"• 候选解全军覆没 (All-Wrong): {len(all_wrong_pool)} 题 ({len(all_wrong_pool)/total*100:.1f}%) -> Policy 上限封死")
print(f"• 答案提取含 UNKNOWN 异常   : {len(unknown_cases)} 题 ({len(unknown_cases)/total*100:.1f}%) -> 正则提取器失效")
print(f"• PRM 成功纠错 (Greedy错->对): {len(prm_rescued)} 题")
print(f"• PRM 负向退化 (Greedy对->错): {len(prm_degraded)} 题")
print(f"• PRM 错失良机 (池中有对未选): {len(prm_missed_oracle)} 题")
print("=" * 70)

if prm_degraded:
    print("\n⚠️ 【重点排查 1：Greedy 对了，但 PRM 挑了错的题目 ID】:")
    for c in prm_degraded:
        print(f"  • 第 [{c['id']}] 题: GT='{c['gt']}' | Greedy='{c['g_pred']}' | PRM 选了='{c['prm_pred']}'")
        for idx, cd in enumerate(c["cands"], 1):
            print(f"     候选#{idx}: 预测={cd['pred']:<10} 正确={cd['corr']} | 均分={cd['avg_s']:.4f} | 短板={cd['min_s']:.4f}")

if prm_missed_oracle:
    print("\n🔍 【重点排查 2：池子里有正确答案，但 PRM 没选中的题目 ID】:")
    for c in prm_missed_oracle:
        print(f"  • 第 [{c['id']}] 题: GT='{c['gt']}' | PRM 选了='{c['prm_pred']}'")
        for idx, cd in enumerate(c["cands"], 1):
            mark = "🎯(真解)" if cd["corr"] else "  "
            print(f"     {mark} 候选#{idx}: 预测={cd['pred']:<10} | 均分={cd['avg_s']:.4f} | 短板={cd['min_s']:.4f}")

if unknown_cases:
    print(f"\n🧩 【重点排查 3：因 UNKNOWN 误判的前 3 个典型题 ID】: {[c['id'] for c in unknown_cases[:5]]}")
