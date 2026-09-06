import os
import glob
from swift.llm import merge_lora

base_output = "/hy-tmp/output/Qwen3-VL-8B-RPRM-Stage2"
checkpoints = glob.glob(f"{base_output}/*/checkpoint-*")
if not checkpoints:
    print("❌ 未检测到训练生成的 checkpoint 目录！")
    exit(1)

# 自动选取最新/最好的检查点
latest_ckpt = max(checkpoints, key=os.path.getmtime)
export_dir = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"

print(f"🚀 [Merge] 正在合并 LoRA 权重: {latest_ckpt}")
print(f"📦 [Export] 导出独立生产模型: {export_dir}")

merge_lora(
    checkpoint_dir=latest_ckpt,
    output_dir=export_dir
)

print(f"✅ 模型合并完成！已成功产出独立就绪模型: {export_dir}")
