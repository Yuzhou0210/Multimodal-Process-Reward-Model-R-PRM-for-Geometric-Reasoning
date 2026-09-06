import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import re
import math
import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

POLICY_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-SFT"
PRM_PATH = "/hy-tmp/models/Qwen3-VL-8B-RPRM-v2"

print("=" * 75)
print("🚀 Initializing Policy Solver and Multimodal Process Reward Model (R-PRM)...")
print("=" * 75)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

policy_proc = AutoProcessor.from_pretrained(POLICY_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
policy_model = AutoModelForImageTextToText.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device, trust_remote_code=True
).eval()

prm_proc = AutoProcessor.from_pretrained(PRM_PATH, trust_remote_code=True, min_pixels=256*28*28, max_pixels=512*28*28)
prm_model = AutoModelForImageTextToText.from_pretrained(
    PRM_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device, trust_remote_code=True
).eval()

prm_tok = prm_proc.tokenizer
yes_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["Yes", " Yes", "yes", " yes"]]))
no_ids = list(set([prm_tok.encode(w, add_special_tokens=False)[-1] for w in ["No", " No", "no", " no"]]))

print("✅ Models loaded successfully into GPU memory! Launching English WebUI...")

# ----------------- Inference Logic -----------------

def generate_policy_solution(image, question, temp=0.0):
    if image is None or not question.strip():
        return "Please provide both an image and the corresponding problem description."
    
    prompt_text = f"{question}\nSolve this geometry problem step by step clearly. Conclude with 'Answer: <choice or value>'."
    messages = [
        {"role": "system", "content": "You are an expert geometry solver. Solve problems step by step with mathematical rigor."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
    ]
    text = policy_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = policy_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)

    do_sample = temp > 0
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, max_new_tokens=800,
            temperature=temp if do_sample else 1.0,
            do_sample=do_sample, top_p=0.9 if do_sample else 1.0
        )
    return policy_proc.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def split_into_steps(text):
    cleaned = re.sub(r"(?m)^[ \t]*---[ \t]*$", "", text)
    cleaned = re.sub(r"(?m)^[ \t]*>[ \t]*", "", cleaned)
    
    if re.search(r"(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步)", cleaned):
        blocks = re.split(r"(?=(?:\*\*Step\s*\d+|Step\s*\d+:|第[一二三四五六七八九十]+步))", cleaned)
    else:
        blocks = re.split(r"\n\s*\n", cleaned)
        
    steps = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip()]
        lines = [l for l in lines if not re.match(r"^(Let's\s+solve|Here\s+is|Sure|Okay|我们来|题目描述|题目分析)", l, re.I)]
        if lines:
            s_str = " ".join(lines)
            if len(s_str) > 10:
                steps.append(s_str)

    if len(steps) >= 2 and len(steps[-1]) < 40 and re.search(r"(Answer|答案)\s*[:：]", steps[-1], re.I):
        steps[-2] = steps[-2] + " " + steps[-1]
        steps.pop()

    return steps if steps else [text.strip()]

def score_single_step(image, question, prev_steps, now_step):
    user_prompt = f"""Question: {question}
Previous Steps: {prev_steps}
Now Step: {now_step}

Evaluate this step concisely (1-2 sentences per point). Your response MUST strictly follow this structure:
Analysis:
1. Visual & Geometric Analysis: (Key elements, shapes, or measurements shown in the image)
2. Now Step Analysis: (Whether the step's logic is mathematically sound)
3. Data Source Analysis: (Whether numbers, formulas, or theorems cited are accurate)
4. Calculation Analysis: (Verification of arithmetic or algebraic calculations)

Conclusion:
Summary of findings.

Verification: Is the step correct (Yes/No)? ["""

    messages = [
        {"role": "system", "content": "You are an expert multimodal mathematics teacher. Your task is to evaluate the correctness of the 'Now Step' with a concise, rigorous reasoning chain."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
    ]
    text = prm_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info([messages])
    inputs = prm_proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = prm_model.generate(**inputs, max_new_tokens=300, return_dict_in_generate=True, output_scores=True)
        gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        raw_review = prm_proc.decode(gen_tokens, skip_special_tokens=True)
        
        decision_idx = next((i for i in range(len(gen_tokens)-1, -1, -1) if gen_tokens[i].item() in yes_ids or gen_tokens[i].item() in no_ids), max(0, len(outputs.scores)-2))
        logits = outputs.scores[decision_idx][0]
        max_yes = max([logits[idx].item() for idx in yes_ids])
        max_no = max([logits[idx].item() for idx in no_ids])
        
        shift = max(max_yes, max_no)
        e_yes = math.exp(max_yes - shift)
        e_no = math.exp(max_no - shift)
        denom = e_yes + e_no
        p_yes = e_yes / denom if denom > 0 else 0.5

    p_yes = 0.0 if math.isnan(p_yes) else p_yes
    return p_yes, raw_review

def extract_answer(text):
    m = re.search(r"(?:Answer|答案|故选|应选|结论为)\s*[:：]\s*[*_`]*([A-Za-z0-9\.\-\_\/°度cm]+)", text, re.I)
    if m: return m.group(1).strip().strip(".").upper()
    tail = text.strip().split("\n")[-1]
    m_choice = re.search(r"\b([A-E])\b", tail)
    return m_choice.group(1) if m_choice else "UNKNOWN"

# ----------------- Visual UI Builders -----------------

def build_step_html(step_idx, step_text, score, review_text):
    if score >= 0.80:
        color = "#10b981"      # Green
        bg_color = "#ecfdf5"
        border_color = "#a7f3d0"
        badge = "✅ Passed / Rigorous"
    elif score >= 0.50:
        color = "#f59e0b"      # Amber
        bg_color = "#fffbeb"
        border_color = "#fde68a"
        badge = "⚠️ Marginal / Ambiguous"
    else:
        color = "#ef4444"      # Red
        bg_color = "#fef2f2"
        border_color = "#fecaca"
        badge = "❌ Fault / Logic or Math Error"

    pct = int(round(score * 100))
    clean_review = review_text.replace("\n", "<br>")

    html = f"""
    <div style="border: 1px solid {border_color}; background-color: {bg_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 700; font-size: 15px; color: #1f2937;">📍 Step {step_idx}</span>
            <span style="background-color: {color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;">{badge}</span>
        </div>
        
        <div style="background-color: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px 12px; font-size: 14px; color: #374151; line-height: 1.5; margin-bottom: 12px;">
            {step_text}
        </div>

        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #4b5563; margin-bottom: 4px;">
                <span>PRM Step Correctness Confidence (P_Yes)</span>
                <span style="color: {color}; font-size: 13px;">{score:.4f} ({pct}%)</span>
            </div>
            <div style="width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 10px; overflow: hidden;">
                <div style="width: {pct}%; background-color: {color}; height: 100%; border-radius: 9999px; transition: width 0.4s ease;"></div>
            </div>
        </div>

        <details style="margin-top: 8px; cursor: pointer;">
            <summary style="font-size: 13px; font-weight: 600; color: #2563eb; outline: none;">
                🔍 Expand 4-Stage Multimodal Verification Report
            </summary>
            <div style="margin-top: 8px; padding: 10px; background-color: white; border-radius: 6px; border: 1px dashed #d1d5db; font-size: 13px; color: #4b5563; line-height: 1.6;">
                {clean_review}
            </div>
        </details>
    </div>
    """
    return html

# ----------------- Callback Handlers -----------------

def run_step_diagnostic(image, question, manual_solution, mode_choice, progress=gr.Progress()):
    if image is None:
        return "⚠️ Please upload a geometry problem image first.", ""
    if not question.strip():
        return "⚠️ Please provide the problem text or question prompt.", ""

    if "Auto-generate via Policy Model" in mode_choice:
        progress(0.1, desc="🤖 Generating step-by-step multimodal solution...")
        solution_text = generate_policy_solution(image, question, temp=0.0)
    else:
        if not manual_solution.strip():
            return "⚠️ Please enter custom deduction steps to inspect.", ""
        solution_text = manual_solution

    progress(0.3, desc="✂️ Parsing deductive steps...")
    steps = split_into_steps(solution_text)
    
    progress(0.4, desc="🔬 PRM verifying visual grounding and mathematical axioms...")
    cards_html = []
    scores = []
    prevs = []
    
    for idx, st in enumerate(steps, 1):
        prev_str = " ".join(prevs) if prevs else "None"
        p_yes, review = score_single_step(image, question, prev_str, st)
        scores.append(p_yes)
        prevs.append(st)
        cards_html.append(build_step_html(idx, st, p_yes, review))
        progress(0.4 + 0.5 * (idx / len(steps)), desc=f"Verified Step {idx}/{len(steps)}...")

    avg_score = sum(scores) / max(1, len(scores))
    min_score = min(scores) if scores else 0.0
    bottleneck_idx = scores.index(min_score) + 1

    summary_banner = f"""
    <div style="display: flex; gap: 12px; margin-bottom: 16px;">
        <div style="flex: 1; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">Total Steps</div>
            <div style="font-size: 20px; font-weight: 700; color: #0f172a;">{len(steps)}</div>
        </div>
        <div style="flex: 1; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">Global PRM Mean</div>
            <div style="font-size: 20px; font-weight: 700; color: #2563eb;">{avg_score:.4f}</div>
        </div>
        <div style="flex: 1; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">Bottleneck (Min)</div>
            <div style="font-size: 20px; font-weight: 700; color: {'#10b981' if min_score>=0.8 else '#ef4444'};">{min_score:.4f}</div>
        </div>
        <div style="flex: 1; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">Weakest Link</div>
            <div style="font-size: 20px; font-weight: 700; color: #d97706;">Step {bottleneck_idx}</div>
        </div>
    </div>
    """

    final_rendered_html = summary_banner + "".join(cards_html)
    return solution_text, final_rendered_html

def run_reranking(image, question, n_samples, progress=gr.Progress()):
    if image is None or not question.strip():
        return "⚠️ Please upload an image and enter the question prompt.", []
        
    n_samples = int(n_samples)
    progress(0.05, desc=f"🎲 Sampling {n_samples} candidate solution trajectories...")
    
    candidates = []
    # Candidate 0: Greedy Guard
    g_sol = generate_policy_solution(image, question, temp=0.0)
    candidates.append({"id": "0 (Greedy Baseline)", "sol": g_sol, "temp": 0.0})
    
    # Candidates 1~N: Exploration Sampling
    for k in range(1, n_samples + 1):
        s_sol = generate_policy_solution(image, question, temp=0.7)
        candidates.append({"id": f"{k} (Sample)", "sol": s_sol, "temp": 0.7})
        
    progress(0.4, desc="⚖️ PRM scoring steps across all candidate trajectories...")
    
    table_rows = []
    for c_i, cand in enumerate(candidates):
        steps = split_into_steps(cand["sol"])
        scores = []
        prevs = []
        for st in steps:
            p_yes, _ = score_single_step(image, question, " ".join(prevs) if prevs else "None", st)
            scores.append(p_yes)
            prevs.append(st)
            
        avg_s = sum(scores) / max(1, len(scores))
        min_s = min(scores) if scores else 0.0
        ans = extract_answer(cand["sol"])
        cand.update({"avg_s": avg_s, "min_s": min_s, "ans": ans, "steps_count": len(steps)})
        progress(0.4 + 0.55 * ((c_i + 1) / len(candidates)))

    # PRM Reranking
    ranked = sorted(candidates, key=lambda x: (round(x["avg_s"], 4), round(x["min_s"], 4)), reverse=True)
    winner_id = ranked[0]["id"]

    for r_idx, c in enumerate(ranked, 1):
        is_winner = "🏆 Selected Winner" if c["id"] == winner_id else "Candidate"
        table_rows.append([
            f"#{r_idx}",
            c["id"],
            c["ans"],
            f"{c['avg_s']:.4f}",
            f"{c['min_s']:.4f}",
            c["steps_count"],
            is_winner
        ])

    winner_summary = f"""
    ### 🏆 PRM Decision Analysis
    * **Optimal Trajectory**: `{winner_id}`
    * **Extracted Conclusion**: **`{ranked[0]['ans']}`**
    * **PRM Mean Confidence**: `{ranked[0]['avg_s']:.4f}` | **Bottleneck Score**: `{ranked[0]['min_s']:.4f}`
    
    ---
    #### Selected Reasoning Trace:
    {ranked[0]['sol']}
    """
    return winner_summary, table_rows

# ----------------- UI Layout -----------------

custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="Multimodal R-PRM Interactive Demo System", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 📐 Multimodal R-PRM: Geometry Reasoning & Process Verification
    **Backbone: Qwen3-VL-8B | 4-Stage Multimodal Verification (Visual / Step / Axiom / Calc) | Best-of-N Monte Carlo Search**
    """)

    with gr.Tabs():
        # TAB 1: Fine-Grained Step Diagnostic
        with gr.TabItem("🔍 Fine-Grained Step Diagnostic"):
            with gr.Row():
                with gr.Column(scale=4):
                    diag_image = gr.Image(label="Upload Geometry Figure", type="pil")
                    diag_question = gr.Textbox(label="Problem Prompt / Question Text", lines=3, placeholder="Enter problem statement...")
                    diag_mode = gr.Radio(
                        ["Auto-generate via Policy Model & Inspect", "Inspect Custom Deduction Steps"],
                        label="Deduction Source",
                        value="Auto-generate via Policy Model & Inspect"
                    )
                    diag_manual = gr.Textbox(label="Custom Deduction Trace (Step-by-Step)", lines=6, placeholder="Paste steps here (e.g., Step 1: ..., Step 2: ...)...", visible=True)
                    diag_btn = gr.Button("🚀 Run Step-by-Step Diagnostic", variant="primary")

                with gr.Column(scale=6):
                    diag_sol_output = gr.Textbox(label="Full Solution Log", lines=6)
                    diag_html_output = gr.HTML(label="Causal Verification Flow")

            diag_btn.click(
                fn=run_step_diagnostic,
                inputs=[diag_image, diag_question, diag_manual, diag_mode],
                outputs=[diag_sol_output, diag_html_output]
            )

        # TAB 2: Best-of-N Reranker
        with gr.TabItem("⚖️ Best-of-N Candidate Reranking (Test-Time Scaling)"):
            with gr.Row():
                with gr.Column(scale=4):
                    rerank_image = gr.Image(label="Upload Geometry Figure", type="pil")
                    rerank_question = gr.Textbox(label="Problem Prompt / Question Text", lines=3, placeholder="Enter problem statement...")
                    rerank_n = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Candidate Budget N (Exploration Budget)")
                    rerank_btn = gr.Button("🎲 Execute Best-of-N Search", variant="primary")

                with gr.Column(scale=6):
                    rerank_summary = gr.Markdown(label="Reranker Summary")
                    rerank_table = gr.Dataframe(
                        headers=["Rank", "Candidate ID", "Answer", "PRM Mean", "Bottleneck", "Steps", "Status"],
                        label="PRM Evaluation Matrix",
                        interactive=False
                    )

            rerank_btn.click(
                fn=run_reranking,
                inputs=[rerank_image, rerank_question, rerank_n],
                outputs=[rerank_summary, rerank_table]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=8080, share=False)
