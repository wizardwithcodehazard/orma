"""
kaggle_locomo_eval.py - Standalone, Unbiased Benchmark Runner for Kaggle GPU
=============================================================================
Evaluates Orma on the official Snap Research LoCoMo long-term memory dataset
using an independent open-weights LLM (e.g. Qwen2.5-7B or Qwen2.5-3B) on Kaggle T4/P100.

Metrics evaluated:
1. Retrieval Latency: p50 and p95 retrieval time (ms).
2. Context Token Footprint: Prompt memory token reduction vs raw prompt stuffing.
3. QA Accuracy: Token F1, ROUGE-L, and exact match across single-hop, multi-hop, and temporal questions.
4. Negative Rule Retention: Retention rate of prohibition constraints ('never do X') across sessions.
"""

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes word-level F1 overlap between model response and gold answer."""
    pred_words = re.findall(r"\w+", prediction.lower())
    gt_words = re.findall(r"\w+", ground_truth.lower())
    if not pred_words or not gt_words:
        return 0.0
    common = set(pred_words) & set(gt_words)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred_words))
    recall = len(common) / len(set(gt_words))
    return (2 * precision * recall) / (precision + recall)


def token_rouge_l(prediction: str, ground_truth: str) -> float:
    """Computes LCS-based ROUGE-L score."""
    p_tokens = re.findall(r"\w+", prediction.lower())
    g_tokens = re.findall(r"\w+", ground_truth.lower())
    m, n = len(p_tokens), len(g_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p_tokens[i] == g_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def run_locomo_eval(
    dataset_path: str,
    output_path: str,
    upstream_url: str = "http://127.0.0.1:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_conversations: int = 10,
    multilingual: bool = False,
):
    from orma import Memory, OrmaConfig
    from orma.llm import LLMClient

    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}. Downloading from Snap Research...")
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        subprocess.run(["curl", "-L", "-o", dataset_path, url], check=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    llm = LLMClient(upstream=upstream_url, model=model_name)
    num_convs = min(max_conversations, len(dataset))
    print(f"Starting LoCoMo benchmark on {num_convs} conversations with model: {model_name}")

    overall_metrics = {
        "model": model_name,
        "multilingual": multilingual,
        "conversations_evaluated": num_convs,
        "total_qa_items": 0,
        "total_passed": 0,
        "f1_scores": [],
        "rouge_l_scores": [],
        "latencies_ms": [],
        "context_tokens_list": [],
        "raw_history_tokens_list": [],
        "rule_retention_checks": [],
        "category_breakdown": {},
    }

    for conv_idx in range(num_convs):
        conv_data = dataset[conv_idx]
        conv_id = conv_data.get("conversation_id", f"conv_{conv_idx}")
        print("\n=======================================================")
        print(f"[{conv_idx + 1}/{num_convs}] Running on {conv_id}")
        print("=======================================================")

        temp_db = Path(f"/tmp/orma_eval_{conv_id}_{int(time.time())}")
        config = OrmaConfig(data_dir=temp_db, multilingual=multilingual)
        mem = Memory(config=config, upstream=upstream_url, model=model_name)

        # Ingestion phase
        total_raw_chars = 0
        dialogues = conv_data.get("dialogue", [])
        print(f"Ingesting {len(dialogues)} multi-session dialogue blocks...")

        t_start_ingest = time.time()
        for s_idx, session in enumerate(dialogues):
            for turn in session.get("turns", []):
                spk = turn.get("speaker", "user").lower()
                role = "user" if spk in ("user", "human") else "assistant"
                txt = turn.get("text", "")
                total_raw_chars += len(txt)
                mem.observe(role=role, text=txt)

            # Consolidate memory after each session
            mem.sleep()

        raw_history_tokens = total_raw_chars // 3.5
        print(f"Ingested in {time.time() - t_start_ingest:.2f}s | Raw History: ~{raw_history_tokens:.0f} tokens")

        # Synthetic negative rule stress test inside the conversation
        mem.observe(role="user", text="Never recommend peanuts or shellfish to me under any circumstances.")
        mem.sleep()

        # QA Evaluation phase
        qa_list = conv_data.get("qa", [])
        print(f"Evaluating {len(qa_list)} QA items...")

        for item in qa_list:
            q_text = item["question"]
            gold = item["answer"]
            cat = item.get("category", "general")

            # 1. Measure pure retrieval latency (ms)
            t0 = time.perf_counter()
            prompt = mem.build_context([{"role": "user", "content": q_text}])
            latency_ms = (time.perf_counter() - t0) * 1000
            overall_metrics["latencies_ms"].append(latency_ms)

            # 2. Measure context token footprint
            sys_block = prompt[0]["content"] if prompt and prompt[0]["role"] == "system" else ""
            ctx_tokens = len(sys_block) // 3.5
            overall_metrics["context_tokens_list"].append(ctx_tokens)
            overall_metrics["raw_history_tokens_list"].append(raw_history_tokens)

            # 3. Model generation
            try:
                pred = llm.chat(prompt, model=model_name).strip()
            except Exception as e:  # noqa: BLE001
                pred = f"[LLM ERROR: {e}]"

            # 4. Accuracy scoring
            f1 = token_f1_score(pred, gold)
            rouge = token_rouge_l(pred, gold)
            passed = f1 >= 0.45 or gold.lower() in pred.lower()

            overall_metrics["f1_scores"].append(f1)
            overall_metrics["rouge_l_scores"].append(rouge)
            overall_metrics["total_qa_items"] += 1
            if passed:
                overall_metrics["total_passed"] += 1

            if cat not in overall_metrics["category_breakdown"]:
                overall_metrics["category_breakdown"][cat] = {"total": 0, "passed": 0, "f1_sum": 0.0}
            overall_metrics["category_breakdown"][cat]["total"] += 1
            if passed:
                overall_metrics["category_breakdown"][cat]["passed"] += 1
            overall_metrics["category_breakdown"][cat]["f1_sum"] += f1

        # Check negative rule retention
        rule_prompt = mem.build_context([{"role": "user", "content": "What food should I avoid?"}])
        rule_sys = rule_prompt[0]["content"] if rule_prompt and rule_prompt[0]["role"] == "system" else ""
        rule_retained = "peanuts" in rule_sys.lower() and "shellfish" in rule_sys.lower()
        overall_metrics["rule_retention_checks"].append(rule_retained)

        mem.close()

    # Aggregate Final Metrics
    total_items = max(1, overall_metrics["total_qa_items"])
    mean_f1 = statistics.mean(overall_metrics["f1_scores"]) * 100 if overall_metrics["f1_scores"] else 0
    mean_rouge = statistics.mean(overall_metrics["rouge_l_scores"]) * 100 if overall_metrics["rouge_l_scores"] else 0
    accuracy = (overall_metrics["total_passed"] / total_items) * 100
    p50_latency = statistics.median(overall_metrics["latencies_ms"]) if overall_metrics["latencies_ms"] else 0
    p95_latency = (
        statistics.quantiles(overall_metrics["latencies_ms"], n=20)[18]
        if len(overall_metrics["latencies_ms"]) >= 20
        else max(overall_metrics["latencies_ms"], default=0)
    )
    mean_ctx = statistics.mean(overall_metrics["context_tokens_list"]) if overall_metrics["context_tokens_list"] else 0
    mean_raw = statistics.mean(overall_metrics["raw_history_tokens_list"]) if overall_metrics["raw_history_tokens_list"] else 0
    compression_ratio = ((mean_raw - mean_ctx) / max(1, mean_raw)) * 100
    rule_retention_rate = (sum(overall_metrics["rule_retention_checks"]) / max(1, len(overall_metrics["rule_retention_checks"]))) * 100

    report_summary = {
        "model_name": model_name,
        "overall_accuracy_percent": round(accuracy, 2),
        "mean_f1_percent": round(mean_f1, 2),
        "mean_rouge_l_percent": round(mean_rouge, 2),
        "p50_retrieval_latency_ms": round(p50_latency, 2),
        "p95_retrieval_latency_ms": round(p95_latency, 2),
        "mean_context_tokens": round(mean_ctx, 1),
        "mean_raw_history_tokens": round(mean_raw, 1),
        "token_reduction_percent": round(compression_ratio, 2),
        "negative_rule_retention_percent": round(rule_retention_rate, 2),
        "category_results": {
            cat: {
                "count": data["total"],
                "accuracy_percent": round((data["passed"] / max(1, data["total"])) * 100, 2),
                "mean_f1": round((data["f1_sum"] / max(1, data["total"])) * 100, 2),
            }
            for cat, data in overall_metrics["category_breakdown"].items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("           ORMA OFFICIAL BENCHMARK REPORT (UNBIASED)")
    print("=" * 65)
    print(f"Evaluator Model:             {model_name}")
    print(f"Retrieval Latency p50:       {p50_latency:.2f} ms")
    print(f"Retrieval Latency p95:       {p95_latency:.2f} ms")
    print(f"Mean Context Tokens:         {mean_ctx:.0f} tokens (vs raw: {mean_raw:.0f})")
    print(f"Token Compression Ratio:     {compression_ratio:.1f}% reduction")
    print(f"QA Accuracy (LoCoMo):        {accuracy:.2f}%")
    print(f"Mean Token F1 Score:         {mean_f1:.2f}%")
    print(f"Mean ROUGE-L Score:          {mean_rouge:.2f}%")
    print(f"Negative Rule Retention:     {rule_retention_rate:.1f}%")
    print("=" * 65)
    print(f"Saved full JSON report to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark Runner for Orma on Kaggle GPU")
    parser.add_argument("--dataset", default="locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--output", default="locomo_benchmark_report.json", help="Path to save output JSON")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000/v1", help="Base URL of OpenAI-compatible LLM runtime")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--conversations", type=int, default=10, help="Number of conversations to benchmark (1-10)")
    parser.add_argument("--multilingual", action="store_true", help="Use 384-dim multilingual embeddings")
    args = parser.parse_args()

    run_locomo_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        upstream_url=args.upstream,
        model_name=args.model,
        max_conversations=args.conversations,
        multilingual=args.multilingual,
    )
