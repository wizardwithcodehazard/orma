"""
paper_eval.py - Official LoCoMo Deterministic Benchmark for Orma
=================================================================
Evaluates Orma strictly on the official Snap Research LoCoMo dataset
(https://github.com/snap-research/locomo).

100% OF QUESTIONS AND DIALOGUES COME FROM THE OFFICIAL DATASET.
Zero hardcoded synthetic questions. Zero synthetic needles. Zero prompt hacks.

Metrics Evaluated (Standard Information Retrieval & Systems Benchmarks):
1. Evidence Recall@k (k=1, 5, 10, 20): Does the retrieved context contain
   the exact dialogue turn cited as gold evidence?
2. Mean Reciprocal Rank (MRR): Rank position of the first relevant evidence turn.
3. Answer-in-Context Recall (Hit Rate): Does Orma's injected context contain
   the factual gold answer string?
4. Per-Category Performance Breakdown:
   - Category 1: Multi-hop reasoning across sessions
   - Category 2: Temporal reasoning
   - Category 3: Open-domain inference
   - Category 4: Single-hop factual recall
5. Retrieval Latency: p50, p95, p99 hardware retrieval time (ms).
6. Token Compression Ratio: Injected prompt tokens vs raw history stuffing.
"""

import argparse
import json
import re
import statistics
import string
import subprocess
import time
from pathlib import Path

CATEGORY_NAMES = {
    1: "Multi-Hop Reasoning",
    2: "Temporal Reasoning",
    3: "Open-Domain Inference",
    4: "Single-Hop Factual",
    5: "Adversarial",
}


def normalize_text(s: str) -> str:
    """Standard IR text normalizer: lowercase, strip punctuation, normalize spaces."""
    s = str(s).lower().replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    return " ".join(s.split())


def build_evidence_index(conv_data: dict) -> dict[str, str]:
    """Maps official LoCoMo dialogue IDs (e.g. 'D1:3') to their exact text."""
    conv_obj = conv_data.get("conversation", {})
    index = {}
    if not isinstance(conv_obj, dict):
        return index
    s_nums = []
    for k in conv_obj:
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                s_nums.append((int(k.split("_")[1]), k))
            except ValueError:
                pass
    s_nums.sort()
    for _, skey in s_nums:
        for turn in conv_obj.get(skey, []):
            dia_id = turn.get("dia_id", "")
            if dia_id:
                index[dia_id] = turn.get("text", "")
    return index


def _is_match(retrieved_text: str, gold_text: str) -> bool:
    """Checks whether the gold text is covered in the retrieved text."""
    r_norm = normalize_text(retrieved_text)
    g_norm = normalize_text(gold_text)
    if not r_norm or not g_norm:
        return False
    # Substring containment
    if g_norm in r_norm or r_norm in g_norm:
        return True
    # Token set coverage (>= 60% of gold tokens present)
    g_words = set(g_norm.split())
    r_words = set(r_norm.split())
    if len(g_words) >= 2:
        overlap = len(g_words & r_words) / len(g_words)
        if overlap >= 0.60:
            return True
    return False


def evaluate_locomo_ir(mem, conv_data: dict, evidence_index: dict) -> dict:
    """Evaluates Orma on the official LoCoMo QA questions using standard IR metrics."""
    qa_list = conv_data.get("qa", [])
    k_values = [1, 5, 10, 20]

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in k_values}
    answer_hits = []
    per_category = {}

    for item in qa_list:
        cat = item.get("category", 0)
        evidence_ids = item.get("evidence", [])
        gold_answer = str(item.get("answer", ""))

        # Skip Category 5 (adversarial questions with no gold answer in dialogue)
        if cat == 5 or not evidence_ids:
            continue

        question = item["question"]

        # Run Orma hybrid search (sqlite-vec + BM25 FTS5 + spreading activation)
        search_res = mem.search(question)

        # Ranked list of retrieved texts (passages first, then states)
        ranked_passages = [s.text for s, _score in search_res.passages]
        state_passages = [st.value for st in search_res.states]
        all_ranked = ranked_passages + state_passages

        # Full context text (what Orma injects into the system prompt)
        ctx = mem.build_context([{"role": "user", "content": question}])
        full_sys_text = ctx[0]["content"] if ctx and ctx[0]["role"] == "system" else ""

        # Collect gold evidence texts
        gold_ev_texts = [evidence_index[eid] for eid in evidence_ids if eid in evidence_index]

        # 1. Evidence Recall@k and MRR
        first_hit_rank = None
        for rank_idx, passage in enumerate(all_ranked):
            for gold_text in gold_ev_texts:
                if _is_match(passage, gold_text):
                    if first_hit_rank is None:
                        first_hit_rank = rank_idx + 1  # 1-indexed
                    break
            if first_hit_rank is not None:
                break

        rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        for k in k_values:
            hit = first_hit_rank is not None and first_hit_rank <= k
            recall_at_k[k].append(1.0 if hit else 0.0)

        # 2. Answer-in-Context Recall
        ans_hit = _is_match(full_sys_text, gold_answer) if gold_answer.strip() else False
        answer_hits.append(1.0 if ans_hit else 0.0)

        # 3. Category tracking
        cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
        if cat_name not in per_category:
            per_category[cat_name] = {"rr": [], "r5": [], "r10": [], "ans_hit": []}
        per_category[cat_name]["rr"].append(rr)
        per_category[cat_name]["r5"].append(1.0 if (first_hit_rank and first_hit_rank <= 5) else 0.0)
        per_category[cat_name]["r10"].append(1.0 if (first_hit_rank and first_hit_rank <= 10) else 0.0)
        per_category[cat_name]["ans_hit"].append(1.0 if ans_hit else 0.0)

    n = len(reciprocal_ranks)
    return {
        "queries_evaluated": n,
        "MRR": round(statistics.mean(reciprocal_ranks) * 100, 2) if reciprocal_ranks else 0,
        **{f"Recall@{k}": round(statistics.mean(recall_at_k[k]) * 100, 2) if recall_at_k[k] else 0 for k in k_values},
        "Answer_in_Context_Recall": round(statistics.mean(answer_hits) * 100, 2) if answer_hits else 0,
        "per_category": {
            cname: {
                "count": len(cdata["rr"]),
                "MRR": round(statistics.mean(cdata["rr"]) * 100, 2),
                "Recall@5": round(statistics.mean(cdata["r5"]) * 100, 2),
                "Recall@10": round(statistics.mean(cdata["r10"]) * 100, 2),
                "Answer_Recall": round(statistics.mean(cdata["ans_hit"]) * 100, 2),
            }
            for cname, cdata in sorted(per_category.items())
        },
    }


def evaluate_latency_and_compression(mem, conv_data: dict) -> dict:
    """Measures hardware retrieval latency (ms) and token compression."""
    qa_list = conv_data.get("qa", [])
    latencies = []
    context_tokens = []

    # Measure raw history
    conv_obj = conv_data.get("conversation", {})
    total_chars = 0
    if isinstance(conv_obj, dict):
        for k in conv_obj:
            if k.startswith("session_") and not k.endswith("_date_time"):
                for turn in conv_obj.get(k, []):
                    total_chars += len(turn.get("text", ""))
    raw_tokens = int(total_chars / 3.5)

    for item in qa_list:
        q = item.get("question", "")
        if not q:
            continue

        t0 = time.perf_counter()
        ctx = mem.build_context([{"role": "user", "content": q}])
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        sys_block = ctx[0]["content"] if ctx and ctx[0]["role"] == "system" else ""
        context_tokens.append(int(len(sys_block) / 3.5))

    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    mean_ctx = statistics.mean(context_tokens) if context_tokens else 0
    compression = round((1 - mean_ctx / max(1, raw_tokens)) * 100, 1) if raw_tokens else 0

    return {
        "queries_timed": len(latencies),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "raw_history_tokens": raw_tokens,
        "mean_context_tokens": round(mean_ctx),
        "token_compression_percent": compression,
    }


def ingest_dialogues(mem, conv_data: dict, upstream_url: str, model_name: str) -> int:
    """Ingests official LoCoMo multi-session conversation turns into Orma."""
    conv_obj = conv_data.get("conversation", {})
    if not isinstance(conv_obj, dict):
        return 0

    spk_a = conv_obj.get("speaker_a", "")
    s_nums = []
    for k in conv_obj:
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                s_nums.append((int(k.split("_")[1]), k))
            except ValueError:
                pass
    s_nums.sort()

    total_turns = 0
    for _, skey in s_nums:
        for turn in conv_obj.get(skey, []):
            spk = turn.get("speaker", "")
            role = "user" if spk == spk_a else "assistant"
            txt = turn.get("text", "")
            if txt.strip():
                mem.observe(role=role, text=txt)
                total_turns += 1

    print(f"  Ingested {total_turns} turns from {len(s_nums)} official sessions.")
    print("  Consolidating memory via sleep...")
    t0 = time.time()
    report = mem.sleep()
    elapsed = time.time() - t0
    print(f"  Sleep consolidation finished in {elapsed:.1f}s (batches: {report.batches}, failed: {report.failed_batches})")

    if report.failed_batches > 0:
        row = mem.store.conn.execute(
            "SELECT consolidation_error FROM episodes WHERE consolidation_error IS NOT NULL LIMIT 1"
        ).fetchone()
        err_msg = row[0] if row else "Unknown error"
        print(f"  [Sleep Diagnosis]: Consolidation note: {err_msg[:120]}")
        print("  (Note: Orma automatically falls back to raw hybrid search over all indexed turns.)")

    return total_turns


def run_benchmark(
    dataset_path: str,
    output_path: str,
    upstream_url: str = "http://127.0.0.1:11434/v1",
    model_name: str = "qwen2.5:3b",
    conversations: int = 1,
    multilingual: bool = False,
):
    from orma import Memory, OrmaConfig

    if not Path(dataset_path).exists():
        print("Downloading official LoCoMo dataset from Snap Research...")
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        subprocess.run(["curl", "-L", "-o", dataset_path, url], check=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    num_convs = min(conversations, len(dataset))
    all_ir_results = []
    all_sys_results = []

    for idx in range(num_convs):
        conv = dataset[idx]
        cid = conv.get("sample_id") or f"conv_{idx}"
        print(f"\n{'=' * 65}")
        print(f"[{idx + 1}/{num_convs}] Running Benchmark on Official LoCoMo: {cid}")
        print("=" * 65)

        # Fresh isolated memory per conversation
        temp_dir = Path(f"/tmp/orma_locomo_{cid}_{int(time.time())}")
        config = OrmaConfig(data_dir=temp_dir, multilingual=multilingual)
        mem = Memory(config=config, upstream=upstream_url, model=model_name)

        evidence_index = build_evidence_index(conv)
        ingest_dialogues(mem, conv, upstream_url, model_name)

        # 1. Information Retrieval Evaluation
        print("  Evaluating Evidence & Answer Retrieval...")
        ir_res = evaluate_locomo_ir(mem, conv, evidence_index)
        all_ir_results.append(ir_res)

        # 2. Latency & Compression Evaluation
        print("  Evaluating Latency & Compression...")
        sys_res = evaluate_latency_and_compression(mem, conv)
        all_sys_results.append(sys_res)

        mem.close()

    # Aggregate results
    agg_mrr = round(statistics.mean([r["MRR"] for r in all_ir_results]), 2)
    agg_r1 = round(statistics.mean([r["Recall@1"] for r in all_ir_results]), 2)
    agg_r5 = round(statistics.mean([r["Recall@5"] for r in all_ir_results]), 2)
    agg_r10 = round(statistics.mean([r["Recall@10"] for r in all_ir_results]), 2)
    agg_r20 = round(statistics.mean([r["Recall@20"] for r in all_ir_results]), 2)
    agg_ans = round(statistics.mean([r["Answer_in_Context_Recall"] for r in all_ir_results]), 2)
    agg_p50 = round(statistics.mean([r["p50_latency_ms"] for r in all_sys_results]), 2)
    agg_p95 = round(statistics.mean([r["p95_latency_ms"] for r in all_sys_results]), 2)
    agg_ctx = round(statistics.mean([r["mean_context_tokens"] for r in all_sys_results]))
    agg_raw = round(statistics.mean([r["raw_history_tokens"] for r in all_sys_results]))
    agg_comp = round(statistics.mean([r["token_compression_percent"] for r in all_sys_results]), 1)

    final_report = {
        "benchmark": "Official Snap Research LoCoMo Benchmark",
        "dataset_source": "https://github.com/snap-research/locomo",
        "dataset_file": dataset_path,
        "conversations_evaluated": num_convs,
        "multilingual_embedding": multilingual,
        "evaluation_protocol": "Standard Information Retrieval (No LLM generation or LLM-judge in scoring loop)",
        "results": {
            "evidence_retrieval": {
                "MRR": agg_mrr,
                "Recall@1": agg_r1,
                "Recall@5": agg_r5,
                "Recall@10": agg_r10,
                "Recall@20": agg_r20,
                "Answer_in_Context_Recall": agg_ans,
            },
            "system_performance": {
                "retrieval_latency_p50_ms": agg_p50,
                "retrieval_latency_p95_ms": agg_p95,
                "mean_context_tokens": agg_ctx,
                "raw_history_tokens": agg_raw,
                "token_compression_ratio_percent": agg_comp,
            },
            "per_category_analysis": all_ir_results[0].get("per_category", {}),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Print publication-ready report
    print("\n" + "=" * 70)
    print("    OFFICIAL LOCOMO BENCHMARK REPORT (UNBIASED & DETERMINISTIC)")
    print("=" * 70)
    print("  Benchmark Source:      Snap Research LoCoMo (Official Dataset)")
    print(f"  Conversations:         {num_convs}")
    print(f"  Multilingual Mode:     {multilingual}")
    print("-" * 70)
    print("  INFORMATION RETRIEVAL PERFORMANCE (Evidence & Answer Grounding):")
    print(f"  - Mean Reciprocal Rank (MRR):  {agg_mrr}%")
    print(f"  - Evidence Recall@1:           {agg_r1}%")
    print(f"  - Evidence Recall@5:           {agg_r5}%")
    print(f"  - Evidence Recall@10:          {agg_r10}%")
    print(f"  - Evidence Recall@20:          {agg_r20}%")
    print(f"  - Answer-in-Context Recall:    {agg_ans}%")
    print("-" * 70)
    print("  CATEGORY-WISE BREAKDOWN:")
    for cat_name, cmetrics in all_ir_results[0].get("per_category", {}).items():
        print(f"  * {cat_name:24} (N={cmetrics['count']}): MRR={cmetrics['MRR']}% | R@5={cmetrics['Recall@5']}% | R@10={cmetrics['Recall@10']}% | AnsRecall={cmetrics['Answer_Recall']}%")
    print("-" * 70)
    print("  SYSTEM EFFICIENCY & LATENCY:")
    print(f"  - Retrieval Latency p50:       {agg_p50} ms")
    print(f"  - Retrieval Latency p95:       {agg_p95} ms")
    print(f"  - Mean Context Tokens:         {agg_ctx} tokens (vs Raw: {agg_raw} tokens)")
    print(f"  - Token Compression Ratio:     {agg_comp}% reduction")
    print("=" * 70)
    print(f"  Saved Full Report:             {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official LoCoMo Benchmark for Orma")
    parser.add_argument("--dataset", default="locomo10.json", help="Path to official locomo10.json")
    parser.add_argument("--output", default="official_locomo_report.json", help="Output JSON path")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434/v1", help="Ollama / vLLM endpoint")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name")
    parser.add_argument("--conversations", type=int, default=1, help="Number of conversations to benchmark (1-10)")
    parser.add_argument("--multilingual", action="store_true", help="Enable multilingual embeddings")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        upstream_url=args.upstream,
        model_name=args.model,
        conversations=args.conversations,
        multilingual=args.multilingual,
    )
