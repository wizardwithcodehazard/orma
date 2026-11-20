"""
paper_eval.py - Official LoCoMo Deterministic Benchmark for Orma
=================================================================
Evaluates Orma strictly on the official Snap Research LoCoMo dataset
(https://github.com/snap-research/locomo).

100% OF QUESTIONS AND DIALOGUES COME FROM THE OFFICIAL DATASET.
Zero synthetic questions. Zero synthetic needles. Zero prompt hacks.

Features:
- Full evaluation across all 10 official LoCoMo conversations.
- Dual evidence matching disclosure:
  1. Exact Match: Normalized substring containment.
  2. Fuzzy Match: Token set overlap >= 60%.
- Answer-in-Context Recall (Exact & Fuzzy coverage) reported separately.
- Category-wise breakdown aggregated across the entire dataset.
- System metrics: p50/p95 latency (ms), context token footprint, token compression ratio.
"""

import argparse
import json
import re
import statistics
import string
import subprocess
import tempfile
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


def is_exact_match(retrieved_text: str, gold_text: str) -> bool:
    """Strict containment check between retrieved sentence and gold dialogue turn."""
    r_norm = normalize_text(retrieved_text)
    g_norm = normalize_text(gold_text)
    if not r_norm or not g_norm:
        return False
    # Either gold turn is inside retrieved block, or retrieved sentence (>= 3 words) is verbatim inside gold turn
    return bool(g_norm in r_norm or (len(r_norm.split()) >= 3 and r_norm in g_norm))


def is_fuzzy_match(retrieved_text: str, gold_text: str) -> bool:
    """Coverage check: exact match OR >= 60% token overlap with the smaller segment."""
    if is_exact_match(retrieved_text, gold_text):
        return True
    r_norm = normalize_text(retrieved_text)
    g_norm = normalize_text(gold_text)
    g_words = set(g_norm.split())
    r_words = set(r_norm.split())
    min_len = min(len(g_words), len(r_words))
    if min_len >= 2:
        overlap = len(g_words & r_words) / min_len
        return overlap >= 0.60
    return False


def evaluate_locomo_ir(mem, conv_data: dict, evidence_index: dict) -> list[dict]:
    """Evaluates Orma on the official LoCoMo QA questions for one conversation."""
    qa_list = conv_data.get("qa", [])
    k_values = [1, 5, 10, 20]
    query_results = []

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

        # 1. Exact Match Rank
        exact_first_hit = None
        for rank_idx, passage in enumerate(all_ranked):
            for gold_text in gold_ev_texts:
                if is_exact_match(passage, gold_text):
                    if exact_first_hit is None:
                        exact_first_hit = rank_idx + 1
                    break
            if exact_first_hit is not None:
                break

        # 2. Fuzzy Match Rank
        fuzzy_first_hit = None
        for rank_idx, passage in enumerate(all_ranked):
            for gold_text in gold_ev_texts:
                if is_fuzzy_match(passage, gold_text):
                    if fuzzy_first_hit is None:
                        fuzzy_first_hit = rank_idx + 1
                    break
            if fuzzy_first_hit is not None:
                break

        # Answer in context coverage
        ans_exact = is_exact_match(full_sys_text, gold_answer) if gold_answer.strip() else False
        ans_fuzzy = is_fuzzy_match(full_sys_text, gold_answer) if gold_answer.strip() else False

        query_results.append({
            "category": cat,
            "category_name": CATEGORY_NAMES.get(cat, f"Category {cat}"),
            "exact_rr": (1.0 / exact_first_hit) if exact_first_hit is not None else 0.0,
            "fuzzy_rr": (1.0 / fuzzy_first_hit) if fuzzy_first_hit is not None else 0.0,
            **{f"exact_r{k}": 1.0 if (exact_first_hit is not None and exact_first_hit <= k) else 0.0 for k in k_values},
            **{f"fuzzy_r{k}": 1.0 if (fuzzy_first_hit is not None and fuzzy_first_hit <= k) else 0.0 for k in k_values},
            "ans_exact": 1.0 if ans_exact else 0.0,
            "ans_fuzzy": 1.0 if ans_fuzzy else 0.0,
        })

    return query_results


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
                s_date = conv_obj.get(f"{k}_date_time", "")
                date_len = len(s_date) + 3 if s_date else 0
                for turn in conv_obj.get(k, []):
                    total_chars += date_len + len(turn.get("text", ""))
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
        "latencies": latencies,
        "context_tokens": context_tokens,
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "raw_history_tokens": raw_tokens,
        "mean_context_tokens": round(mean_ctx),
        "token_compression_percent": compression,
    }


def ingest_dialogues(mem, conv_data: dict, upstream_url: str, model_name: str, enable_sleep: bool = True) -> int:
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
        s_date = conv_obj.get(f"{skey}_date_time", "")
        for turn in conv_obj.get(skey, []):
            spk = turn.get("speaker", "")
            role = "user" if spk == spk_a else "assistant"
            txt = turn.get("text", "")
            if txt.strip():
                obs_text = f"[{s_date}] {txt}" if s_date else txt
                mem.observe(role=role, text=obs_text)
                total_turns += 1

    print(f"  Ingested {total_turns} turns from {len(s_nums)} official sessions.")
    if enable_sleep:
        print("  Consolidating memory via sleep...")
        t0 = time.time()
        report = mem.sleep()
        elapsed = time.time() - t0
        print(f"  Sleep consolidation finished in {elapsed:.1f}s (batches: {report.batches}, failed: {report.failed_batches})")

        if report.failed_batches > 0:
            try:
                row = mem.store.conn.execute(
                    "SELECT sleep_error FROM episodes WHERE sleep_error IS NOT NULL LIMIT 1"
                ).fetchone()
                err_msg = row[0] if row else "Consolidation skipped"
                print(f"  [Sleep Note]: {err_msg[:120]}")
            except Exception:  # noqa: BLE001, S110
                pass
            print("  (Note: Orma automatically falls back to raw hybrid search over all indexed turns.)")
    else:
        print("  Running in Pure Episodic Memory Mode (Zero-LLM baseline).")

    return total_turns


def run_benchmark(
    dataset_path: str,
    output_path: str,
    upstream_url: str = "http://127.0.0.1:11434/v1",
    model_name: str = "qwen2.5:3b",
    conversations: int = 10,
    multilingual: bool = False,
):
    from orma import Memory, OrmaConfig
    from orma.llm import LLMClient

    if not Path(dataset_path).exists():
        print("Downloading official LoCoMo dataset from Snap Research...")
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        subprocess.run(["curl", "-L", "-o", dataset_path, url], check=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    num_convs = min(conversations, len(dataset))
    all_queries = []
    all_latencies = []
    all_context_tokens = []
    all_raw_tokens = []

    # Check if upstream LLM endpoint is live
    enable_sleep = True
    base_check_url = upstream_url.rstrip("/v1").rstrip("/") + "/"
    try:
        import urllib.request
        resp = urllib.request.urlopen(base_check_url, timeout=2.0)
        print(f"  [LLM Engine]: Upstream {upstream_url} is active (status {resp.status}).")
    except Exception as e:  # noqa: BLE001
        enable_sleep = False
        print(f"  [LLM Engine]: Upstream {upstream_url} is offline ({e}).")
        print("  --> Operating in Pure Episodic Memory Mode (Zero-LLM Baseline).")

    for idx in range(num_convs):
        conv = dataset[idx]
        cid = conv.get("sample_id") or f"conv_{idx}"
        print(f"\n{'=' * 65}")
        print(f"[{idx + 1}/{num_convs}] Running Benchmark on Official LoCoMo: {cid}")
        print("=" * 65)

        # Fresh isolated memory per conversation in tempdir
        temp_dir = Path(tempfile.gettempdir()) / f"orma_locomo_{cid}_{int(time.time())}_{idx}"
        config = OrmaConfig(data_dir=temp_dir, multilingual=multilingual, max_passages=20, sleep_batch_episodes=60)
        llm = LLMClient(upstream_url, model=model_name, timeout=600.0) if enable_sleep else None
        mem = Memory(config=config, upstream=upstream_url, model=model_name, llm=llm)

        evidence_index = build_evidence_index(conv)
        ingest_dialogues(mem, conv, upstream_url, model_name, enable_sleep=enable_sleep)

        # 1. Information Retrieval Evaluation
        print("  Evaluating Evidence & Answer Retrieval...")
        q_results = evaluate_locomo_ir(mem, conv, evidence_index)
        all_queries.extend(q_results)

        # 2. Latency & Compression Evaluation
        print("  Evaluating Latency & Compression...")
        sys_res = evaluate_latency_and_compression(mem, conv)
        all_latencies.extend(sys_res["latencies"])
        all_context_tokens.extend(sys_res["context_tokens"])
        all_raw_tokens.append(sys_res["raw_history_tokens"])

        mem.close()

    total_n = len(all_queries)
    k_vals = [1, 5, 10, 20]

    # Overall Metrics
    exact_mrr = round(statistics.mean([q["exact_rr"] for q in all_queries]) * 100, 2) if all_queries else 0
    fuzzy_mrr = round(statistics.mean([q["fuzzy_rr"] for q in all_queries]) * 100, 2) if all_queries else 0

    exact_recall = {k: round(statistics.mean([q[f"exact_r{k}"] for q in all_queries]) * 100, 2) for k in k_vals}
    fuzzy_recall = {k: round(statistics.mean([q[f"fuzzy_r{k}"] for q in all_queries]) * 100, 2) for k in k_vals}

    exact_ans_recall = round(statistics.mean([q["ans_exact"] for q in all_queries]) * 100, 2) if all_queries else 0
    fuzzy_ans_recall = round(statistics.mean([q["ans_fuzzy"] for q in all_queries]) * 100, 2) if all_queries else 0

    # System Performance Metrics
    p50_lat = round(statistics.median(all_latencies), 2) if all_latencies else 0
    p95_lat = round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 2) if all_latencies else 0
    mean_ctx = round(statistics.mean(all_context_tokens)) if all_context_tokens else 0
    mean_raw = round(statistics.mean(all_raw_tokens)) if all_raw_tokens else 0
    compression = round((1 - mean_ctx / max(1, mean_raw)) * 100, 1) if mean_raw else 0

    # Category Breakdown across full dataset
    categories_eval = {}
    for cat_id in sorted(CATEGORY_NAMES.keys()):
        cat_q = [q for q in all_queries if q["category"] == cat_id]
        if not cat_q:
            continue
        cname = CATEGORY_NAMES[cat_id]
        categories_eval[cname] = {
            "count": len(cat_q),
            "exact_mrr": round(statistics.mean([q["exact_rr"] for q in cat_q]) * 100, 2),
            "fuzzy_mrr": round(statistics.mean([q["fuzzy_rr"] for q in cat_q]) * 100, 2),
            "exact_r5": round(statistics.mean([q["exact_r5"] for q in cat_q]) * 100, 2),
            "fuzzy_r5": round(statistics.mean([q["fuzzy_r5"] for q in cat_q]) * 100, 2),
            "exact_r10": round(statistics.mean([q["exact_r10"] for q in cat_q]) * 100, 2),
            "fuzzy_r10": round(statistics.mean([q["fuzzy_r10"] for q in cat_q]) * 100, 2),
            "exact_ans_recall": round(statistics.mean([q["ans_exact"] for q in cat_q]) * 100, 2),
            "fuzzy_ans_recall": round(statistics.mean([q["ans_fuzzy"] for q in cat_q]) * 100, 2),
        }

    is_full_eval = (num_convs == len(dataset))
    eval_scope = "FULL DATASET EVALUATION" if is_full_eval else f"PILOT EVALUATION ({num_convs} OF {len(dataset)} CONVERSATIONS)"

    final_report = {
        "benchmark": "Official Snap Research LoCoMo Benchmark",
        "evaluation_scope": eval_scope,
        "dataset_source": "https://github.com/snap-research/locomo",
        "dataset_file": dataset_path,
        "conversations_evaluated": num_convs,
        "total_dataset_conversations": len(dataset),
        "queries_evaluated": total_n,
        "multilingual_embedding": multilingual,
        "evaluation_protocol": "Standard Information Retrieval (Dual Exact & Fuzzy Disclosure)",
        "results": {
            "evidence_retrieval_exact": {
                "MRR": exact_mrr,
                **{f"Recall@{k}": exact_recall[k] for k in k_vals},
                "Answer_in_Context_Recall": exact_ans_recall,
            },
            "evidence_retrieval_fuzzy": {
                "MRR": fuzzy_mrr,
                **{f"Recall@{k}": fuzzy_recall[k] for k in k_vals},
                "Answer_in_Context_Recall": fuzzy_ans_recall,
            },
            "system_performance": {
                "retrieval_latency_p50_ms": p50_lat,
                "retrieval_latency_p95_ms": p95_lat,
                "mean_context_tokens": mean_ctx,
                "mean_raw_history_tokens": mean_raw,
                "token_compression_ratio_percent": compression,
            },
            "per_category_analysis": categories_eval,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Print publication-ready report
    print("\n" + "=" * 76)
    print(f"      OFFICIAL LOCOMO BENCHMARK REPORT ({eval_scope})")
    print("=" * 76)
    print("  Benchmark Source:      Snap Research LoCoMo (Official Dataset)")
    print(f"  Conversations:         {num_convs} of {len(dataset)}")
    print(f"  Total QA Queries:      {total_n}")
    print(f"  Multilingual Mode:     {multilingual}")
    print("-" * 76)
    print("  EVIDENCE RETRIEVAL PERFORMANCE (DUAL DISCLOSURE):")
    print(f"  - Exact MRR:                   {exact_mrr}%  | Fuzzy MRR:        {fuzzy_mrr}%")
    print(f"  - Exact Recall@1:              {exact_recall[1]}%  | Fuzzy Recall@1:   {fuzzy_recall[1]}%")
    print(f"  - Exact Recall@5:              {exact_recall[5]}%  | Fuzzy Recall@5:   {fuzzy_recall[5]}%")
    print(f"  - Exact Recall@10:             {exact_recall[10]}%  | Fuzzy Recall@10:  {fuzzy_recall[10]}%")
    print(f"  - Exact Recall@20:             {exact_recall[20]}%  | Fuzzy Recall@20:  {fuzzy_recall[20]}%")
    print(f"  - Exact Answer-in-Context:     {exact_ans_recall}%  | Fuzzy Answer-in-Context: {fuzzy_ans_recall}%")
    print("-" * 76)
    print("  CATEGORY-WISE BREAKDOWN (ACROSS ALL CONVERSATIONS):")
    for cat_name, m in categories_eval.items():
        print(f"  * {cat_name:24} (N={m['count']:3d}): Exact R@10={m['exact_r10']:5.2f}% (Fuzzy={m['fuzzy_r10']:5.2f}%) | Exact Ans={m['exact_ans_recall']:5.2f}%")
    print("-" * 76)
    print("  SYSTEM EFFICIENCY & LATENCY:")
    print(f"  - Retrieval Latency p50:       {p50_lat} ms")
    print(f"  - Retrieval Latency p95:       {p95_lat} ms")
    print(f"  - Mean Context Tokens:         {mean_ctx} tokens (vs Raw: {mean_raw} tokens)")
    print(f"  - Token Compression Ratio:     {compression}% reduction")
    print("=" * 76)
    print(f"  Saved Full Report:             {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official LoCoMo Benchmark for Orma")
    parser.add_argument("--dataset", default="locomo10.json", help="Path to official locomo10.json")
    parser.add_argument("--output", default="official_locomo_report.json", help="Output JSON path")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434/v1", help="Ollama / vLLM endpoint")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name")
    parser.add_argument("--conversations", type=int, default=10, help="Number of conversations to benchmark (1-10)")
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
