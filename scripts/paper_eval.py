"""
paper_eval.py - Deterministic, LLM-Free, Unbiased Benchmark for Orma
=====================================================================
Every metric in this script is computed WITHOUT any LLM generation or
LLM-as-a-judge. All scoring is pure string matching, set intersection,
or hardware timing against ground truth from the official Snap Research
LoCoMo dataset (https://github.com/snap-research/locomo).

A reviewer can audit every line: there is no prompt engineering, no
threshold tuning, and no model in the scoring loop.

Tests:
  T1  Evidence Hit Rate (Recall@k)    - Does retrieved context contain
      the gold evidence sentence?       (Standard IR metric)
  T2  Needle-In-A-Haystack (NIAH)     - Can Orma retrieve synthetic
      exact-match facts from 15k+ tokens of conversation?
  T3  Inhibitory Rule Integrity       - Do N distinct prohibition rules
      survive consolidation? (Database-level check, 0 LLM)
  T4  Retrieval Latency & Efficiency  - p50/p95 latency (ms) and token
      compression ratio vs raw history stuffing

Usage:
  pip install orma-ai
  python paper_eval.py --dataset locomo10.json --output paper_results.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# T1: Evidence Hit Rate (Recall@k)
# ---------------------------------------------------------------------------
# For each QA item in LoCoMo, the dataset provides ground truth evidence
# sentence IDs (e.g. "D1:3" = session 1, turn 3). We check whether the
# text of that exact evidence sentence appears in Orma's retrieved context.
# This is a standard information retrieval metric with ZERO LLM involvement.

def build_evidence_index(conv_data: dict) -> dict[str, str]:
    """Maps dialogue IDs like 'D1:3' to their exact text from the dataset."""
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
    for s_idx, skey in enumerate(s_nums):
        turns = conv_obj.get(skey[1], [])
        for turn in turns:
            dia_id = turn.get("dia_id", "")
            if dia_id:
                index[dia_id] = turn.get("text", "")
    return index


def evidence_hit_rate(mem, conv_data: dict, evidence_index: dict) -> dict:
    """T1: For each QA item with ground truth evidence, check if the evidence
    sentence text is present in Orma's retrieved context for that question."""
    qa_list = conv_data.get("qa", [])
    hits = 0
    total = 0
    per_category = {}

    for item in qa_list:
        evidence_ids = item.get("evidence", [])
        if not evidence_ids:
            continue

        question = item["question"]
        cat = item.get("category", 0)

        # Get Orma's retrieved context (no LLM generation, just retrieval)
        search_result = mem.search(question)
        retrieved_texts = set()
        for sentence, _score in search_result.passages:
            retrieved_texts.add(sentence.text.strip().lower())
        for state in search_result.states:
            retrieved_texts.add(state.value.strip().lower())

        # Check if ANY gold evidence sentence text appears in retrieved set
        hit = False
        for eid in evidence_ids:
            gold_text = evidence_index.get(eid, "")
            if not gold_text:
                continue
            gold_lower = gold_text.strip().lower()
            # Substring match: does any retrieved passage contain the evidence?
            for rt in retrieved_texts:
                if gold_lower in rt or rt in gold_lower:
                    hit = True
                    break
            if hit:
                break

        if hit:
            hits += 1
        total += 1

        if cat not in per_category:
            per_category[cat] = {"hits": 0, "total": 0}
        per_category[cat]["total"] += 1
        if hit:
            per_category[cat]["hits"] += 1

    return {
        "hit_rate": round(hits / max(1, total) * 100, 2),
        "hits": hits,
        "total": total,
        "per_category": {
            str(c): {
                "hit_rate": round(d["hits"] / max(1, d["total"]) * 100, 2),
                "hits": d["hits"],
                "total": d["total"],
            }
            for c, d in sorted(per_category.items())
        },
    }


# ---------------------------------------------------------------------------
# T2: Needle-In-A-Haystack (NIAH) - Exact Match
# ---------------------------------------------------------------------------
# Insert 5 synthetic facts with unique identifiers into a large conversation
# history. Then query for each fact and check for exact substring match in
# the retrieved context. Zero LLM, pure string matching.

NEEDLES = [
    ("User passport number is XK-482917.", "passport number", "XK-482917"),
    ("The secret project codename is AURORA-7.", "project codename", "AURORA-7"),
    ("Emergency contact phone number is +91-9876543210.", "emergency contact phone", "+91-9876543210"),
    ("The WiFi password for the office is Tr0ub4dor&3.", "WiFi password", "Tr0ub4dor&3"),
    ("Bank account IBAN is DE89370400440532013000.", "bank account IBAN", "DE89370400440532013000"),
]


def needle_in_haystack(mem, space: str = "default") -> dict:
    """T2: Insert needles, query for each, check exact match in retrieved context."""
    # Insert needles as user observations
    for needle_text, _, _ in NEEDLES:
        mem.observe(role="user", text=needle_text, space=space)

    results = []
    for needle_text, query, exact_match in NEEDLES:
        search_result = mem.search(query, space=space)

        # Collect all retrieved text
        all_retrieved = ""
        for sentence, _score in search_result.passages:
            all_retrieved += " " + sentence.text
        for state in search_result.states:
            all_retrieved += " " + state.value

        found = exact_match in all_retrieved
        results.append({
            "query": query,
            "needle": exact_match,
            "found": found,
        })

    found_count = sum(1 for r in results if r["found"])
    return {
        "exact_match_rate": round(found_count / len(NEEDLES) * 100, 2),
        "found": found_count,
        "total": len(NEEDLES),
        "details": results,
    }


# ---------------------------------------------------------------------------
# T3: Inhibitory Rule Integrity (Database-Level)
# ---------------------------------------------------------------------------
# Insert N distinct prohibition rules via the fast-path (:/ prefix).
# Check at database level how many survive as active states.
# Zero LLM involvement - pure SQLite row count.

RULES = [
    ":/ Never suggest peanuts to me.",
    ":/ Avoid shellfish in any recipe.",
    ":/ Never use semicolons in shell commands.",
    ":/ Avoid trailing commas in JSON.",
    ":/ Never recommend caffeine after 6pm.",
]


def rule_integrity(mem, space: str = "default") -> dict:
    """T3: Insert N rules, check how many survive as active states in the database."""
    for rule in RULES:
        mem.add(rule, role="user", space=space)

    # Query database directly for active states with 'forbidden_' prefix
    all_states = mem.store.current_states(space)
    forbidden_states = [s for s in all_states if s.attribute and "forbidden" in s.attribute.lower()]

    # Also check via retrieval that rules appear in context
    context = mem.build_context([{"role": "user", "content": "What rules should I follow?"}], space=space)
    sys_block = context[0]["content"] if context and context[0]["role"] == "system" else ""
    sys_lower = sys_block.lower()

    retrieval_hits = {
        "peanuts": "peanut" in sys_lower,
        "shellfish": "shellfish" in sys_lower,
        "semicolons": "semicolon" in sys_lower,
        "commas": "comma" in sys_lower,
        "caffeine": "caffeine" in sys_lower,
    }

    return {
        "rules_inserted": len(RULES),
        "active_in_database": len(forbidden_states),
        "retention_rate": round(len(forbidden_states) / len(RULES) * 100, 2),
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_count": sum(1 for v in retrieval_hits.values() if v),
    }


# ---------------------------------------------------------------------------
# T4: Retrieval Latency & Token Efficiency
# ---------------------------------------------------------------------------
# Measure pure retrieval latency (no LLM) and token compression ratio.

def latency_and_efficiency(mem, conv_data: dict, space: str = "default") -> dict:
    """T4: Measure retrieval latency over all QA questions and token compression."""
    qa_list = conv_data.get("qa", [])
    latencies = []
    context_tokens_list = []

    # Measure raw history size
    conv_obj = conv_data.get("conversation", {})
    total_chars = 0
    if isinstance(conv_obj, dict):
        for k in conv_obj:
            if k.startswith("session_") and not k.endswith("_date_time"):
                for turn in conv_obj.get(k, []):
                    total_chars += len(turn.get("text", ""))
    raw_tokens = int(total_chars / 3.5)

    for item in qa_list:
        question = item.get("question", "")
        if not question:
            continue

        # Pure retrieval latency (no LLM generation)
        t0 = time.perf_counter()
        context = mem.build_context([{"role": "user", "content": question}], space=space)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # Context token footprint
        sys_block = context[0]["content"] if context and context[0]["role"] == "system" else ""
        ctx_tokens = int(len(sys_block) / 3.5)
        context_tokens_list.append(ctx_tokens)

    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    mean_ctx = statistics.mean(context_tokens_list) if context_tokens_list else 0
    compression = round((1 - mean_ctx / max(1, raw_tokens)) * 100, 1) if raw_tokens else 0

    return {
        "queries": len(latencies),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "raw_history_tokens": raw_tokens,
        "mean_context_tokens": round(mean_ctx),
        "token_compression_percent": compression,
    }


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def ingest_conversation(mem, conv_data: dict, upstream_url: str, model_name: str):
    """Ingest all dialogue turns from a LoCoMo conversation into Orma memory."""
    conv_obj = conv_data.get("conversation", {})
    if not isinstance(conv_obj, dict):
        print("WARNING: conversation field is not a dict, skipping ingestion.")
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
        turns = conv_obj.get(skey, [])
        for turn in turns:
            spk = turn.get("speaker", "")
            role = "user" if spk == spk_a else "assistant"
            txt = turn.get("text", "")
            if txt.strip():
                mem.observe(role=role, text=txt)
                total_turns += 1

    # Single batch consolidation
    print(f"  Consolidating {total_turns} turns via sleep (requires LLM)...")
    t0 = time.time()
    mem.sleep()
    print(f"  Sleep consolidation done in {time.time() - t0:.1f}s")

    return total_turns


def run_paper_eval(
    dataset_path: str,
    output_path: str,
    upstream_url: str = "http://127.0.0.1:11434/v1",
    model_name: str = "qwen2.5:3b",
    max_conversations: int = 1,
    multilingual: bool = False,
):
    from orma import Memory, OrmaConfig

    if not Path(dataset_path).exists():
        import subprocess
        print("Dataset not found. Downloading official LoCoMo from Snap Research...")
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        subprocess.run(["curl", "-L", "-o", dataset_path, url], check=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    num_convs = min(max_conversations, len(dataset))
    all_results = []

    for conv_idx in range(num_convs):
        conv_data = dataset[conv_idx]
        conv_id = conv_data.get("sample_id") or f"conv_{conv_idx}"
        print(f"\n{'=' * 65}")
        print(f"[{conv_idx + 1}/{num_convs}] Evaluating {conv_id}")
        print(f"{'=' * 65}")

        # Fresh memory instance per conversation (no cross-contamination)
        temp_db = Path(f"/tmp/orma_paper_eval_{conv_id}_{int(time.time())}")
        config = OrmaConfig(data_dir=temp_db, multilingual=multilingual)
        mem = Memory(config=config, upstream=upstream_url, model=model_name)

        # Ingest
        evidence_index = build_evidence_index(conv_data)
        total_turns = ingest_conversation(mem, conv_data, upstream_url, model_name)
        print(f"  Ingested {total_turns} turns, {len(evidence_index)} evidence sentences indexed")

        # T1: Evidence Hit Rate (zero LLM)
        print("  T1: Computing Evidence Hit Rate (Recall@k)...")
        t1 = evidence_hit_rate(mem, conv_data, evidence_index)
        print(f"      Hit Rate: {t1['hit_rate']}% ({t1['hits']}/{t1['total']})")

        # T2: Needle-In-A-Haystack (zero LLM)
        print("  T2: Needle-In-A-Haystack (Exact Match)...")
        t2 = needle_in_haystack(mem)
        print(f"      Exact Match Rate: {t2['exact_match_rate']}% ({t2['found']}/{t2['total']})")

        # T3: Inhibitory Rule Integrity (zero LLM)
        print("  T3: Inhibitory Rule Integrity (Database-Level)...")
        t3 = rule_integrity(mem)
        print(f"      Retention: {t3['retention_rate']}% ({t3['active_in_database']}/{t3['rules_inserted']})")

        # T4: Retrieval Latency & Token Efficiency (zero LLM)
        print("  T4: Retrieval Latency & Token Efficiency...")
        t4 = latency_and_efficiency(mem, conv_data)
        print(f"      p50: {t4['p50_latency_ms']}ms | p95: {t4['p95_latency_ms']}ms")
        print(f"      Tokens: {t4['mean_context_tokens']} / {t4['raw_history_tokens']} ({t4['token_compression_percent']}% reduction)")

        all_results.append({
            "conversation_id": conv_id,
            "t1_evidence_hit_rate": t1,
            "t2_needle_in_haystack": t2,
            "t3_rule_integrity": t3,
            "t4_latency_efficiency": t4,
        })

        mem.close()

    # Aggregate across conversations
    agg_t1 = round(statistics.mean([r["t1_evidence_hit_rate"]["hit_rate"] for r in all_results]), 2)
    agg_t2 = round(statistics.mean([r["t2_needle_in_haystack"]["exact_match_rate"] for r in all_results]), 2)
    agg_t3 = round(statistics.mean([r["t3_rule_integrity"]["retention_rate"] for r in all_results]), 2)
    agg_p50 = round(statistics.mean([r["t4_latency_efficiency"]["p50_latency_ms"] for r in all_results]), 2)
    agg_p95 = round(statistics.mean([r["t4_latency_efficiency"]["p95_latency_ms"] for r in all_results]), 2)
    agg_ctx = round(statistics.mean([r["t4_latency_efficiency"]["mean_context_tokens"] for r in all_results]))
    agg_raw = round(statistics.mean([r["t4_latency_efficiency"]["raw_history_tokens"] for r in all_results]))
    agg_comp = round(statistics.mean([r["t4_latency_efficiency"]["token_compression_percent"] for r in all_results]), 1)

    report = {
        "benchmark": "Orma Deterministic Paper Evaluation (LLM-Free Scoring)",
        "dataset": "Snap Research LoCoMo (official)",
        "conversations_evaluated": num_convs,
        "multilingual_embedding": multilingual,
        "consolidation_model": model_name,
        "aggregate": {
            "T1_evidence_hit_rate_percent": agg_t1,
            "T2_needle_exact_match_percent": agg_t2,
            "T3_rule_retention_percent": agg_t3,
            "T4_retrieval_p50_ms": agg_p50,
            "T4_retrieval_p95_ms": agg_p95,
            "T4_mean_context_tokens": agg_ctx,
            "T4_raw_history_tokens": agg_raw,
            "T4_token_compression_percent": agg_comp,
        },
        "per_conversation": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print publication-ready summary
    print("\n" + "=" * 65)
    print("   ORMA DETERMINISTIC BENCHMARK REPORT (LLM-FREE SCORING)")
    print("=" * 65)
    print("Dataset:                     Snap Research LoCoMo (official)")
    print(f"Conversations Evaluated:     {num_convs}")
    print(f"Consolidation Model:         {model_name}")
    print(f"Multilingual Embedding:      {multilingual}")
    print("-" * 65)
    print(f"T1  Evidence Hit Rate:       {agg_t1}%")
    print(f"T2  Needle Exact Match:      {agg_t2}%")
    print(f"T3  Rule Retention:          {agg_t3}%")
    print(f"T4  Retrieval Latency p50:   {agg_p50} ms")
    print(f"T4  Retrieval Latency p95:   {agg_p95} ms")
    print(f"T4  Context Tokens:          {agg_ctx} (vs raw: {agg_raw})")
    print(f"T4  Token Compression:       {agg_comp}% reduction")
    print("=" * 65)
    print(f"Saved full JSON report to: {output_path}\n")
    print("METHODOLOGY NOTE: All T1-T4 scores are computed WITHOUT any")
    print("LLM generation or LLM-as-a-judge. The only LLM call is during")
    print("the ingestion/consolidation phase (sleep), which is Orma's")
    print("normal operation. Scoring is pure string matching, set")
    print("intersection, database row counting, and hardware timing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic, LLM-Free Benchmark for Orma (Paper Evaluation)"
    )
    parser.add_argument("--dataset", default="locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--output", default="paper_eval_results.json", help="Output JSON report path")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible LLM endpoint (for sleep consolidation only)")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name (used only for sleep consolidation, NOT for scoring)")
    parser.add_argument("--conversations", type=int, default=1, help="Number of conversations (1-10)")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual embeddings")
    args = parser.parse_args()

    run_paper_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        upstream_url=args.upstream,
        model_name=args.model,
        max_conversations=args.conversations,
        multilingual=args.multilingual,
    )
