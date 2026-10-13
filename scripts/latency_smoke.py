"""
latency_smoke.py - Recall/encode latency and hub-robustness check with real embeddings.

No LLM needed: sleep extraction is stubbed. Stores three facts about the user's sister plus
2,000 filler facts about the user, then checks that a sister-related query still recalls the
allergy fact (the fan effect stops the hub from flooding the prompt) and reports latency.

    python scripts/latency_smoke.py /tmp/smruti-smoke      # data dir is wiped and reused

Reference (M1 MacBook Air, bge-small): recall p50 ~26 ms, encode p50 ~5 ms.
"""

import shutil
import statistics
import sys
import time

from smruti import Memory, SmrutiConfig

EXTRACTION = {
    "user_name": "", "procedures": [], "summary_updates": [], "events": [],
    "entities": [{"name": "Pune", "type": "city"}, {"name": "Asha", "type": "person"}],
    "facts": [
        {"subject": "user", "predicate": "lives_in", "object": "Pune", "cardinality": "one", "time": ""},
        {"subject": "user", "predicate": "has_sister", "object": "Asha", "cardinality": "many", "time": ""},
        {"subject": "Asha", "predicate": "is_allergic_to", "object": "peanuts", "cardinality": "many", "time": ""},
    ],
}


class StubLLM:
    def chat_json(self, messages, model=None, **kwargs):
        return EXTRACTION


def main(data_dir: str) -> None:
    shutil.rmtree(data_dir, ignore_errors=True)
    mem = Memory(path=data_dir, llm=StubLLM(), model="stub", config=SmrutiConfig())
    mem.observe("user", "I live in Pune and my sister Asha is allergic to peanuts")
    report = mem.sleep()
    print("sleep:", report.consolidation, "failed batches:", report.failed_batches)

    store = mem.store
    user = store.find_entity("default", "user").id
    mentioned = store.add_predicate("default", "mentioned", "many")
    start = time.perf_counter()
    for i in range(2000):
        store.add_fact("default", user, mentioned, None, f"topic {i}: note about project {i % 97} and hobby {i % 13}", 1.0, [0])
    print(f"inserted 2000 filler facts in {time.perf_counter() - start:.1f}s")

    query = [{"role": "user", "content": "I'm baking cookies for my sister, any ideas?"}]
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        result = mem.recaller.build("default", query)
        timings.append((time.perf_counter() - start) * 1000)
    facts_part = result.memory_text.split("Recent notes")[0]
    print(f"recall p50={statistics.median(timings):.1f}ms max={max(timings):.1f}ms tokens={result.tokens}")
    print("allergy fact recalled:", "- Asha is allergic to peanuts" in facts_part)

    timings = []
    for i in range(20):
        start = time.perf_counter()
        mem.observe("user", f"random new message number {i} about gardening")
        timings.append((time.perf_counter() - start) * 1000)
    print(f"encode p50={statistics.median(timings):.1f}ms max={max(timings):.1f}ms")
    mem.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smruti-smoke")
