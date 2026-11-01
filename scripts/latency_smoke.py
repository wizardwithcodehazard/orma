"""
latency_smoke.py - Recall and add latency, and a needle-in-a-haystack check, with real embeddings.

No LLM needed. Keeps one sentence about the user's sister plus 2,000 filler notes, then checks that a
sister-related query still recalls the allergy sentence among them, and reports latency.

    python scripts/latency_smoke.py /tmp/orma-smoke      # data dir is wiped and reused
"""

from __future__ import annotations

import shutil
import statistics
import sys
import time
from pathlib import Path

from orma import Memory, OrmaConfig

FILLER_TOPICS = [
    "Grocery list: oat milk, sourdough bread, black tea, apples, ginger.",
    "Debugged the connection pool timeout in the async worker; increasing max_overflow fixed it.",
    "Meeting notes: team sync moved to Thursdays 10am UTC to include European contributors.",
    "Book recommendation from Priya: The Design of Everyday Things, by Don Norman.",
    "Flight to Frankfurt is on Lufthansa LH755, departing at 03:15 from terminal 2.",
    "Recipe idea: roasted chickpeas with cumin, smoked paprika and lime juice.",
    "Replaced the front brake pads on the bicycle; 15mm wrench needed for the axle nuts.",
    "Drafting RFC on structured logging: JSON format with trace_id propagation over HTTP headers.",
    "Dentist appointment rescheduled to next Tuesday at 4:30pm with Dr. Rao.",
    "Ideas for the weekend hike: Rajmachi fort trail via Lonavala, about 14km round trip.",
]


def main(data_dir: str) -> None:
    p = Path(data_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {p}")
    config = OrmaConfig(data_dir=p)
    mem = Memory(config=config)

    # 1. Target needle
    print("Writing needle...")
    mem.add("My sister Asha is allergic to peanuts.", role="note")

    # 2. Filler records
    total = 2000
    print(f"Writing {total} filler notes...")
    start_all = time.time()
    add_times: list[float] = []
    for i in range(total):
        topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
        text = f"Note #{i + 1}: {topic}"
        t0 = time.time()
        mem.add(text, role="note")
        add_times.append((time.time() - t0) * 1000)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} notes added...")
    print(f"Added {total} notes in {time.time() - start_all:.2f}s "
          f"(median: {statistics.median(add_times):.1f}ms, p95: {statistics.quantiles(add_times, n=20)[18]:.1f}ms)")

    # 3. Recall check
    query = "What is Asha allergic to?"
    print(f"\nRecalling: '{query}'...")
    recall_times: list[float] = []
    found = False
    for _ in range(5):
        t0 = time.time()
        res = mem.search(query)
        recall_times.append((time.time() - t0) * 1000)
        sentences = [s.text for s, _ in res.passages]
        if any("allergic" in s for s in sentences):
            found = True

    print(f"Recall median: {statistics.median(recall_times):.1f}ms "
          f"(p95: {statistics.quantiles(recall_times, n=20)[18]:.1f}ms)")
    print(f"Needle found in top passages: {found}")
    assert found, "Target sentence not found among 2000 filler notes!"

    mem.close()
    print("Smoke test passed successfully!")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/orma-smoke")
