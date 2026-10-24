"""
latency_smoke.py - Recall and add latency, and a needle-in-a-haystack check, with real embeddings.

No LLM needed. Keeps one sentence about the user's sister plus 2,000 filler notes, then checks that a
sister-related query still recalls the allergy sentence among them, and reports latency.

    python scripts/latency_smoke.py /tmp/smruti-smoke      # data dir is wiped and reused
"""

import shutil
import statistics
import sys
import time

from orma import Memory, SmrutiConfig

NEEDLE = "My sister Asha is allergic to peanuts, so I always check recipes when I cook for her."


def main(data_dir: str) -> None:
    shutil.rmtree(data_dir, ignore_errors=True)
    mem = Memory(path=data_dir, config=SmrutiConfig())
    mem.add(NEEDLE, at=time.time() - 90 * 86400)  # said three months ago
    start = time.perf_counter()
    for i in range(2000):
        mem.add(f"Note {i}: project {i % 97} needs a review of module {i % 13} before the {i % 28 + 1}th.")
    print(f"added 2000 filler notes in {time.perf_counter() - start:.1f}s")

    query = [{"role": "user", "content": "I'm baking cookies for my sister, any ideas?"}]
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        result = mem.recaller.build("default", query)
        timings.append((time.perf_counter() - start) * 1000)
    print(f"recall p50={statistics.median(timings):.1f}ms max={max(timings):.1f}ms tokens={result.tokens}")
    print("allergy sentence recalled:", NEEDLE in result.memory_text)

    timings = []
    for i in range(20):
        start = time.perf_counter()
        mem.add(f"Random new note number {i} about gardening and tomatoes.")
        timings.append((time.perf_counter() - start) * 1000)
    print(f"add p50={statistics.median(timings):.1f}ms max={max(timings):.1f}ms")
    mem.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smruti-smoke")
