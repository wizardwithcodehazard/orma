"""
demo_ollama.py - End-to-end check against a real local model, with and without memory.

Runs a short scripted conversation through a running `orma proxy`, consolidates it with
`orma sleep`, then asks questions that need memory and prints the answers next to the plain
model's. Works with any OpenAI-compatible runtime (Ollama, llama.cpp, LM Studio).

Requirements:
    pip install httpx
    ollama pull llama3.2:3b                                    # or your model of choice

Run with:
    export ORMA_DIR=/tmp/orma-demo && rm -rf $ORMA_DIR
    python -m orma proxy --model llama3.2:3b --sleep-poll 0     # terminal 1
    # or for llama.cpp:
    # python -m orma proxy --upstream http://127.0.0.1:9931/v1 --model MODEL --sleep-poll 0

    python scripts/demo_ollama.py                              # terminal 2

Options: --proxy (default http://127.0.0.1:8080/v1), --upstream (default
http://localhost:11434/v1), --model (default llama3.2:3b). The CLI is run as `python -m orma`,
so this checkout is used even if another `orma` command is on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import httpx


def orma(*args: str) -> None:
    start = time.time()
    out = subprocess.run([sys.executable, "-m", "orma", *args], check=False, capture_output=True, text=True,
                         encoding="utf-8")
    print(f"\n$ orma {' '.join(args)}  ({time.time() - start:.1f}s)\n{(out.stdout + out.stderr).strip()}")


def ask(url: str, model: str, messages: list[dict], label: str) -> str:
    start = time.time()
    r = httpx.post(f"{url}/chat/completions", json={"model": model, "messages": messages, "temperature": 0.2},
                   timeout=120.0)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"].strip()
    print(f"\n--- {label} ({time.time() - start:.1f}s) ---\n{reply}")
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--upstream", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()

    print(f"Targeting proxy {args.proxy}, upstream {args.upstream}, model {args.model}")

    # Conversation 1: introduce facts
    print("\n=== Conversation 1: teaching facts ===")
    ask(args.proxy, args.model, [
        {"role": "user", "content": "Hi! I'm Sahil. I live in Pune and work on compilers."},
    ], "User turn 1")

    ask(args.proxy, args.model, [
        {"role": "user", "content": "Hi! I'm Sahil. I live in Pune and work on compilers."},
        {"role": "assistant", "content": "Nice to meet you, Sahil! What kind of compilers do you work on?"},
        {"role": "user", "content": "My sister Asha is allergic to peanuts. We went trekking to Sinhagad last Saturday."},
    ], "User turn 2")

    # Background consolidation
    print("\n=== Consolidating ===")
    orma("sleep", "--model", args.model, "--upstream", args.upstream)
    orma("facts")

    # Conversation 2: an update (new home city)
    print("\n=== Conversation 2: changing a fact ===")
    ask(args.proxy, args.model, [
        {"role": "user", "content": "I moved to Bangalore last month for a new job."},
    ], "User turn 3")

    orma("sleep", "--model", args.model, "--upstream", args.upstream)
    orma("facts", "--history")

    # Recall check: ask memory-dependent questions
    print("\n=== Questions ===")
    for question in (
        "Where do I live?",
        "Can I serve peanuts at a dinner with my sister?",
        "What did I do last Saturday?",
    ):
        print(f"\n=================================================================\nQ: {question}")
        orma("recall", question)
        ask(args.proxy, args.model, [{"role": "user", "content": question}], "WITH orma")
        ask(args.upstream, args.model, [{"role": "user", "content": question}], "WITHOUT memory (raw model)")

    orma("stats")


if __name__ == "__main__":
    main()
