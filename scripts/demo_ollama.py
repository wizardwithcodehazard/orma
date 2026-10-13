"""
demo_ollama.py - End-to-end check against a real local model, with and without memory.

Runs a short scripted conversation through a running `smruti proxy`, consolidates it with
`smruti sleep`, then asks questions that need memory and prints the answers next to plain
Ollama's. Use a throwaway memory directory so your real memory is untouched:

    export SMRUTI_DIR=/tmp/smruti-demo && rm -rf $SMRUTI_DIR
    smruti proxy --model llama3.2:3b --sleep-poll 0          # terminal 1
    python scripts/demo_ollama.py                            # terminal 2

Options: --proxy (default http://localhost:8080/v1), --ollama (default
http://localhost:11434/v1), --model (default llama3.2:3b).
"""

import argparse
import shutil
import subprocess
import time

import httpx

CONVERSATION_1 = [
    "Hi! I'm Sahil. I live in Pune and I work as a backend developer at a fintech startup.",
    "My sister Asha is allergic to peanuts, so I always double-check recipes when I cook for her.",
    "These days I'm learning Rust in the evenings.",
    "Last Saturday I went trekking to Sinhagad fort with Asha, it was amazing.",
]
CONVERSATION_2 = ["Quick update: I moved to Bangalore last month for a new job."]
QUESTIONS = [
    "I'm baking cookies for my sister this weekend. Any recipe ideas?",
    "Which city should I search in for good weekend cafes near me?",
    "When did I last go trekking, and who came with me?",
]


def ask(base: str, model: str, messages: list[dict], label: str) -> str:
    start = time.time()
    response = httpx.post(f"{base}/chat/completions", timeout=600,
                          json={"model": model, "messages": messages, "temperature": 0})
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    print(f"\n[{label}] ({time.time() - start:.1f}s)\n{text[:600]}")
    return text


def converse(proxy: str, model: str, turns: list[str], label: str) -> None:
    messages: list[dict] = []
    for user in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ask(proxy, model, messages, f"{label} | {user[:50]}")})


def smruti(*args: str) -> None:
    start = time.time()
    out = subprocess.run([shutil.which("smruti") or "smruti", *args], check=False, capture_output=True, text=True)
    print(f"\n$ smruti {' '.join(args)}  ({time.time() - start:.1f}s)\n{(out.stdout + out.stderr).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", default="http://localhost:8080/v1")
    parser.add_argument("--ollama", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()

    print("=== Conversation 1 (through the proxy)")
    converse(args.proxy, args.model, CONVERSATION_1, "conv1")
    smruti("sleep", "--model", args.model, "--upstream", args.ollama)
    smruti("facts")

    print("\n=== Conversation 2 (a change)")
    converse(args.proxy, args.model, CONVERSATION_2, "conv2")
    smruti("sleep", "--model", args.model, "--upstream", args.ollama)
    smruti("facts", "--history")

    print("\n=== New chat: questions that need memory")
    for question in QUESTIONS:
        smruti("recall", question)
        ask(args.proxy, args.model, [{"role": "user", "content": question}], "WITH smruti")
        ask(args.ollama, args.model, [{"role": "user", "content": question}], "WITHOUT memory (plain Ollama)")
    smruti("stats")


if __name__ == "__main__":
    main()
