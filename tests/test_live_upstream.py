import os
import httpx
import pytest
from orma import Memory, OrmaConfig

# Check if local llama.cpp or Ollama is running
CANDIDATE_UPSTREAMS = [
    ("http://127.0.0.1:9931/v1", "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"),
    ("http://localhost:11434/v1", "llama3.2:3b"),
]

def find_live_upstream():
    for url, model in CANDIDATE_UPSTREAMS:
        try:
            r = httpx.get(f"{url.rstrip('/v1')}/health", timeout=1.0)
            if r.status_code == 200:
                return url, model
        except Exception:
            pass
        try:
            # Try a lightweight chat ping
            r = httpx.post(
                f"{url}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                timeout=2.0,
            )
            if r.status_code == 200:
                return url, model
        except Exception:
            pass
    return None, None


@pytest.fixture(scope="module")
def live_server():
    url, model = find_live_upstream()
    if not url:
        pytest.skip("No live LLM server detected (checked 127.0.0.1:9931 and localhost:11434)")
    return url, model


def test_live_chat_and_sleep_consolidation(live_server, tmp_path):
    """End-to-end integration test with a real local LLM (llama.cpp / Ollama)."""
    upstream_url, model_name = live_server
    config = OrmaConfig(data_dir=tmp_path, timezone="UTC")
    mem = Memory(upstream=upstream_url, model=model_name, config=config)

    # 1. Add conversation turns
    mem.add("Hi! I'm Sahil. I live in Pune and work on compilers.", role="user")
    mem.add("My sister Asha is allergic to peanuts.", role="user")

    # 2. Trigger sleep consolidation on the live model
    report = mem.sleep()
    assert report.batches >= 1

    # 3. Check that states were extracted from the live model
    facts = list(mem.facts(include_closed=True))
    assert len(facts) >= 1

    # 4. Prompt context injection
    context = mem.build_context([
        {"role": "user", "content": "Where do I live?"}
    ])
    assert len(context) > 1
    sys_prompt = context[0]["content"]
    assert "## Memory" in sys_prompt

    # 5. Live chat response through Memory SDK
    reply = mem.chat([
        {"role": "user", "content": "Where do I live?"}
    ])
    assert isinstance(reply, str)
    assert len(reply) > 0
    # The model should be able to answer Pune based on the injected context
    assert "pune" in reply.lower()

    mem.close()
