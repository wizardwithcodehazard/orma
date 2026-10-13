"""
memory.py - Python SDK: one object wiring the store, encoder, recaller and sleeper together.

    mem = Memory(upstream="http://localhost:11434/v1", model="llama3.2:3b")
    reply = mem.chat([{"role": "user", "content": "I live in Pune"}])
    mem.sleep()
"""

import time
from pathlib import Path

from smruti.config import SmrutiConfig
from smruti.encode import Encoder, EncodeResult
from smruti.llm import LLMClient
from smruti.messages import message_text
from smruti.recall import Recaller
from smruti.sleep import Sleeper, SleepReport
from smruti.store import Fact, Store


class Memory:
    def __init__(self, path=None, upstream: str | None = None, model: str | None = None,
                 config: SmrutiConfig | None = None, embedder=None, llm=None):
        self.config = config or SmrutiConfig()
        if path is not None:
            self.config.data_dir = Path(path)
        self.store = Store(self.config, embedder)
        self.encoder = Encoder(self.store, self.config)
        self.recaller = Recaller(self.store, self.config)
        self.llm = llm or (LLMClient(upstream, model=model) if upstream else None)
        self.model = model

    def build_context(self, messages: list[dict], space: str = "default", budget: int | None = None,
                      now=None) -> list[dict]:
        """The messages to send upstream: memory merged into the system prompt, trimmed to budget."""
        return self.recaller.build(space, messages, budget, now=now).messages

    def observe(self, role: str, text: str, space: str = "default", outcome: str | None = None,
                now=None) -> EncodeResult:
        return self.encoder.encode(space, role, text, outcome, now=now)

    def chat(self, messages: list[dict], space: str = "default", now=None) -> str:
        llm = self._require_llm()
        reply = llm.chat(self.build_context(messages, space, now=now), model=self.model)
        user_text = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
        if user_text:
            self.observe("user", user_text, space, now=now)
        self.observe("assistant", reply, space, now=now)
        return reply

    def sleep(self, space: str = "default", now=None) -> SleepReport:
        return Sleeper(self.store, self.config, self._require_llm(), self.model).run(space, now)

    def due_spaces(self, now=None) -> list[str]:
        """Spaces whose pending turns are due for consolidation."""
        return Sleeper(self.store, self.config, self.llm, self.model).due_spaces(now)

    def stats(self, space: str = "default") -> dict[str, int]:
        return self.store.stats(space)

    def facts(self, entity: str | None = None, space: str = "default", include_closed: bool = False) -> list[Fact]:
        subject_id = None
        if entity is not None:
            found = self.store.find_entity(space, entity)
            if found is None:
                return []
            subject_id = found.id
        if include_closed:
            return self.store.fact_history(space, subject_id)
        return self.store.current_facts(space, subject_id=subject_id)

    def forget(self, fact_id: int, space: str = "default") -> bool:
        """Ends a fact's validity now; returns False if it is not a current fact of the space."""
        found = self.store.facts([fact_id])
        if not found or found[0].space != space or found[0].valid_to is not None:
            return False
        self.store.close_fact(fact_id, time.time())
        return True

    def close(self) -> None:
        self.store.close()
        if hasattr(self.llm, "close"):
            self.llm.close()

    def _require_llm(self):
        if self.llm is None:
            raise RuntimeError("This operation needs an LLM: pass upstream=... or llm=...")
        return self.llm
