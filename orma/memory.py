"""
memory.py - The Python SDK: one object for a general memory layer.

    mem = Memory(upstream="http://localhost:11434/v1", model="llama3.2:3b")
    mem.add("I moved to Bangalore last month.")              # a note; searchable at once
    mem.add_file("meeting-notes.md")                          # a document, split into records
    mem.sleep()                                               # consolidate: states and events
    mem.search("where do I live?")                            # states, passages, events
    mem.build_context([{"role": "user", "content": "..."}])  # memory merged into a chat prompt

Anything can be added: chat turns (observe), notes, documents, tool output. Each record keeps its role
(user, assistant, tool, note, document), source and speaker; in user and note records "I" is the user.
"""

import time
from pathlib import Path

from orma.config import SmrutiConfig
from orma.encode import Encoder, EncodeResult
from orma.llm import LLMClient
from orma.messages import message_text
from orma.recall import Recaller, SearchResult
from orma.sleep import Sleeper, SleepReport
from orma.store import State, Store

RECORD_CHARS = 2000  # long text is kept as several records of about this size, split between paragraphs


def _parts(text: str, size: int) -> list[str]:
    """text in parts of at most about size characters, split between paragraphs where possible."""
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + 2 + len(paragraph) > size:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    return [p for p in parts + [current] if p.strip()]


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

    # --- adding --------------------------------------------------------------------

    def add(self, text: str, space: str = "default", role: str = "note", source: str = "api", speaker: str = "",
            at: float | None = None) -> list[EncodeResult]:
        """Keeps text as one or more records (long text is split between paragraphs). Notes are the user's own
        words unless a speaker is given."""
        speaker = speaker or ("user" if role in ("user", "note") else role)
        return [self.encoder.encode(space, role, part, now=at, source=source, speaker=speaker)
                for part in _parts(text, RECORD_CHARS)]

    def add_file(self, path, space: str = "default", role: str = "document", at: float | None = None) \
            -> list[EncodeResult]:
        """Keeps a text file as document records, dated by its modification time; its speaker is its name."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        at = path.stat().st_mtime if at is None else at
        return self.add(text, space, role=role, source=f"file:{path.name}", speaker=path.name, at=at)

    def observe(self, role: str, text: str, space: str = "default", outcome: str | None = None,
                now=None) -> EncodeResult:
        """One chat turn, as the proxy records it."""
        return self.encoder.encode(space, role, text, outcome, now=now, source="chat",
                                   speaker="user" if role == "user" else role)

    # --- reading -------------------------------------------------------------------

    def search(self, query: str, space: str = "default", now=None) -> SearchResult:
        return self.recaller.search(space, query, now)

    def build_context(self, messages: list[dict], space: str = "default", budget: int | None = None,
                      now=None) -> list[dict]:
        """The messages to send to a chat model: memory merged into the system prompt, trimmed to budget."""
        return self.recaller.build(space, messages, budget, now=now).messages

    def chat(self, messages: list[dict], space: str = "default", now=None) -> str:
        llm = self._require_llm()
        reply = llm.chat(self.build_context(messages, space, now=now), model=self.model)
        user_text = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
        if user_text:
            self.observe("user", user_text, space, now=now)
        self.observe("assistant", reply, space, now=now)
        return reply

    # --- consolidating -------------------------------------------------------------

    def sleep(self, space: str = "default", now=None) -> SleepReport:
        """Consolidates pending records into states and events (the slow, LLM step)."""
        return Sleeper(self.store, self.config, self._require_llm(), self.model).run(space, now)

    def rebuild(self, space: str = "default", now=None) -> SleepReport:
        """Forgets every state and event and reads all kept records again, e.g. after switching to a stronger
        model. Records kept by an older version are split into sentences first."""
        for episode in self.store.episodes_without_sentences(space):
            self.encoder.split(episode)
        self.store.clear_consolidation(space)
        return self.sleep(space, now)

    def due_spaces(self, now=None) -> list[str]:
        """Spaces whose pending records are due for consolidation."""
        return Sleeper(self.store, self.config, self.llm, self.model).due_spaces(now)

    # --- inspecting ----------------------------------------------------------------

    def stats(self, space: str = "default") -> dict[str, int]:
        return self.store.stats(space)

    def facts(self, entity: str | None = None, space: str = "default", include_closed: bool = False) -> list[State]:
        """States (what is true now; with include_closed, also what used to be), optionally of one entity."""
        subject_id = None
        if entity is not None:
            found = self.store.find_entity(space, entity)
            if found is None:
                return []
            subject_id = found.id
        if include_closed:
            return self.store.state_history(space, subject_id)
        return self.store.current_states(space, subject_id=subject_id)

    def forget(self, state_id: int, space: str = "default") -> bool:
        """Ends a state now; returns False if it is not a current state of the space."""
        found = self.store.states([state_id])
        if not found or found[0].space != space or found[0].valid_to is not None:
            return False
        self.store.close_state(state_id, time.time())
        return True

    def close(self) -> None:
        self.store.close()
        if hasattr(self.llm, "close"):
            self.llm.close()

    def _require_llm(self):
        if self.llm is None:
            raise RuntimeError("This operation needs an LLM: pass upstream=... or llm=...")
        return self.llm

