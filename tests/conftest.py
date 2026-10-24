import zlib

import numpy as np
import pytest

from orma.config import SmrutiConfig
from orma.embed import normalize


class FakeEmbedder:
    """Deterministic bag-of-words hashing: texts that share words get high cosine similarity."""

    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().replace("_", " ").split():
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        return normalize(out)


class NoisyEmbedder(FakeEmbedder):
    """FakeEmbedder plus a component shared by every text, so unrelated texts have cosine ~0.5 like a
    real small embedding model (bge-small scores unrelated short texts ~0.4-0.6) instead of ~0."""

    def embed(self, texts):
        shared = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float32)
        return normalize(super().embed(texts) + shared)


class FakeLLM:
    """Returns queued JSON extractions (or raises queued exceptions) and a fixed chat reply."""

    def __init__(self, extractions=(), chat_reply="ok"):
        self.extractions = list(extractions)
        self.chat_reply = chat_reply
        self.calls = []
        self.json_kwargs = []  # schema, temperature, ... of each chat_json call

    def chat_json(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        self.json_kwargs.append(kwargs)
        result = self.extractions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        return self.chat_reply


@pytest.fixture
def config(tmp_path):
    # UTC, so calendar days in tests do not depend on the machine's timezone.
    return SmrutiConfig(data_dir=tmp_path, embedding_dim=FakeEmbedder.dim, timezone="UTC")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def store(config, embedder):
    from orma.store import Store

    s = Store(config, embedder)
    yield s
    s.close()


@pytest.fixture
def noisy_store(config):
    from orma.store import Store

    s = Store(config, NoisyEmbedder())
    yield s
    s.close()


def _seed_state(store, space, attribute, value, subject="user", text=None, now=1.0, valid_to=None):
    """A state backed by a real record and sentence, as sleep would leave it."""
    text = text or f"{subject} {attribute.replace('_', ' ')} {value}"
    episode_id = store.add_episode(space, "user", text, None, 1.0, store.embed(text), now=now, speaker="user")
    (sentence_id,) = store.add_sentences(space, store.episodes([episode_id])[0], [text], [store.embed(text)])
    subject_id = store.user_entity(space) if subject == "user" else (
        getattr(store.find_entity(space, subject), "id", None) or store.add_entity(space, subject, "thing"))
    found = store.find_attribute(space, attribute)
    attribute_id = found.id if found else store.add_attribute(space, attribute)
    store.mark_consolidated([episode_id], now)  # as sleep leaves it
    return store.add_state(space, subject_id, attribute_id, value, now, sentence_id, now, valid_to=valid_to)


@pytest.fixture
def seed_state():
    return _seed_state
