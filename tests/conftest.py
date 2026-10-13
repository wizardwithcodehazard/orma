import zlib

import numpy as np
import pytest

from smruti.config import SmrutiConfig
from smruti.embed import normalize


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
    # One extraction per sleep batch, so queued FakeLLM extractions map one-to-one onto batches;
    # replay-and-vote tests set sleep_votes=2 explicitly.
    return SmrutiConfig(data_dir=tmp_path, embedding_dim=FakeEmbedder.dim, sleep_votes=1)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def store(config, embedder):
    from smruti.store import Store

    s = Store(config, embedder)
    yield s
    s.close()


@pytest.fixture
def noisy_store(config):
    from smruti.store import Store

    s = Store(config, NoisyEmbedder())
    yield s
    s.close()
