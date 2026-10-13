"""
config.py - Every tunable of the memory layer in one place, so each component can be
ablated or swept in experiments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    env_dir = os.environ.get("SMRUTI_DIR")
    return Path(env_dir).expanduser().resolve() if env_dir else Path.home() / ".smruti"


@dataclass
class SmrutiConfig:
    # Storage
    data_dir: Path = field(default_factory=default_data_dir)
    db_filename: str = "memory.db"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Encode: adaptive novelty gate
    novelty_percentile: float = 10.0
    novelty_warmup: int = 50
    novelty_window: int = 500

    # Recall
    budget_tokens: int = 4096
    chars_per_token: float = 3.5
    entity_link_k: int = 8
    entity_link_threshold: float = 0.6
    max_hops: int = 2
    hop_decay: float = 0.6
    search_k: int = 20
    recency_half_life_days: float = 14.0
    core_min_confidence: float = 0.7
    min_recent_turns: int = 2

    # Sleep
    sleep_after_episodes: int = 20
    sleep_idle_seconds: float = 120.0
    sleep_batch_episodes: int = 12
    sleep_max_attempts: int = 3
    entity_merge_threshold: float = 0.85
    predicate_merge_threshold: float = 0.8
    procedure_merge_threshold: float = 0.85
    summary_max_tokens: int = 200
    episode_retention_days: float = 7.0
    procedure_ttl_days: float = 30.0
    procedure_min_confidence: float = 0.5

    # Ablation switches
    enable_novelty_gate: bool = True
    enable_graph: bool = True
    enable_core_block: bool = True

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_filename
