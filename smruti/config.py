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
    max_hops: int = 2
    hop_decay: float = 0.6
    search_k: int = 20
    rrf_k: int = 60
    # Similarity floor = this percentile of the space's nearest-neighbour similarity distribution (from
    # the novelty buffer); the median is what a typical turn's closest memory reaches, i.e. the noise level.
    similarity_floor_percentile: float = 50.0
    max_candidate_facts: int = 64
    max_candidate_events: int = 16
    recency_half_life_days: float = 14.0
    core_min_confidence: float = 0.7
    min_recent_turns: int = 2
    recall_note_chars: int = 300  # longest pending-episode note shown in recall

    # Sleep
    sleep_after_episodes: int = 20
    sleep_idle_seconds: float = 120.0
    sleep_batch_episodes: int = 12
    sleep_max_attempts: int = 3
    sleep_prompt_tokens: int = 3000
    sleep_max_episode_chars: int = 1500
    max_tool_chars: int = 4000  # tool output kept per recorded turn (head and tail)
    sleep_context_predicates: int = 30
    sleep_lease_seconds: float = 600.0
    entity_merge_threshold: float = 0.85
    predicate_merge_threshold: float = 0.8
    procedure_merge_threshold: float = 0.85
    summary_max_tokens: int = 200
    episode_retention_days: float = 7.0
    procedure_ttl_days: float = 30.0
    procedure_min_confidence: float = 0.5
    # Replay and vote (an experiment; each extra vote costs a full extraction): each batch is extracted
    # sleep_votes times (the first pass at temperature 0, replays at sleep_vote_temperature). The first pass is
    # consolidated; a fact a replay also states (rendered-text cosine >= vote_match_threshold) counts as one
    # extra observation.
    sleep_votes: int = 1  # 1 = single pass, no voting
    sleep_vote_temperature: float = 0.7
    vote_match_threshold: float = 0.8

    # Ablation switches
    enable_novelty_gate: bool = True
    enable_graph: bool = True
    enable_core_block: bool = True
    enable_similarity_weighting: bool = True
    enable_episode_recall: bool = True
    enable_schema_decoding: bool = True  # constrain extraction to JSON_SCHEMA (else plain JSON mode)
    enable_grounding: bool = True  # drop extracted facts, entities and user names the batch's log does not contain
    enable_events: bool = True  # store dated events during sleep and recall them

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_filename
