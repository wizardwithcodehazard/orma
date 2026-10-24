"""
config.py - Every tunable of the memory layer in one place, so each component can be
ablated or swept in experiments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    env_dir = os.environ.get("ORMA_DIR") or os.environ.get("SMRUTI_DIR")
    return Path(env_dir).expanduser().resolve() if env_dir else Path.home() / ".orma"


@dataclass
class OrmaConfig:
    # Storage
    data_dir: Path = field(default_factory=default_data_dir)
    db_filename: str = "memory.db"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Calendar: the user's timezone ('' = this machine's, 'UTC', or an IANA name like Asia/Kolkata) decides what
    # day a timestamp falls on; date_order ('DMY', 'MDY', 'YMD' or '' for dateparser's default) reads "08/10/2026".
    timezone: str = ""
    date_order: str = ""

    # Add: every record is kept in full; its sentences are the unit of recall and of citation in sleep.
    novelty_duplicate_similarity: float = 0.97  # a record this close to one already kept is counted, not stored
    novelty_warmup: int = 50  # samples before recall calibrates similarity against the space's own distribution
    novelty_window: int = 500
    max_sentence_chars: int = 400  # longer sentences are split into pieces of at most this size
    max_tool_chars: int = 4000  # tool output kept per recorded turn by the proxy (head and tail)

    # Recall (no LLM, no writes)
    budget_tokens: int = 4096
    chars_per_token: float = 3.5
    search_k: int = 30
    rrf_k: int = 60
    # Similarity floor = this percentile of the space's nearest-neighbour similarity distribution: what an
    # arbitrary record reaches by chance (small embedders give unrelated texts ~0.5), so it carries no evidence.
    similarity_floor_percentile: float = 50.0
    base_level_decay: float = 0.5  # ACT-R d: a memory's strength is the sum of (age_days + 1) ** -d over its uses
    use_weight: float = 0.3  # how much that strength can move a memory's rank: a bounded boost, never a filter
    hawkes_alpha: float = 0.4  # amplitude of self-exciting temporal burst
    hawkes_half_life_seconds: float = 1800.0  # 30-min conversational excitation half-life
    ebbinghaus_stability_days: float = 2.0  # base memory stability in days
    ebbinghaus_gain: float = 0.5  # stability scaling per spaced rehearsal / citation
    max_passages: int = 12
    max_events: int = 6
    max_query_states: int = 8  # states of others matched by the query (the user's own states are always shown)
    activation_min_threshold: float = 0.05  # minimum spreading activation score for associated states
    state_budget_share: float = 0.25  # "Now true" takes at most this share of the budget
    min_recent_turns: int = 2
    recall_roles: tuple[str, ...] = ("user", "note", "document", "tool")  # whose sentences recall shows

    # Sleep (one LLM call per batch, in the background)
    sleep_after_episodes: int = 20
    sleep_idle_seconds: float = 120.0
    sleep_batch_episodes: int = 12
    sleep_max_attempts: int = 3
    sleep_prompt_tokens: int = 3000
    context_chars: int = 200  # an assistant turn is shown to sleep as this much context, never as evidence
    citable_roles: tuple[str, ...] = ("user", "note", "document", "tool")  # whose lines a memory may cite
    sleep_context_states: int = 20
    sleep_context_attributes: int = 30
    sleep_lease_seconds: float = 600.0
    entity_merge_threshold: float = 0.85
    attribute_merge_threshold: float = 0.8
    # The two closed word classes the holder check needs, English by default (replace both for another language):
    # words by which a speaker means themself, and words that refer back to someone already named.
    first_person_words: tuple[str, ...] = ("i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves")
    third_person_words: tuple[str, ...] = ("she", "her", "hers", "herself", "he", "him", "his", "himself", "they",
                                           "them", "their", "theirs", "it", "its", "itself")

    # Ablation switches
    enable_semantic: bool = True  # sleep builds states and events; off = recall over kept sentences only (baseline)
    enable_novelty_gate: bool = True
    enable_similarity_weighting: bool = True
    enable_base_level: bool = True  # rank by recency and frequency of use (ACT-R); off = by match alone
    enable_hawkes: bool = True  # temporal self-exciting burst intensity for conversational momentum
    enable_ebbinghaus: bool = True  # spaced repetition memory stability scaling
    enable_spreading_activation: bool = True  # ACT-R associative spreading activation with Fan Effect
    enable_schema_decoding: bool = True  # constrain extraction to the JSON schema (else plain JSON mode)
    enable_grounding: bool = True  # a state's value must be in (or paraphrased by) the line it cites
    enable_date_phrases: bool = True  # resolve date words ("last Saturday") against when they were said (else ISO only)

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_filename


# Backwards compatibility alias
SmrutiConfig = OrmaConfig
